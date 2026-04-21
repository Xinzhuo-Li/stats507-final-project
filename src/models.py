from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def build_linear_regression() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = 8,
    min_samples_leaf: int = 3,
    random_state: int = 42,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


def naive_price_prediction(features: pd.DataFrame) -> np.ndarray:
    """Use the current adjusted close as the naive future price prediction."""
    return features["adj_close"].to_numpy()


@dataclass
class LSTMConfig:
    sequence_length: int = 20
    hidden_size: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 40
    random_state: int = 42


if nn is not None:
    class SequenceRegressor(nn.Module):  # type: ignore[misc]
        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :]).squeeze(-1)
else:
    SequenceRegressor = None  # type: ignore[assignment]


class LSTMForecaster:
    """Lightweight LSTM forecaster for tabular time series sequences."""

    def __init__(self, config: LSTMConfig | None = None) -> None:
        if torch is None:
            raise ImportError("torch is required to use the LSTM model.")

        self.config = config or LSTMConfig()
        self.model: SequenceRegressor | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.target_mean: float | None = None
        self.target_std: float | None = None

    def _build_sequences(self, x: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        seq_len = self.config.sequence_length
        sequences = []
        targets = []
        for idx in range(seq_len, len(x)):
            sequences.append(x[idx - seq_len : idx])
            if y is not None:
                targets.append(y[idx])

        sequence_array = np.asarray(sequences, dtype=np.float32)
        target_array = None if y is None else np.asarray(targets, dtype=np.float32)
        return sequence_array, target_array

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "LSTMForecaster":
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        self.feature_mean = x_train.mean(axis=0)
        self.feature_std = x_train.std(axis=0)
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
        x_scaled = (x_train - self.feature_mean) / self.feature_std
        self.target_mean = float(y_train.mean())
        self.target_std = float(y_train.std())
        if self.target_std < 1e-8:
            self.target_std = 1.0
        y_scaled = (y_train - self.target_mean) / self.target_std

        train_sequences, train_targets = self._build_sequences(x_scaled, y_scaled)
        if train_targets is None or len(train_targets) == 0:
            raise ValueError("Not enough rows to build LSTM training sequences.")

        dataset = TensorDataset(
            torch.tensor(train_sequences, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        self.model = SequenceRegressor(input_size=x_train.shape[1], hidden_size=self.config.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if (
            self.model is None
            or self.feature_mean is None
            or self.feature_std is None
            or self.target_mean is None
            or self.target_std is None
        ):
            raise ValueError("The LSTM model must be fit before calling predict.")

        x_scaled = (x - self.feature_mean) / self.feature_std
        sequences, _ = self._build_sequences(x_scaled)
        if len(sequences) == 0:
            return np.array([], dtype=float)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(sequences, dtype=torch.float32)).cpu().numpy()
        predictions = (predictions * self.target_std) + self.target_mean
        return predictions
