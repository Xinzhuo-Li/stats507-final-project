from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common regression metrics."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    non_zero_mask = np.abs(y_true) > 1e-8
    if non_zero_mask.any():
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float("nan")

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
    }


def evaluate_regimes(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Evaluate model performance in low- and high-volatility regimes."""
    rows: list[dict[str, float | int | str]] = []
    for regime_name, mask in {
        "low_volatility": predictions["roll_std_10"] < threshold,
        "high_volatility": predictions["roll_std_10"] >= threshold,
    }.items():
        regime_df = predictions.loc[mask]
        if regime_df.empty:
            continue

        metrics = regression_metrics(
            regime_df["actual"].to_numpy(),
            regime_df["prediction"].to_numpy(),
        )
        metrics.update(
            {
                "model": regime_df["model"].iloc[0],
                "horizon": int(regime_df["horizon"].iloc[0]),
                "regime": regime_name,
                "n_samples": int(len(regime_df)),
            }
        )
        rows.append(metrics)

    return pd.DataFrame(rows)
