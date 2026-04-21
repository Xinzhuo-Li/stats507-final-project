from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "daily_return",
    "log_volume",
    "intraday_range",
    "lag_adj_close_1",
    "lag_adj_close_2",
    "lag_adj_close_3",
    "lag_adj_close_5",
    "lag_adj_close_10",
    "lag_return_1",
    "lag_return_2",
    "lag_return_3",
    "roll_mean_5",
    "roll_mean_10",
    "roll_std_5",
    "roll_std_10",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged, rolling, and volatility-oriented features."""
    data = df.copy()

    data["daily_return"] = data["adj_close"].pct_change()
    data["log_volume"] = np.log(data["volume"].clip(lower=1))
    data["intraday_range"] = (data["high"] - data["low"]) / data["open"]

    for lag in [1, 2, 3, 5, 10]:
        data[f"lag_adj_close_{lag}"] = data["adj_close"].shift(lag)

    for lag in [1, 2, 3]:
        data[f"lag_return_{lag}"] = data["daily_return"].shift(lag)

    for window in [5, 10]:
        data[f"roll_mean_{window}"] = data["adj_close"].rolling(window=window).mean()
        data[f"roll_std_{window}"] = data["daily_return"].rolling(window=window).std()

    return data


def add_horizon_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create the future adjusted close target for a given horizon."""
    data = df.copy()
    data["target"] = data["adj_close"].shift(-horizon)
    data["horizon"] = horizon
    return data


def prepare_model_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create a clean feature frame for a specific prediction horizon."""
    data = add_horizon_target(build_features(df), horizon)
    required_columns = ["date", "target"] + FEATURE_COLUMNS
    data = data.dropna(subset=required_columns).reset_index(drop=True)
    return data
