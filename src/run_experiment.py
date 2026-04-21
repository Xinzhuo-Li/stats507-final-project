from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

from .data_utils import chronological_split, load_stock_data
from .evaluate import evaluate_regimes, regression_metrics
from .features import FEATURE_COLUMNS, prepare_model_frame
from .models import LSTMForecaster, build_linear_regression, build_random_forest, naive_price_prediction, torch
from .visualize import plot_metric_by_horizon, plot_predictions, plot_regime_metric

RANDOM_FOREST_PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 3},
    {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 3},
    {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 3},
]


def evaluate_split(
    split_df: pd.DataFrame,
    predictions: pd.Series | pd.Index | list[float],
    model_name: str,
    horizon: int,
    split_name: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    prediction_series = pd.Series(predictions, dtype=float)
    aligned_df = split_df.iloc[-len(prediction_series) :].copy().reset_index(drop=True)
    aligned_df["prediction"] = prediction_series.to_numpy()
    aligned_df["actual"] = aligned_df["target"]
    aligned_df["model"] = model_name
    aligned_df["horizon"] = horizon
    aligned_df["split"] = split_name

    metrics = regression_metrics(
        aligned_df["actual"].to_numpy(),
        aligned_df["prediction"].to_numpy(),
    )
    metrics.update(
        {
            "model": model_name,
            "horizon": horizon,
            "split": split_name,
            "n_samples": int(len(aligned_df)),
        }
    )
    return metrics, aligned_df[["date", "actual", "prediction", "roll_std_10", "model", "horizon", "split"]]


def tune_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    horizon: int,
) -> tuple[object, pd.DataFrame, dict[str, int | None]]:
    """Select Random Forest hyperparameters using validation RMSE."""
    tuning_rows: list[dict[str, float | int | None]] = []
    best_model = None
    best_params: dict[str, int | None] | None = None
    best_rmse = float("inf")

    for params in RANDOM_FOREST_PARAM_GRID:
        model = build_random_forest(**params)
        model.fit(x_train.to_numpy(), y_train.to_numpy())
        val_pred = model.predict(x_val.to_numpy())
        metrics = regression_metrics(y_val.to_numpy(), val_pred)

        tuning_rows.append(
            {
                "model": "random_forest",
                "horizon": horizon,
                "n_estimators": int(params["n_estimators"]),
                "max_depth": -1 if params["max_depth"] is None else int(params["max_depth"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "validation_rmse": float(metrics["rmse"]),
                "validation_mae": float(metrics["mae"]),
                "validation_r2": float(metrics["r2"]),
            }
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = float(metrics["rmse"])
            best_model = model
            best_params = params

    if best_model is None or best_params is None:
        raise ValueError("Random Forest tuning did not produce a valid model.")

    tuning_df = pd.DataFrame(tuning_rows).sort_values("validation_rmse").reset_index(drop=True)
    return best_model, tuning_df, best_params


def run_horizon_experiment(
    frame: pd.DataFrame,
    horizon: int,
    include_lstm: bool,
) -> tuple[list[dict[str, float | int | str]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    regime_rows: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []

    train_df, val_df, test_df = chronological_split(frame)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    x_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["target"]
    threshold = float(train_df["roll_std_10"].median())

    tabular_models = {
        "naive_baseline": None,
        "linear_regression": build_linear_regression(),
    }

    for model_name, model in tabular_models.items():
        if model is None:
            val_pred = naive_price_prediction(val_df)
            test_pred = naive_price_prediction(test_df)
        else:
            model.fit(x_train.to_numpy(), y_train.to_numpy())
            val_pred = model.predict(val_df[FEATURE_COLUMNS].to_numpy())
            test_pred = model.predict(test_df[FEATURE_COLUMNS].to_numpy())

        for split_name, split_df, preds in [
            ("validation", val_df, val_pred),
            ("test", test_df, test_pred),
        ]:
            metrics, prediction_df = evaluate_split(split_df, preds, model_name, horizon, split_name)
            metrics_rows.append(metrics)
            prediction_frames.append(prediction_df)
            if split_name == "test":
                regime_rows.append(evaluate_regimes(prediction_df, threshold))

    tuned_rf, rf_tuning_df, best_rf_params = tune_random_forest(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        horizon=horizon,
    )
    tuning_frames.append(rf_tuning_df)
    print(
        f"Horizon {horizon}: tuned Random Forest params "
        f"{best_rf_params} with validation RMSE {rf_tuning_df.iloc[0]['validation_rmse']:.2f}"
    )

    val_pred = tuned_rf.predict(val_df[FEATURE_COLUMNS].to_numpy())
    test_pred = tuned_rf.predict(test_df[FEATURE_COLUMNS].to_numpy())
    for split_name, split_df, preds in [
        ("validation", val_df, val_pred),
        ("test", test_df, test_pred),
    ]:
        metrics, prediction_df = evaluate_split(split_df, preds, "random_forest", horizon, split_name)
        metrics_rows.append(metrics)
        prediction_frames.append(prediction_df)
        if split_name == "test":
            regime_rows.append(evaluate_regimes(prediction_df, threshold))

    if include_lstm:
        if torch is None:
            print("Skipping LSTM because torch is not installed.")
        else:
            lstm = LSTMForecaster()
            lstm.fit(x_train.to_numpy(), y_train.to_numpy())

            val_pred = lstm.predict(val_df[FEATURE_COLUMNS].to_numpy())
            test_pred = lstm.predict(test_df[FEATURE_COLUMNS].to_numpy())

            for split_name, split_df, preds in [
                ("validation", val_df, val_pred),
                ("test", test_df, test_pred),
            ]:
                if len(preds) == 0:
                    continue
                metrics, prediction_df = evaluate_split(split_df, preds, "lstm", horizon, split_name)
                metrics_rows.append(metrics)
                prediction_frames.append(prediction_df)
                if split_name == "test":
                    regime_rows.append(evaluate_regimes(prediction_df, threshold))

    return (
        metrics_rows,
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(regime_rows, ignore_index=True),
        pd.concat(tuning_frames, ignore_index=True),
    )


def save_outputs(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(tables_dir / "all_metrics.csv", index=False)
    predictions_df.to_csv(tables_dir / "all_predictions.csv", index=False)
    regime_df.to_csv(tables_dir / "volatility_regime_metrics.csv", index=False)
    tuning_df.to_csv(tables_dir / "random_forest_tuning.csv", index=False)

    test_metrics = metrics_df.loc[metrics_df["split"] == "test"].copy()
    test_metrics.to_csv(tables_dir / "test_metrics.csv", index=False)
    test_metrics.pivot(index="horizon", columns="model", values="rmse").to_csv(
        tables_dir / "rmse_by_horizon.csv"
    )

    plot_metric_by_horizon(test_metrics, figures_dir / "rmse_by_horizon.png", metric="rmse")
    plot_metric_by_horizon(test_metrics, figures_dir / "mae_by_horizon.png", metric="mae")
    plot_regime_metric(regime_df, figures_dir / "rmse_by_regime.png", metric="rmse")

    best_row = test_metrics.sort_values("rmse").iloc[0]
    best_predictions = predictions_df.loc[
        (predictions_df["split"] == "test")
        & (predictions_df["model"] == best_row["model"])
        & (predictions_df["horizon"] == best_row["horizon"])
    ]
    plot_predictions(best_predictions, figures_dir / "best_model_predictions.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stock forecasting experiments.")
    parser.add_argument(
        "--input-path",
        default="data/yahoo_data.xlsx",
        help="Path to the input Excel file.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 3, 7],
        help="Prediction horizons to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where tables and figures will be saved.",
    )
    parser.add_argument(
        "--include-lstm",
        action="store_true",
        help="Include the optional PyTorch LSTM model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    raw_df = load_stock_data(input_path)
    print(
        "Loaded dataset:",
        {
            "rows": len(raw_df),
            "start_date": raw_df["date"].min().date().isoformat(),
            "end_date": raw_df["date"].max().date().isoformat(),
        },
    )

    all_metrics: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    regime_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []

    for horizon in args.horizons:
        frame = prepare_model_frame(raw_df, horizon)
        metrics_rows, prediction_df, regime_df, tuning_df = run_horizon_experiment(
            frame=frame,
            horizon=horizon,
            include_lstm=args.include_lstm,
        )
        all_metrics.extend(metrics_rows)
        prediction_frames.append(prediction_df)
        regime_frames.append(regime_df)
        tuning_frames.append(tuning_df)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    regime_df = pd.concat(regime_frames, ignore_index=True)
    tuning_df = pd.concat(tuning_frames, ignore_index=True)

    save_outputs(metrics_df, predictions_df, regime_df, tuning_df, output_dir)

    print("\nTest-set RMSE summary:")
    print(
        metrics_df.loc[metrics_df["split"] == "test", ["model", "horizon", "rmse", "mae", "r2"]]
        .sort_values(["horizon", "rmse"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
