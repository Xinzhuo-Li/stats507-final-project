from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


def plot_metric_by_horizon(metrics_df: pd.DataFrame, output_path: str | Path, metric: str = "rmse") -> None:
    """Plot a model comparison chart across prediction horizons."""
    plt.figure(figsize=(9, 5))
    sns.barplot(data=metrics_df, x="horizon", y=metric, hue="model")
    plt.title(f"{metric.upper()} by Prediction Horizon")
    plt.xlabel("Prediction Horizon (days)")
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_regime_metric(regime_df: pd.DataFrame, output_path: str | Path, metric: str = "rmse") -> None:
    """Plot a volatility regime comparison chart."""
    plt.figure(figsize=(10, 5))
    sns.barplot(data=regime_df, x="regime", y=metric, hue="model")
    plt.title(f"{metric.upper()} by Volatility Regime")
    plt.xlabel("Volatility Regime")
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_predictions(predictions: pd.DataFrame, output_path: str | Path, max_points: int = 120) -> None:
    """Plot actual vs predicted prices for a sample window."""
    plot_df = predictions.sort_values("date").tail(max_points)
    plt.figure(figsize=(11, 5))
    plt.plot(plot_df["date"], plot_df["actual"], label="Actual", linewidth=2)
    plt.plot(plot_df["date"], plot_df["prediction"], label="Predicted", linewidth=2)
    plt.title(f"Actual vs Predicted Adjusted Close ({plot_df['model'].iloc[0]}, h={plot_df['horizon'].iloc[0]})")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
