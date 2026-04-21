# Stock Price Forecasting Across Horizons and Volatility Regimes

This project studies how forecast performance changes across prediction horizons and market volatility conditions using historical Yahoo Finance daily data from 2018 to 2023.

## Project Goal

The project compares three modeling families:

- `Linear Regression` as a transparent baseline
- `Random Forest` as a non-linear tabular model
- `LSTM` as a sequence model implemented with PyTorch

The analysis answers two questions:

1. How does predictive accuracy change for `t+1`, `t+3`, and `t+7` horizons?
2. How does market volatility affect model performance?

## Repository Structure

```text
.
├── data/
│   └── yahoo_data.xlsx
├── notebooks/
├── outputs/
│   ├── figures/
│   └── tables/
├── report/
├── src/
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── features.py
│   ├── models.py
│   ├── run_experiment.py
│   └── visualize.py
├── Final_project_guideline.pdf
├── project_proposal.docx
└── requirements.txt
```

## Workflow

1. Load and clean the daily stock price data.
2. Build lagged, rolling, return, and volatility-based features.
3. Construct targets for multiple horizons.
4. Train models with chronological train/validation/test splits.
5. Tune the Random Forest with the validation split.
6. Evaluate overall performance and performance under high/low volatility.
7. Export result tables and figures for the final report.

## Running the Project

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the experiment pipeline:

```bash
python3 -m src.run_experiment
```

By default, the script:

- uses `data/yahoo_data.xlsx`
- evaluates horizons `1`, `3`, and `7`
- saves tables to `outputs/tables/`
- saves figures to `outputs/figures/`
- exports Random Forest tuning results to `outputs/tables/random_forest_tuning.csv`
- skips the LSTM model if `torch` is not installed

To include the LSTM model:

```bash
python3 -m src.run_experiment --include-lstm
```

## Planned Deliverables

- Reproducible Python pipeline
- Tables comparing model performance across horizons
- Volatility regime analysis
- Figures for the summary report
- IEEE-style final report based on the generated outputs
