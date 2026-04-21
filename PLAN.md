# Project Execution Plan

## Phase 1: Data Foundation

- Load `yahoo_data.xlsx` and standardize the schema.
- Parse dates, sort chronologically, and remove invalid rows.
- Confirm the usable date range and total observations.

## Phase 2: Feature Engineering

- Build lagged adjusted-close features.
- Compute daily returns.
- Create rolling mean and rolling volatility features.
- Add volume and intraday-range features.
- Construct separate targets for horizons `1`, `3`, and `7`.

## Phase 3: Modeling

- Train a naive baseline using the current adjusted close.
- Train `Linear Regression`.
- Train `Random Forest`.
- Optionally train `LSTM` when PyTorch is available.

## Phase 4: Evaluation

- Use chronological train/validation/test splits.
- Report `RMSE`, `MAE`, `R2`, and `MAPE`.
- Compare performance across horizons.
- Split test predictions into low- and high-volatility regimes.

## Phase 5: Deliverables

- Export metric tables to `outputs/tables/`.
- Export figures to `outputs/figures/`.
- Use `report/report_outline.md` as the starting point for the final paper.

## Current Implementation Status

- [x] Repository structure created
- [x] Data pipeline modules created
- [x] Evaluation and visualization modules created
- [x] Dependencies installed locally
- [x] First end-to-end experiment run completed
- [x] LSTM run completed
