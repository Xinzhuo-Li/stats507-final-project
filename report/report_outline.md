# Final Report Outline

## Title

Stock Price Forecasting Across Prediction Horizons and Volatility Regimes

## Abstract

- State the forecasting task and the dataset.
- Summarize the models compared.
- Report the main quantitative findings by horizon and volatility regime.

## Introduction

- Motivation for stock forecasting and its difficulty.
- Why horizon length matters.
- Why volatility matters.
- Project objective: compare model families under different operating conditions.

## Related Work

- Classical time-series methods
- Machine learning for tabular financial features
- Deep learning approaches such as RNN/LSTM
- Be careful to use only real references

## Method

### Dataset

- Yahoo Finance daily data from 2018 to 2023
- Variables: open, high, low, close, adjusted close, volume

### Problem Formulation

- Input: historical price and volume-based features at day `t`
- Output: future adjusted close at horizon `t+1`, `t+3`, `t+7`

### Feature Engineering

- Lagged prices and returns
- Rolling means
- Rolling standard deviations
- Volume-based and intraday range features

### Models

- Naive baseline
- Linear Regression
- Random Forest
- LSTM

### Evaluation

- Chronological train/validation/test split
- RMSE, MAE, R2, MAPE
- High- vs low-volatility regime analysis

## Results

- Table: test metrics across horizons
- Figure: RMSE by horizon
- Figure: RMSE by volatility regime
- Figure: actual vs predicted prices for the best-performing model
- Interpretation of where each model helps or fails

## Conclusion

- Summarize the strongest model by condition
- Discuss the effect of horizon length on accuracy
- Discuss the effect of volatility on performance
- State limitations and possible next steps
