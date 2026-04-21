# Notebook Plan

If you want a notebook-based presentation layer on top of the `src/` code, use the following sequence:

1. `01_data_overview.ipynb`
   - Inspect raw data
   - Plot the adjusted close series
   - Describe missing values and date coverage

2. `02_feature_engineering.ipynb`
   - Preview lag and rolling features
   - Show horizon target construction
   - Verify train/validation/test splits

3. `03_model_comparison.ipynb`
   - Compare naive baseline, Linear Regression, and Random Forest
   - Display metric tables and error plots

4. `04_lstm_analysis.ipynb`
   - Train and inspect the LSTM model
   - Compare sequence-model behavior to tabular models
