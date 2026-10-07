import numpy as np
import pandas as pd


# Load cleaned prices and return daily log returns.
def load_log_returns(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    stock_cols = sorted([column for column in df.columns if column != "Date"], key=lambda name: (name.split("_", 1)[0], name))
    log_returns = np.log(df[stock_cols]).diff().iloc[1:].reset_index(drop=True)
    log_returns.insert(0, "Date", df["Date"].iloc[1:].reset_index(drop=True))
    return log_returns


# Rolling Pearson correlation matrices over the log returns; returns (matrices, end-dates, stock_names).
def rolling_correlations(log_returns, window_length, window_shift):
    stock_names = log_returns.columns[1:].tolist()
    correlation_matrices = []
    window_dates = []

    for i in range(0, len(log_returns) - window_length + 1, window_shift):
        window = log_returns.iloc[i:i + window_length]
        correlation_matrices.append(window.iloc[:, 1:].corr().values)
        window_dates.append(window["Date"].iloc[-1])

    return correlation_matrices, window_dates, stock_names
