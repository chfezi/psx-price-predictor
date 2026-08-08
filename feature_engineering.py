"""
Phase 2: Feature Engineering.

Adds return, technical-indicator, lag, rolling-stat, intraday, date, and
target features to each cleaned stock CSV in processed/ (from Phase 1),
following Phases/phase_2.md. Overwrites each CSV in place with the
featured version.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

FEATURE_COLUMNS_FOR_LEAKAGE_CHECK = [
    "Return_1d", "Return_5d", "Return_10d", "Return_20d", "Log_Return",
    "High_Low_Range", "Open_Close_Range",
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "RSI_14",
    "MACD_line", "Signal_line", "MACD_histogram",
    "BB_upper", "BB_middle", "BB_lower", "ATR_14", "Volatility_20",
    "Volume_SMA_20", "Volume_Ratio",
    "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5", "Close_lag_10",
    "Return_lag_1", "Return_lag_2", "Return_lag_5",
    "Volume_lag_1", "Volume_lag_5",
    "Rolling_Max_20", "Rolling_Min_20",
    "Price_Range", "Body", "Upper_Wick", "Lower_Wick", "Range_Percentage",
    "DayOfWeek", "Month",
]


def add_return_features(df):
    close = df["Close"]

    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)
    df["Return_10d"] = close.pct_change(10)
    df["Return_20d"] = close.pct_change(20)

    df["Log_Return"] = np.log(close / close.shift(1))

    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Open"]
    df["Open_Close_Range"] = (df["Close"] - df["Open"]) / df["Open"]

    return df


def add_technical_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()
    df["SMA_200"] = close.rolling(window=200).mean()

    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df["RSI_14"] = calculate_rsi(close, 14)
    df["RSI_14"] = df["RSI_14"].replace([np.inf, -np.inf], np.nan)

    df["MACD_line"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_line"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
    df["MACD_histogram"] = df["MACD_line"] - df["Signal_line"]

    df["BB_middle"] = df["SMA_20"]
    std = close.rolling(window=20).std()
    df["BB_upper"] = df["SMA_20"] + (2 * std)
    df["BB_lower"] = df["SMA_20"] - (2 * std)

    def calculate_atr(high, low, close, period=14):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    df["ATR_14"] = calculate_atr(high, low, close, 14)
    df["ATR_14"] = df["ATR_14"].replace([np.inf, -np.inf], np.nan)

    df["Volatility_20"] = close.pct_change().rolling(window=20).std()

    df["Volume_SMA_20"] = volume.rolling(window=20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA_20"]
    df["Volume_Ratio"] = df["Volume_Ratio"].replace([np.inf, -np.inf], np.nan)

    return df


def add_lag_features(df):
    for lag in [1, 2, 3, 5, 10]:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

    for lag in [1, 2, 5]:
        df[f"Return_lag_{lag}"] = df["Return_1d"].shift(lag)

    for lag in [1, 5]:
        df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)

    return df


def add_rolling_stats(df):
    close = df["Close"]

    df["Rolling_Max_20"] = close.rolling(window=20).max()
    df["Rolling_Min_20"] = close.rolling(window=20).min()

    return df


def add_intraday_features(df):
    df["Price_Range"] = df["High"] - df["Low"]
    df["Body"] = df["Close"] - df["Open"]
    df["Upper_Wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["Lower_Wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    df["Range_Percentage"] = df["Price_Range"] / df["Open"]

    return df


def add_date_features(df):
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

    return df


def add_target_variables(df):
    df["Target_High"] = df["High"].shift(-1)
    df["Target_Low"] = df["Low"].shift(-1)

    return df


def engineer_features(df):
    df = add_return_features(df)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df = add_intraday_features(df)
    df = add_date_features(df)
    df = add_target_variables(df)
    return df


def report_missing_values(df, ticker):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"{ticker}: {len(missing)} columns have missing values (expected, from rolling windows)")
        print(f"  Rows affected: up to {missing.max()} rows (leading rows only)")
    else:
        print(f"{ticker}: no missing values")


def check_no_leakage(df, feature_columns):
    correlations = df[feature_columns + ["Target_High"]].corr()["Target_High"].sort_values(ascending=False)
    print("Feature correlation with Target_High:")
    print(correlations)

    suspicious = correlations[(correlations.abs() > 0.98) & (correlations.index != "Target_High")]
    if len(suspicious) > 0:
        print("\nWARNING: These features may be leaking future data:")
        print(suspicious)
    else:
        print("\nNo obvious leakage detected.")


def main():
    stock_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_cleaned.csv")]

    print(f"Found {len(stock_files)} stocks to process\n")

    for filename in sorted(stock_files):
        ticker = filename.replace("_cleaned.csv", "")
        filepath = PROCESSED_DIR / filename

        df = pd.read_csv(filepath)
        df["Date"] = pd.to_datetime(df["Date"])

        df = engineer_features(df)

        report_missing_values(df, ticker)
        check_no_leakage(df, FEATURE_COLUMNS_FOR_LEAKAGE_CHECK)

        df.to_csv(filepath, index=False)
        print(f"{ticker}: saved with {len(df.columns)} total columns\n")

    print("Feature engineering complete for all stocks.")


if __name__ == "__main__":
    main()
