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

# Phase 6: stationary ratio/return features, checked separately against the
# new return target below so the raw-price leakage check above stays an
# exact before/after comparison.
STATIONARY_FEATURE_COLUMNS = [
    "Close_to_SMA_20", "Close_to_SMA_50", "Close_to_SMA_200",
    "Close_to_EMA_12", "Close_to_EMA_26",
    "Return_lag_3", "Return_lag_10",
    "MACD_Pct", "MACD_Hist_Pct", "ATR_Pct",
    "BB_PercentB", "BB_Bandwidth",
]

# Forecast horizons, in trading days. 1d duplicates Target_Open/Close Return
# above under a horizon-suffixed name (see add_horizon_targets) so every
# horizon, including 1-day, can be trained through the same horizon-indexed
# pipeline in train_models_phase9.py.
HORIZONS = [1, 5, 10, 20, 60]


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


def add_stationary_features(df):
    """
    Phase 6: raw price-level features (SMA/EMA values, MACD, ATR, Bollinger
    Bands) don't transfer across the 25 stocks' very different price scales
    in a single unified model. These ratio/return equivalents do.
    """
    close = df["Close"]

    for window in [20, 50, 200]:
        sma_col = f"SMA_{window}"
        if sma_col in df.columns:
            df[f"Close_to_SMA_{window}"] = close / df[sma_col]

    for span in [12, 26]:
        ema_col = f"EMA_{span}"
        if ema_col in df.columns:
            df[f"Close_to_EMA_{span}"] = close / df[ema_col]

    for lag in [3, 10]:
        df[f"Return_lag_{lag}"] = df["Return_1d"].shift(lag)

    if "MACD_line" in df.columns:
        df["MACD_Pct"] = df["MACD_line"] / close
    if "MACD_histogram" in df.columns:
        df["MACD_Hist_Pct"] = df["MACD_histogram"] / close

    if "ATR_14" in df.columns:
        df["ATR_Pct"] = df["ATR_14"] / close

    if {"BB_upper", "BB_lower", "BB_middle"}.issubset(df.columns):
        band_width = df["BB_upper"] - df["BB_lower"]
        df["BB_PercentB"] = (close - df["BB_lower"]) / band_width
        df["BB_Bandwidth"] = band_width / df["BB_middle"]

    for col in STATIONARY_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

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
    df["Target_Open"] = df["Open"].shift(-1)
    df["Target_Close"] = df["Close"].shift(-1)

    return df


def add_return_targets(df):
    """
    Phase 9: return-based counterparts of Target_Open/Target_Close, replacing
    Target_High_Return/Target_Low_Return now that the pipeline predicts
    Open/Close instead of High/Low (Phases/phase_9.md Step 2). Same clip
    rationale as the High/Low version this replaces: PSX's daily circuit
    limits mean single-day moves past roughly 15 percent are data errors,
    not real prices, so the clip protects training from that handful of bad
    rows rather than reflecting a real return ceiling. See
    Phases/phase_9_notes.md.
    """
    df["Target_Open_Return"] = (df["Target_Open"] - df["Close"]) / df["Close"]
    df["Target_Close_Return"] = (df["Target_Close"] - df["Close"]) / df["Close"]

    for col in ["Target_Open_Return", "Target_Close_Return"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).clip(-0.15, 0.15)

    return df


def add_horizon_targets(df, horizons=HORIZONS):
    """
    Same horizon-extension shape as Phase 7's High/Low version, now over
    Open/Close: adds both the raw (unclipped) price target - ground truth
    for evaluation - and the clipped return target used for training, for
    every horizon including 1d, so train_models_phase9.py can train every
    horizon (1/5/10/20/60) through one uniform pipeline.

    Clip widens with horizon as +/-0.15*sqrt(n): a flat +/-15% would cut off
    the bulk of genuine 60-day moves, not just bad-data tails, since real
    return spread grows with the forecast window. sqrt(n) scaling follows
    from how return volatility scales under a random-walk assumption, and
    reduces to exactly the 1-day clip at n=1.
    """
    for n in horizons:
        open_col = f"Target_Open_{n}d"
        close_col = f"Target_Close_{n}d"
        df[open_col] = df["Open"].shift(-n)
        df[close_col] = df["Close"].shift(-n)

        clip = 0.15 * np.sqrt(n)
        df[f"Target_Open_Return_{n}d"] = (
            (df[open_col] - df["Close"]) / df["Close"]
        ).replace([np.inf, -np.inf], np.nan).clip(-clip, clip)
        df[f"Target_Close_Return_{n}d"] = (
            (df[close_col] - df["Close"]) / df["Close"]
        ).replace([np.inf, -np.inf], np.nan).clip(-clip, clip)

    return df


def engineer_features(df):
    df = add_return_features(df)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_stationary_features(df)
    df = add_rolling_stats(df)
    df = add_intraday_features(df)
    df = add_date_features(df)
    df = add_target_variables(df)
    df = add_return_targets(df)
    df = add_horizon_targets(df)
    return df


def report_missing_values(df, ticker):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"{ticker}: {len(missing)} columns have missing values (expected, from rolling windows)")
        print(f"  Rows affected: up to {missing.max()} rows (leading rows only)")
    else:
        print(f"{ticker}: no missing values")


def check_no_leakage(df, feature_columns, target_column="Target_Close"):
    correlations = df[feature_columns + [target_column]].corr()[target_column].sort_values(ascending=False)
    print(f"Feature correlation with {target_column}:")
    print(correlations)

    suspicious = correlations[(correlations.abs() > 0.98) & (correlations.index != target_column)]
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
        check_no_leakage(df, FEATURE_COLUMNS_FOR_LEAKAGE_CHECK, "Target_Close")
        check_no_leakage(
            df,
            FEATURE_COLUMNS_FOR_LEAKAGE_CHECK + STATIONARY_FEATURE_COLUMNS,
            "Target_Close_Return",
        )
        for n in HORIZONS:
            check_no_leakage(
                df,
                FEATURE_COLUMNS_FOR_LEAKAGE_CHECK + STATIONARY_FEATURE_COLUMNS,
                f"Target_Close_Return_{n}d",
            )

        df.to_csv(filepath, index=False)
        print(f"{ticker}: saved with {len(df.columns)} total columns\n")

    print("Feature engineering complete for all stocks.")


if __name__ == "__main__":
    main()
