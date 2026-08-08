"""
Phase 3: Merge and Prepare for Modeling.

Combines all 25 featured stock CSVs from Phase 2 (processed/) into one
master dataset with a Ticker column, encodes the ticker, and produces
time-based train/test/validate splits, following Phases/phase_3.md.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TRAIN_CUTOFF = pd.Timestamp("2024-01-01")
TEST_CUTOFF = pd.Timestamp("2025-01-01")

FEATURE_COLUMNS = [
    "Ticker_encoded",
    "Open", "High", "Low", "Close", "Volume",
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

TARGET_COLUMNS = ["Target_High", "Target_Low"]


def merge_all_stocks(processed_dir, output_path):
    all_dfs = []

    for filename in sorted(os.listdir(processed_dir)):
        if filename.endswith("_cleaned.csv"):
            ticker = filename.replace("_cleaned.csv", "")

            df = pd.read_csv(processed_dir / filename)
            df["Ticker"] = ticker

            all_dfs.append(df)
            print(f"Loaded {ticker}: {len(df)} rows")

    master_df = pd.concat(all_dfs, ignore_index=True)

    master_df["Date"] = pd.to_datetime(master_df["Date"])
    master_df = master_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_path, index=False)

    print(f"\nMaster dataset created: {len(master_df)} total rows")
    print(f"Columns: {len(master_df.columns)}")
    print(f"Date range: {master_df['Date'].min().date()} to {master_df['Date'].max().date()}")
    print(f"Unique tickers: {master_df['Ticker'].nunique()}")

    return master_df


def main():
    master_df = merge_all_stocks(PROCESSED_DIR, DATA_DIR / "master_dataset.csv")

    print("\nRows per ticker:")
    print(master_df.groupby("Ticker").size().sort_values())

    print("\nMissing values per column (top 10):")
    print(master_df.isnull().sum().sort_values(ascending=False).head(10))

    # Step 2: Encode the Ticker column
    le = LabelEncoder()
    master_df["Ticker_encoded"] = le.fit_transform(master_df["Ticker"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "ticker_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\nTicker encoding:")
    print(dict(zip(le.classes_, le.transform(le.classes_))))

    # Re-save master_dataset.csv now that Ticker_encoded exists (Step 1 wrote
    # it before this column was added; downstream consumers like app.py need it)
    master_df.to_csv(DATA_DIR / "master_dataset.csv", index=False)

    # Step 3: Time-based train/test/validate split
    train_df = master_df[master_df["Date"] < TRAIN_CUTOFF].copy()
    test_df = master_df[(master_df["Date"] >= TRAIN_CUTOFF) & (master_df["Date"] < TEST_CUTOFF)].copy()
    validate_df = master_df[master_df["Date"] >= TEST_CUTOFF].copy()

    print(f"\nTrain: {len(train_df)} rows, {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"Test: {len(test_df)} rows, {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    print(f"Validate: {len(validate_df)} rows, {validate_df['Date'].min().date()} to {validate_df['Date'].max().date()}")

    # Step 4: Drop rows with missing values (guard against inf first, per troubleshooting note)
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()
    validate_df = validate_df.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\nTrain after dropna: {len(train_df)} rows")
    print(f"Test after dropna: {len(test_df)} rows")
    print(f"Validate after dropna: {len(validate_df)} rows")

    # Step 5: Define feature and target columns
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMNS]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMNS]

    X_validate = validate_df[FEATURE_COLUMNS]
    y_validate = validate_df[TARGET_COLUMNS]

    print(f"\nFeature count: {len(FEATURE_COLUMNS)}")
    print(f"X_train shape: {X_train.shape}")

    # Step 7: Save the prepared splits
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    validate_df.to_csv(DATA_DIR / "validate.csv", index=False)
    print("\nSaved train.csv, test.csv, validate.csv to data/")

    # Sanity checks
    print("\n--- Sanity Checks ---")
    print("Tickers in train:", train_df["Ticker"].nunique())
    print("Tickers in test:", test_df["Ticker"].nunique())
    print("Tickers in validate:", validate_df["Ticker"].nunique())

    print("Train max date:", train_df["Date"].max())
    print("Test min date:", test_df["Date"].min())
    print("Test max date:", test_df["Date"].max())
    print("Validate min date:", validate_df["Date"].min())

    print("NaN in train:", X_train.isnull().sum().sum())
    print("NaN in test:", X_test.isnull().sum().sum())
    print("NaN in validate:", X_validate.isnull().sum().sum())

    missing_train = sorted(set(master_df["Ticker"].unique()) - set(train_df["Ticker"].unique()))
    missing_test = sorted(set(master_df["Ticker"].unique()) - set(test_df["Ticker"].unique()))
    missing_validate = sorted(set(master_df["Ticker"].unique()) - set(validate_df["Ticker"].unique()))
    if missing_train:
        print("WARNING: tickers missing from train:", missing_train)
    if missing_test:
        print("WARNING: tickers missing from test:", missing_test)
    if missing_validate:
        print("WARNING: tickers missing from validate:", missing_validate)


if __name__ == "__main__":
    main()
