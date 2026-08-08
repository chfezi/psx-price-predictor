"""
Phase 1: Data Cleaning.

Cleans each raw PSX stock CSV in psx_data_8years/ (FFBL excluded) and
writes the cleaned version to processed/, following Phases/phase_1.md.
"""

import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "psx_data_8years"
PROCESSED_DIR = BASE_DIR / "processed"
EXCLUDED_TICKERS = {"FFBL"}


def ticker_from_filename(filename):
    return filename.replace("_psx_data.csv", "")


def clean_stock_data(input_path, output_path):
    """Clean a single stock CSV file."""

    print(f"\nCleaning {input_path.name}...")

    # Load
    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df)} rows")

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    print("  Dates parsed")

    # Fix data types
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    print("  Data types fixed")

    # Remove duplicates
    duplicates = df.duplicated(subset=["Date"]).sum()
    df = df.drop_duplicates(subset=["Date"], keep="first")
    print(f"  Removed {duplicates} duplicates")

    # Remove rows with missing OHLCV
    missing_before = df.isnull().sum().sum()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    missing_after = df.isnull().sum().sum()
    print(f"  Removed rows with missing OHLCV ({missing_before} -> {missing_after} nulls)")

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)
    print("  Sorted by date")

    # Validation
    errors = 0

    # Check High >= Low
    if (df["High"] < df["Low"]).any():
        print(f"  WARNING: {(df['High'] < df['Low']).sum()} rows have High < Low")
        errors += 1

    # Check Close within High-Low range
    impossible_close = (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
    if impossible_close.any():
        print(f"  WARNING: {impossible_close.sum()} rows have Close outside High-Low range")
        errors += 1

    # Check negative prices
    if (df[["Open", "High", "Low", "Close"]] < 0).any().any():
        print("  WARNING: Negative prices found")
        df = df[(df[["Open", "High", "Low", "Close"]] >= 0).all(axis=1)]
        errors += 1

    # Check negative volume
    if (df["Volume"] < 0).any():
        print("  WARNING: Negative volumes found")
        df = df[df["Volume"] >= 0]
        errors += 1

    if errors == 0:
        print("  Validation: OK")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Final: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def main():
    stock_files = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.endswith(".csv") and ticker_from_filename(f) not in EXCLUDED_TICKERS
    )

    print(f"Found {len(stock_files)} stock files to clean ({', '.join(sorted(EXCLUDED_TICKERS))} excluded)\n")

    summary = []
    for filename in stock_files:
        ticker = ticker_from_filename(filename)
        input_path = RAW_DIR / filename
        output_path = PROCESSED_DIR / f"{ticker}_cleaned.csv"

        df = clean_stock_data(input_path, output_path)
        summary.append(
            {
                "Ticker": ticker,
                "Rows": len(df),
                "Start Date": df["Date"].min().date(),
                "End Date": df["Date"].max().date(),
                "Days Covered": (df["Date"].max() - df["Date"].min()).days,
            }
        )

    print(f"\nAll {len(stock_files)} stocks cleaned!")

    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.to_string(index=False))

    summary_path = PROCESSED_DIR / "cleaning_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
