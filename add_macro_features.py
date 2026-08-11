"""
One-off script for the macro-features experiment (see
Phases/frontend_notes.md): merges data/macro_data.csv (built by
build_macro_data.py) onto data/train.csv, data/test.csv, and
data/validate.csv by Date, producing train_macro.csv/test_macro.csv/
validate_macro.csv - same rows, same train/test/validate split as the
existing Phase 9 files, with two new columns (KSE100_Return_1d,
USDPKR_Return_1d) added. Existing train.csv/test.csv/validate.csv are left
untouched.

Macro data is one row per calendar date, broadcast across all 25 tickers
via a plain merge on Date (every ticker sees the same market-wide return on
a given day, which is exactly what these features are meant to represent).
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
MACRO_COLUMNS = ["KSE100_Return_1d", "USDPKR_Return_1d"]


def add_macro(split_name):
    df = pd.read_csv(DATA_DIR / f"{split_name}.csv")
    macro = pd.read_csv(DATA_DIR / "macro_data.csv")[["Date"] + MACRO_COLUMNS]

    df["Date"] = pd.to_datetime(df["Date"])
    macro["Date"] = pd.to_datetime(macro["Date"])

    before_rows = len(df)
    merged = df.merge(macro, on="Date", how="left")
    assert len(merged) == before_rows, f"{split_name}: merge changed row count"

    missing = merged[MACRO_COLUMNS].isna().any(axis=1).sum()
    if missing:
        merged[MACRO_COLUMNS] = merged[MACRO_COLUMNS].fillna(0.0)
        print(f"{split_name}: {missing}/{before_rows} rows had no macro match for their Date, filled with 0.0")

    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    out_path = DATA_DIR / f"{split_name}_macro.csv"
    merged.to_csv(out_path, index=False)
    print(f"{split_name}: wrote {out_path} ({len(merged)} rows, {len(merged.columns)} columns)")


def main():
    for split_name in ("train", "test", "validate"):
        add_macro(split_name)


if __name__ == "__main__":
    main()
