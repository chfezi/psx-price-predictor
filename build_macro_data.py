"""
One-off script: builds data/macro_data.csv (daily KSE-100 index level and
USD/PKR rate, plus their 1-day returns) from real sourced data, for the
macro-features experiment requested mid-conversation (see
Phases/frontend_notes.md). Not part of the regular pipeline - run once,
then add_macro_features.py merges the output onto train/test/validate.csv.

Sources, spliced at the boundary where PSX's own feed starts (no gap,
~7-week overlap, verified against each other):
- KSE-100: Yahoo Finance (^KSE) for 2018-01-02 through PSX's feed start,
  then PSX's own timeseries API (dps.psx.com.pk/timeseries/eod/KSE100)
  from 2021-08-12 onward - Yahoo's ^KSE feed goes mostly None after
  2021-09-30, PSX's is the live authoritative source going forward.
- USD/PKR: Yahoo Finance (PKR=X), complete 2018-01-01 to present with only
  a handful of missing days (forward-filled).
"""

import json
from pathlib import Path

import pandas as pd

SCRATCH = Path(
    r"C:\Users\fezan\AppData\Local\Temp\claude\c--Users-fezan-OneDrive-Desktop-data-scrapping"
    r"\7a7213de-a53c-496c-9608-d91c98496057\scratchpad"
)
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_kse():
    yahoo = json.load(open(SCRATCH / "kse_full.json"))
    r = yahoo["chart"]["result"][0]
    ts, closes = r["timestamp"], r["indicators"]["quote"][0]["close"]
    yahoo_df = pd.DataFrame({"Date": pd.to_datetime(ts, unit="s").normalize(), "KSE100_Close": closes}).dropna()

    psx = json.load(open(SCRATCH / "kse100_eod.json"))
    psx_df = pd.DataFrame(psx["data"], columns=["ts", "close", "vol", "avg"])
    psx_df["Date"] = pd.to_datetime(psx_df["ts"], unit="s").dt.normalize()
    psx_df = psx_df[["Date", "close"]].rename(columns={"close": "KSE100_Close"})

    psx_start = psx_df["Date"].min()
    yahoo_df = yahoo_df[yahoo_df["Date"] < psx_start]
    combined = pd.concat([yahoo_df, psx_df], ignore_index=True)
    combined = combined.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return combined


def load_usdpkr():
    usd = json.load(open(SCRATCH / "usdpkr_full.json"))
    r = usd["chart"]["result"][0]
    ts, closes = r["timestamp"], r["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"Date": pd.to_datetime(ts, unit="s").normalize(), "USDPKR_Close": closes})
    return df.dropna().drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


def main():
    kse = load_kse()
    print(f"KSE-100: {kse['Date'].min().date()} to {kse['Date'].max().date()}, {len(kse)} rows")

    usd = load_usdpkr()
    print(f"USD/PKR: {usd['Date'].min().date()} to {usd['Date'].max().date()}, {len(usd)} rows")

    macro = pd.merge(kse, usd, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
    macro["KSE100_Close"] = macro["KSE100_Close"].ffill()
    macro["USDPKR_Close"] = macro["USDPKR_Close"].ffill()
    macro["KSE100_Return_1d"] = macro["KSE100_Close"].pct_change()
    macro["USDPKR_Return_1d"] = macro["USDPKR_Close"].pct_change()
    macro = macro.dropna().reset_index(drop=True)

    print(f"\nCombined macro series: {macro['Date'].min().date()} to {macro['Date'].max().date()}, {len(macro)} rows")
    print(macro.head())
    print(macro.tail())

    out_path = DATA_DIR / "macro_data.csv"
    macro.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
