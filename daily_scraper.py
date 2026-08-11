"""
Phase 10: Daily Data Automation - incremental scraper.

Fetches only the newest trading day(s) per stock and appends them to the raw
CSVs in psx_data_8years/, so the pipeline can run on a schedule instead of
re-scraping 8 years of history every time. See Phases/datascrap.md.

Step 0 of that plan called for hitting PSX's JSON endpoint
(https://dps.psx.com.pk/timeseries/eod/{SYMBOL}) first and checking the
response shape before writing a parser against it. Doing that showed each
row is [timestamp, close, volume, open] - Close, Volume and Open only, no
High/Low - so it can't produce the OHLCV rows clean_data.py expects further
down the pipeline. The POST /historical endpoint PSX_DATA_SCRAPPER.py
already uses returns full OHLCV, so this script reuses that endpoint (and
its fetch_month/parse_rows helpers) instead, just scoped to the month(s)
since each ticker's last stored date rather than 8 years back.
"""

import time
from datetime import date

import pandas as pd
import requests

from clean_data import EXCLUDED_TICKERS, RAW_DIR, ticker_from_filename
from PSX_DATA_SCRAPPER import fetch_month, parse_rows

TICKER_DELAY_SECONDS = 1.5


def load_tickers():
    return sorted(
        ticker_from_filename(f.name)
        for f in RAW_DIR.iterdir()
        if f.suffix == ".csv" and ticker_from_filename(f.name) not in EXCLUDED_TICKERS
    )


def months_since(start_date, end_date):
    months = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def scrape_new_rows(session, symbol, latest_date):
    all_rows = []
    for year, month in months_since(latest_date, date.today()):
        html = fetch_month(session, symbol, month, year)
        if html:
            all_rows.extend(parse_rows(html))

    new_rows = [r for r in all_rows if date.fromisoformat(r["Date"]) > latest_date]
    new_rows.sort(key=lambda r: r["Date"])
    return new_rows


def update_ticker(session, ticker):
    csv_path = RAW_DIR / f"{ticker}_psx_data.csv"
    if not csv_path.exists():
        print(f"[!] {ticker}: no existing raw CSV at {csv_path}, skipping")
        return "failed"

    existing = pd.read_csv(csv_path)
    latest_date = pd.to_datetime(existing["Date"]).dt.date.max()

    new_rows = scrape_new_rows(session, ticker, latest_date)
    if not new_rows:
        print(f"[=] {ticker}: no new rows (latest stored date {latest_date})")
        return "no_new_rows"

    new_df = pd.DataFrame(new_rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    new_df.to_csv(csv_path, mode="a", header=False, index=False)
    print(f"[+] {ticker}: added {len(new_rows)} row(s), {new_rows[0]['Date']} to {new_rows[-1]['Date']}")
    return "updated"


def main():
    tickers = load_tickers()
    print(f"[OK] Daily scrape for {len(tickers)} tickers: {', '.join(tickers)}")

    summary = {"updated": 0, "no_new_rows": 0, "failed": 0}
    with requests.Session() as session:
        for i, ticker in enumerate(tickers):
            try:
                result = update_ticker(session, ticker)
            except Exception as exc:
                print(f"[!] {ticker}: failed - {exc}")
                result = "failed"
            summary[result] += 1

            if i < len(tickers) - 1:
                time.sleep(TICKER_DELAY_SECONDS)

    print(
        f"\n[OK] Daily scrape complete: {summary['updated']} updated, "
        f"{summary['no_new_rows']} with no new rows, {summary['failed']} failed"
    )


if __name__ == "__main__":
    main()
