"""
PSX historical data scraper.

Pulls ~8 years of daily OHLCV data for a fixed set of PSX symbols from the
PSX Data Portal (dps.psx.com.pk) and saves one CSV per symbol.

This talks directly to the endpoint the portal's own "Search by Symbol"
form uses (POST /historical with symbol/month/year), verified against the
live site — no browser/Selenium required.
"""

import csv
import random
import time
from datetime import date, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://dps.psx.com.pk/historical"
SYMBOLS = [
    "BAHL", "COLG", "DGKC", "EFERT", "ENGROH", "FFC", "HBL", "HUBC",
    "ILP", "INDU", "LUCK", "MARI", "MCB", "MEBL", "MLCF", "NESTLE", "NETSOL",
    "OGDC", "PAKT", "PPL", "PSO", "SYS", "TRG", "UBL", "UNITY",
]
YEARS_BACK = 8
OUTPUT_DIR = Path(__file__).resolve().parent / "psx_data_8years"
REQUEST_DELAY_SECONDS = 0.6
MAX_RETRIES = 3
TIMEOUT_SECONDS = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}


def fetch_month(session, symbol, month, year):
    data = {"symbol": symbol, "month": str(month), "year": str(year)}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(BASE_URL, data=data, headers=HEADERS, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                print(f"    [!] Failed {symbol} {year}-{month:02d} after {MAX_RETRIES} attempts: {exc}")
                return None
            time.sleep(1.5 * attempt)
    return None


def parse_rows(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("#historicalTable tbody tr"):
        cells = tr.find_all("td")
        if len(cells) != 6:
            continue
        date_text = cells[0].get_text(strip=True)
        try:
            row_date = datetime.strptime(date_text, "%b %d, %Y").date()
        except ValueError:
            continue
        try:
            open_, high, low, close = (
                float(cells[i].get_text(strip=True).replace(",", "")) for i in range(1, 5)
            )
            volume = int(cells[5].get_text(strip=True).replace(",", ""))
        except ValueError:
            continue
        rows.append(
            {
                "Date": row_date.isoformat(),
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
    return rows


def month_range(years_back):
    today = date.today()
    start_year = today.year - years_back
    months = []
    for year in range(start_year, today.year + 1):
        last_month = today.month if year == today.year else 12
        for month in range(1, last_month + 1):
            months.append((year, month))
    return months


def scrape_symbol(session, symbol):
    print(f"[->] Scraping {symbol}...")
    all_rows = []
    for year, month in month_range(YEARS_BACK):
        html = fetch_month(session, symbol, month, year)
        if html:
            all_rows.extend(parse_rows(html))
        time.sleep(REQUEST_DELAY_SECONDS + random.uniform(0, 0.3))
    all_rows.sort(key=lambda r: r["Date"])
    return all_rows


def save_csv(symbol, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{symbol}_psx_data.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Open", "High", "Low", "Close", "Volume"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"    [OK] Saved {len(rows)} rows -> {out_path}")


def main():
    print(f"[OK] Starting PSX historical scraper for {len(SYMBOLS)} symbols, {YEARS_BACK} years back")
    with requests.Session() as session:
        for symbol in SYMBOLS:
            rows = scrape_symbol(session, symbol)
            save_csv(symbol, rows)
    print("[OK] Done.")


if __name__ == "__main__":
    main()
