"""
KSE-100 expansion Stage 1: one-time historical backfill for the 73 new
tickers, following Phases/kse_100_expand.md.

Separate script from the Phase 10 daily scraper (daily_scraper.py), which
only appends the newest day(s) to an existing CSV. This pulls the full
2018-2026 range for tickers that don't have a CSV yet, using the same
POST /historical endpoint and fetch_month/parse_rows helpers
PSX_DATA_SCRAPPER.py and daily_scraper.py already use - not the
timeseries/eod endpoint the plan doc names, see that doc's "Deviation"
note for why (no High/Low in that endpoint's rows).

Renamed tickers (RENAMED_TICKERS below) get both their old and new symbol
scraped and merged into one continuous series under the new symbol's
filename, since the old scrip code stops returning new rows at the rename
date but still holds the pre-rename history.

Saves one CSV per ticker to psx_data_8years/, same convention
PSX_DATA_SCRAPPER.py uses ({SYMBOL}_psx_data.csv), so clean_data.py picks
them up automatically without any changes.
"""

import sys
import time
from pathlib import Path

from PSX_DATA_SCRAPPER import (
    OUTPUT_DIR,
    REQUEST_DELAY_SECONDS,
    fetch_month,
    month_range,
    parse_rows,
    save_csv,
)
import random

import requests

START_YEAR = 2018

NEW_COMPANIES = {
    "PGLC": "Pak-Gulf Leasing Company",
    "PSX": "Pakistan Stock Exchange",
    "DHPL": "DH Partners",
    "BAFL": "Bank Alfalah",
    "BOP": "The Bank of Punjab",
    "FABL": "Faysal Bank",
    "HMB": "Habib Metropolitan Bank",
    "NBP": "National Bank of Pakistan",
    "SCBPL": "Standard Chartered Bank Pakistan",
    "AKBL": "Askari Bank",
    "ABL": "Allied Bank",
    "AICL": "Adamjee Insurance",
    "GADT": "Gadoon Textile Mills",
    "YOUW": "Yousaf Weaving Mills",
    "NML": "Nishat Mills",
    "MEHT": "Mehmood Textile Mills",
    "KTML": "Kohinoor Textile Mills",
    "BNWM": "Bannu Woollen Mills",
    "IBFL": "Ibrahim Fibres",
    "JDWS": "JDW Sugar Mills",
    "BWCL": "Bestway Cement",
    "FCCL": "Fauji Cement Company",
    "KOHC": "Kohat Cement Company",
    "CHCC": "Cherat Cement Company",
    "POWER": "Power Cement",
    "PIOC": "Pioneer Cement",
    "ATRL": "Attock Refinery",
    "CNERGY": "Cnergyico PK",
    "KAPCO": "Kot Addu Power Company",
    "KEL": "K-Electric",
    "APL": "Attock Petroleum",
    "SNGP": "Sui Northern Gas Pipelines",
    "SSGC": "Sui Southern Gas Company",
    "POL": "Pakistan Oilfields",
    "ISL": "International Steels",
    "INIL": "International Industries",
    "MTL": "Millat Tractors",
    "SAZEW": "Sazgar Engineering Works",
    "ATLH": "Atlas Honda",
    "HCAR": "Honda Atlas Cars Pakistan",
    "GAL": "Ghandhara Automobiles",
    "GHNI": "Ghandhara Industries",
    "THALL": "Thal",
    "PAEL": "Pak Elektron",
    "PIBTL": "Pakistan International Bulk Terminal",
    "AIRLINK": "Air Link Communication",
    "PTC": "Pakistan Telecommunication Company",
    "HUMNL": "Hum Network",
    "AHCL": "Arif Habib Corporation",
    "FATIMA": "Fatima Fertilizer Company",
    "SEARL": "The Searle Company",
    "AGP": "AGP",
    "GLAXO": "GlaxoSmithKline Pakistan",
    "ABOT": "Abbott Laboratories Pakistan",
    "CPHL": "Citi Pharma",
    "HALEON": "Haleon Pakistan",
    "HINOON": "Highnoon Laboratories",
    "LCI": "Lucky Core Industries",
    "LOTCHEM": "Lotte Chemical Pakistan",
    "PKGS": "Packages",
    "SSOM": "S.S. Oil Mills",
    "SRVI": "Service Industries",
    "FFL": "Fauji Foods",
    "MUREB": "Murree Brewery Company",
    "NATF": "National Foods",
    "RMPL": "Rafhan Maize Products",
    "UPFL": "Unilever Pakistan Foods",
    "TGL": "Tariq Glass Industries",
    "GHGL": "Ghani Glass",
    "PSEL": "Pakistan Services",
    "PABC": "Pakistan Aluminium Beverage Cans",
    "SHFA": "Shifa International Hospitals",
    "JVDC": "Javedan Corporation",
}

RENAMED_TICKERS = {
    "GAL": "GHNL",
    "HALEON": "GSKCH",
    "LCI": "ICI",
}


def months_from(start_year, end_date):
    months = []
    for year in range(start_year, end_date.year + 1):
        last_month = end_date.month if year == end_date.year else 12
        for month in range(1, last_month + 1):
            months.append((year, month))
    return months


def scrape_symbol_full(session, symbol):
    print(f"[->] Scraping {symbol}...")
    all_rows = []
    for year, month in month_range(2026 - START_YEAR):
        html = fetch_month(session, symbol, month, year)
        if html:
            all_rows.extend(parse_rows(html))
        time.sleep(REQUEST_DELAY_SECONDS + random.uniform(0, 0.3))
    return all_rows


def backfill_ticker(session, symbol):
    new_rows = scrape_symbol_full(session, symbol)

    old_symbol = RENAMED_TICKERS.get(symbol)
    if old_symbol:
        print(f"    [i] {symbol} was renamed from {old_symbol}, scraping old symbol too")
        old_rows = scrape_symbol_full(session, old_symbol)
        by_date = {r["Date"]: r for r in old_rows}
        by_date.update({r["Date"]: r for r in new_rows})
        rows = sorted(by_date.values(), key=lambda r: r["Date"])
        print(f"    [i] merged {old_symbol} ({len(old_rows)} rows) + {symbol} ({len(new_rows)} rows) -> {len(rows)} unique dates")
    else:
        rows = sorted(new_rows, key=lambda r: r["Date"])

    save_csv(symbol, rows)
    return len(rows)


def main():
    requested = sys.argv[1:]
    symbols = requested if requested else sorted(NEW_COMPANIES.keys())

    unknown = [s for s in symbols if s not in NEW_COMPANIES]
    if unknown:
        print(f"[!] Unknown symbol(s), not in NEW_COMPANIES: {unknown}")
        sys.exit(1)

    already_present = [s for s in symbols if (OUTPUT_DIR / f"{s}_psx_data.csv").exists()]
    if already_present:
        print(f"[i] Already have a raw CSV, will overwrite: {already_present}")

    print(f"[OK] Backfilling {len(symbols)} symbol(s), {START_YEAR}-2026")

    results = {}
    with requests.Session() as session:
        for symbol in symbols:
            try:
                row_count = backfill_ticker(session, symbol)
                results[symbol] = row_count
            except Exception as exc:
                print(f"[!] {symbol}: failed - {exc}")
                results[symbol] = None

    print("\n[OK] Backfill summary:")
    for symbol, count in results.items():
        status = f"{count} rows" if count is not None else "FAILED"
        print(f"  {symbol}: {status}")

    zero_row = [s for s, c in results.items() if c == 0]
    if zero_row:
        print(f"\n[!] Zero rows returned for: {zero_row} - check symbol against dps.psx.com.pk/screener")

    failed = [s for s, c in results.items() if c is None]
    if failed:
        print(f"\n[!] Failed entirely: {failed}")


if __name__ == "__main__":
    main()
