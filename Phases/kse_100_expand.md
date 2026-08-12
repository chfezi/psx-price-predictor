# KSE-100 expansion: target list and full pipeline

Source for the company list: PSX recomposition notice PSX/N-305, effective
April 1, 2026. Your confirmed 25 tickers (from `COMPANY_NAMES`) match 23 of
the 100 KSE-100 constituents. Two of your tickers aren't currently KSE-100
members: NETSOL isn't a constituent at all, and UNITY (Unity Foods) was
removed in the April 2026 recomposition, replaced by Arif Habib Corporation.
Neither needs any action, they stay in your dataset as is. That leaves 77
companies to add.

## Already in your dataset (23 of 100)

| Ticker | Company | Sector |
|---|---|---|
| BAHL | Bank AL Habib | Commercial banks |
| COLG | Colgate-Palmolive Pakistan | Food and personal care |
| DGKC | D. G. Khan Cement | Cement |
| EFERT | Engro Fertilizers | Fertilizer |
| ENGROH | Engro Holdings | Investment/securities |
| FFC | Fauji Fertilizer | Fertilizer |
| HBL | Habib Bank | Commercial banks |
| HUBC | Hub Power Company | Power generation |
| ILP | Interloop | Textile composite |
| INDU | Indus Motor Company | Automobile assembler |
| LUCK | Lucky Cement | Cement |
| MARI | Mari Energies (formerly Mari Petroleum) | Oil and gas exploration |
| MCB | MCB Bank | Commercial banks |
| MEBL | Meezan Bank | Commercial banks |
| MLCF | Maple Leaf Cement | Cement |
| NESTLE | Nestle Pakistan | Food and personal care |
| OGDC | Oil and Gas Development Co. | Oil and gas exploration |
| PAKT | Pakistan Tobacco | Tobacco |
| PPL | Pakistan Petroleum | Oil and gas exploration |
| PSO | Pakistan State Oil | Oil and gas marketing |
| SYS | Systems Limited | Technology and communication |
| TRG | TRG Pakistan | Technology and communication |
| UBL | United Bank | Commercial banks |

Not currently KSE-100 members, kept as is: NETSOL (NetSol Technologies),
UNITY (Unity Foods).

## Not yet in your dataset (77 of 100)

Four of these are a closed-end mutual fund, a modaraba, and two REITs. They
track a fund or property NAV rather than an operating business, worth
deciding on before adding.

### Close-end mutual fund, modaraba, REIT (reconsider)
- HBL Growth Fund
- First Habib Modaraba
- Dolmen City REIT
- TPL REIT Fund I

### Leasing, investment, securities companies
- Pak-Gulf Leasing Company
- Pakistan Stock Exchange
- DH Partners

### Commercial banks
- Bank Alfalah
- The Bank of Punjab
- Faysal Bank
- Habib Metropolitan Bank
- National Bank of Pakistan
- Standard Chartered Bank Pakistan
- Askari Bank
- Allied Bank

### Insurance
- Adamjee Insurance

### Textile (spinning, weaving, composite, woollen, synthetics)
- Gadoon Textile Mills
- Yousaf Weaving Mills
- Nishat Mills
- Mehmood Textile Mills
- Kohinoor Textile Mills
- Bannu Woollen Mills
- Ibrahim Fibres

### Sugar
- JDW Sugar Mills

### Cement
- Bestway Cement
- Fauji Cement Company
- Kohat Cement Company
- Cherat Cement Company
- Power Cement
- Pioneer Cement

### Refinery
- Attock Refinery
- Cnergyico PK

### Power generation and distribution
- Kot Addu Power Company
- K-Electric

### Oil and gas marketing / exploration
- Attock Petroleum
- Sui Northern Gas Pipelines
- Sui Southern Gas Company
- Pakistan Oilfields

### Engineering
- International Steels
- International Industries

### Automobile assembler and parts
- Millat Tractors
- Sazgar Engineering Works
- Atlas Honda
- Honda Atlas Cars Pakistan
- Ghandhara Automobiles
- Ghandhara Industries
- Thal

### Cables and electrical goods
- Pak Elektron

### Transport
- Pakistan International Bulk Terminal

### Technology and communication
- Air Link Communication
- Pakistan Telecommunication Company
- Hum Network

### Fertilizer
- Arif Habib Corporation
- Fatima Fertilizer Company

### Pharmaceutical
- The Searle Company
- AGP
- GlaxoSmithKline Pakistan
- Abbott Laboratories Pakistan
- Citi Pharma
- Haleon Pakistan
- Highnoon Laboratories

### Chemical
- Lucky Core Industries (formerly ICI Pakistan)
- Lotte Chemical Pakistan

### Paper, board and packaging
- Packages

### Vanaspati and allied
- S.S. Oil Mills

### Leather and tanneries
- Service Industries

### Food and personal care
- Fauji Foods
- Murree Brewery Company
- National Foods
- Rafhan Maize Products
- Unilever Pakistan Foods

### Glass and ceramics
- Tariq Glass Industries
- Ghani Glass

### Miscellaneous
- Pakistan Services
- Pakistan Aluminium Beverage Cans
- Shifa International Hospitals

### Property
- Javedan Corporation

## Ticker skeleton (verified, drop into COMPANY_NAMES)

Decision: the 4 fund/REIT/modaraba entries (HBL Growth Fund, First Habib
Modaraba, Dolmen City REIT, TPL REIT Fund I) are excluded. They track a
NAV rather than an operating business and don't fit the same
feature/target pipeline. 73 companies added, not 77.

Every symbol below was cross-checked against a live dps.psx.com.pk company
page (confirming the listed legal name matches) plus a second source.
Four were renamed and their scrip code changed within the 2018-2026
backfill window - Stage 1 needs to pull both symbols and merge:

| New symbol | Company | Old symbol | Renamed |
|---|---|---|---|
| GAL | Ghandhara Automobiles (formerly Ghandhara Nissan) | GHNL | Apr 2023 |
| HALEON | Haleon Pakistan (formerly GSK Consumer Healthcare) | GSKCH | Jan 2023 |
| LCI | Lucky Core Industries (formerly ICI Pakistan) | ICI | Dec 2022 |

(Lotte Chemical Pakistan was renamed from Lotte Pakistan PTA in 2013,
before the backfill window starts, so LOTCHEM alone covers 2018-2026 - no
merge needed there.)

Two pairs of similarly-named but distinct listed entities, don't conflate:
Ghandhara Automobiles (GAL) vs Ghandhara Industries (GHNI); Atlas Honda
(ATLH, motorcycles) vs Honda Atlas Cars Pakistan (HCAR, cars).

```python
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

# Renamed tickers: scrape both the old and new symbol's history and merge
# into one continuous series under the new symbol before Stage 2.
RENAMED_TICKERS = {
    "GAL": "GHNL",
    "HALEON": "GSKCH",
    "LCI": "ICI",
}
```

### DHPL excluded after Stage 4

DHPL (DH Partners) is a 2024 demerger entity (spun out of Dawood
Hercules/Engro Holdings) with only ~276 rows of history, none of which
fall in the 2018-2023 train or 2024 test splits - every row lands in the
2025-2026 validate window. `evaluate_per_stock_h` in
`train_models_phase9.py` builds its per-stock comparison table by
iterating tickers present in the *test* set, so DHPL would never get a row
in `phase9_model_comparison_per_stock.csv`, and
`generate_predictions.py`'s `choose_best_model()` calls `.idxmax()` on
that (empty, for DHPL) filtered frame - a crash, not a silent gap.

Added to `clean_data.py`'s `EXCLUDED_TICKERS` (same mechanism as the
existing FFBL exclusion) rather than building a no-backtest fallback path.
Lands at 97/100 KSE-100 constituents instead of 98. Revisit once DHPL has
2+ years of history to actually backtest against - its raw CSV is kept in
psx_data_8years/ for that.

### Deviation from the endpoint named above

This doc's Stage 1 section (below) names the `timeseries/eod/{SYMBOL}`
endpoint for the backfill. Phase 10's own notes (`daily_scraper.py`'s
docstring) already found that endpoint returns only
`[timestamp, close, volume, open]` - no High/Low - which `clean_data.py`'s
Close-in-range check and several `feature_engineering.py` features (ATR,
Bollinger Bands, wicks) depend on. The backfill script instead reuses the
`POST /historical` endpoint and `fetch_month`/`parse_rows` from
`PSX_DATA_SCRAPPER.py`, same as `daily_scraper.py` already does, since
that one is proven to return full OHLCV.

## Full pipeline, ticker verification through deployment

### Stage 0: verify tickers

Look up each of the 77 companies on `dps.psx.com.pk/screener` and fill in
the real symbol. Decide on the 4 fund/REIT/modaraba entries here too. Cut
the list down to whatever you're actually adding before moving on.

### Stage 1: one-time historical backfill

Write a separate script from the Phase 10 daily scraper. Same
`dps.psx.com.pk/timeseries/eod/{SYMBOL}` endpoint, called once per new
ticker with the full 2018 to 2026 date range instead of a single day. Save
each ticker's raw output the same way `PSX_DATA_SCRAPPER.py` did originally,
one CSV per ticker.

A few of these companies were renamed or restructured (Mari Petroleum
became Mari Energies, ICI Pakistan became Lucky Core Industries). Check
whether the historical scrip code changed at the same time. If the old
history sits under a different symbol, you may need to pull it separately
and merge it with the current symbol's data before treating it as one
continuous series.

### Stage 2: clean_data.py

Run unchanged. It already validates missing values, duplicates, date
formatting, and the Close-outside-[Low,High] check that Phase 9 folded in.
Watch the console output for each new ticker. A few of the newly added
names may be thin-traded like PAKT, NESTLE, COLG, and ILP already are,
that's expected and not a bug on its own.

### Stage 3: feature_engineering.py

Run unchanged, once per new ticker's cleaned CSV. Produces the same ~47
features plus Target_Open/Target_Close and the horizon targets
(Target_{Open,Close}_{Return}_{1,5,10,20,60}d).

### Stage 4: merge_and_split.py

This is where the new tickers actually join the dataset. The LabelEncoder
for Ticker needs to fit on all tickers together (existing 25 or 23, plus
whichever new ones you're adding), not be extended after the fact, so this
step has to rerun over the full combined set rather than appending encoded
rows from a separately-fit encoder. Same time-based split as before, train
2018 to 2023, test 2024, validate 2025 to 2026.

### Stage 5: retrain, not fine-tune

All 50 models (5 horizons x 5 model types x 2 targets) need a full retrain
on the combined dataset, not an incremental update. The Ticker feature is
categorical: XGBoost and Random Forest split on it directly, and the
LSTM/GRU pipeline uses the same encoder for scaling and sequence
construction. Neither setup can absorb new categories into an
already-trained model without retraining from scratch.

Use `train_models_phase9.py` as the base. Expect meaningfully longer
training time since the dataset grows with every ticker added, run
LSTM/GRU and XGBoost's GPU path if available, Random Forest and Linear
Regression stay CPU-only either way, as noted in Phase 9. Keep the
per-epoch loss logging and the training log CSV the same way Phase 9 set
it up.

### Stage 6: post-training updates

- `manage_model_storage.py`: update the ticker list and confirm filename
  patterns still cover the full set.
- `COMPANY_NAMES` in `app.py`: merge in the verified symbols from the
  skeleton above.
- Whatever script assembles `phase9_predictions.csv`: rerun for the new
  tickers, and check nothing in it or in the dashboard hardcodes a count
  of 25 (summary stat cards, array sizes, loop bounds).
- Phase 10's daily scraper: add the new tickers to its list so daily
  updates cover all of them going forward, not just the original set.

### Stage 7: storage

Ten Random Forest pickles already run about 1.1 GB at 25 stocks across 5
horizons, and pickle size grows with training set size as well as ticker
count. Expect this to grow further. Plan on GitHub Release assets or
external storage for the model files rather than trying to keep them in
git, same as the existing phase6/phase9 files already need.

### Stage 8: validate before treating it as live

- Confirm accuracy, directional accuracy, and improvement-over-naive for
  the new tickers land in a similar range to the existing 23, rather than
  assuming the shared model generalizes to them for free.
- Check the error-distribution range and ensemble weights compute cleanly
  for the new tickers, since both are derived from each ticker's own
  backtested errors rather than hardcoded.
- Spot-check a few of the newly added, more thinly traded names the same
  way PAKT, NESTLE, COLG, and ILP were checked, rather than assuming every
  addition behaves like the liquid names.