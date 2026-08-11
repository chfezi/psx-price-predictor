# Phase 10: Daily Data Automation

## Objective

Automate the PSX data pipeline so it runs on every trading day without manual work, using GitHub Actions. Replace the old Selenium-based `PSX_DATA_SCRAPPER.py` with a lightweight incremental scraper that only fetches the newest day's data per stock, then runs the existing cleaning, feature engineering, and merge scripts so the dataset stays model-ready at all times.

## Why the old scraper is not reused

`PSX_DATA_SCRAPPER.py` has three problems that rule it out for automation:

- It calls `input("Press ENTER to start scraping...")`, which blocks forever in a non-interactive environment like GitHub Actions.
- It only covers 6 stocks (SYS, MEBL, HUBC, ENGRO, OGDC, FFC), while the project covers 25.
- It re-scrapes the full 8-year history on every run using Selenium and generic selectors against the live site's search box and table. This is slow, likely to get a cloud IP blocked, and fragile if PSX changes its page layout even slightly.

## New approach: direct EOD endpoint, no Selenium or Chrome

PSX's data portal exposes end-of-day data directly through a JSON endpoint per symbol:

```
https://dps.psx.com.pk/timeseries/eod/{SYMBOL}
```

This same endpoint is used by multiple independent open-source PSX tools (the `psxdata` Python library and the PSX MCP Server both scrape it directly). Fetching it with plain `requests` avoids Chrome and Selenium entirely, so it is far faster and more reliable to run on a schedule.

### Step 0: verify the response shape first, do not assume it

Before writing the parser, hit the endpoint for one symbol (e.g. HUBC) in a throwaway script and print the raw response. Confirm:

- exact field names (date, open, high, low, close, volume; keys may not match these exactly)
- date format (unix timestamp vs string, and which timezone)
- whether a `User-Agent` header is required to avoid a block (recommended to always send a normal browser UA string regardless)
- how many rows the endpoint returns by default (recent N days vs full history)

Write the parser against what is actually observed in this step, not against assumptions.

## Step 1: get the ticker list dynamically

Do not hand-type the 25 tickers into the new script, there is real risk of a typo breaking one stock silently. Read the ticker list from the existing dataset instead, for example the unique values in `master_dataset.csv`'s `Ticker` column, or the classes of the saved `LabelEncoder`. This guarantees the scraper always matches exactly what the rest of the project already tracks.

## Step 2: new incremental scraper (`daily_scraper.py`)

For each ticker:

1. Load its existing raw CSV and find the latest date already stored.
2. Fetch the EOD endpoint for that symbol.
3. Keep only rows dated after the latest stored date.
4. If there are no new rows (weekend, PSX holiday, endpoint not updated yet), log it and move on. This is a normal outcome, not an error.
5. Append new rows to the raw CSV, matching the exact column names and order that `clean_data.py` already expects.
6. Wrap each request in a retry (e.g. 3 attempts with short backoff) and a timeout.
7. Add a small delay between tickers (1-2 seconds) to avoid tripping anti-bot protection.
8. Log a per-ticker summary (success, rows added, or failed) to stdout, so it is visible in the Actions run log.

## Step 3: run the rest of the pipeline

After scraping, run in this order:

1. `clean_data.py`
2. `feature_engineering.py`
3. `merge_and_split.py`

This keeps `master_dataset.csv` and the per-stock processed files current. Model training or retraining is out of scope here and stays a separate, manual phase. This automation only refreshes data.

## Step 4: GitHub Actions workflow

Create `.github/workflows/daily_scrape.yml`:

- Trigger on a cron schedule for weekdays, timed for after PSX market close. Regular PSX trading generally ends by mid-afternoon Pakistan time, so schedule the run for early evening PKT to stay safe (e.g. 18:00 PKT = 13:00 UTC). Also add a `workflow_dispatch` trigger so it can be run manually for testing.
- `actions/checkout@v4`, then `actions/setup-python@v5`.
- Install a slim dependency set for this job only: pandas, numpy, requests, and scikit-learn if `feature_engineering.py` needs it. Do not install the full `requirements.txt`, tensorflow, torch, xgboost, shap, and streamlit are not needed for a data-only job and would only slow it down.
- Run `daily_scraper.py`, then the three pipeline scripts in order.
- Check for changes with `git diff --quiet`. If there are changes, commit and push with a message like `Automated data update {date}`. If nothing changed, skip the commit cleanly, this is expected on holidays.
- Use the default `GITHUB_TOKEN` with `contents: write` permission set in the workflow. No separate personal access token is needed.

## Notes and risks worth flagging back after implementation

- Cloud IPs (GitHub-hosted runners) can occasionally get rate-limited by sites with anti-bot protection. If the scraper works for a while and then starts failing consistently, this is the likely cause. The fix is usually adding delay/headers, or shifting the schedule slightly.
- The earlier-found bug where 66 rows across the dataset have Close outside the [Low, High] range still needs a real fix in `clean_data.py`, since it will now also affect newly appended rows going forward, not just the historical backfill.
- News and sentiment data are still a separate, later addition per the original project notes, not part of this phase.