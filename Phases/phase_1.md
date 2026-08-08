# Phase 1: Data Cleaning

## Objective

Clean each of your 25 stock CSVs individually (excluding FFBL). Handle missing values, remove duplicates, fix date formats, and validate data quality before moving to feature engineering.

Input: Raw CSVs in `raw/` folder (25 stocks, no FFBL)
Output: Cleaned CSVs in `processed/` folder

**Note:** FFBL is excluded because it stopped trading on 2024-12-20 due to merger into FFC. We only train on stocks with complete data through 2026-08-06.

---

## What You're Cleaning

Each stock CSV from PSX scraper likely has these columns:

```
Date,Open,High,Low,Close,Volume
2018-01-02,185.50,187.00,185.00,186.50,123456
2018-01-03,186.50,188.00,186.00,187.00,134567
```

**Common issues in raw data:**

1. Missing values (NaN in OHLCV columns)
2. Duplicate rows for the same date
3. Date format inconsistencies (DD/MM/YYYY vs MM/DD/YYYY vs YYYY-MM-DD)
4. Negative prices or volumes (data corruption)
5. Wrong data types (dates as strings, volumes as text)
6. Duplicate index values or malformed rows

---

## Cleaning Steps (In Order)

### Step 1: Load the CSV

```python
import pandas as pd
import os

# Load one stock
df = pd.read_csv('raw/SYS.csv')

# Check what you got
print(df.head())
print(df.info())
print(df.shape)  # Rows, columns
```

**What to look for:**

- Is Date a string or datetime object?
- Any columns with weird names (extra spaces, lowercase)?
- Data types correct? (Open/High/Low/Close should be float, Volume should be int)

---

### Step 2: Fix Date Format

PSX portal might give dates in different formats. Standardize to YYYY-MM-DD.

```python
# Try parsing as datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# If that fails, try other formats:
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Check result
print(df['Date'].dtype)  # Should be datetime64[ns]
print(df['Date'].head())
```

---

### Step 3: Fix Data Types

Make sure columns are the right type.

```python
# Convert OHLCV to correct types
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce', downcast='integer')

# errors='coerce' converts invalid values to NaN
# You'll handle these NaNs in next step

print(df.dtypes)
```

---

### Step 4: Remove Duplicates

Check for and remove exact duplicate rows (same date).

```python
# How many duplicates?
print(f"Duplicates before: {df.duplicated(subset=['Date']).sum()}")

# Remove duplicates - keep first occurrence
df = df.drop_duplicates(subset=['Date'], keep='first')

print(f"Duplicates after: {df.duplicated(subset=['Date']).sum()}")
print(f"Total rows now: {len(df)}")
```

**Why keep first? Convention. Could keep 'last' if you prefer.**

---

### Step 5: Remove Rows with Missing OHLCV

Drop any row that has NaN in price or volume columns.

```python
# Check missing values
print("Missing values before:")
print(df.isnull().sum())

# Drop rows with missing OHLCV
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

print("\nMissing values after:")
print(df.isnull().sum())
print(f"Total rows now: {len(df)}")
```

**Note:** First 200 rows will have NaNs in technical indicators (added later). That's fine. Only remove NaNs in raw OHLCV here.

---

### Step 6: Sort by Date

Ensure data is chronologically ordered, oldest first.

```python
# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

# Verify
print(df[['Date', 'Close']].head(10))
print(df[['Date', 'Close']].tail(10))
```

---

### Step 7: Data Validation

Check for impossible values.

```python
# High should be >= Close >= Low
impossible_high_low = df['High'] < df['Low']
if impossible_high_low.any():
    print(f"WARNING: {impossible_high_low.sum()} rows have High < Low")
    print(df[impossible_high_low])

# Close should be between High and Low (mostly)
impossible_close = (df['Close'] > df['High']) | (df['Close'] < df['Low'])
if impossible_close.any():
    print(f"WARNING: {impossible_close.sum()} rows have Close outside High-Low range")
    print(df[impossible_close])

# No negative prices
negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any(axis=1)
if negative_prices.any():
    print(f"ERROR: {negative_prices.sum()} rows have negative prices")
    print(df[negative_prices])
    df = df[~negative_prices]  # Remove them

# No negative or zero volumes
# (Zero volume is OK - means no trading that day)
negative_volume = df['Volume'] < 0
if negative_volume.any():
    print(f"ERROR: {negative_volume.sum()} rows have negative volume")
    df = df[~negative_volume]

print("Validation complete!")
```

---

### Step 8: Save Cleaned CSV

```python
# Create output directory if not exists
os.makedirs('processed/', exist_ok=True)

# Save cleaned data
output_path = 'processed/SYS_cleaned.csv'
df.to_csv(output_path, index=False)

print(f"Saved cleaned data to {output_path}")
print(f"Rows: {len(df)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
```

---

## Full Cleaning Script for One Stock

```python
import pandas as pd
import os

def clean_stock_data(input_path, output_path):
    """Clean a single stock CSV file."""
    
    print(f"\nCleaning {input_path}...")
    
    # Load
    df = pd.read_csv(input_path)
    print(f"  Loaded: {len(df)} rows")
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    print(f"  Dates parsed")
    
    # Fix data types
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    print(f"  Data types fixed")
    
    # Remove duplicates
    duplicates = df.duplicated(subset=['Date']).sum()
    df = df.drop_duplicates(subset=['Date'], keep='first')
    print(f"  Removed {duplicates} duplicates")
    
    # Remove rows with missing OHLCV
    missing_before = df.isnull().sum().sum()
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    missing_after = df.isnull().sum().sum()
    print(f"  Removed rows with missing OHLCV ({missing_before} -> {missing_after} nulls)")
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"  Sorted by date")
    
    # Validation
    errors = 0
    
    # Check High >= Low
    if (df['High'] < df['Low']).any():
        print(f"  WARNING: {(df['High'] < df['Low']).sum()} rows have High < Low")
        errors += 1
    
    # Check negative prices
    if (df[['Open', 'High', 'Low', 'Close']] < 0).any().any():
        print(f"  WARNING: Negative prices found")
        df = df[(df[['Open', 'High', 'Low', 'Close']] >= 0).all(axis=1)]
        errors += 1
    
    # Check negative volume
    if (df['Volume'] < 0).any():
        print(f"  WARNING: Negative volumes found")
        df = df[df['Volume'] >= 0]
        errors += 1
    
    if errors == 0:
        print(f"  Validation: OK")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  Final: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df

# Clean SYS
clean_stock_data('raw/SYS.csv', 'processed/SYS_cleaned.csv')
```

---

## Batch Process All 25 Stocks (Excluding FFBL)

Once you have the function above working, clean all stocks except FFBL:

```python
import os

raw_dir = 'raw/'
processed_dir = 'processed/'

# List all CSV files, excluding FFBL
stock_files = [f for f in os.listdir(raw_dir) 
               if f.endswith('.csv') and f != 'FFBL.csv']

print(f"Found {len(stock_files)} stock files to clean (FFBL excluded)\n")

for filename in sorted(stock_files):
    input_path = os.path.join(raw_dir, filename)
    output_filename = filename.replace('.csv', '_cleaned.csv')
    output_path = os.path.join(processed_dir, output_filename)
    
    clean_stock_data(input_path, output_path)

print(f"\nAll {len(stock_files)} stocks cleaned!")
print("Note: FFBL was excluded (incomplete data due to merger)")
```

---

## Summary Report After Cleaning

After cleaning all 26 stocks, generate a summary:

```python
import os
import pandas as pd

processed_dir = 'processed/'

summary = []

for filename in sorted(os.listdir(processed_dir)):
    if filename.endswith('_cleaned.csv'):
        ticker = filename.replace('_cleaned.csv', '')
        df = pd.read_csv(os.path.join(processed_dir, filename))
        df['Date'] = pd.to_datetime(df['Date'])
        
        summary.append({
            'Ticker': ticker,
            'Rows': len(df),
            'Start Date': df['Date'].min().date(),
            'End Date': df['Date'].max().date(),
            'Days Covered': (df['Date'].max() - df['Date'].min()).days
        })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('data/cleaning_summary.csv', index=False)
```

**Expected output (25 stocks, FFBL excluded):**

```
 Ticker  Rows   Start Date     End Date  Days Covered
    SYS  2120  2018-01-02  2026-08-06          3111
   MEBL  2100  2018-01-02  2026-08-06          3111
   HUBC  2110  2018-01-02  2026-08-06          3111
    FFC  2095  2018-01-02  2026-08-06          3111
  PAKT  1809  2018-01-02  2026-08-06          3111
  NESTLE 1650 2018-01-02  2026-08-06          3111
  ...
```

All 25 stocks will have data through 2026-08-06 (no FFBL with early cutoff).

---

## Known Data Issues

**FFBL (Fauji Fertilizer Bin Qasim):**
- EXCLUDED from training (stops at 2024-12-20 due to merger into FFC)
- Do not process this file

**PAKT, NESTLE, COLG, ILP:**
- Fewer rows than others (1,600-1,800 vs 2,100+)
- Reason: Thin-traded stocks with no trading on certain days
- This is legitimate - don't fill or interpolate
- Keep as is

---

## Checklist

When Phase 1 is complete, you should have:

- [ ] 25 cleaned CSVs in `processed/` folder (FFBL excluded)
- [ ] Each cleaned CSV has proper date format (YYYY-MM-DD)
- [ ] No missing values in OHLCV columns
- [ ] No duplicate rows
- [ ] Data sorted chronologically
- [ ] All prices and volumes valid (no negatives)
- [ ] Summary report generated showing row counts per stock (25 total)

After this, move to **Phase 2: Feature Engineering** (technical indicators).

---

## Troubleshooting

**Date parsing fails:**
- Check your raw data format first: `head raw/SYS.csv`
- Try different format strings: `'%d/%m/%Y'`, `'%m/%d/%Y'`, etc.
- If mixed formats, parse in try/except loop

**Data type conversion fails:**
- Some cells might have text (e.g., "N/A", "-", empty strings)
- Use `errors='coerce'` to convert these to NaN
- Then drop rows with NaN in OHLCV

**After cleaning, too few rows:**
- Check if you're dropping too many rows
- Print which rows were dropped and why
- Adjust validation thresholds if needed

**Duplicates not found:**
- Some duplicates might have slightly different times
- If Time column exists, include it: `drop_duplicates(['Date', 'Time'])`

---

## Next Step

After cleaning all 26 stocks, move to **Phase 2: Feature Engineering** to add technical indicators (SMA, EMA, RSI, MACD, etc.) to each cleaned CSV.
