# Phase 3: Merge and Prepare for Modeling

## Objective

Combine all 25 featured stock CSVs from Phase 2 into one master dataset with a `Ticker` column, then prepare clean train, test, and validation splits ready for Phase 4.

This phase does not do any prediction or modeling. It only merges, splits, and packages the data. Price range prediction (tomorrow's High to Low corridor) happens in Phase 4, once the four models (XGBoost, Random Forest, LSTM, Linear Regression) are trained on the data prepared here.

Input: Featured CSVs in `processed/` (from Phase 2, 25 stocks, ~50 columns each)
Output: `data/master_dataset.csv`, `data/train.csv`, `data/test.csv`, `data/validate.csv`, `ticker_encoder.pkl`

---

## Step 1: Merge All 25 Stocks

Add a `Ticker` column to each stock's data and combine into one dataframe.

```python
import pandas as pd
import os

def merge_all_stocks(processed_dir, output_path):
    all_dfs = []
    
    for filename in sorted(os.listdir(processed_dir)):
        if filename.endswith('_cleaned.csv'):
            ticker = filename.replace('_cleaned.csv', '')
            
            df = pd.read_csv(os.path.join(processed_dir, filename))
            df['Ticker'] = ticker
            
            all_dfs.append(df)
            print(f"Loaded {ticker}: {len(df)} rows")
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    master_df['Date'] = pd.to_datetime(master_df['Date'])
    master_df = master_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_df.to_csv(output_path, index=False)
    
    print(f"\nMaster dataset created: {len(master_df)} total rows")
    print(f"Columns: {len(master_df.columns)}")
    print(f"Date range: {master_df['Date'].min().date()} to {master_df['Date'].max().date()}")
    print(f"Unique tickers: {master_df['Ticker'].nunique()}")
    
    return master_df

master_df = merge_all_stocks('processed/', 'data/master_dataset.csv')
```

**Quick validation after merging:**

```python
print("\nRows per ticker:")
print(master_df.groupby('Ticker').size().sort_values())

print("\nMissing values per column (top 10):")
print(master_df.isnull().sum().sort_values(ascending=False).head(10))
```

The row counts per ticker here should match what Phase 1's summary report showed. The missing values should only appear in the technical indicator columns, and only in each stock's earliest rows.

---

## Step 2: Encode the Ticker Column

Most models need the ticker as a number, not a string. Use label encoding since all 25 tickers are known in advance and there is no ordinal relationship implied.

```python
from sklearn.preprocessing import LabelEncoder
import pickle

le = LabelEncoder()
master_df['Ticker_encoded'] = le.fit_transform(master_df['Ticker'])

# Save the encoder so Phase 4 and future predictions use the same mapping
os.makedirs('models/', exist_ok=True)
pickle.dump(le, open('models/ticker_encoder.pkl', 'wb'))

print(dict(zip(le.classes_, le.transform(le.classes_))))
```

Keep the original `Ticker` column in the dataset alongside `Ticker_encoded`. You will want the readable ticker name later for grouping results by stock in Phase 4's accuracy comparison.

---

## Step 3: Time-Based Train, Test, Validate Split

Stock data has a time order, so the split must respect that. Do not shuffle rows or use a random split. Random splitting would let the model see future rows during training and evaluate on rows that come before them, which inflates accuracy in a way that will not hold up in production.

```python
train_cutoff = pd.Timestamp('2024-01-01')
test_cutoff = pd.Timestamp('2025-01-01')

train_df = master_df[master_df['Date'] < train_cutoff].copy()
test_df = master_df[(master_df['Date'] >= train_cutoff) & (master_df['Date'] < test_cutoff)].copy()
validate_df = master_df[master_df['Date'] >= test_cutoff].copy()

print(f"Train: {len(train_df)} rows, {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
print(f"Test: {len(test_df)} rows, {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
print(f"Validate: {len(validate_df)} rows, {validate_df['Date'].min().date()} to {validate_df['Date'].max().date()}")
```

This gives roughly:

- **Train:** 2018 to 2023 (about 6 years, the bulk of the data, used to fit the models)
- **Test:** 2024 (used during Phase 4 to compare the four models against each other and pick the best one per stock)
- **Validate:** 2025 to 2026 (held out completely, touched only once at the very end, to confirm the chosen model still performs well on data it has never influenced in any way)

---

## Step 4: Drop Rows with Missing Values

Now that the split is time-based, drop the NaN rows. These come from the rolling window features (SMA_200 and similar) needing history that was not available in each stock's earliest days. Because the split is chronological and these NaNs sit at the very start of each stock's history, they fall almost entirely inside the training set.

```python
train_df = train_df.dropna()
test_df = test_df.dropna()
validate_df = validate_df.dropna()

print(f"Train after dropna: {len(train_df)} rows")
print(f"Test after dropna: {len(test_df)} rows")
print(f"Validate after dropna: {len(validate_df)} rows")
```

Do not fill these with zero or interpolate. A dropped row is honest. A filled row pretends the model had information it did not actually have.

---

## Step 5: Define Feature and Target Columns

Separate what goes into the model (features) from what the model is trying to predict (targets).

```python
feature_columns = [
    'Ticker_encoded',
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d', 'Log_Return',
    'High_Low_Range', 'Open_Close_Range',
    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI_14',
    'MACD_line', 'Signal_line', 'MACD_histogram',
    'BB_upper', 'BB_middle', 'BB_lower', 'ATR_14', 'Volatility_20',
    'Volume_SMA_20', 'Volume_Ratio',
    'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10',
    'Return_lag_1', 'Return_lag_2', 'Return_lag_5',
    'Volume_lag_1', 'Volume_lag_5',
    'Rolling_Max_20', 'Rolling_Min_20',
    'Price_Range', 'Body', 'Upper_Wick', 'Lower_Wick', 'Range_Percentage',
    'DayOfWeek', 'Month'
]

target_columns = ['Target_High', 'Target_Low']

X_train = train_df[feature_columns]
y_train = train_df[target_columns]

X_test = test_df[feature_columns]
y_test = test_df[target_columns]

X_validate = validate_df[feature_columns]
y_validate = validate_df[target_columns]

print(f"Feature count: {len(feature_columns)}")
print(f"X_train shape: {X_train.shape}")
```

`Date` and the plain `Ticker` string column are excluded from the feature list. `Date` is not a usable numeric input on its own, and `Ticker` is already represented by `Ticker_encoded`.

---

## Step 6: Feature Scaling (Deferred to Phase 4)

Scaling is not done here, on purpose. Tree-based models (XGBoost, Random Forest) do not need scaled inputs at all. LSTM and Linear Regression do, but the scaler must be fit only on training data and then applied to test and validation, and this needs to happen right before each model sees the data, not once globally here. Phase 4 will handle this per model, using the splits produced in this phase as the starting point.

---

## Step 7: Save the Prepared Splits

```python
os.makedirs('data/', exist_ok=True)

train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
validate_df.to_csv('data/validate.csv', index=False)

print("Saved train.csv, test.csv, validate.csv to data/")
```

---

## Sanity Checks Before Moving to Phase 4

```python
# Confirm every ticker appears in all three splits
print("Tickers in train:", train_df['Ticker'].nunique())
print("Tickers in test:", test_df['Ticker'].nunique())
print("Tickers in validate:", validate_df['Ticker'].nunique())
# All three should show 25

# Confirm no date overlap between splits
print("Train max date:", train_df['Date'].max())
print("Test min date:", test_df['Date'].min())
print("Test max date:", test_df['Date'].max())
print("Validate min date:", validate_df['Date'].min())
# Train max should be before test min, test max before validate min

# Confirm no NaN remains
print("NaN in train:", X_train.isnull().sum().sum())
print("NaN in test:", X_test.isnull().sum().sum())
print("NaN in validate:", X_validate.isnull().sum().sum())
# All three should be 0
```

If any ticker is missing from test or validate, that stock's IPO date, delisting, or a data gap needs a second look before Phase 4. If the NaN count is not zero anywhere, Step 4 needs to be rerun on that split.

---

## Checklist

When Phase 3 is complete, you should have:

- [ ] `data/master_dataset.csv` with all 25 stocks combined, `Ticker` and `Ticker_encoded` columns present
- [ ] `models/ticker_encoder.pkl` saved for later use
- [ ] `data/train.csv`, `data/test.csv`, `data/validate.csv` split by date, not shuffled
- [ ] No missing values in any of the three splits
- [ ] Feature and target columns clearly separated
- [ ] All 25 tickers present in each split
- [ ] No date overlap between train, test, and validate

---

## Troubleshooting

**A ticker is missing from test or validate:**
Check that stock's date range from the Phase 1 summary. If it started trading later than 2018 or has a shorter history, it may simply have no rows in an earlier split, which is fine. It becomes a problem only if it is missing from test or validate despite having recent data in `processed/`.

**NaN count is not zero after dropna:**
A column added in Phase 2 might contain infinities instead of NaN, most likely from RSI or ATR when the denominator was zero for a stretch. Run `df.replace([np.inf, -np.inf], np.nan)` before the dropna step and rerun.

**Train set feels too small or too large relative to test:**
Adjust `train_cutoff` and `test_cutoff`. The 2024 and 2025 boundaries used above assume the standard 2018 to 2026 data range from Phase 1. If your actual coverage differs, shift the cutoffs proportionally, keeping train as the largest portion, test as a solid middle stretch, and validate as the most recent, smallest, and completely untouched-until-the-end portion.

---

## Next Step

Phase 4 will use `train.csv`, `test.csv`, and `validate.csv` to train the four models (XGBoost, Random Forest, LSTM, Linear Regression) for each stock, report accuracy per model per stock so the best one can be picked, and generate the actual price range prediction (tomorrow's Low to High corridor) as the final output.