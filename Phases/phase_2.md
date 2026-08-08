# Phase 2: Feature Engineering

## Objective

Add a comprehensive set of predictive features to each of the 25 cleaned stock CSVs from Phase 1. These features will feed four models per stock (XGBoost, Random Forest, LSTM, Linear Regression), so the feature set needs to support accurate predictions across all four, not just one.

Input: Cleaned CSVs in `processed/` (from Phase 1, 25 stocks, no FFBL)
Output: Featured CSVs in `processed/` (same folder, features appended)

## Why a Comprehensive Feature Set

The goal here is accuracy, not minimalism. Tree-based models (XGBoost, Random Forest) handle redundant or correlated features well. They assign features low importance if the features do not help, so including extra signal rarely hurts and often helps. Linear Regression is more sensitive to correlated features, but it acts as a baseline here rather than the primary model, so a slightly noisier feature set is an acceptable tradeoff for now. Feature pruning based on importance scores happens later, in the modeling phase, once you can see what each model actually uses.

Every feature added below only uses information available at or before the current row's date. None of them peek into the future. The target columns (tomorrow's High and Low) are created separately at the end and clearly marked so there is no confusion between predictors and targets.

---

## Step 1: Returns and Price Behavior

Raw prices are harder to predict directly because they carry the stock's absolute price level (a Rs. 7,600 stock and a Rs. 150 stock behave very differently in raw terms). Returns normalize this.

```python
import pandas as pd
import numpy as np

def add_return_features(df):
    close = df['Close']
    
    # Percentage returns over different windows
    df['Return_1d'] = close.pct_change(1)
    df['Return_5d'] = close.pct_change(5)
    df['Return_10d'] = close.pct_change(10)
    df['Return_20d'] = close.pct_change(20)
    
    # Log returns (more stable for modeling, handles compounding better)
    df['Log_Return'] = np.log(close / close.shift(1))
    
    # Daily range as percentage of open
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Open']
    df['Open_Close_Range'] = (df['Close'] - df['Open']) / df['Open']
    
    return df
```

---

## Step 2: Technical Indicators

Standard indicators that capture trend, momentum, and volatility.

```python
def add_technical_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Simple Moving Averages
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['SMA_200'] = close.rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI_14'] = calculate_rsi(close, 14)
    
    # MACD
    df['MACD_line'] = df['EMA_12'] - df['EMA_26']
    df['Signal_line'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD_line'] - df['Signal_line']
    
    # Bollinger Bands
    df['BB_middle'] = df['SMA_20']
    std = close.rolling(window=20).std()
    df['BB_upper'] = df['SMA_20'] + (2 * std)
    df['BB_lower'] = df['SMA_20'] - (2 * std)
    
    # ATR (Average True Range) - measures volatility
    def calculate_atr(high, low, close, period=14):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    df['ATR_14'] = calculate_atr(high, low, close, 14)
    
    # Volatility (rolling std of returns - distinct from ATR, captures return volatility not price range)
    df['Volatility_20'] = close.pct_change().rolling(window=20).std()
    
    # Volume indicators
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20']
    
    return df
```

Note: `SMA_20` already is the rolling mean of Close, and `Volatility_20` already is the rolling standard deviation of returns. We are not duplicating these under different names elsewhere in this pipeline.

---

## Step 3: Lag Features

Give the models direct access to recent history as separate columns, rather than relying only on rolling windows.

```python
def add_lag_features(df):
    # Close price lags
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    
    # Return lags
    for lag in [1, 2, 5]:
        df[f'Return_lag_{lag}'] = df['Return_1d'].shift(lag)
    
    # Volume lags
    for lag in [1, 5]:
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
    
    return df
```

This step must run after `add_return_features`, since it lags the `Return_1d` column created there.

---

## Step 4: Rolling Statistics (Selective)

Only the rolling statistics that are not already covered by SMA or Volatility.

```python
def add_rolling_stats(df):
    close = df['Close']
    
    df['Rolling_Max_20'] = close.rolling(window=20).max()
    df['Rolling_Min_20'] = close.rolling(window=20).min()
    
    return df
```

Rolling mean and rolling standard deviation are intentionally skipped here. `SMA_20` and `Volatility_20` already cover that information, and adding near-duplicate columns only adds noise for Linear Regression and unnecessary computation for the tree models.

---

## Step 5: Intraday and Candle Features

Cheap to compute, and they capture information about the shape of each day's trading that OHLC alone does not spell out directly.

```python
def add_intraday_features(df):
    df['Price_Range'] = df['High'] - df['Low']
    df['Body'] = df['Close'] - df['Open']
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Range_Percentage'] = df['Price_Range'] / df['Open']
    
    return df
```

---

## Step 6: Date Features (Optional, Low Priority)

Include these for the model to test, but do not expect them to carry much weight. Never encode the date itself as a raw number.

```python
def add_date_features(df):
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 4=Friday
    df['Month'] = df['Date'].dt.month
    
    return df
```

---

## Step 7: Target Variables

The target is tomorrow's High and Low. Create these now, per stock, before merging, since each CSV is already a single continuous ticker and does not need a groupby.

```python
def add_target_variables(df):
    df['Target_High'] = df['High'].shift(-1)
    df['Target_Low'] = df['Low'].shift(-1)
    
    return df
```

The last row of each stock will have no target (there is no "tomorrow" for it) and will be dropped during the train split step in Phase 3.

---

## Step 8: Data Leakage Check

Before moving on, confirm that nothing in the feature set uses future information.

```python
def check_no_leakage(df, feature_columns):
    """
    Sanity check: correlate each feature with the target using only
    the training portion, and manually inspect any feature with a 
    suspiciously perfect correlation (close to 1.0), which usually 
    signals a leak.
    """
    correlations = df[feature_columns + ['Target_High']].corr()['Target_High'].sort_values(ascending=False)
    print("Feature correlation with Target_High:")
    print(correlations)
    
    suspicious = correlations[(correlations.abs() > 0.98) & (correlations.index != 'Target_High')]
    if len(suspicious) > 0:
        print("\nWARNING: These features may be leaking future data:")
        print(suspicious)
    else:
        print("\nNo obvious leakage detected.")
```

A correlation near 1.0 between a feature and the target is a red flag. It usually means the feature accidentally includes the same future value the target is supposed to predict. Everything built in Steps 1 through 6 above uses `shift()` with positive values or `rolling()` windows ending at the current row, so this should come back clean, but it is worth running once per stock to be sure.

---

## Step 9: Missing Value Handling

The rolling windows (especially `SMA_200`) mean the first 200 rows of each stock will have NaN values in some columns. Do not fill these with zero or interpolate them. Leave them as NaN here; they get dropped in Phase 3 during the train/test split, once per stock, so you don't lose data unnecessarily at this stage.

```python
def report_missing_values(df, ticker):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"{ticker}: {len(missing)} columns have missing values (expected, from rolling windows)")
        print(f"  Rows affected: up to {missing.max()} rows (leading rows only)")
    else:
        print(f"{ticker}: no missing values")
```

---

## Full Feature Engineering Function (One Stock)

Combining all steps in the correct order.

```python
def engineer_features(df):
    df = add_return_features(df)
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df = add_intraday_features(df)
    df = add_date_features(df)
    df = add_target_variables(df)
    return df
```

Order matters here. Lag features depend on `Return_1d` existing first, and everything else is independent, but keeping this order keeps the function readable and easy to debug if one step produces unexpected NaNs.

---

## Batch Process All 25 Stocks

```python
import os

processed_dir = 'processed/'

feature_columns_for_leakage_check = [
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

stock_files = [f for f in os.listdir(processed_dir) if f.endswith('_cleaned.csv')]

print(f"Found {len(stock_files)} stocks to process\n")

for filename in sorted(stock_files):
    ticker = filename.replace('_cleaned.csv', '')
    filepath = os.path.join(processed_dir, filename)
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = engineer_features(df)
    
    report_missing_values(df, ticker)
    check_no_leakage(df, feature_columns_for_leakage_check)
    
    # Save with features, same filename (overwrite)
    df.to_csv(filepath, index=False)
    print(f"{ticker}: saved with {len(df.columns)} total columns\n")

print("Feature engineering complete for all 25 stocks.")
```

---

## Feature Set Summary

| Category | Features | Count |
|---|---|---|
| Raw OHLCV | Open, High, Low, Close, Volume | 5 |
| Returns | Return_1d, Return_5d, Return_10d, Return_20d, Log_Return, High_Low_Range, Open_Close_Range | 7 |
| Technical Indicators | SMA_20/50/200, EMA_12/26, RSI_14, MACD_line, Signal_line, MACD_histogram, BB_upper/middle/lower, ATR_14, Volatility_20, Volume_SMA_20, Volume_Ratio | 16 |
| Lag Features | Close_lag (1,2,3,5,10), Return_lag (1,2,5), Volume_lag (1,5) | 10 |
| Rolling Stats | Rolling_Max_20, Rolling_Min_20 | 2 |
| Intraday | Price_Range, Body, Upper_Wick, Lower_Wick, Range_Percentage | 5 |
| Date | DayOfWeek, Month | 2 |
| Targets | Target_High, Target_Low | 2 |

Total predictor columns: about 47, plus Date and the two targets.

---

## Note on Per-Stock Accuracy Tracking

Since the plan is to train four models (XGBoost, Random Forest, LSTM, Linear Regression) and compare accuracy per stock, the `Ticker` column gets added during the merge step in Phase 3, not here. Keep the individual stock CSVs separate through this phase so each one can still be inspected and validated on its own before they get combined. Once merged, the evaluation step in Phase 4 will group results by `Ticker` so you can see, for example, that XGBoost wins on MEBL while Random Forest wins on HUBC, and pick accordingly per stock.

---

## Checklist

When Phase 2 is complete, you should have:

- [ ] All 25 stock CSVs updated with the full feature set (about 47 predictor columns plus Date and targets)
- [ ] Leakage check run on each stock, no suspicious correlations flagged
- [ ] Missing value report reviewed (NaNs only in the first ~200 rows per stock, from rolling windows)
- [ ] Target_High and Target_Low columns present and correctly shifted
- [ ] No feature accidentally built from future data

---

## Troubleshooting

**Leakage check flags a feature:**
Check whether that feature uses `shift()` with a negative number by mistake, or a rolling window that somehow includes a future index. Recheck the exact line where that feature is built.

**Too many NaN rows after merging in Phase 3:**
This is expected from SMA_200 needing 200 prior rows. If a stock has fewer than 200 total rows, all of `SMA_200` will be NaN for that stock. Check row counts from the Phase 1 summary report before worrying about this.

**RSI or ATR producing infinite values:**
Happens when the denominator (loss, in RSI's case) is zero for a stretch of unchanged prices. Replace infinities with NaN after calculation: `df['RSI_14'] = df['RSI_14'].replace([np.inf, -np.inf], np.nan)`.

---

## Next Step

After this, move to **Phase 3: Merge into Master Dataset**, where all 25 featured CSVs get combined into one file with a `Ticker` column, ready for the train and test split and then model training in Phase 4.