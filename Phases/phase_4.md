# Phase 4: Model Training, Evaluation, and Range Prediction

## Objective

Train four models (Linear Regression, Random Forest, XGBoost, LSTM) on the prepared data from Phase 3, evaluate each one's accuracy overall and broken down per stock, select the best model per stock, and generate the final output: tomorrow's predicted price range (Low to High) for each stock.

Each model type gets trained twice, once to predict `Target_High` and once to predict `Target_Low`. This keeps the comparison consistent across all four model types and makes the accuracy numbers directly comparable.

Input: `data/train.csv`, `data/test.csv`, `data/validate.csv` (from Phase 3)
Output: 8 trained models (4 types x 2 targets), an accuracy comparison table per stock, and a final range prediction function

---

## Step 1: Load the Prepared Data

```python
import pandas as pd
import numpy as np

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
validate_df = pd.read_csv('data/validate.csv')

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

X_train = train_df[feature_columns]
X_test = test_df[feature_columns]
X_validate = validate_df[feature_columns]

y_train_high = train_df['Target_High']
y_train_low = train_df['Target_Low']
y_test_high = test_df['Target_High']
y_test_low = test_df['Target_Low']
y_validate_high = validate_df['Target_High']
y_validate_low = validate_df['Target_Low']
```

---

## Step 2: Feature Scaling (for Linear Regression and LSTM only)

Tree models do not need this. Fit the scaler on training data only, then apply the same fit to test and validate, never the other way around.

```python
from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validate_scaled = scaler.transform(X_validate)

pickle.dump(scaler, open('models/feature_scaler.pkl', 'wb'))
```

---

## Step 3: Evaluation Function

Regression models do not have a single "accuracy" number the way classifiers do, so this pipeline reports four metrics together and adds one intuitive accuracy-style percentage for quick comparison.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred, model_name, target_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    accuracy_pct = 100 - mape  # intuitive stand-in, not a substitute for the other three
    
    return {
        'Model': model_name,
        'Target': target_name,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'R2': round(r2, 4),
        'Accuracy_%': round(accuracy_pct, 2)
    }
```

`MAE` is the average error in rupees, easy to explain to anyone. `RMSE` punishes large misses more than small ones. `MAPE` expresses the error as a percentage, which is what makes stocks at very different price levels comparable to each other. `R2` shows how much of the price movement the model actually explains versus a naive guess. `Accuracy_%` is just `100 - MAPE`, included because it reads intuitively, but the other four numbers are the ones that matter for real judgment.

---

## Step 4: Model 1 - Linear Regression (Baseline)

```python
from sklearn.linear_model import LinearRegression

lr_high = LinearRegression()
lr_high.fit(X_train_scaled, y_train_high)
pred_lr_high = lr_high.predict(X_test_scaled)

lr_low = LinearRegression()
lr_low.fit(X_train_scaled, y_train_low)
pred_lr_low = lr_low.predict(X_test_scaled)

results = []
results.append(evaluate_predictions(y_test_high, pred_lr_high, 'Linear Regression', 'High'))
results.append(evaluate_predictions(y_test_low, pred_lr_low, 'Linear Regression', 'Low'))

pickle.dump(lr_high, open('models/lr_high.pkl', 'wb'))
pickle.dump(lr_low, open('models/lr_low.pkl', 'wb'))
```

Linear Regression is the baseline here, mainly to show whether the more complex models are actually earning their keep. Given the correlated lag and price-level features flagged back in Phase 2, do not be surprised if this model's coefficients look unstable even while its accuracy looks reasonable. If accuracy is noticeably worse than the other three, that is expected and fine. It is here for comparison, not because it needs to win.

---

## Step 5: Model 2 - Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

rf_high = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_high.fit(X_train, y_train_high)
pred_rf_high = rf_high.predict(X_test)

rf_low = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_low.fit(X_train, y_train_low)
pred_rf_low = rf_low.predict(X_test)

results.append(evaluate_predictions(y_test_high, pred_rf_high, 'Random Forest', 'High'))
results.append(evaluate_predictions(y_test_low, pred_rf_low, 'Random Forest', 'Low'))

pickle.dump(rf_high, open('models/rf_high.pkl', 'wb'))
pickle.dump(rf_low, open('models/rf_low.pkl', 'wb'))
```

Random Forest uses the unscaled features directly. `n_estimators=200` and `max_depth=15` are reasonable starting points. If training feels slow given 25 stocks worth of data, `n_jobs=-1` uses all available CPU cores.

---

## Step 6: Model 3 - XGBoost

```python
import xgboost as xgb

xgb_high = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_high.fit(X_train, y_train_high, eval_set=[(X_test, y_test_high)], verbose=False)
pred_xgb_high = xgb_high.predict(X_test)

xgb_low = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_low.fit(X_train, y_train_low, eval_set=[(X_test, y_test_low)], verbose=False)
pred_xgb_low = xgb_low.predict(X_test)

results.append(evaluate_predictions(y_test_high, pred_xgb_high, 'XGBoost', 'High'))
results.append(evaluate_predictions(y_test_low, pred_xgb_low, 'XGBoost', 'Low'))

xgb_high.save_model('models/xgb_high.json')
xgb_low.save_model('models/xgb_low.json')
```

The `eval_set` parameter lets XGBoost track test performance during training, useful later if you want to add early stopping to prevent overfitting.

---

## Step 7: Model 4 - LSTM

LSTM needs sequences, not flat rows. Each input sample is a window of the past N days' features for one stock, so the sequences must be built per ticker and never cross from one stock into another.

```python
def create_sequences(df, feature_columns, target_column, ticker_column='Ticker', window_size=30):
    X_sequences = []
    y_targets = []
    
    for ticker in df[ticker_column].unique():
        ticker_df = df[df[ticker_column] == ticker].sort_values('Date').reset_index(drop=True)
        
        features = ticker_df[feature_columns].values
        targets = ticker_df[target_column].values
        
        for i in range(window_size, len(ticker_df)):
            X_sequences.append(features[i-window_size:i])
            y_targets.append(targets[i])
    
    return np.array(X_sequences), np.array(y_targets)

WINDOW_SIZE = 30

# Scale features first (fit on train, reuse for test and validate)
train_df_scaled = train_df.copy()
test_df_scaled = test_df.copy()

train_df_scaled[feature_columns] = scaler.transform(train_df[feature_columns])
test_df_scaled[feature_columns] = scaler.transform(test_df[feature_columns])

X_train_seq_high, y_train_seq_high = create_sequences(train_df_scaled, feature_columns, 'Target_High', window_size=WINDOW_SIZE)
X_test_seq_high, y_test_seq_high = create_sequences(test_df_scaled, feature_columns, 'Target_High', window_size=WINDOW_SIZE)

X_train_seq_low, y_train_seq_low = create_sequences(train_df_scaled, feature_columns, 'Target_Low', window_size=WINDOW_SIZE)
X_test_seq_low, y_test_seq_low = create_sequences(test_df_scaled, feature_columns, 'Target_Low', window_size=WINDOW_SIZE)

print(f"LSTM training shape: {X_train_seq_high.shape}")  # (samples, 30, num_features)
```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# High model
lstm_high = build_lstm_model((WINDOW_SIZE, len(feature_columns)))
lstm_high.fit(
    X_train_seq_high, y_train_seq_high,
    validation_data=(X_test_seq_high, y_test_seq_high),
    epochs=50, batch_size=64, callbacks=[early_stop], verbose=1
)
pred_lstm_high = lstm_high.predict(X_test_seq_high).flatten()

# Low model
lstm_low = build_lstm_model((WINDOW_SIZE, len(feature_columns)))
lstm_low.fit(
    X_train_seq_low, y_train_seq_low,
    validation_data=(X_test_seq_low, y_test_seq_low),
    epochs=50, batch_size=64, callbacks=[early_stop], verbose=1
)
pred_lstm_low = lstm_low.predict(X_test_seq_low).flatten()

results.append(evaluate_predictions(y_test_seq_high, pred_lstm_high, 'LSTM', 'High'))
results.append(evaluate_predictions(y_test_seq_low, pred_lstm_low, 'LSTM', 'Low'))

lstm_high.save('models/lstm_high.keras')
lstm_low.save('models/lstm_low.keras')
```

`window_size=30` means the model looks at the last 30 trading days to predict tomorrow. `EarlyStopping` halts training once validation loss stops improving, which helps avoid overfitting given how much data 25 stocks worth of history adds up to.

Note that the LSTM's test set has fewer rows than the other three models' test sets. Building sequences consumes the first `window_size` rows of each stock's test period, since there is no way to build a 30-day window for the first 30 rows. Keep this in mind when comparing row counts across models, though the accuracy metrics themselves remain directly comparable.

---

## Step 8: Overall Comparison Table

```python
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(['Target', 'MAPE'])
print(results_df.to_string(index=False))

results_df.to_csv('data/model_comparison_overall.csv', index=False)
```

**Expected shape of output:**

```
       Model Target    MAE   RMSE  MAPE     R2  Accuracy_%
XGBoost         High  12.40  18.90  1.85  0.961       98.15
Random Forest   High  13.10  19.50  1.95  0.955       98.05
LSTM            High  14.20  21.00  2.10  0.948       97.90
Linear Regression High 22.80  31.40  3.40  0.882       96.60
XGBoost          Low  11.90  17.80  1.79  0.963       98.21
...
```

This table alone tells you which model type wins overall, but the real decision in this project needs to happen per stock, not in aggregate, since a model that works well on average across 25 stocks may still lose to a different model on any individual one.

---

## Step 9: Per-Stock Accuracy Breakdown

```python
def evaluate_per_stock(test_df, predictions_dict, target_column, ticker_column='Ticker'):
    """
    predictions_dict: {'Model Name': predicted_array}
    All prediction arrays must align row-for-row with test_df (same length, same order).
    """
    per_stock_results = []
    
    for model_name, predictions in predictions_dict.items():
        temp_df = test_df.reset_index(drop=True).copy()
        temp_df['Prediction'] = predictions
        
        for ticker in temp_df[ticker_column].unique():
            ticker_data = temp_df[temp_df[ticker_column] == ticker]
            
            mae = mean_absolute_error(ticker_data[target_column], ticker_data['Prediction'])
            mape = np.mean(np.abs((ticker_data[target_column] - ticker_data['Prediction']) / ticker_data[target_column])) * 100
            
            per_stock_results.append({
                'Ticker': ticker,
                'Model': model_name,
                'Target': target_column,
                'MAE': round(mae, 2),
                'MAPE': round(mape, 2),
                'Accuracy_%': round(100 - mape, 2)
            })
    
    return pd.DataFrame(per_stock_results)

predictions_high = {
    'Linear Regression': pred_lr_high,
    'Random Forest': pred_rf_high,
    'XGBoost': pred_xgb_high
    # LSTM excluded here due to different row count from sequence windowing;
    # evaluate it separately using X_test_seq_high's corresponding ticker slice
}

per_stock_high = evaluate_per_stock(test_df, predictions_high, 'Target_High')
print(per_stock_high.sort_values(['Ticker', 'Accuracy_%'], ascending=[True, False]))

per_stock_high.to_csv('data/model_comparison_per_stock.csv', index=False)
```

**Selecting the best model per stock:**

```python
best_per_stock = per_stock_high.loc[per_stock_high.groupby('Ticker')['Accuracy_%'].idxmax()]
print(best_per_stock[['Ticker', 'Model', 'Accuracy_%']])
```

This gives a table like:

```
Ticker            Model  Accuracy_%
  ENGRO         XGBoost       98.40
   FFC   Random Forest       97.85
  HUBC         XGBoost       98.10
  MEBL   Random Forest       97.60
   SYS         XGBoost       98.55
...
```

Repeat the same per-stock evaluation for `Target_Low` separately, since the best model for predicting a stock's High is not guaranteed to be the best model for predicting its Low.

---

## Step 10: Generate the Final Price Range Prediction

The model predicts the next row in sequence for a given stock, which in practice means the next trading day, since the data only contains trading days to begin with. The model itself is never told what that date actually is (`Date` was excluded from `feature_columns` in Phase 3), so a separate helper is needed to attach a real calendar date to the output.

### Finding the Next Trading Date

PSX trades Monday through Friday and closes for a set of public holidays each year. Weekends are easy to skip programmatically, but holidays need to be supplied, since they shift from year to year (Eid and other Islamic calendar dates in particular).

```python
from datetime import datetime, timedelta

# Known PSX holidays - update this list each year from the official PSX calendar
psx_holidays_2026 = [
    '2026-02-05',  # Kashmir Solidarity Day
    '2026-03-23',  # Pakistan Day
    '2026-05-01',  # Labour Day
    '2026-08-14',  # Independence Day
    '2026-12-25',  # Quaid-e-Azam Day
    # add Eid-ul-Fitr, Eid-ul-Adha, Ashura, and any other moving holidays
    # once PSX confirms the dates for the year
]

def get_next_trading_date(current_date, holidays=None):
    """
    current_date: datetime or string 'YYYY-MM-DD', the most recent date in your data
    holidays: list of 'YYYY-MM-DD' strings for known PSX closures
    Returns the next valid trading date as a string.
    """
    if isinstance(current_date, str):
        current_date = datetime.strptime(current_date, '%Y-%m-%d')
    
    holidays = holidays or []
    holiday_dates = [datetime.strptime(h, '%Y-%m-%d').date() for h in holidays]
    
    next_date = current_date + timedelta(days=1)
    
    while next_date.weekday() >= 5 or next_date.date() in holiday_dates:
        next_date += timedelta(days=1)
    
    return next_date.strftime('%Y-%m-%d')
```

`weekday() >= 5` catches Saturday (5) and Sunday (6). This function only knows about holidays you give it, so the `psx_holidays_2026` list needs to stay current. An outdated list will still work, it will just occasionally label a prediction with a date the market is actually closed.

### Producing the Range with a Date Label

```python
def predict_price_range(ticker, latest_features_row, latest_date, best_model_lookup, holidays=None):
    """
    ticker: stock symbol, e.g. 'SYS'
    latest_features_row: a single row of feature values (today's data) as a DataFrame
    latest_date: the date of latest_features_row, as a string 'YYYY-MM-DD'
    best_model_lookup: dict like {'SYS': {'High': xgb_high, 'Low': rf_low}}
    holidays: optional list of known PSX holiday date strings
    """
    models_for_ticker = best_model_lookup[ticker]
    
    predicted_high = models_for_ticker['High'].predict(latest_features_row)[0]
    predicted_low = models_for_ticker['Low'].predict(latest_features_row)[0]
    
    if predicted_low > predicted_high:
        predicted_low, predicted_high = predicted_high, predicted_low
    
    predicted_date = get_next_trading_date(latest_date, holidays)
    
    return {
        'Ticker': ticker,
        'Predicted_Date': predicted_date,
        'Predicted_Low': round(predicted_low, 2),
        'Predicted_High': round(predicted_high, 2),
        'Range': f"{round(predicted_low, 2)} - {round(predicted_high, 2)}"
    }

# Example usage
best_model_lookup = {
    'SYS': {'High': xgb_high, 'Low': xgb_low},
    'MEBL': {'High': rf_high, 'Low': rf_low},
    # ... populate for all 25 stocks based on Step 9's results
}

latest_row = X_test[X_test['Ticker_encoded'] == 0].tail(1)  # replace 0 with actual encoded value for the ticker
latest_date = test_df[test_df['Ticker_encoded'] == 0]['Date'].max()  # most recent date available for this ticker

result = predict_price_range('SYS', latest_row, latest_date, best_model_lookup, holidays=psx_holidays_2026)
print(f"{result['Ticker']} predicted range for {result['Predicted_Date']}: {result['Range']}")
```

The safety check swapping `predicted_low` and `predicted_high` if they come out reversed exists because each target is predicted independently. Two separate models occasionally disagree on which one is bigger, especially near the edges of a model's confidence, so this guards against printing a nonsensical range.

The date label is calculated entirely after prediction and never fed back into the model. It exists only so the printed output reads as "SYS predicted range for 2026-08-10" instead of an unlabeled number, which matters once this runs daily rather than as a one-off experiment.

---

## Step 11: Final Validation (Run Once, At the End)

Only after the best models are chosen using the test set in Steps 8 and 9, run them once against `validate_df`, the portion of data neither the model nor your model-selection decisions have touched at all.

```python
final_high_model = xgb_high  # replace with whichever model won overall or per stock
final_low_model = xgb_low

pred_validate_high = final_high_model.predict(X_validate)
pred_validate_low = final_low_model.predict(X_validate)

final_validation_high = evaluate_predictions(y_validate_high, pred_validate_high, 'Final Model', 'High')
final_validation_low = evaluate_predictions(y_validate_low, pred_validate_low, 'Final Model', 'Low')

print(final_validation_high)
print(final_validation_low)
```

If this validation accuracy is close to the test accuracy from Step 8, the model generalizes well. If it drops noticeably, the model was likely tuned too closely to the test set and needs revisiting before being considered final.

---

## Checklist

When Phase 4 is complete, you should have:

- [ ] All 8 models trained (Linear Regression, Random Forest, XGBoost, LSTM, each for High and Low) and saved to `models/`
- [ ] `data/model_comparison_overall.csv` showing aggregate accuracy across all 25 stocks
- [ ] `data/model_comparison_per_stock.csv` showing which model wins on which stock
- [ ] A `best_model_lookup` mapping each ticker to its best High and Low model
- [ ] `predict_price_range()` tested and producing sane, non-reversed ranges
- [ ] Final validation run exactly once on `validate_df`, confirming the chosen models generalize

---

## Troubleshooting

**LSTM training is very slow:**
Reduce `epochs`, increase `batch_size`, or reduce `window_size` from 30 to something smaller like 15 or 20. A GPU helps significantly if available.

**XGBoost or Random Forest overfits (train accuracy much higher than test accuracy):**
Reduce `max_depth`, increase `min_samples_leaf` (Random Forest) or `min_child_weight` (XGBoost), or lower `n_estimators`.

**One stock's accuracy is far worse than the rest:**
Check that stock's row count and date coverage from the Phase 1 summary. Thin-traded stocks like PAKT, NESTLE, COLG, and ILP have less training data per stock, so somewhat lower accuracy on those specific tickers is expected rather than a bug.

**Predicted Low comes out higher than Predicted High:**
This is what the swap check in Step 10 handles. If it happens frequently for a specific stock, it usually means that stock's models are weak (check its individual accuracy in the per-stock table) rather than a bug in the range logic itself.

---

## Next Steps (Beyond This Pipeline)

With Phase 4 complete, the core prediction system is functional: data collected, cleaned, featured, merged, and now generating price ranges per stock with the best model chosen per stock. The two items noted from the original project plan, adding news data and automating daily scraping, remain open for later and were never part of Phases 1 through 4.