# Phase 5: Streamlit Interface

## Objective

Build a single-page Streamlit dashboard for demoing the PSX price range predictions. No multi-page navigation, no "click to see the next screen" flow. Selecting a stock from a dropdown updates everything on the page instantly. Charts and the prediction cards are always visible together, side by side, not hidden behind tabs or buttons.

Input: `data/master_dataset.csv`, trained models from `models/`, `data/model_comparison_per_stock.csv` (all from Phases 3 and 4)
Output: `app.py`, run with `streamlit run app.py`

---

## Design Principles for This Interface

1. **One page, no navigation.** Everything the demo needs to show lives on a single screen, scrollable if needed, but never behind a "Next" or "Submit" button.
2. **The dropdown is the only input.** Changing the selected stock is the one interaction, and Streamlit reruns the page automatically when it changes. No separate "Predict" button required.
3. **Side by side, not stacked one at a time.** Use `st.columns()` so the predicted range, the latest close, and the best model all sit next to each other in one glance.
4. **Charts load with the page, not on demand.** The price chart renders immediately for whichever stock is selected, not after an extra click.
5. **An overview table up top.** Before picking a specific stock, the person watching the demo can already see every stock's predicted range at once, in one table, with zero interaction.

---

## Step 1: Project Setup

```
psx-stock-prediction/
├── app.py                          # the Streamlit app
├── requirements.txt
├── data/
│   ├── master_dataset.csv
│   └── model_comparison_per_stock.csv
└── models/
    ├── ticker_encoder.pkl
    ├── feature_scaler.pkl
    ├── xgb_high.json
    ├── xgb_low.json
    ├── rf_high.pkl
    ├── rf_low.pkl
    ├── lr_high.pkl
    └── lr_low.pkl
```

```
# requirements.txt
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
```

Install with `pip install -r requirements.txt`.

---

## Step 2: Load Data and Models Once (Cached)

Streamlit reruns the whole script on every interaction, so loading the master dataset and every model fresh each time would make the dropdown feel slow. Caching keeps this instant.

```python
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="PSX Stock Price Range Predictor", layout="wide")

@st.cache_data
def load_master_data():
    df = pd.read_csv('data/master_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_per_stock_comparison():
    return pd.read_csv('data/model_comparison_per_stock.csv')

@st.cache_resource
def load_models():
    import xgboost as xgb
    
    xgb_high = xgb.XGBRegressor()
    xgb_high.load_model('models/xgb_high.json')
    xgb_low = xgb.XGBRegressor()
    xgb_low.load_model('models/xgb_low.json')
    
    rf_high = pickle.load(open('models/rf_high.pkl', 'rb'))
    rf_low = pickle.load(open('models/rf_low.pkl', 'rb'))
    
    lr_high = pickle.load(open('models/lr_high.pkl', 'rb'))
    lr_low = pickle.load(open('models/lr_low.pkl', 'rb'))
    
    scaler = pickle.load(open('models/feature_scaler.pkl', 'rb'))
    
    return {
        'XGBoost': {'High': xgb_high, 'Low': xgb_low},
        'Random Forest': {'High': rf_high, 'Low': rf_low},
        'Linear Regression': {'High': lr_high, 'Low': lr_low},
        'scaler': scaler
    }

master_df = load_master_data()
comparison_df = load_per_stock_comparison()
model_map = load_models()
```

**Note on LSTM:** the live interface intentionally leaves LSTM out. It needs a 30-day sequence window built per prediction rather than a single row, which adds real complexity for a demo that just needs to show the concept working. If LSTM won the comparison for a given stock in Phase 4, the app below still shows XGBoost, Random Forest, or Linear Regression, whichever ranked next best for that stock. Adding LSTM back in later mainly means building the sequence-window logic into `predict_for_ticker()` below.

---

## Step 3: Helper Functions

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

psx_holidays_2026 = [
    '2026-02-05', '2026-03-23', '2026-05-01', '2026-08-14', '2026-12-25',
    # add Eid and other moving holidays once confirmed for the year
]

def get_next_trading_date(current_date, holidays=None):
    if isinstance(current_date, str):
        current_date = datetime.strptime(current_date, '%Y-%m-%d')
    holidays = holidays or []
    holiday_dates = [datetime.strptime(h, '%Y-%m-%d').date() for h in holidays]
    next_date = current_date + timedelta(days=1)
    while next_date.weekday() >= 5 or next_date.date() in holiday_dates:
        next_date += timedelta(days=1)
    return next_date.strftime('%Y-%m-%d')

def apply_range_sanity_check(ticker_df, predicted_low, predicted_high, max_range_multiplier=3):
    """
    Bound the predicted range using the stock's own recent volatility (ATR),
    so a live prediction can never balloon into an unrealistic range even if
    one of the two models (often Linear Regression, due to multicollinearity
    among the lag and price-level features) extrapolates poorly on a row
    whose values sit slightly outside what it saw during training.
    """
    latest_atr = ticker_df['ATR_14'].iloc[-1]
    max_allowed_range = latest_atr * max_range_multiplier
    predicted_range = predicted_high - predicted_low
    
    if predicted_range > max_allowed_range:
        midpoint = (predicted_high + predicted_low) / 2
        predicted_low = midpoint - (max_allowed_range / 2)
        predicted_high = midpoint + (max_allowed_range / 2)
    
    return predicted_low, predicted_high

def predict_for_ticker(ticker, override_high_model=None, override_low_model=None):
    """
    override_high_model / override_low_model: pass a model name string
    ('XGBoost', 'Random Forest', 'Linear Regression') to force that model,
    or leave as None to use whichever model won the accuracy comparison
    for that stock in Phase 4.
    """
    ticker_df = master_df[master_df['Ticker'] == ticker].sort_values('Date')
    latest_row = ticker_df.tail(1)
    latest_date = latest_row['Date'].values[0]
    
    X_latest = latest_row[feature_columns]
    
    stock_comparison = comparison_df[comparison_df['Ticker'] == ticker]
    
    best_high_row = stock_comparison[stock_comparison['Target'] == 'Target_High'].sort_values('Accuracy_%', ascending=False).iloc[0]
    best_low_row = stock_comparison[stock_comparison['Target'] == 'Target_Low'].sort_values('Accuracy_%', ascending=False).iloc[0]
    
    high_model_name = override_high_model if override_high_model else best_high_row['Model']
    low_model_name = override_low_model if override_low_model else best_low_row['Model']
    
    # Look up this specific model's own accuracy for the stock, not just the winner's
    high_accuracy = stock_comparison[
        (stock_comparison['Target'] == 'Target_High') & (stock_comparison['Model'] == high_model_name)
    ]['Accuracy_%'].values[0]
    low_accuracy = stock_comparison[
        (stock_comparison['Target'] == 'Target_Low') & (stock_comparison['Model'] == low_model_name)
    ]['Accuracy_%'].values[0]
    
    X_high_input = model_map['scaler'].transform(X_latest) if high_model_name == 'Linear Regression' else X_latest
    X_low_input = model_map['scaler'].transform(X_latest) if low_model_name == 'Linear Regression' else X_latest
    
    predicted_high = model_map[high_model_name]['High'].predict(X_high_input)[0]
    predicted_low = model_map[low_model_name]['Low'].predict(X_low_input)[0]
    
    if predicted_low > predicted_high:
        predicted_low, predicted_high = predicted_high, predicted_low
    
    predicted_low, predicted_high = apply_range_sanity_check(ticker_df, predicted_low, predicted_high)
    
    predicted_date = get_next_trading_date(pd.Timestamp(latest_date), psx_holidays_2026)
    
    return {
        'predicted_date': predicted_date,
        'predicted_low': round(predicted_low, 2),
        'predicted_high': round(predicted_high, 2),
        'high_model_used': high_model_name,
        'low_model_used': low_model_name,
        'high_accuracy': round(high_accuracy, 2),
        'low_accuracy': round(low_accuracy, 2),
        'latest_close': latest_row['Close'].values[0],
        'latest_date': latest_date
    }
```

`ATR_14` already exists as a feature from Phase 2 and reflects that specific stock's own typical daily trading range, so the bound scales correctly for both a low-volatility stock like a utility and a high-volatility one, rather than using one fixed number for all 25 stocks. `max_range_multiplier=3` allows some room above a completely ordinary day without letting the range balloon to multiples of the stock's actual price. Adjust this multiplier up or down after watching how it behaves across a few days of live predictions.

This sanity check treats the symptom safely, but the better long-term fix is addressing why Linear Regression wins the accuracy comparison in the first place when its live behavior is this unstable. Two options worth considering in Phase 4:

1. **Exclude Linear Regression from the "best model" selection entirely**, keeping it purely as the baseline comparison it was always meant to be, and only letting XGBoost, Random Forest, or LSTM be chosen as the live-serving model.
2. **Regularize it instead of dropping it** by swapping `LinearRegression()` for `Ridge(alpha=1.0)` in Phase 4's Step 4, which keeps the same linear structure but shrinks the unstable coefficients caused by the correlated lag and price-level features, making it far less likely to extrapolate wildly on a new row.

---

## Step 4: Page Layout

### Top Section: Every Stock at a Glance

This renders immediately, before any dropdown interaction, so the full picture is visible right away.

```python
st.title("PSX Stock Price Range Predictor")

st.subheader("All Stocks - Next Trading Day Predicted Range")

all_tickers = sorted(master_df['Ticker'].unique())

overview_rows = []
for ticker in all_tickers:
    result = predict_for_ticker(ticker)
    overview_rows.append({
        'Ticker': ticker,
        'Latest Close': result['latest_close'],
        'Predicted Low': result['predicted_low'],
        'Predicted High': result['predicted_high'],
        'For Date': result['predicted_date']
    })

overview_df = pd.DataFrame(overview_rows)
st.dataframe(overview_df, use_container_width=True, hide_index=True)

st.divider()
```

### Stock Selector and Manual Model Choice (All Reactive, No Extra Button)

```python
available_models = [m for m in model_map.keys() if m != 'scaler']

col_pick1, col_pick2, col_pick3 = st.columns(3)

with col_pick1:
    selected_ticker = st.selectbox("Select a stock for details", all_tickers)

with col_pick2:
    high_model_choice = st.selectbox("Model for High prediction", ['Auto (Best)'] + available_models)

with col_pick3:
    low_model_choice = st.selectbox("Model for Low prediction", ['Auto (Best)'] + available_models)

override_high = None if high_model_choice == 'Auto (Best)' else high_model_choice
override_low = None if low_model_choice == 'Auto (Best)' else low_model_choice

result = predict_for_ticker(selected_ticker, override_high, override_low)
```

Leaving both on `Auto (Best)` keeps the original behavior, using whichever model won the accuracy comparison for that stock. Picking a specific model from either dropdown forces that one instead, which is useful for comparing predictions side by side or for checking whether a model like Linear Regression is producing an odd result on a particular stock. Both dropdowns rerun the page the same way the ticker dropdown does, so no separate button is needed here either.

### Side-by-Side Prediction Cards

```python
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Range", f"{result['predicted_low']} - {result['predicted_high']}")
    st.caption(f"For {result['predicted_date']}")

with col2:
    st.metric("Latest Close", f"{result['latest_close']}")
    st.caption(f"As of {pd.Timestamp(result['latest_date']).date()}")

with col3:
    st.metric("Model Used (High)", result['high_model_used'], f"{result['high_accuracy']}% accuracy")
    st.metric("Model Used (Low)", result['low_model_used'], f"{result['low_accuracy']}% accuracy")

st.divider()
```

### Chart: Always Visible, Updates with the Selected Stock

```python
st.subheader(f"{selected_ticker} - Price History and Prediction")

ticker_history = master_df[master_df['Ticker'] == selected_ticker].sort_values('Date').tail(90)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ticker_history['Date'], y=ticker_history['Close'],
    mode='lines', name='Close Price', line=dict(color='#1f77b4', width=2)
))
fig.add_trace(go.Scatter(
    x=ticker_history['Date'], y=ticker_history['High'],
    mode='lines', name='Daily High', line=dict(width=1, dash='dot', color='lightgray')
))
fig.add_trace(go.Scatter(
    x=ticker_history['Date'], y=ticker_history['Low'],
    mode='lines', name='Daily Low', line=dict(width=1, dash='dot', color='lightgray')
))

predicted_date_ts = pd.Timestamp(result['predicted_date'])
fig.add_trace(go.Scatter(
    x=[predicted_date_ts, predicted_date_ts],
    y=[result['predicted_low'], result['predicted_high']],
    mode='lines+markers', name='Predicted Range (Next Day)',
    line=dict(color='orange', width=5)
))

fig.update_layout(height=450, hovermode='x unified', legend=dict(orientation='h', y=1.1))
st.plotly_chart(fig, use_container_width=True)

st.divider()
```

### Model Comparison, Side by Side (High vs Low)

```python
st.subheader(f"Model Accuracy Comparison - {selected_ticker}")

stock_comparison = comparison_df[comparison_df['Ticker'] == selected_ticker]

col_a, col_b = st.columns(2)

with col_a:
    st.write("**Predicting High**")
    high_comp = stock_comparison[stock_comparison['Target'] == 'Target_High'][['Model', 'MAE', 'Accuracy_%']]
    st.dataframe(high_comp.sort_values('Accuracy_%', ascending=False), hide_index=True, use_container_width=True)

with col_b:
    st.write("**Predicting Low**")
    low_comp = stock_comparison[stock_comparison['Target'] == 'Target_Low'][['Model', 'MAE', 'Accuracy_%']]
    st.dataframe(low_comp.sort_values('Accuracy_%', ascending=False), hide_index=True, use_container_width=True)
```

---

## Step 5: Running the App

```bash
streamlit run app.py
```

This opens a browser tab, usually at `http://localhost:8501`. The overview table, the selected stock's cards, the chart, and the comparison tables all render on one scrollable page.

---

## Full Page Flow (What the Demo Actually Shows)

Top to bottom, all on one screen:

1. Title
2. Table of all 25 stocks' predicted ranges for the next trading day, visible without any interaction, always using each stock's best model
3. Three dropdowns side by side: pick a stock, and optionally override the model used for High and for Low (defaults to the best model if left on Auto)
4. Three cards side by side: predicted range, latest close, and which model actually produced each half of the prediction along with its accuracy
5. A chart of the last 90 days of price history with the predicted range marked at the next trading date
6. Two tables side by side comparing all model accuracies for that specific stock, split by High and Low

Changing any of the three dropdowns updates sections 4, 5, and 6 instantly. Section 2 stays visible throughout as a constant reference point and always reflects the auto-selected best model, regardless of what is chosen in section 3.

---

## Checklist

When Phase 5 is complete, you should have:

- [ ] `app.py` running locally with `streamlit run app.py`
- [ ] The overview table showing all 25 stocks' predictions without needing to select anything
- [ ] Selecting a stock from the dropdown updates the cards, chart, and comparison tables without a separate button click
- [ ] Manually overriding the High or Low model from the dropdowns updates the prediction accordingly, and switching back to Auto restores the best-model result
- [ ] The chart renders immediately and shows both historical prices and the predicted range together
- [ ] Predicted range for each stock passes a sanity check (bounded by recent ATR, never unrealistically wide)
- [ ] Model comparison tables show accuracy for every model, not just the winning one

---

## Troubleshooting

**App feels slow when switching stocks:**
Confirm `@st.cache_data` and `@st.cache_resource` are actually applied to the loading functions. If the master CSV or model files are being reloaded from disk on every dropdown change, caching was likely skipped somewhere.

**Chart looks empty for a specific stock:**
Check that stock's row count from the Phase 1 summary. Thin-traded stocks like PAKT, NESTLE, COLG, or ILP may have fewer recent rows in the last 90-day window if there were gaps in trading.

**Predicted range and chart's predicted marker don't line up visually:**
This usually means the `predicted_date_ts` calculation and the x-axis date range are not on the same scale. Confirm `master_df['Date']` is a proper `datetime64` column, not still a string.

**Predicted range (High minus Low) looks unrealistically wide for a single day:**
Check `model_comparison_per_stock.csv` for that ticker. If Linear Regression won for High or Low, its coefficients are likely unstable from the correlated lag and price-level features flagged in Phase 2, and it can extrapolate badly on a live row whose values sit slightly outside its training range, unlike the tree models. The `apply_range_sanity_check()` function above bounds this using the stock's own ATR. For the root cause rather than just the symptom, consider excluding Linear Regression from live serving in Phase 4, or replacing it with `Ridge(alpha=1.0)` to regularize the coefficients.

**KeyError on a model name in `model_map`:**
This means `comparison_df` has a `Model` value that does not exactly match the keys in `model_map` (`XGBoost`, `Random Forest`, `Linear Regression`). Check for extra spaces or a naming mismatch between Phase 4's comparison CSV and this file.

---

## Next Step

This covers the demo interface. If daily automation gets built later (the scraper running on its own each day), this same `app.py` keeps working without changes, since `predict_for_ticker()` always reads whatever is most recent in `master_dataset.csv` rather than a fixed date.