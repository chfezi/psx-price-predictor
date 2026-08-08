"""
Phase 5: Streamlit Interface.

Single-page dashboard for demoing PSX price range predictions, built on
the merged dataset (Phase 3) and trained models (Phase 4), following
Phases/phase_5.md.

Run with: streamlit run app.py
"""

import pickle
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# rf_high.pkl / rf_low.pkl are ~130MB each, too large for a normal git push,
# so they're kept out of the repo (.gitignore) and fetched from a GitHub
# Release the first time the app runs. Fill these in after creating the
# release (see README / setup instructions).
MODEL_DOWNLOAD_URLS = {
    "rf_high.pkl": "https://github.com/chfezi/psx-price-predictor/releases/download/v1.0-models/rf_high.pkl",
    "rf_low.pkl": "https://github.com/chfezi/psx-price-predictor/releases/download/v1.0-models/rf_low.pkl",
}


def ensure_models_downloaded():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODEL_DOWNLOAD_URLS.items():
        dest = MODELS_DIR / filename
        if dest.exists() or not url:
            continue
        with st.spinner(f"Downloading {filename} (first run only)..."):
            urllib.request.urlretrieve(url, dest)


st.set_page_config(page_title="PSX Stock Price Range Predictor", layout="wide")

ensure_models_downloaded()

FEATURE_COLUMNS = [
    "Ticker_encoded",
    "Open", "High", "Low", "Close", "Volume",
    "Return_1d", "Return_5d", "Return_10d", "Return_20d", "Log_Return",
    "High_Low_Range", "Open_Close_Range",
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "RSI_14",
    "MACD_line", "Signal_line", "MACD_histogram",
    "BB_upper", "BB_middle", "BB_lower", "ATR_14", "Volatility_20",
    "Volume_SMA_20", "Volume_Ratio",
    "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5", "Close_lag_10",
    "Return_lag_1", "Return_lag_2", "Return_lag_5",
    "Volume_lag_1", "Volume_lag_5",
    "Rolling_Max_20", "Rolling_Min_20",
    "Price_Range", "Body", "Upper_Wick", "Lower_Wick", "Range_Percentage",
    "DayOfWeek", "Month",
]

PSX_HOLIDAYS_2026 = [
    "2026-02-05", "2026-03-23", "2026-05-01", "2026-08-14", "2026-12-25",
]

LSTM_WINDOW_SIZE = 30


def get_model_input(ticker_df, latest_row, model_name, scaler):
    if model_name == "LSTM":
        seq = ticker_df.tail(LSTM_WINDOW_SIZE)[FEATURE_COLUMNS]
        seq_scaled = scaler.transform(seq)
        return seq_scaled.reshape(1, LSTM_WINDOW_SIZE, len(FEATURE_COLUMNS))
    if model_name == "Linear Regression":
        return scaler.transform(latest_row[FEATURE_COLUMNS])
    return latest_row[FEATURE_COLUMNS]


def get_prediction(model, model_name, model_input):
    if model_name == "LSTM":
        return float(model.predict(model_input, verbose=0)[0][0])
    return float(model.predict(model_input)[0])


@st.cache_data
def load_master_data():
    df = pd.read_csv(DATA_DIR / "master_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_per_stock_comparison():
    return pd.read_csv(DATA_DIR / "model_comparison_per_stock.csv")


@st.cache_resource
def load_models():
    import xgboost as xgb
    from tensorflow.keras.models import load_model

    xgb_high = xgb.XGBRegressor()
    xgb_high.load_model(str(MODELS_DIR / "xgb_high.json"))
    xgb_low = xgb.XGBRegressor()
    xgb_low.load_model(str(MODELS_DIR / "xgb_low.json"))

    with open(MODELS_DIR / "rf_high.pkl", "rb") as f:
        rf_high = pickle.load(f)
    with open(MODELS_DIR / "rf_low.pkl", "rb") as f:
        rf_low = pickle.load(f)

    with open(MODELS_DIR / "lr_high.pkl", "rb") as f:
        lr_high = pickle.load(f)
    with open(MODELS_DIR / "lr_low.pkl", "rb") as f:
        lr_low = pickle.load(f)

    with open(MODELS_DIR / "feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    lstm_high = load_model(str(MODELS_DIR / "lstm_high.keras"))
    lstm_low = load_model(str(MODELS_DIR / "lstm_low.keras"))

    return {
        "XGBoost": {"High": xgb_high, "Low": xgb_low},
        "Random Forest": {"High": rf_high, "Low": rf_low},
        "Linear Regression": {"High": lr_high, "Low": lr_low},
        "LSTM": {"High": lstm_high, "Low": lstm_low},
        "scaler": scaler,
    }


master_df = load_master_data()
comparison_df = load_per_stock_comparison()
model_map = load_models()


def get_next_trading_date(current_date, holidays=None):
    if isinstance(current_date, str):
        current_date = datetime.strptime(current_date, "%Y-%m-%d")
    holidays = holidays or []
    holiday_dates = [datetime.strptime(h, "%Y-%m-%d").date() for h in holidays]
    next_date = current_date + timedelta(days=1)
    while next_date.weekday() >= 5 or next_date.date() in holiday_dates:
        next_date += timedelta(days=1)
    return next_date.strftime("%Y-%m-%d")


def apply_range_sanity_check(ticker_df, predicted_low, predicted_high, max_range_multiplier=3):
    """
    Bound the predicted range using the stock's own recent volatility (ATR),
    so a live prediction can never balloon into an unrealistic range even if
    one of the two models (often Linear Regression, due to multicollinearity
    among the lag and price-level features) extrapolates poorly on a row
    whose values sit slightly outside what it saw during training.
    """
    latest_atr = ticker_df["ATR_14"].iloc[-1]
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
    ticker_df = master_df[master_df["Ticker"] == ticker].sort_values("Date")
    latest_row = ticker_df.tail(1)
    latest_date = latest_row["Date"].values[0]

    X_latest = latest_row[FEATURE_COLUMNS]

    stock_comparison = comparison_df[comparison_df["Ticker"] == ticker]

    # "Naive Baseline" (predict tomorrow = today) is shown for comparison but
    # isn't a loadable model, so it's excluded when picking the auto-selected
    # "best" model even if it happens to score highest for this stock/target.
    selectable_comparison = stock_comparison[stock_comparison["Model"] != "Naive Baseline"]
    best_high_row = selectable_comparison[selectable_comparison["Target"] == "Target_High"].sort_values("Accuracy_%", ascending=False).iloc[0]
    best_low_row = selectable_comparison[selectable_comparison["Target"] == "Target_Low"].sort_values("Accuracy_%", ascending=False).iloc[0]

    high_model_name = override_high_model if override_high_model else best_high_row["Model"]
    low_model_name = override_low_model if override_low_model else best_low_row["Model"]

    # Look up this specific model's own accuracy for the stock, not just the winner's
    high_accuracy = stock_comparison[
        (stock_comparison["Target"] == "Target_High") & (stock_comparison["Model"] == high_model_name)
    ]["Accuracy_%"].values[0]
    low_accuracy = stock_comparison[
        (stock_comparison["Target"] == "Target_Low") & (stock_comparison["Model"] == low_model_name)
    ]["Accuracy_%"].values[0]

    # Linear Regression and LSTM were fit on scaler-transformed features; the
    # tree models were fit on raw features and must not be scaled here. LSTM
    # additionally needs the last 30 days as a sequence, not a single row.
    X_high_input = get_model_input(ticker_df, latest_row, high_model_name, model_map["scaler"])
    X_low_input = get_model_input(ticker_df, latest_row, low_model_name, model_map["scaler"])

    predicted_high = get_prediction(model_map[high_model_name]["High"], high_model_name, X_high_input)
    predicted_low = get_prediction(model_map[low_model_name]["Low"], low_model_name, X_low_input)

    if predicted_low > predicted_high:
        predicted_low, predicted_high = predicted_high, predicted_low

    predicted_low, predicted_high = apply_range_sanity_check(ticker_df, predicted_low, predicted_high)

    predicted_date = get_next_trading_date(pd.Timestamp(latest_date), PSX_HOLIDAYS_2026)

    return {
        "predicted_date": predicted_date,
        "predicted_low": round(float(predicted_low), 2),
        "predicted_high": round(float(predicted_high), 2),
        "high_model_used": high_model_name,
        "low_model_used": low_model_name,
        "high_accuracy": round(float(high_accuracy), 2),
        "low_accuracy": round(float(low_accuracy), 2),
        "latest_close": latest_row["Close"].values[0],
        "latest_date": latest_date,
    }


# --- Top Section: Every Stock at a Glance ---
st.title("PSX Stock Price Range Predictor")

st.subheader("All Stocks - Next Trading Day Predicted Range")

all_tickers = sorted(master_df["Ticker"].unique())

overview_rows = []
for ticker in all_tickers:
    result = predict_for_ticker(ticker)
    overview_rows.append(
        {
            "Ticker": ticker,
            "Latest Close": result["latest_close"],
            "Predicted Low": result["predicted_low"],
            "Predicted High": result["predicted_high"],
            "For Date": result["predicted_date"],
        }
    )

overview_df = pd.DataFrame(overview_rows)
st.dataframe(overview_df, use_container_width=True, hide_index=True)

st.divider()

# --- Stock Selector and Manual Model Choice (All Reactive, No Extra Button) ---
available_models = [m for m in model_map.keys() if m != "scaler"]

col_pick1, col_pick2, col_pick3 = st.columns(3)

with col_pick1:
    selected_ticker = st.selectbox("Select a stock for details", all_tickers)

with col_pick2:
    high_model_choice = st.selectbox("Model for High prediction", ["Auto (Best)"] + available_models)

with col_pick3:
    low_model_choice = st.selectbox("Model for Low prediction", ["Auto (Best)"] + available_models)

override_high = None if high_model_choice == "Auto (Best)" else high_model_choice
override_low = None if low_model_choice == "Auto (Best)" else low_model_choice

result = predict_for_ticker(selected_ticker, override_high, override_low)

# --- Side-by-Side Prediction Cards ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Range", f"{result['predicted_low']} - {result['predicted_high']}")
    st.caption(f"For {result['predicted_date']}")

with col2:
    st.metric("Latest Close", f"{result['latest_close']}")
    st.caption(f"As of {pd.Timestamp(result['latest_date']).date()}")

with col3:
    st.metric("Model Used (High)", result["high_model_used"])
    st.metric("Model Used (Low)", result["low_model_used"])

st.divider()

# --- Chart: Always Visible, Updates with the Selected Stock ---
st.subheader(f"{selected_ticker} - Price History and Prediction")

ticker_history = master_df[master_df["Ticker"] == selected_ticker].sort_values("Date").tail(90)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ticker_history["Date"], y=ticker_history["Close"],
    mode="lines", name="Close Price", line=dict(color="#1f77b4", width=2),
))
fig.add_trace(go.Scatter(
    x=ticker_history["Date"], y=ticker_history["High"],
    mode="lines", name="Daily High", line=dict(width=1, dash="dot", color="lightgray"),
))
fig.add_trace(go.Scatter(
    x=ticker_history["Date"], y=ticker_history["Low"],
    mode="lines", name="Daily Low", line=dict(width=1, dash="dot", color="lightgray"),
))

predicted_date_ts = pd.Timestamp(result["predicted_date"])
fig.add_trace(go.Scatter(
    x=[predicted_date_ts, predicted_date_ts],
    y=[result["predicted_low"], result["predicted_high"]],
    mode="lines+markers", name="Predicted Range (Next Day)",
    line=dict(color="orange", width=5),
))

fig.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)
