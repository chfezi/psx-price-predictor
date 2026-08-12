"""
Assembly step the Phase 8 dashboard needs (Phases/frontend.md "Data the
dashboard expects"): runs each stock's latest feature row through the single
best Phase 9 model for every (target, horizon) cell, and writes one row per
stock to data/phase9_predictions.csv, in the schema app.py's
load_predictions() reads.

Single best model, not the ensemble blend: an ensemble version of this
script shipped for a while (see Phases/frontend_notes.md), but blending in
LSTM/GRU's weaker cells dragged the served Improvement_over_Naive_% down
compared to just serving whichever one of the 5 model types actually has
the best real backtested Improvement_over_Naive_% for that specific
(ticker, target, horizon) - reverted per that explicit tradeoff. "Best" is
read straight from data/phase9_model_comparison_per_stock.csv, the same
per-stock evaluation table train_models_phase9.py already produced; always
picks the best of the 5 (no naive-baseline fallback), matching the last
single-model-era decision documented in Phases/frontend_notes.md.

The confidence band's Std_Error, and the Improvement_over_Naive_%/
Directional_Accuracy_% shown on the detail view, are the chosen model's own
numbers - read here and baked straight into phase9_predictions.csv - rather
than the ensemble's, so the dashboard's stats describe the model actually
being served.

Only needs whichever (horizon, target, model_type) combo actually wins a
given stock's serving slot, not all 50 - checks models_evaluation_only/ as
well as models/ regardless, since manage_model_storage.py's own selection
rule differs slightly and may have moved a combo there that still wins here.
"""

import pickle
from pathlib import Path

import pandas as pd
import torch
import xgboost as xgb

from model_utils import WINDOW_SIZE
from train_models_phase9 import FEATURE_COLUMNS, GRUModel, HORIZONS, TARGETS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EVAL_ONLY_DIR = BASE_DIR / "models_evaluation_only"

MODEL_FILE_INFO = {
    "Linear Regression": ("lr", "pkl"),
    "Random Forest": ("rf", "pkl"),
    "XGBoost": ("xgb", "json"),
    "LSTM": ("lstm", "keras"),
    "GRU": ("gru", "pt"),
}

_model_cache = {}


def load_comparison_table():
    return pd.read_csv(DATA_DIR / "phase9_model_comparison_per_stock.csv")


def load_error_distributions():
    with open(MODELS_DIR / "phase9_error_distributions.pkl", "rb") as f:
        return pickle.load(f)


def load_ensemble_weights():
    """Not used for serving any more (see module docstring), but
    evaluate_ensemble.py still imports this to run its own ensemble-vs-
    single-model diagnostic against data/test.csv."""
    with open(MODELS_DIR / "phase9_ensemble_weights.pkl", "rb") as f:
        return pickle.load(f)


def model_path(model_type, target, horizon):
    prefix, ext = MODEL_FILE_INFO[model_type]
    filename = f"phase9_{prefix}_{target.lower()}_{horizon}d.{ext}"
    for directory in (MODELS_DIR, EVAL_ONLY_DIR):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{filename} not found in models/ or models_evaluation_only/")


def load_model(model_type, target, horizon):
    key = (model_type, target, horizon)
    if key in _model_cache:
        return _model_cache[key]

    path = model_path(model_type, target, horizon)
    if model_type == "XGBoost":
        model = xgb.XGBRegressor()
        model.load_model(str(path))
    elif model_type in ("Random Forest", "Linear Regression"):
        with open(path, "rb") as f:
            model = pickle.load(f)
    elif model_type == "LSTM":
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(str(path))
    elif model_type == "GRU":
        model = GRUModel(input_size=len(FEATURE_COLUMNS))
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    _model_cache[key] = model
    return model


def predict_price(model_type, target, horizon, ticker_df, scaler):
    """Point prediction for one (ticker's latest row, target, horizon, model
    type) cell, in price space.

    LSTM/GRU window: model_utils.create_sequences builds training sequences
    as features[i - window_size:i] - the WINDOW_SIZE days strictly BEFORE
    the anchor day i, never including day i itself (the target/price/close
    at i come from the anchor row, the sequence is what came before it).
    ticker_df.tail(WINDOW_SIZE) would instead take the last WINDOW_SIZE rows
    INCLUDING today (the anchor day being predicted from) as the sequence's
    final step - a real train/inference mismatch, not a stylistic choice,
    caught after the fact (see Phases/frontend_notes.md). iloc[-(WINDOW_SIZE
    + 1):-1] is today's row excluded, the WINDOW_SIZE rows before it kept -
    matching training.
    """
    latest_close = float(ticker_df["Close"].values[-1])
    model = load_model(model_type, target, horizon)

    if model_type in ("LSTM", "GRU"):
        seq = ticker_df.iloc[-(WINDOW_SIZE + 1):-1][FEATURE_COLUMNS]
        seq_scaled = scaler.transform(seq).reshape(1, WINDOW_SIZE, len(FEATURE_COLUMNS))
        if model_type == "LSTM":
            pred_return = float(model.predict(seq_scaled, verbose=0)[0][0])
        else:
            with torch.no_grad():
                x = torch.tensor(seq_scaled, dtype=torch.float32)
                pred_return = float(model(x).numpy().flatten()[0])
    else:
        latest_row = ticker_df.tail(1)[FEATURE_COLUMNS]
        if model_type == "Linear Regression":
            x = scaler.transform(latest_row)
            pred_return = float(model.predict(x)[0])
        else:  # Random Forest, XGBoost
            pred_return = float(model.predict(latest_row)[0])

    return latest_close * (1 + pred_return)


def choose_best_model(comparison_df, ticker, target, horizon):
    """Whichever of the 5 model types has the best real backtested
    Improvement_over_Naive_% for this (ticker, target, horizon) cell, always
    - no naive-baseline fallback on a tie/loss (see module docstring)."""
    rows = comparison_df[
        (comparison_df["Ticker"] == ticker)
        & (comparison_df["Target"] == f"Target_{target}")
        & (comparison_df["Horizon"] == horizon)
    ]
    best = rows.loc[rows["Improvement_over_Naive_%"].idxmax()]
    return best["Model"], float(best["Improvement_over_Naive_%"]), float(best["Directional_Accuracy_%"])


def main():
    master_df = pd.read_csv(DATA_DIR / "master_dataset.csv")
    master_df["Date"] = pd.to_datetime(master_df["Date"])

    with open(MODELS_DIR / "phase9_feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    comparison_df = load_comparison_table()
    error_distributions = load_error_distributions()

    rows = []
    for ticker in sorted(master_df["Ticker"].unique()):
        ticker_df = master_df[master_df["Ticker"] == ticker].sort_values("Date")
        latest = ticker_df.tail(1)

        row = {
            "Ticker": ticker,
            "Data_Date": latest["Date"].values[0].astype("datetime64[D]").astype(str),
            "Yesterday_Open": float(latest["Open"].values[0]),
            "Yesterday_Close": float(latest["Close"].values[0]),
        }
        for horizon in HORIZONS:
            for target in TARGETS:
                model_type, improvement, dir_acc = choose_best_model(comparison_df, ticker, target, horizon)
                pred = predict_price(model_type, target, horizon, ticker_df, scaler)
                std_error = error_distributions[ticker][target][horizon][model_type]["std_error"]

                row[f"Pred_{target}_{horizon}d"] = round(pred, 2)
                row[f"Model_{target}_{horizon}d"] = model_type
                row[f"Improvement_{target}_{horizon}d"] = round(improvement, 2)
                row[f"Directional_Accuracy_{target}_{horizon}d"] = round(dir_acc, 2)
                row[f"Std_Error_{target}_{horizon}d"] = round(std_error, 4)

        rows.append(row)
        print(f"{ticker}: done")

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / "phase9_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
