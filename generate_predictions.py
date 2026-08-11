"""
Assembly step the Phase 8 dashboard needs (Phases/frontend.md "Data the
dashboard expects"): runs each stock's latest feature row through every
Phase 9 model for every (target, horizon) cell, blends the 5 model types'
predictions with the ensemble weights train_models_phase9.py already
computed, and writes one row per stock to data/phase9_predictions.csv, in
the schema app.py's load_predictions() reads.

Ensemble, not single-best-model: earlier versions of this script served
whichever single model type had the best Improvement_over_Naive_% per
(ticker, target, horizon). That number is a single train/test split's
estimate, though, with sampling noise of its own - picking a single winner
by it throws away the other 4 models' information entirely. Blending all 5
with models/phase9_ensemble_weights.pkl's inverse-MAE weights (lower a
model's per-stock MAE, the more weight its prediction gets - see
model_utils.compute_ensemble_weights) was computed during training but
never actually used anywhere until now; using it is the cheap, already-paid-
for improvement over picking one model and discarding the rest.

Needs every one of the 50 (horizon, target, model_type) combos available
somewhere, unlike the single-best-model version which only needed whichever
combo actually won per stock - checks models_evaluation_only/ as well as
models/, since manage_model_storage.py moved 6 combos there that were never
a single-model serving choice for any stock, but are still inputs to the
blend for whichever ticker/horizon/target they weren't the winner for.
"""

import pickle
from pathlib import Path

import pandas as pd
import torch
import xgboost as xgb

from model_utils import WINDOW_SIZE, ensemble_predict
from train_models_phase9 import FEATURE_COLUMNS, GRUModel, HORIZONS, MODEL_TYPES, TARGETS

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


def load_ensemble_weights():
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


def predict_ensemble(ticker, target, horizon, ticker_df, scaler, ensemble_weights):
    """Blends all 5 model types' point predictions for one (ticker, target,
    horizon) cell using that ticker's own inverse-MAE weights, and returns
    the blend plus whichever model type carries the most weight in it, for
    display."""
    weights = ensemble_weights[ticker][target][horizon]
    predictions_by_type = {
        model_type: predict_price(model_type, target, horizon, ticker_df, scaler)
        for model_type in MODEL_TYPES
    }
    blended = ensemble_predict(predictions_by_type, weights)
    top_model = max(weights, key=weights.get)
    return blended, top_model


def main():
    master_df = pd.read_csv(DATA_DIR / "master_dataset.csv")
    master_df["Date"] = pd.to_datetime(master_df["Date"])

    with open(MODELS_DIR / "phase9_feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    ensemble_weights = load_ensemble_weights()

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
                pred, top_model = predict_ensemble(ticker, target, horizon, ticker_df, scaler, ensemble_weights)
                row[f"Pred_{target}_{horizon}d"] = round(pred, 2)
                row[f"Model_{target}_{horizon}d"] = "Ensemble"
                row[f"Top_Model_{target}_{horizon}d"] = top_model

        rows.append(row)
        print(f"{ticker}: done")

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / "phase9_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
