"""
Diagnostic requested after wiring the ensemble blend into
generate_predictions.py: does blending all 5 models actually beat the naive
baseline and the single best model more often, on the held-out test set
(data/test.csv) - the same set train_models_phase9.py scored every model
against - rather than just on the one live row generate_predictions.py
predicts from.

Mirrors train_models_phase9.py's own evaluation methodology, so the numbers
here are directly comparable to data/phase9_model_comparison_overall.csv and
_per_stock.csv: tabular models (LR/RF/XGBoost) score every test_h row for a
horizon/target; LSTM/GRU score only rows with a full 30-day lookback
available within the test set itself (train_models_phase9.create_sequences_h).
A blend needs all 5 predictions for the same row, so evaluation here is
restricted to the sequence-eligible rows only - the same subset LSTM/GRU
alone were scored on in training, which is why this script's own per-model
MAEs (computed fresh on that subset, not read from the existing CSVs) are
used as the "best single model" reference rather than pulling LR/RF/XGBoost's
numbers straight from phase9_model_comparison_overall.csv, which scored them
on a few more rows (the first 30 per ticker) than this evaluation can reach.

Alignment (tabular predictions rebuilt per-ticker to match create_sequences_h's
row order) is checked with np.array_equal/np.allclose before scoring, not
assumed - same practice as Phases/phase_9_notes.md Decision 3.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from generate_predictions import load_ensemble_weights, load_model
from model_utils import WINDOW_SIZE, compute_backtested_error_distribution, ensemble_predict, to_price
from train_models_phase9 import (
    FEATURE_COLUMNS, HORIZONS, MODEL_TYPES, TARGETS,
    create_sequences_h, get_training_rows_for_horizon,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def evaluate_horizon_target(test_h, target, horizon, scaler, ensemble_weights):
    return_col = f"Target_{target}_Return_{horizon}d"
    price_col = f"Target_{target}_{horizon}d"

    lr = load_model("Linear Regression", target, horizon)
    rf = load_model("Random Forest", target, horizon)
    xgb_model = load_model("XGBoost", target, horizon)
    lstm = load_model("LSTM", target, horizon)
    gru = load_model("GRU", target, horizon)

    test_h_scaled_df = test_h.copy()
    test_h_scaled_df[FEATURE_COLUMNS] = scaler.transform(test_h[FEATURE_COLUMNS])
    test_h_scaled_df["Close_raw"] = test_h["Close"].values

    X_test_seq, _, y_test_seq_price, close_seq, tickers_seq = create_sequences_h(
        test_h_scaled_df, FEATURE_COLUMNS, return_col, price_col)

    lstm_ret = lstm.predict(X_test_seq, verbose=0).flatten()
    with torch.no_grad():
        gru_ret = gru(torch.tensor(X_test_seq, dtype=torch.float32)).numpy().flatten()
    lstm_price = to_price(close_seq, lstm_ret)
    gru_price = to_price(close_seq, gru_ret)

    # Rebuild tabular predictions with the exact per-ticker sort-and-slice
    # create_sequences_h used internally, so row j here lines up with
    # tickers_seq[j]/close_seq[j]/y_test_seq_price[j].
    aligned_lr, aligned_rf, aligned_xgb = [], [], []
    aligned_actual, aligned_close, aligned_tickers = [], [], []

    for ticker in test_h["Ticker"].unique():
        t_df = test_h[test_h["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(t_df) <= WINDOW_SIZE:
            continue
        t_feat = t_df[FEATURE_COLUMNS]
        t_feat_scaled = scaler.transform(t_feat)
        t_close = t_df["Close"].values

        aligned_lr.extend(to_price(t_close, lr.predict(t_feat_scaled))[WINDOW_SIZE:])
        aligned_rf.extend(to_price(t_close, rf.predict(t_feat))[WINDOW_SIZE:])
        aligned_xgb.extend(to_price(t_close, xgb_model.predict(t_feat))[WINDOW_SIZE:])
        aligned_actual.extend(t_df[price_col].values[WINDOW_SIZE:])
        aligned_close.extend(t_close[WINDOW_SIZE:])
        aligned_tickers.extend([ticker] * (len(t_df) - WINDOW_SIZE))

    aligned_tickers = np.array(aligned_tickers)
    aligned_close = np.array(aligned_close, dtype=float)
    aligned_actual = np.array(aligned_actual, dtype=float)

    assert np.array_equal(aligned_tickers, tickers_seq), "ticker order mismatch vs create_sequences_h"
    assert np.allclose(aligned_close, close_seq), "close mismatch vs create_sequences_h"
    assert np.allclose(aligned_actual, y_test_seq_price), "target mismatch vs create_sequences_h"

    single_preds = {
        "Linear Regression": np.array(aligned_lr, dtype=float),
        "Random Forest": np.array(aligned_rf, dtype=float),
        "XGBoost": np.array(aligned_xgb, dtype=float),
        "LSTM": lstm_price,
        "GRU": gru_price,
    }

    blended = np.zeros(len(aligned_tickers))
    for i, ticker in enumerate(aligned_tickers):
        weights = ensemble_weights[ticker][target][horizon]
        preds_by_type = {m: single_preds[m][i] for m in MODEL_TYPES}
        blended[i] = ensemble_predict(preds_by_type, weights)

    naive_pred = aligned_close  # today's Close held constant, both targets

    def improvement(mae, naive_mae):
        return 100 * (naive_mae - mae) / naive_mae if naive_mae > 0 else None

    naive_mae = mean_absolute_error(aligned_actual, naive_pred)
    ensemble_mae = mean_absolute_error(aligned_actual, blended)

    actual_return = (aligned_actual - aligned_close) / aligned_close
    ensemble_return = (blended - aligned_close) / aligned_close
    ensemble_dir_acc = 100 * float(np.mean(np.sign(actual_return) == np.sign(ensemble_return)))

    per_model_mae = {m: mean_absolute_error(aligned_actual, preds) for m, preds in single_preds.items()}
    best_single_model = min(per_model_mae, key=per_model_mae.get)
    best_single_mae = per_model_mae[best_single_model]

    overall_row = {
        "Target": target, "Horizon": horizon, "N_Rows": len(aligned_tickers),
        "Naive_MAE": round(naive_mae, 2),
        "Ensemble_MAE": round(ensemble_mae, 2),
        "Ensemble_Improvement_over_Naive_%": round(improvement(ensemble_mae, naive_mae), 2),
        "Ensemble_Directional_Accuracy_%": round(ensemble_dir_acc, 2),
        "Best_Single_Model": best_single_model,
        "Best_Single_MAE": round(best_single_mae, 2),
        "Best_Single_Improvement_over_Naive_%": round(improvement(best_single_mae, naive_mae), 2),
        "Ensemble_Beats_Best_Single": bool(ensemble_mae < best_single_mae),
    }

    per_ticker_rows = []
    for ticker in np.unique(aligned_tickers):
        mask = aligned_tickers == ticker
        t_naive_mae = mean_absolute_error(aligned_actual[mask], naive_pred[mask])
        t_ensemble_mae = mean_absolute_error(aligned_actual[mask], blended[mask])
        t_model_mae = {m: mean_absolute_error(aligned_actual[mask], preds[mask]) for m, preds in single_preds.items()}
        t_best_model = min(t_model_mae, key=t_model_mae.get)
        t_best_mae = t_model_mae[t_best_model]
        t_dir_acc = 100 * float(np.mean(np.sign(actual_return[mask]) == np.sign(ensemble_return[mask])))
        # Backtested error distribution of the ensemble blend itself, per
        # ticker/target/horizon - the Phases/frontend.md detail view's
        # confidence-band bounds are sized from this (Std_Error, via
        # model_utils.predict_range_from_error_distribution), not recomputed
        # live in app.py.
        error_stats = compute_backtested_error_distribution(aligned_actual[mask], blended[mask])
        per_ticker_rows.append({
            "Ticker": ticker, "Target": target, "Horizon": horizon, "N_Rows": int(mask.sum()),
            "Naive_MAE": round(t_naive_mae, 2),
            "Best_Single_Model": t_best_model, "Best_Single_MAE": round(t_best_mae, 2),
            "Best_Single_Improvement_over_Naive_%": improvement(t_best_mae, t_naive_mae),
            "Ensemble_MAE": round(t_ensemble_mae, 2),
            "Ensemble_Beats_Single": bool(t_ensemble_mae < t_best_mae),
            "Ensemble_Improvement_over_Naive_%": improvement(t_ensemble_mae, t_naive_mae),
            "Ensemble_Directional_Accuracy_%": round(t_dir_acc, 2),
            "Std_Error": round(error_stats["std_error"], 4),
            "P10": round(error_stats["p10"], 4),
            "P90": round(error_stats["p90"], 4),
        })

    return overall_row, per_ticker_rows


def main():
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    with open(MODELS_DIR / "phase9_feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    ensemble_weights = load_ensemble_weights()

    overall_rows = []
    per_ticker_rows = []

    for horizon in HORIZONS:
        test_h = get_training_rows_for_horizon(test_df, horizon)
        for target in TARGETS:
            print(f"Evaluating {target} {horizon}d...")
            overall_row, ticker_rows = evaluate_horizon_target(test_h, target, horizon, scaler, ensemble_weights)
            overall_rows.append(overall_row)
            per_ticker_rows.extend(ticker_rows)

    overall_df = pd.DataFrame(overall_rows).sort_values(["Horizon", "Target"])
    overall_df.to_csv(DATA_DIR / "phase9_ensemble_vs_single_overall.csv", index=False)

    per_ticker_df = pd.DataFrame(per_ticker_rows).sort_values(["Horizon", "Target", "Ticker"])
    per_ticker_df.to_csv(DATA_DIR / "phase9_ensemble_comparison.csv", index=False)

    print("\n=== Ensemble vs naive vs best-single-model, all tickers combined ===")
    print(overall_df.to_string(index=False))

    beats_single_rate = per_ticker_df["Ensemble_Beats_Single"].mean() * 100
    print(f"\nEnsemble beats the best single model in {beats_single_rate:.1f}% of (ticker, target, horizon) "
          f"cells ({int(per_ticker_df['Ensemble_Beats_Single'].sum())}/{len(per_ticker_df)}).")

    beats_naive_rate = (per_ticker_df["Ensemble_Improvement_over_Naive_%"] > 0).mean() * 100
    print(f"Ensemble beats naive baseline in {beats_naive_rate:.1f}% of (ticker, target, horizon) cells.")

    single_beats_naive_rate = (per_ticker_df["Best_Single_Improvement_over_Naive_%"] > 0).mean() * 100
    print(f"Best single model (per ticker/target/horizon) beats naive baseline in "
          f"{single_beats_naive_rate:.1f}% of cells.")


if __name__ == "__main__":
    main()
