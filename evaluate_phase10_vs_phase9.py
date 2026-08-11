"""
Head-to-head: Phase 9 (no macro features) vs Phase 10 (same everything +
KSE100_Return_1d/USDPKR_Return_1d) on both data/test.csv and
data/validate.csv, using the exact same ensemble-blend methodology as
evaluate_ensemble.py/evaluate_ensemble_validate.py - same row alignment
checks, same naive baseline, same metrics - just parameterized by which
phase's models/features/data to load, so the two are directly comparable.

See Phases/frontend_notes.md for why this experiment exists.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_utils import WINDOW_SIZE, bootstrap_paired_comparison, ensemble_predict, to_price
from train_models_phase9 import GRUModel, HORIZONS, MODEL_TYPES, TARGETS, create_sequences_h, get_training_rows_for_horizon

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

PHASE9_FEATURES = [
    "Ticker_encoded", "Open", "High", "Low", "Close", "Volume",
    "Return_1d", "Return_5d", "Return_10d", "Return_20d", "Log_Return",
    "High_Low_Range", "Open_Close_Range",
    "Close_to_SMA_20", "Close_to_SMA_50", "Close_to_SMA_200",
    "Close_to_EMA_12", "Close_to_EMA_26", "RSI_14",
    "MACD_Pct", "MACD_Hist_Pct", "BB_PercentB", "BB_Bandwidth",
    "ATR_Pct", "Volatility_20", "Volume_SMA_20", "Volume_Ratio",
    "Return_lag_1", "Return_lag_2", "Return_lag_3", "Return_lag_5", "Return_lag_10",
    "Volume_lag_1", "Volume_lag_5", "Rolling_Max_20", "Rolling_Min_20",
    "Price_Range", "Body", "Upper_Wick", "Lower_Wick", "Range_Percentage",
    "DayOfWeek", "Month",
]
PHASE10_FEATURES = PHASE9_FEATURES + ["KSE100_Return_1d", "USDPKR_Return_1d"]

PHASES = {
    "phase9": {"prefix": "phase9", "features": PHASE9_FEATURES, "test_file": "test.csv", "validate_file": "validate.csv"},
    "phase10": {"prefix": "phase10", "features": PHASE10_FEATURES, "test_file": "test_macro.csv", "validate_file": "validate_macro.csv"},
}

_model_cache = {}


def model_path(prefix, model_type, target, horizon):
    file_prefix, ext = MODEL_FILE_INFO[model_type]
    filename = f"{prefix}_{file_prefix}_{target.lower()}_{horizon}d.{ext}"
    for directory in (MODELS_DIR, EVAL_ONLY_DIR):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{filename} not found in models/ or models_evaluation_only/")


def load_model(prefix, model_type, target, horizon, n_features):
    key = (prefix, model_type, target, horizon)
    if key in _model_cache:
        return _model_cache[key]

    path = model_path(prefix, model_type, target, horizon)
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
        model = GRUModel(input_size=n_features)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    _model_cache[key] = model
    return model


def evaluate_horizon_target(phase_key, test_h, target, horizon, scaler, ensemble_weights):
    cfg = PHASES[phase_key]
    prefix, feature_columns = cfg["prefix"], cfg["features"]
    n_features = len(feature_columns)

    return_col = f"Target_{target}_Return_{horizon}d"
    price_col = f"Target_{target}_{horizon}d"

    lr = load_model(prefix, "Linear Regression", target, horizon, n_features)
    rf = load_model(prefix, "Random Forest", target, horizon, n_features)
    xgb_model = load_model(prefix, "XGBoost", target, horizon, n_features)
    lstm = load_model(prefix, "LSTM", target, horizon, n_features)
    gru = load_model(prefix, "GRU", target, horizon, n_features)

    test_h_scaled_df = test_h.copy()
    test_h_scaled_df[feature_columns] = scaler.transform(test_h[feature_columns])
    test_h_scaled_df["Close_raw"] = test_h["Close"].values

    X_test_seq, _, y_test_seq_price, close_seq, tickers_seq = create_sequences_h(
        test_h_scaled_df, feature_columns, return_col, price_col)

    lstm_ret = lstm.predict(X_test_seq, verbose=0).flatten()
    with torch.no_grad():
        gru_ret = gru(torch.tensor(X_test_seq, dtype=torch.float32)).numpy().flatten()
    lstm_price = to_price(close_seq, lstm_ret)
    gru_price = to_price(close_seq, gru_ret)

    aligned_lr, aligned_rf, aligned_xgb = [], [], []
    aligned_actual, aligned_close, aligned_tickers = [], [], []

    for ticker in test_h["Ticker"].unique():
        t_df = test_h[test_h["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(t_df) <= WINDOW_SIZE:
            continue
        t_feat = t_df[feature_columns]
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

    assert np.array_equal(aligned_tickers, tickers_seq), f"{phase_key}: ticker order mismatch"
    assert np.allclose(aligned_close, close_seq), f"{phase_key}: close mismatch"
    assert np.allclose(aligned_actual, y_test_seq_price), f"{phase_key}: target mismatch"

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

    naive_pred = aligned_close
    naive_mae = mean_absolute_error(aligned_actual, naive_pred)
    ensemble_mae = mean_absolute_error(aligned_actual, blended)
    ensemble_improvement = 100 * (naive_mae - ensemble_mae) / naive_mae if naive_mae > 0 else None

    naive_rmse = np.sqrt(mean_squared_error(aligned_actual, naive_pred))
    ensemble_rmse = np.sqrt(mean_squared_error(aligned_actual, blended))
    ensemble_rmse_improvement = 100 * (naive_rmse - ensemble_rmse) / naive_rmse if naive_rmse > 0 else None

    actual_return = (aligned_actual - aligned_close) / aligned_close
    ensemble_return = (blended - aligned_close) / aligned_close
    ensemble_dir_acc = 100 * float(np.mean(np.sign(actual_return) == np.sign(ensemble_return)))

    per_model_mae = {m: mean_absolute_error(aligned_actual, preds) for m, preds in single_preds.items()}
    best_single_model = min(per_model_mae, key=per_model_mae.get)
    best_single_mae = per_model_mae[best_single_model]
    best_single_improvement = 100 * (naive_mae - best_single_mae) / naive_mae if naive_mae > 0 else None

    summary = {
        "Phase": phase_key, "Target": target, "Horizon": horizon, "N_Rows": len(aligned_tickers),
        "Naive_MAE": round(naive_mae, 2),
        "Ensemble_MAE": round(ensemble_mae, 2),
        "Ensemble_Improvement_over_Naive_%": round(ensemble_improvement, 2),
        "Naive_RMSE": round(naive_rmse, 2),
        "Ensemble_RMSE": round(ensemble_rmse, 2),
        "Ensemble_RMSE_Improvement_over_Naive_%": round(ensemble_rmse_improvement, 2),
        "Ensemble_Directional_Accuracy_%": round(ensemble_dir_acc, 2),
        "Best_Single_Model": best_single_model,
        "Best_Single_MAE": round(best_single_mae, 2),
        "Best_Single_Improvement_over_Naive_%": round(best_single_improvement, 2),
    }
    raw = {
        "tickers": aligned_tickers,
        "actual": aligned_actual,
        "close": aligned_close,
        "naive_pred": naive_pred,
        "ensemble_pred": blended,
        "single_preds": single_preds,
    }
    return summary, raw


def run_phase(phase_key, data_file, collect_raw=False):
    cfg = PHASES[phase_key]
    df = pd.read_csv(DATA_DIR / data_file)

    with open(MODELS_DIR / f"{cfg['prefix']}_feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / f"{cfg['prefix']}_ensemble_weights.pkl", "rb") as f:
        ensemble_weights = pickle.load(f)

    rows = []
    raw_by_cell = {}
    for horizon in HORIZONS:
        test_h = get_training_rows_for_horizon(df, horizon)
        for target in TARGETS:
            print(f"  [{phase_key}] {target} {horizon}d on {data_file}...")
            summary, raw = evaluate_horizon_target(phase_key, test_h, target, horizon, scaler, ensemble_weights)
            rows.append(summary)
            if collect_raw:
                raw_by_cell[(target, horizon)] = raw

    if collect_raw:
        return rows, raw_by_cell
    return rows


def holm_bonferroni(p_values, alpha=0.05):
    """
    Hand-rolled since statsmodels isn't installed in this environment.
    Controls family-wise error rate like Bonferroni but with more power:
    sort p-values ascending, compare p_(i) against alpha/(m-i+1) (1-indexed),
    stop rejecting at the first failure. Returns (adjusted_p, reject) arrays
    in the ORIGINAL input order, not sorted order.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m)
    reject_sorted = np.zeros(m, dtype=bool)
    still_rejecting = True
    running_max = 0.0
    for i in range(m):
        threshold = alpha / (m - i)
        adjusted = sorted_p[i] * (m - i)
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = min(running_max, 1.0)
        if still_rejecting and sorted_p[i] <= threshold:
            reject_sorted[i] = True
        else:
            still_rejecting = False

    adjusted = np.empty(m)
    reject = np.empty(m, dtype=bool)
    adjusted[order] = adjusted_sorted
    reject[order] = reject_sorted
    return adjusted, reject


def compare_phase_pair_on_cell(target, horizon, summary9, raw9, summary10, raw10, n_resamples=2000):
    """
    Statistical comparison of Phase 9 vs Phase 10's ensemble blend for one
    (target, horizon) cell on validate.csv, going beyond the raw MAE gap
    reported in data/phase10_vs_phase9_comparison.csv: adds RMSE-based and
    directional-accuracy-based winners (to see if "who wins" depends on
    which metric you look at), plus a ticker-cluster bootstrap and Wilcoxon
    signed-rank test on the Phase9-vs-Phase10 gap (to see if that gap is
    distinguishable from noise at all, given only 25 tickers and serially
    correlated rows within each ticker). See the plan doc for why a cluster
    bootstrap was chosen over Diebold-Mariano here.
    """
    assert np.array_equal(raw9["tickers"], raw10["tickers"]), f"{target}@{horizon}d: ticker order mismatch between phases"
    assert np.allclose(raw9["actual"], raw10["actual"]), f"{target}@{horizon}d: actual mismatch between phases"
    assert np.allclose(raw9["close"], raw10["close"]), f"{target}@{horizon}d: close mismatch between phases"

    tickers = raw9["tickers"]
    actual = raw9["actual"]
    close = raw9["close"]

    per_ticker_imp9, per_ticker_imp10 = [], []
    for ticker in np.unique(tickers):
        mask = tickers == ticker
        naive_mae_t = mean_absolute_error(actual[mask], close[mask])
        imp9_t = 100 * (naive_mae_t - mean_absolute_error(actual[mask], raw9["ensemble_pred"][mask])) / naive_mae_t
        imp10_t = 100 * (naive_mae_t - mean_absolute_error(actual[mask], raw10["ensemble_pred"][mask])) / naive_mae_t
        per_ticker_imp9.append(imp9_t)
        per_ticker_imp10.append(imp10_t)
    per_ticker_imp9 = np.array(per_ticker_imp9)
    per_ticker_imp10 = np.array(per_ticker_imp10)

    boot = bootstrap_paired_comparison(
        tickers, actual, close, raw9["ensemble_pred"], raw10["ensemble_pred"], n_resamples=n_resamples)
    wilcoxon_p = float(scipy.stats.wilcoxon(per_ticker_imp10 - per_ticker_imp9).pvalue)

    mae_winner = "phase9" if summary9["Ensemble_MAE"] < summary10["Ensemble_MAE"] else "phase10"
    rmse_winner = "phase9" if summary9["Ensemble_RMSE"] < summary10["Ensemble_RMSE"] else "phase10"
    diracc_winner = "phase9" if summary9["Ensemble_Directional_Accuracy_%"] > summary10["Ensemble_Directional_Accuracy_%"] else "phase10"

    return {
        "Target": target, "Horizon": horizon, "N_Tickers": len(np.unique(tickers)),
        "Phase9_MAE": summary9["Ensemble_MAE"], "Phase10_MAE": summary10["Ensemble_MAE"], "MAE_Winner": mae_winner,
        "Phase9_RMSE": summary9["Ensemble_RMSE"], "Phase10_RMSE": summary10["Ensemble_RMSE"], "RMSE_Winner": rmse_winner,
        "Phase9_DirAcc_%": summary9["Ensemble_Directional_Accuracy_%"], "Phase10_DirAcc_%": summary10["Ensemble_Directional_Accuracy_%"], "DirAcc_Winner": diracc_winner,
        "Phase9_Improvement_%": summary9["Ensemble_Improvement_over_Naive_%"], "Phase10_Improvement_%": summary10["Ensemble_Improvement_over_Naive_%"],
        "Bootstrap_Pct_Resamples_Phase10_Wins": round(100 * boot["pct_b_wins"], 1),
        "Bootstrap_p_raw": round(boot["p_raw"], 4),
        "Bootstrap_CI95_Diff_Low": round(boot["ci95_low"], 2),
        "Bootstrap_CI95_Diff_High": round(boot["ci95_high"], 2),
        "Wilcoxon_p_value": round(wilcoxon_p, 4),
        "Winners_Agree_Across_Metrics": mae_winner == rmse_winner == diracc_winner,
    }


def main():
    all_rows = []

    print("=== test.csv ===")
    for phase_key, cfg in PHASES.items():
        rows = run_phase(phase_key, cfg["test_file"])
        for r in rows:
            r["Eval_Set"] = "test"
        all_rows.extend(rows)

    print("=== validate.csv ===")
    validate_raw = {}
    for phase_key, cfg in PHASES.items():
        rows, raw_by_cell = run_phase(phase_key, cfg["validate_file"], collect_raw=True)
        for r in rows:
            r["Eval_Set"] = "validate"
        all_rows.extend(rows)
        validate_raw[phase_key] = {"rows": {(r["Target"], r["Horizon"]): r for r in rows}, "raw": raw_by_cell}

    result_df = pd.DataFrame(all_rows)
    result_df = result_df.sort_values(["Eval_Set", "Horizon", "Target", "Phase"])
    result_df.to_csv(DATA_DIR / "phase10_vs_phase9_comparison.csv", index=False)

    print("\n=== Phase 9 (no macro) vs Phase 10 (+ macro), test.csv ===")
    print(result_df[result_df["Eval_Set"] == "test"].to_string(index=False))

    print("\n=== Phase 9 (no macro) vs Phase 10 (+ macro), validate.csv ===")
    print(result_df[result_df["Eval_Set"] == "validate"].to_string(index=False))

    print(f"\nSaved full comparison to data/phase10_vs_phase9_comparison.csv")

    print("\n=== Statistical significance: Phase 9 vs Phase 10, validate.csv only ===")
    sig_rows = []
    for horizon in HORIZONS:
        for target in TARGETS:
            summary9 = validate_raw["phase9"]["rows"][(target, horizon)]
            raw9 = validate_raw["phase9"]["raw"][(target, horizon)]
            summary10 = validate_raw["phase10"]["rows"][(target, horizon)]
            raw10 = validate_raw["phase10"]["raw"][(target, horizon)]
            print(f"  Bootstrapping {target} {horizon}d...")
            sig_rows.append(compare_phase_pair_on_cell(target, horizon, summary9, raw9, summary10, raw10))

    sig_df = pd.DataFrame(sig_rows)
    adjusted_p, reject = holm_bonferroni(sig_df["Bootstrap_p_raw"].values, alpha=0.05)
    sig_df["Bootstrap_p_holm"] = np.round(adjusted_p, 4)
    sig_df["Significant_after_correction"] = reject
    sig_df = sig_df.sort_values(["Horizon", "Target"])

    sig_out_path = DATA_DIR / "phase10_vs_phase9_significance_validate.csv"
    sig_df.to_csv(sig_out_path, index=False)

    print(sig_df.to_string(index=False))
    print(f"\nSaved significance comparison to {sig_out_path}")

    flagged = sig_df[~sig_df["Winners_Agree_Across_Metrics"] | ~sig_df["Significant_after_correction"]]
    if len(flagged):
        print("\nCells where metrics disagree on the winner, or the MAE gap is NOT significant after Holm correction:")
        print(flagged[["Target", "Horizon", "MAE_Winner", "RMSE_Winner", "DirAcc_Winner",
                        "Bootstrap_p_raw", "Bootstrap_p_holm", "Significant_after_correction"]].to_string(index=False))


if __name__ == "__main__":
    main()
