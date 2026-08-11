"""
Same comparison as evaluate_ensemble.py, run against data/validate.csv
instead of data/test.csv.

Why this run is the trustworthy one: both the ensemble weights
(models/phase9_ensemble_weights.pkl) and evaluate_ensemble.py's own
"best single model per ticker" pick are derived from data/test.csv's own
errors - so scoring against test.csv again is partly circular, and the
per-ticker "best single model" figure in particular benefits from having
been chosen with foreknowledge of the exact errors it's then scored on.
data/validate.csv was never touched by training, weighting, or model
selection anywhere in this project (see Phases/phase_9_notes.md), so this
run is the first genuinely out-of-sample look at whether the ensemble
blend, and the "best single model" idea, hold up on data neither has seen.
"""

import pickle

import pandas as pd

from evaluate_ensemble import DATA_DIR, MODELS_DIR, evaluate_horizon_target
from generate_predictions import load_ensemble_weights
from train_models_phase9 import HORIZONS, TARGETS, get_training_rows_for_horizon


def main():
    validate_df = pd.read_csv(DATA_DIR / "validate.csv")

    with open(MODELS_DIR / "phase9_feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    ensemble_weights = load_ensemble_weights()

    overall_rows = []
    per_ticker_rows = []

    for horizon in HORIZONS:
        validate_h = get_training_rows_for_horizon(validate_df, horizon)
        for target in TARGETS:
            print(f"Evaluating {target} {horizon}d on validate.csv...")
            overall_row, ticker_rows = evaluate_horizon_target(validate_h, target, horizon, scaler, ensemble_weights)
            overall_rows.append(overall_row)
            per_ticker_rows.extend(ticker_rows)

    overall_df = pd.DataFrame(overall_rows).sort_values(["Horizon", "Target"])
    overall_df.to_csv(DATA_DIR / "phase9_ensemble_vs_single_overall_validate.csv", index=False)

    per_ticker_df = pd.DataFrame(per_ticker_rows).sort_values(["Horizon", "Target", "Ticker"])
    per_ticker_df.to_csv(DATA_DIR / "phase9_ensemble_comparison_validate.csv", index=False)

    print("\n=== Ensemble vs naive vs best-single-model, all tickers combined (validate.csv) ===")
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
