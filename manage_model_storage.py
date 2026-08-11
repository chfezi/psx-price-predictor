"""
Phase 8 Step 5: Manage Model Storage.

Computes the same serving-model set app.py uses (Improvement_over_Naive_%
based, naive-baseline fallback - Phases/phase_8.md Step 1) directly from
data/phase9_model_comparison_per_stock.csv, then sets aside the handful of
Phase 9 (horizon, target, model_type) combos that are never the serving
choice for any of the 25 stocks. Small utility files (ticker_encoder.pkl)
are left in place - not worth the churn at their size.

Phase 4 and Phase 6 model files are no longer moved here - Phase 9 deletes
them outright instead of setting them aside (Phases/phase_9.md Step 6), so
those move loops are gone. Not hardcoded to the specific combos found once -
recomputes from the CSV each run, so this stays correct if models are
retrained. See Phases/phase_9_notes.md.
"""

import shutil
from itertools import product
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EVAL_ONLY_DIR = BASE_DIR / "models_evaluation_only"

HORIZONS = [1, 5, 10, 20, 60]
TARGETS = ["Open", "Close"]
MODEL_FILE_INFO = {
    "Linear Regression": ("lr", "pkl"),
    "Random Forest": ("rf", "pkl"),
    "XGBoost": ("xgb", "json"),
    "LSTM": ("lstm", "keras"),
    "GRU": ("gru", "pt"),
}


def compute_needed_combos():
    """Same pick_serving_model logic app.py uses - see Phases/phase_8.md Step 1."""
    df = pd.read_csv(DATA_DIR / "phase9_model_comparison_per_stock.csv")
    df["Target_clean"] = df["Target"].str.replace("Target_", "")

    needed = set()
    for (_, target, horizon), group in df.groupby(["Ticker", "Target_clean", "Horizon"]):
        positive = group[group["Improvement_over_Naive_%"] > 0]
        if len(positive) == 0:
            continue
        best_model = positive.loc[positive["Improvement_over_Naive_%"].idxmax(), "Model"]
        needed.add((horizon, target, best_model))
    return needed


def dir_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def move_file(filename, source_dir=MODELS_DIR):
    src = source_dir / filename
    if not src.exists():
        print(f"  (already missing, skipping) {filename}")
        return
    dest = EVAL_ONLY_DIR / filename
    shutil.move(str(src), str(dest))
    print(f"  moved {filename}")


def main():
    EVAL_ONLY_DIR.mkdir(parents=True, exist_ok=True)

    before_size = dir_size(MODELS_DIR)

    needed = compute_needed_combos()
    all_combos = set(product(HORIZONS, TARGETS, MODEL_FILE_INFO.keys()))
    unneeded = sorted(all_combos - needed)

    print(f"Serving-model combos needed: {len(needed)} / {len(all_combos)} possible")
    print(f"Never selected as serving model for any stock ({len(unneeded)}):")
    for horizon, target, model_type in unneeded:
        print(f"  {horizon}d {target} {model_type}")

    print("\nMoving never-selected Phase 9 combos...")
    for horizon, target, model_type in unneeded:
        prefix, ext = MODEL_FILE_INFO[model_type]
        filename = f"phase9_{prefix}_{target.lower()}_{horizon}d.{ext}"
        move_file(filename)

    after_size = dir_size(MODELS_DIR)
    eval_only_size = dir_size(EVAL_ONLY_DIR)

    print(f"\nmodels/: {before_size / 1e6:.0f}MB -> {after_size / 1e6:.0f}MB")
    print(f"models_evaluation_only/: {eval_only_size / 1e6:.0f}MB")
    print("\nRemaining in models/:")
    for f in sorted(MODELS_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name} ({f.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
