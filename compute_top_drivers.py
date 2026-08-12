"""
Precomputes the Phase 6-style SHAP top-3 driver tags for the stock detail
view (Phases/frontend.md "Top drivers"), one row per (Ticker, Target,
Horizon) cell that's actually served by a tree model, so app.py only ever
reads a CSV - never loads a model or shap directly, the same way
generate_predictions.py already keeps model loading out of the live
dashboard process.

Reads data/phase9_predictions.csv's Model_{target}_{h}d columns to see
which single model generate_predictions.py actually served for each cell
(see that script's docstring - single best model per cell, not a blend),
and only explains cells served by XGBoost or Random Forest, per
Phases/phase_6_notes.md Decision 4 and Phases/phase_8.md's rule: "only
shown when the serving model supports SHAP (tree models); when the serving
model is [something else], skip this section entirely rather than showing
empty or fabricated drivers." Must be rerun after generate_predictions.py
whenever the served model for a cell changes.

Both tree models are unified across all 25 tickers (Ticker_encoded is a
feature, not a per-ticker model) per train_models_phase9.py, so only one
XGBoost and one Random Forest model - and one shap.TreeExplainer each -
need to be built per (target, horizon), then reused for every ticker's own
latest feature row that was actually served by that model type.
"""

from pathlib import Path

import pandas as pd
import shap

from generate_predictions import load_model
from train_models_phase9 import FEATURE_COLUMNS, HORIZONS, TARGETS

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TREE_MODELS = ("XGBoost", "Random Forest")
TOP_N = 3


def main():
    master_df = pd.read_csv(DATA_DIR / "master_dataset.csv")
    master_df["Date"] = pd.to_datetime(master_df["Date"])
    preds_df = pd.read_csv(DATA_DIR / "phase9_predictions.csv").set_index("Ticker")

    latest_rows = {
        ticker: master_df[master_df["Ticker"] == ticker].sort_values("Date").tail(1)[FEATURE_COLUMNS]
        for ticker in sorted(master_df["Ticker"].unique())
    }

    rows = []
    for target in TARGETS:
        for horizon in HORIZONS:
            explainers = {}  # built lazily, only for model types actually served in this cell
            for ticker, X_row in latest_rows.items():
                served_model = preds_df.loc[ticker, f"Model_{target}_{horizon}d"]
                if served_model not in TREE_MODELS:
                    continue  # Phase 8 rule: skip, don't fabricate drivers for non-tree models

                if served_model not in explainers:
                    explainers[served_model] = shap.TreeExplainer(load_model(served_model, target, horizon))

                shap_values = explainers[served_model].shap_values(X_row)
                contributions = list(zip(FEATURE_COLUMNS, shap_values[0]))
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)

                row = {"Ticker": ticker, "Target": target, "Horizon": horizon, "Model": served_model}
                for i, (feature, value) in enumerate(contributions[:TOP_N], start=1):
                    row[f"Feature_{i}"] = feature
                    row[f"Direction_{i}"] = "up" if value > 0 else "down"
                rows.append(row)
            print(f"{target} {horizon}d: done")

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / "phase9_top_drivers.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path} ({len(out_df)}/{len(TARGETS) * len(HORIZONS) * len(latest_rows)} "
          f"cells served by a tree model)")


if __name__ == "__main__":
    main()
