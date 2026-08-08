"""
Adds a per-stock 'Naive Baseline' row to data/model_comparison_per_stock.csv:
predicting tomorrow's High/Low as simply today's High/Low, no model at all.
This exposes how much (if any) real skill the trained models have beyond a
copy-paste guess, since stock prices barely move day to day in percentage
terms and MAE/Accuracy_% alone make any model look deceptively good.

Run once after train_models.py; sources predictions from data/test.csv
(same test split the trained models were scored on) so the comparison is
apples-to-apples.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def mape_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return round(mae, 2), round(mape, 2), round(100 - mape, 2)


def main():
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    comparison_df = pd.read_csv(DATA_DIR / "model_comparison_per_stock.csv")

    comparison_df = comparison_df[comparison_df["Model"] != "Naive Baseline"]

    rows = []
    for ticker, ticker_df in test_df.groupby("Ticker"):
        mae, mape, acc = mape_accuracy(ticker_df["Target_High"], ticker_df["High"])
        rows.append({"Ticker": ticker, "Model": "Naive Baseline", "Target": "Target_High", "MAE": mae, "MAPE": mape, "Accuracy_%": acc})

        mae, mape, acc = mape_accuracy(ticker_df["Target_Low"], ticker_df["Low"])
        rows.append({"Ticker": ticker, "Model": "Naive Baseline", "Target": "Target_Low", "MAE": mae, "MAPE": mape, "Accuracy_%": acc})

    baseline_df = pd.DataFrame(rows)
    combined = pd.concat([comparison_df, baseline_df], ignore_index=True)
    combined = combined.sort_values(["Ticker", "Target", "Accuracy_%"], ascending=[True, True, False])
    combined.to_csv(DATA_DIR / "model_comparison_per_stock.csv", index=False)

    print(f"Added {len(baseline_df)} Naive Baseline rows ({test_df['Ticker'].nunique()} tickers x 2 targets)")
    print(combined[combined["Model"] == "Naive Baseline"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
