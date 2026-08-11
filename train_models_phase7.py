"""
Phase 7: Multi-Horizon Forecasting and GRU.

Trains the Phase 6 model roster (Linear Regression, Random Forest, XGBoost,
LSTM) plus a new GRU model type, once per horizon (1/5/10/20/60 trading
days) per target (High/Low), all inside the unchanged unified architecture
(one model per horizon per type across all 25 stocks, Ticker as a feature).
Extends Phase 6's backtested error distributions and ensemble weights with
a horizon key, and extends the evaluation table with Horizon, Directional
Accuracy, and Improvement over Naive Baseline, following Phases/phase_7.md.

No app.py changes and no live-serving decisions here - see
Phases/phase_7.md ("Frontend changes ... are Phase 8, not this phase").

"Validation set" mapping and the test/validate split discipline are
unchanged from Phase 6 (see train_models_phase6.py's module docstring):
everything here is computed on data/test.csv; data/validate.csv is not
touched.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from train_models_phase6 import (
    NEW_FEATURE_COLUMNS,
    WINDOW_SIZE,
    compute_backtested_error_distribution,
    compute_ensemble_weights,
    evaluate_per_stock,
    evaluate_predictions,
    to_price,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

HORIZONS = [1, 5, 10, 20, 60]
TARGETS = ["High", "Low"]
MODEL_TYPES = ["Linear Regression", "Random Forest", "XGBoost", "LSTM", "GRU"]

GRU_HIDDEN_SIZE = 32
GRU_NUM_LAYERS = 1
GRU_EPOCHS = 50
GRU_BATCH_SIZE = 64
GRU_PATIENCE = 10
GRU_LR = 0.001


# --- horizon-aware evaluation helpers (build on Phase 6's evaluators) ---

def evaluate_predictions_h(y_true_price, y_pred_price, close_values, naive_pred_price, model_name, target_name, horizon):
    result = evaluate_predictions(y_true_price, y_pred_price, model_name, target_name)
    result["Horizon"] = horizon

    y_true_price = np.asarray(y_true_price, dtype=float)
    y_pred_price = np.asarray(y_pred_price, dtype=float)
    close_values = np.asarray(close_values, dtype=float)
    actual_return = (y_true_price - close_values) / close_values
    predicted_return = (y_pred_price - close_values) / close_values
    result["Directional_Accuracy_%"] = round(100 * float(np.mean(np.sign(actual_return) == np.sign(predicted_return))), 2)

    naive_mae = mean_absolute_error(y_true_price, naive_pred_price)
    result["Improvement_over_Naive_%"] = round(100 * (naive_mae - result["MAE"]) / naive_mae, 2) if naive_mae > 0 else None
    return result


def evaluate_per_stock_h(tickers, y_true_price, close_values, naive_pred_price, predictions_dict, target_name, horizon):
    base = evaluate_per_stock(tickers, y_true_price, predictions_dict, target_name)
    base["Horizon"] = horizon

    tickers = np.asarray(tickers)
    y_true_price = np.asarray(y_true_price, dtype=float)
    close_values = np.asarray(close_values, dtype=float)
    naive_pred_price = np.asarray(naive_pred_price, dtype=float)
    actual_return = (y_true_price - close_values) / close_values

    extra_rows = []
    for model_name, preds in predictions_dict.items():
        preds = np.asarray(preds, dtype=float)
        predicted_return = (preds - close_values) / close_values
        for ticker in np.unique(tickers):
            mask = tickers == ticker
            dir_acc = 100 * float(np.mean(np.sign(actual_return[mask]) == np.sign(predicted_return[mask])))
            naive_mae = mean_absolute_error(y_true_price[mask], naive_pred_price[mask])
            model_mae = mean_absolute_error(y_true_price[mask], preds[mask])
            improvement = 100 * (naive_mae - model_mae) / naive_mae if naive_mae > 0 else None
            extra_rows.append({
                "Ticker": ticker, "Model": model_name, "Target": target_name,
                "Directional_Accuracy_%": round(dir_acc, 2),
                "Improvement_over_Naive_%": round(improvement, 2) if improvement is not None else None,
            })
    extra_df = pd.DataFrame(extra_rows)
    return base.merge(extra_df, on=["Ticker", "Model", "Target"])


def create_sequences_h(df, feature_columns, return_target_column, price_target_column,
                        close_column="Close_raw", ticker_column="Ticker", window_size=WINDOW_SIZE):
    """
    Same shape as train_models_phase6.create_sequences, except the price-space
    conversion value is read from close_column ("Close_raw") rather than
    "Close". "Close" is itself one of feature_columns and gets standardized
    for model input - reading raw Close from the same column for price
    conversion, after restoring it post-scaling, would silently overwrite
    the scaled Close *feature* fed to the model with an unscaled one (a
    latent issue in Phase 6's LSTM prep this avoids by keeping the two
    uses on separate columns; see Phases/phase_7_notes.md).
    """
    X_sequences, y_returns, y_prices, closes, tickers = [], [], [], [], []

    for ticker in df[ticker_column].unique():
        ticker_df = df[df[ticker_column] == ticker].sort_values("Date").reset_index(drop=True)

        features = ticker_df[feature_columns].values
        returns = ticker_df[return_target_column].values
        prices = ticker_df[price_target_column].values
        close_vals = ticker_df[close_column].values

        for i in range(window_size, len(ticker_df)):
            X_sequences.append(features[i - window_size:i])
            y_returns.append(returns[i])
            y_prices.append(prices[i])
            closes.append(close_vals[i])
            tickers.append(ticker)

    return (
        np.array(X_sequences), np.array(y_returns), np.array(y_prices),
        np.array(closes), np.array(tickers),
    )


def get_training_rows_for_horizon(df, horizon):
    """
    Drop rows missing this horizon's target. Phases/phase_7.md Step 3: don't
    do this to the master dataset (merge_and_split.py no longer does - see
    its HORIZON_TARGET_COLUMNS comment), only when building a horizon's own
    training set. In this dataset train.csv/test.csv turn out to have zero
    such rows at every horizon (data runs through 2026, so even a 60-day
    lookup from the last training row lands on real future data still
    inside the full history) - see Phases/phase_7_notes.md.
    """
    return df.dropna(subset=[f"Target_High_Return_{horizon}d", f"Target_Low_Return_{horizon}d"]).reset_index(drop=True)


# --- GRU ---

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=GRU_HIDDEN_SIZE, num_layers=GRU_NUM_LAYERS):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)


def train_gru(X_train_seq, y_train_seq, X_test_seq, y_test_seq, input_size, label):
    """
    Hyperparameters (hidden_size, num_layers, lr, batch_size, epochs,
    patience) are locked module-level constants, identical across every
    horizon this is called for - Phases/phase_7.md Step 4.
    """
    torch.manual_seed(42)
    model = GRUModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=GRU_BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    loss_history = []

    for epoch in range(GRU_EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_test_t), y_test_t).item()

        print(f"  GRU {label} epoch {epoch + 1}/{GRU_EPOCHS} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")
        loss_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= GRU_PATIENCE:
                print(f"  GRU {label} early stopping at epoch {epoch + 1} (best val_loss {best_val_loss:.6f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).numpy().flatten()
    return model, test_pred, loss_history


def build_error_distributions_h(tickers, y_true_price, preds_by_model_price, target_label, horizon, store):
    tickers = np.asarray(tickers)
    y_true_price = np.asarray(y_true_price, dtype=float)
    for model_name, preds in preds_by_model_price.items():
        preds = np.asarray(preds, dtype=float)
        for ticker in np.unique(tickers):
            mask = tickers == ticker
            stats = compute_backtested_error_distribution(y_true_price[mask], preds[mask])
            store.setdefault(ticker, {}).setdefault(target_label, {}).setdefault(horizon, {})[model_name] = stats


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    def build_lstm_model(input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    with open(MODELS_DIR / "phase6_feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    overall_results = []
    per_stock_results = []
    error_distributions = {}
    ensemble_weights = {}
    loss_history_rows = []

    for horizon in HORIZONS:
        print(f"\n=== Horizon {horizon}d ===")
        train_h = get_training_rows_for_horizon(train_df, horizon)
        test_h = get_training_rows_for_horizon(test_df, horizon)
        print(f"Rows: train={len(train_h)} (of {len(train_df)}), test={len(test_h)} (of {len(test_df)})")

        X_train_h = train_h[NEW_FEATURE_COLUMNS]
        X_test_h = test_h[NEW_FEATURE_COLUMNS]
        X_train_h_scaled = scaler.transform(X_train_h)
        X_test_h_scaled = scaler.transform(X_test_h)
        test_close = test_h["Close"].values

        train_h_scaled_df = train_h.copy()
        test_h_scaled_df = test_h.copy()
        train_h_scaled_df[NEW_FEATURE_COLUMNS] = X_train_h_scaled
        test_h_scaled_df[NEW_FEATURE_COLUMNS] = X_test_h_scaled
        # "Close" is one of NEW_FEATURE_COLUMNS and is now standardized above -
        # correct for model input, but create_sequences_h also needs the raw
        # Close for price conversion. Carried on a separate column so the
        # scaled Close feature isn't overwritten (see create_sequences_h docstring).
        train_h_scaled_df["Close_raw"] = train_h["Close"].values
        test_h_scaled_df["Close_raw"] = test_h["Close"].values

        for target in TARGETS:
            return_col = f"Target_{target}_Return_{horizon}d"
            price_col = f"Target_{target}_{horizon}d"
            raw_col = "High" if target == "High" else "Low"

            y_train_ret = train_h[return_col]
            y_test_ret = test_h[return_col]
            y_test_price = test_h[price_col].values
            naive_pred_price = test_h[raw_col].values

            price_preds = {}

            print(f"Training Linear Regression ({target}, {horizon}d)...")
            lr = LinearRegression()
            lr.fit(X_train_h_scaled, y_train_ret)
            price_preds["Linear Regression"] = to_price(test_close, lr.predict(X_test_h_scaled))
            with open(MODELS_DIR / f"phase7_lr_{target.lower()}_{horizon}d.pkl", "wb") as f:
                pickle.dump(lr, f)

            print(f"Training Random Forest ({target}, {horizon}d)...")
            rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            rf.fit(X_train_h, y_train_ret)
            price_preds["Random Forest"] = to_price(test_close, rf.predict(X_test_h))
            with open(MODELS_DIR / f"phase7_rf_{target.lower()}_{horizon}d.pkl", "wb") as f:
                pickle.dump(rf, f)

            print(f"Training XGBoost ({target}, {horizon}d)...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            )
            xgb_model.fit(X_train_h, y_train_ret, eval_set=[(X_test_h, y_test_ret)], verbose=False)
            price_preds["XGBoost"] = to_price(test_close, xgb_model.predict(X_test_h))
            xgb_model.save_model(str(MODELS_DIR / f"phase7_xgb_{target.lower()}_{horizon}d.json"))

            print(f"Building sequences ({target}, {horizon}d)...")
            X_train_seq, y_train_seq_ret, _, _, _ = create_sequences_h(
                train_h_scaled_df, NEW_FEATURE_COLUMNS, return_col, price_col)
            X_test_seq, y_test_seq_ret, y_test_seq_price, close_seq, tickers_seq = create_sequences_h(
                test_h_scaled_df, NEW_FEATURE_COLUMNS, return_col, price_col)

            print(f"Training LSTM ({target}, {horizon}d)...")
            lstm_model = build_lstm_model((WINDOW_SIZE, len(NEW_FEATURE_COLUMNS)))
            early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            lstm_fit_history = lstm_model.fit(
                X_train_seq, y_train_seq_ret,
                validation_data=(X_test_seq, y_test_seq_ret),
                epochs=50, batch_size=64, callbacks=[early_stop], verbose=2,
            )
            pred_lstm_price = to_price(close_seq, lstm_model.predict(X_test_seq).flatten())
            lstm_model.save(str(MODELS_DIR / f"phase7_lstm_{target.lower()}_{horizon}d.keras"))
            for i, (tl, vl) in enumerate(zip(lstm_fit_history.history["loss"], lstm_fit_history.history["val_loss"])):
                loss_history_rows.append({
                    "Model": "LSTM", "Target": target, "Horizon": horizon,
                    "Epoch": i + 1, "Train_Loss": tl, "Val_Loss": vl,
                })

            print(f"Training GRU ({target}, {horizon}d)...")
            gru_model, pred_gru_ret, gru_loss_history = train_gru(
                X_train_seq, y_train_seq_ret, X_test_seq, y_test_seq_ret,
                input_size=len(NEW_FEATURE_COLUMNS), label=f"{target} {horizon}d",
            )
            pred_gru_price = to_price(close_seq, pred_gru_ret)
            torch.save(gru_model.state_dict(), str(MODELS_DIR / f"phase7_gru_{target.lower()}_{horizon}d.pt"))
            for row in gru_loss_history:
                loss_history_rows.append({
                    "Model": "GRU", "Target": target, "Horizon": horizon,
                    "Epoch": row["epoch"], "Train_Loss": row["train_loss"], "Val_Loss": row["val_loss"],
                })

            seq_price_preds = {"LSTM": pred_lstm_price, "GRU": pred_gru_price}
            # Naive baseline for the sequence rows: today's High/Low held constant,
            # built in the exact per-ticker order create_sequences_h iterates
            # (df["Ticker"].unique(), first-appearance order, each ticker's own
            # rows sorted by Date, sliced [WINDOW_SIZE:]) so it lines up row-for-row
            # with tickers_seq/close_seq/y_test_seq_price. Sourced from test_h (raw,
            # unscaled), not test_h_scaled_df, since High/Low are also feature
            # columns and are standardized in the latter.
            naive_pred_seq_parts = []
            for ticker in test_h_scaled_df["Ticker"].unique():
                ticker_rows = test_h[test_h["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
                naive_pred_seq_parts.append(ticker_rows[raw_col].values[WINDOW_SIZE:])
            naive_pred_seq_price = np.concatenate(naive_pred_seq_parts) if naive_pred_seq_parts else np.array([])

            for model_name, preds in price_preds.items():
                overall_results.append(evaluate_predictions_h(
                    y_test_price, preds, test_close, naive_pred_price, model_name, target, horizon))
            for model_name, preds in seq_price_preds.items():
                overall_results.append(evaluate_predictions_h(
                    y_test_seq_price, preds, close_seq, naive_pred_seq_price, model_name, target, horizon))

            per_stock_results.append(evaluate_per_stock_h(
                test_h["Ticker"].values, y_test_price, test_close, naive_pred_price, price_preds, f"Target_{target}", horizon))
            per_stock_results.append(evaluate_per_stock_h(
                tickers_seq, y_test_seq_price, close_seq, naive_pred_seq_price, seq_price_preds, f"Target_{target}", horizon))

            build_error_distributions_h(test_h["Ticker"].values, y_test_price, price_preds, target, horizon, error_distributions)
            build_error_distributions_h(tickers_seq, y_test_seq_price, seq_price_preds, target, horizon, error_distributions)

        print(f"Horizon {horizon}d complete.")

    loss_history_df = pd.DataFrame(loss_history_rows)
    loss_history_df.to_csv(DATA_DIR / "phase7_training_loss_history.csv", index=False)
    print(f"\nSaved epoch-by-epoch train/val loss history ({len(loss_history_df)} rows) to phase7_training_loss_history.csv.")

    overall_df = pd.DataFrame(overall_results).sort_values(["Horizon", "Target", "MAPE"])
    overall_df.to_csv(DATA_DIR / "phase7_model_comparison_overall.csv", index=False)
    print("\n=== Phase 7 Overall Model Comparison (all horizons) ===")
    print(overall_df.to_string(index=False))

    per_stock_df = pd.concat(per_stock_results, ignore_index=True)
    per_stock_df = per_stock_df.sort_values(["Horizon", "Ticker", "Target", "Accuracy_%"], ascending=[True, True, True, False])
    per_stock_df.to_csv(DATA_DIR / "phase7_model_comparison_per_stock.csv", index=False)

    with open(MODELS_DIR / "phase7_error_distributions.pkl", "wb") as f:
        pickle.dump(error_distributions, f)
    print(f"\nSaved backtested error distributions for {len(error_distributions)} tickers across {len(HORIZONS)} horizons.")

    # Ensemble weights: inverse-MAE across the 5 model types, per ticker/target/horizon
    for horizon in HORIZONS:
        for target in TARGETS:
            target_col = f"Target_{target}"
            subset = per_stock_df[(per_stock_df["Horizon"] == horizon) & (per_stock_df["Target"] == target_col)]
            for ticker in subset["Ticker"].unique():
                mae_by_model = subset[subset["Ticker"] == ticker].set_index("Model")["MAE"].to_dict()
                weights = compute_ensemble_weights(mae_by_model)
                ensemble_weights.setdefault(ticker, {}).setdefault(target, {})[horizon] = weights

    with open(MODELS_DIR / "phase7_ensemble_weights.pkl", "wb") as f:
        pickle.dump(ensemble_weights, f)

    # Range-width-widens-with-horizon check
    width_rows = []
    for ticker, by_target in error_distributions.items():
        for target, by_horizon in by_target.items():
            if 1 not in by_horizon or 60 not in by_horizon:
                continue
            for model_name in MODEL_TYPES:
                if model_name not in by_horizon[1] or model_name not in by_horizon[60]:
                    continue
                std_1d = by_horizon[1][model_name]["std_error"]
                std_60d = by_horizon[60][model_name]["std_error"]
                width_rows.append({
                    "Ticker": ticker, "Target": target, "Model": model_name,
                    "Std_1d": round(std_1d, 2), "Std_60d": round(std_60d, 2),
                    "Widens": bool(std_60d > std_1d),
                })
    width_df = pd.DataFrame(width_rows)
    width_df.to_csv(DATA_DIR / "phase7_range_width_by_horizon.csv", index=False)
    print(f"\nRange widens from 1d to 60d in {width_df['Widens'].mean() * 100:.1f}% of ticker/target/model combinations.")
    if (~width_df["Widens"]).any():
        print("Combinations where range does NOT widen from 1d to 60d:")
        print(width_df[~width_df["Widens"]].to_string(index=False))

    print("\nPhase 7 training complete.")


if __name__ == "__main__":
    main()
