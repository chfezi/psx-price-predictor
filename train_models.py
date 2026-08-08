"""
Phase 4: Model Training, Evaluation, and Range Prediction.

Trains four model types (Linear Regression, Random Forest, XGBoost, LSTM)
on the Phase 3 splits, each for Target_High and Target_Low, evaluates them
overall and per stock, picks the best model per stock/target, and exposes
a price-range prediction helper, following Phases/phase_4.md.
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

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

WINDOW_SIZE = 30

PSX_HOLIDAYS_2026 = [
    "2026-02-05", "2026-03-23", "2026-05-01", "2026-08-14", "2026-12-25",
]


def evaluate_predictions(y_true, y_pred, model_name, target_name):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    accuracy_pct = 100 - mape

    return {
        "Model": model_name,
        "Target": target_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4),
        "Accuracy_%": round(accuracy_pct, 2),
    }


def create_sequences(df, feature_columns, target_column, ticker_column="Ticker", window_size=WINDOW_SIZE):
    X_sequences, y_targets, tickers = [], [], []

    for ticker in df[ticker_column].unique():
        ticker_df = df[df[ticker_column] == ticker].sort_values("Date").reset_index(drop=True)

        features = ticker_df[feature_columns].values
        targets = ticker_df[target_column].values

        for i in range(window_size, len(ticker_df)):
            X_sequences.append(features[i - window_size:i])
            y_targets.append(targets[i])
            tickers.append(ticker)

    return np.array(X_sequences), np.array(y_targets), np.array(tickers)


def evaluate_per_stock(tickers, y_true, predictions_dict, target_name):
    per_stock_results = []
    y_true = np.asarray(y_true, dtype=float)
    tickers = np.asarray(tickers)

    for model_name, predictions in predictions_dict.items():
        predictions = np.asarray(predictions, dtype=float)
        for ticker in np.unique(tickers):
            mask = tickers == ticker
            mae = mean_absolute_error(y_true[mask], predictions[mask])
            mape = np.mean(np.abs((y_true[mask] - predictions[mask]) / y_true[mask])) * 100

            per_stock_results.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "Target": target_name,
                    "MAE": round(mae, 2),
                    "MAPE": round(mape, 2),
                    "Accuracy_%": round(100 - mape, 2),
                }
            )

    return pd.DataFrame(per_stock_results)


def get_next_trading_date(current_date, holidays=None):
    if isinstance(current_date, str):
        current_date = datetime.strptime(current_date, "%Y-%m-%d")
    elif isinstance(current_date, pd.Timestamp):
        current_date = current_date.to_pydatetime()

    holidays = holidays or []
    holiday_dates = [datetime.strptime(h, "%Y-%m-%d").date() for h in holidays]

    next_date = current_date + timedelta(days=1)
    while next_date.weekday() >= 5 or next_date.date() in holiday_dates:
        next_date += timedelta(days=1)

    return next_date.strftime("%Y-%m-%d")


def predict_price_range(ticker, latest_features_row, latest_date, best_model_lookup, scaler, holidays=None):
    """
    latest_features_row: a single row of RAW (unscaled) feature values, as a DataFrame.
    best_model_lookup[ticker]['High'/'Low'] is {'model': fitted_estimator, 'needs_scaling': bool}.
    Linear Regression was fit on scaler-transformed features, so it needs the row
    scaled before predicting; the tree models were fit on raw features and don't.
    """
    models_for_ticker = best_model_lookup[ticker]

    high_entry = models_for_ticker["High"]
    low_entry = models_for_ticker["Low"]

    high_input = scaler.transform(latest_features_row) if high_entry["needs_scaling"] else latest_features_row
    low_input = scaler.transform(latest_features_row) if low_entry["needs_scaling"] else latest_features_row

    predicted_high = high_entry["model"].predict(high_input)[0]
    predicted_low = low_entry["model"].predict(low_input)[0]

    if predicted_low > predicted_high:
        predicted_low, predicted_high = predicted_high, predicted_low

    predicted_date = get_next_trading_date(latest_date, holidays)

    return {
        "Ticker": ticker,
        "Predicted_Date": predicted_date,
        "Predicted_Low": round(float(predicted_low), 2),
        "Predicted_High": round(float(predicted_high), 2),
        "Range": f"{round(float(predicted_low), 2)} - {round(float(predicted_high), 2)}",
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load prepared data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    validate_df = pd.read_csv(DATA_DIR / "validate.csv")

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    X_validate = validate_df[FEATURE_COLUMNS]

    y_train_high = train_df["Target_High"]
    y_train_low = train_df["Target_Low"]
    y_test_high = test_df["Target_High"]
    y_test_low = test_df["Target_Low"]
    y_validate_high = validate_df["Target_High"]
    y_validate_low = validate_df["Target_Low"]

    # Step 2: Feature scaling (Linear Regression + LSTM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_validate_scaled = scaler.transform(X_validate)
    with open(MODELS_DIR / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    results = []

    # Step 4: Linear Regression
    print("Training Linear Regression...")
    lr_high = LinearRegression()
    lr_high.fit(X_train_scaled, y_train_high)
    pred_lr_high = lr_high.predict(X_test_scaled)

    lr_low = LinearRegression()
    lr_low.fit(X_train_scaled, y_train_low)
    pred_lr_low = lr_low.predict(X_test_scaled)

    results.append(evaluate_predictions(y_test_high, pred_lr_high, "Linear Regression", "High"))
    results.append(evaluate_predictions(y_test_low, pred_lr_low, "Linear Regression", "Low"))

    with open(MODELS_DIR / "lr_high.pkl", "wb") as f:
        pickle.dump(lr_high, f)
    with open(MODELS_DIR / "lr_low.pkl", "wb") as f:
        pickle.dump(lr_low, f)

    # Step 5: Random Forest
    print("Training Random Forest...")
    rf_high = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_high.fit(X_train, y_train_high)
    pred_rf_high = rf_high.predict(X_test)

    rf_low = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_low.fit(X_train, y_train_low)
    pred_rf_low = rf_low.predict(X_test)

    results.append(evaluate_predictions(y_test_high, pred_rf_high, "Random Forest", "High"))
    results.append(evaluate_predictions(y_test_low, pred_rf_low, "Random Forest", "Low"))

    with open(MODELS_DIR / "rf_high.pkl", "wb") as f:
        pickle.dump(rf_high, f)
    with open(MODELS_DIR / "rf_low.pkl", "wb") as f:
        pickle.dump(rf_low, f)

    # Step 6: XGBoost
    print("Training XGBoost...")
    xgb_high = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    xgb_high.fit(X_train, y_train_high, eval_set=[(X_test, y_test_high)], verbose=False)
    pred_xgb_high = xgb_high.predict(X_test)

    xgb_low = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    xgb_low.fit(X_train, y_train_low, eval_set=[(X_test, y_test_low)], verbose=False)
    pred_xgb_low = xgb_low.predict(X_test)

    results.append(evaluate_predictions(y_test_high, pred_xgb_high, "XGBoost", "High"))
    results.append(evaluate_predictions(y_test_low, pred_xgb_low, "XGBoost", "Low"))

    xgb_high.save_model(str(MODELS_DIR / "xgb_high.json"))
    xgb_low.save_model(str(MODELS_DIR / "xgb_low.json"))

    # Step 7: LSTM
    print("Building LSTM sequences...")
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[FEATURE_COLUMNS] = scaler.transform(train_df[FEATURE_COLUMNS])
    test_df_scaled[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    X_train_seq_high, y_train_seq_high, _ = create_sequences(train_df_scaled, FEATURE_COLUMNS, "Target_High")
    X_test_seq_high, y_test_seq_high, tickers_test_seq_high = create_sequences(test_df_scaled, FEATURE_COLUMNS, "Target_High")

    X_train_seq_low, y_train_seq_low, _ = create_sequences(train_df_scaled, FEATURE_COLUMNS, "Target_Low")
    X_test_seq_low, y_test_seq_low, tickers_test_seq_low = create_sequences(test_df_scaled, FEATURE_COLUMNS, "Target_Low")

    print(f"LSTM training shape: {X_train_seq_high.shape}")

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

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print("Training LSTM (High)...")
    lstm_high = build_lstm_model((WINDOW_SIZE, len(FEATURE_COLUMNS)))
    lstm_high.fit(
        X_train_seq_high, y_train_seq_high,
        validation_data=(X_test_seq_high, y_test_seq_high),
        epochs=50, batch_size=64, callbacks=[early_stop], verbose=2,
    )
    pred_lstm_high = lstm_high.predict(X_test_seq_high).flatten()

    print("Training LSTM (Low)...")
    lstm_low = build_lstm_model((WINDOW_SIZE, len(FEATURE_COLUMNS)))
    lstm_low.fit(
        X_train_seq_low, y_train_seq_low,
        validation_data=(X_test_seq_low, y_test_seq_low),
        epochs=50, batch_size=64, callbacks=[early_stop], verbose=2,
    )
    pred_lstm_low = lstm_low.predict(X_test_seq_low).flatten()

    results.append(evaluate_predictions(y_test_seq_high, pred_lstm_high, "LSTM", "High"))
    results.append(evaluate_predictions(y_test_seq_low, pred_lstm_low, "LSTM", "Low"))

    lstm_high.save(str(MODELS_DIR / "lstm_high.keras"))
    lstm_low.save(str(MODELS_DIR / "lstm_low.keras"))

    # Step 8: Overall comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["Target", "MAPE"])
    print("\n=== Overall Model Comparison ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(DATA_DIR / "model_comparison_overall.csv", index=False)

    # Step 9: Per-stock accuracy breakdown (LR, RF, XGBoost use test_df row order; LSTM uses its own ticker-aligned sequence order)
    predictions_high = {
        "Linear Regression": pred_lr_high,
        "Random Forest": pred_rf_high,
        "XGBoost": pred_xgb_high,
    }
    predictions_low = {
        "Linear Regression": pred_lr_low,
        "Random Forest": pred_rf_low,
        "XGBoost": pred_xgb_low,
    }

    per_stock_high = evaluate_per_stock(test_df["Ticker"].values, y_test_high, predictions_high, "Target_High")
    per_stock_low = evaluate_per_stock(test_df["Ticker"].values, y_test_low, predictions_low, "Target_Low")

    lstm_per_stock_high = evaluate_per_stock(tickers_test_seq_high, y_test_seq_high, {"LSTM": pred_lstm_high}, "Target_High")
    lstm_per_stock_low = evaluate_per_stock(tickers_test_seq_low, y_test_seq_low, {"LSTM": pred_lstm_low}, "Target_Low")

    per_stock_high = pd.concat([per_stock_high, lstm_per_stock_high], ignore_index=True)
    per_stock_low = pd.concat([per_stock_low, lstm_per_stock_low], ignore_index=True)

    per_stock_all = pd.concat([per_stock_high, per_stock_low], ignore_index=True)
    per_stock_all = per_stock_all.sort_values(["Ticker", "Target", "Accuracy_%"], ascending=[True, True, False])
    print("\n=== Per-Stock Accuracy (all models) ===")
    print(per_stock_all.to_string(index=False))
    per_stock_all.to_csv(DATA_DIR / "model_comparison_per_stock.csv", index=False)

    # Best model per stock (LR/RF/XGBoost only, since these are the models the
    # prediction helper below can score a single feature row with directly;
    # LSTM needs a full 30-day sequence and is compared separately above)
    selectable_high = per_stock_high
    selectable_low = per_stock_low

    best_high = selectable_high.loc[selectable_high.groupby("Ticker")["Accuracy_%"].idxmax()]
    best_low = selectable_low.loc[selectable_low.groupby("Ticker")["Accuracy_%"].idxmax()]

    print("\n=== Best Model per Stock (High) ===")
    print(best_high[["Ticker", "Model", "Accuracy_%"]].to_string(index=False))
    print("\n=== Best Model per Stock (Low) ===")
    print(best_low[["Ticker", "Model", "Accuracy_%"]].to_string(index=False))

    # "needs_scaling" marks models fit on scaler.transform()'d features (Linear
    # Regression); tree models were fit on raw features and must NOT be scaled.
    model_objects = {
        "Linear Regression": {"High": lr_high, "Low": lr_low, "needs_scaling": True},
        "Random Forest": {"High": rf_high, "Low": rf_low, "needs_scaling": False},
        "XGBoost": {"High": xgb_high, "Low": xgb_low, "needs_scaling": False},
    }

    best_model_lookup = {}
    for ticker in per_stock_high["Ticker"].unique():
        high_model_name = best_high.set_index("Ticker").loc[ticker, "Model"]
        low_model_name = best_low.set_index("Ticker").loc[ticker, "Model"]
        best_model_lookup[ticker] = {
            "High": {
                "model": model_objects[high_model_name]["High"],
                "needs_scaling": model_objects[high_model_name]["needs_scaling"],
            },
            "Low": {
                "model": model_objects[low_model_name]["Low"],
                "needs_scaling": model_objects[low_model_name]["needs_scaling"],
            },
        }

    with open(MODELS_DIR / "best_model_lookup.pkl", "wb") as f:
        pickle.dump(
            {
                "high_model_name": best_high.set_index("Ticker")["Model"].to_dict(),
                "low_model_name": best_low.set_index("Ticker")["Model"].to_dict(),
            },
            f,
        )

    # Step 10: Demonstrate the final price range prediction for one stock
    demo_ticker = "SYS"
    demo_row = X_test[test_df["Ticker"] == demo_ticker].tail(1)
    demo_date = test_df[test_df["Ticker"] == demo_ticker]["Date"].max()

    demo_result = predict_price_range(demo_ticker, demo_row, demo_date, best_model_lookup, scaler, holidays=PSX_HOLIDAYS_2026)
    print(f"\n{demo_result['Ticker']} predicted range for {demo_result['Predicted_Date']}: {demo_result['Range']}")

    # Step 11: Final validation, run once, using the overall best model per target
    overall_high = results_df[results_df["Target"] == "High"].sort_values("MAPE").iloc[0]
    overall_low = results_df[results_df["Target"] == "Low"].sort_values("MAPE").iloc[0]

    final_high_model = model_objects.get(overall_high["Model"], {}).get("High")
    final_low_model = model_objects.get(overall_low["Model"], {}).get("Low")

    if final_high_model is None:
        final_high_model = xgb_high
        print(f"\nOverall winner for High was {overall_high['Model']} (sequence model); using XGBoost for final validation instead.")
    if final_low_model is None:
        final_low_model = xgb_low
        print(f"Overall winner for Low was {overall_low['Model']} (sequence model); using XGBoost for final validation instead.")

    pred_validate_high = final_high_model.predict(X_validate)
    pred_validate_low = final_low_model.predict(X_validate)

    final_validation_high = evaluate_predictions(y_validate_high, pred_validate_high, "Final Model", "High")
    final_validation_low = evaluate_predictions(y_validate_low, pred_validate_low, "Final Model", "Low")

    print("\n=== Final Validation (validate_df, touched once) ===")
    print(final_validation_high)
    print(final_validation_low)

    pd.DataFrame([final_validation_high, final_validation_low]).to_csv(DATA_DIR / "final_validation.csv", index=False)

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
