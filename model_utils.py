"""
Shared model evaluation/ensemble utilities.

Moved out of train_models_phase6.py (unchanged) so Phase 9's training script
can import them without depending on train_models_phase6.py or, through it,
train_models.py - both deleted in Phase 9. See Phases/phase_9.md "Why this
needs a shared module first" and Phases/phase_9_notes.md.

None of these functions are High/Low or Open/Close-specific: they take price
arrays and model names as arguments and do generic evaluation or ensemble
math.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

WINDOW_SIZE = 30
Z_SCORE_80PCT = 1.28


def evaluate_predictions(y_true, y_pred, model_name, target_name):
    """
    No Accuracy_% (100 - MAPE) here on purpose: it reads as a headline
    quality score but stock prices barely move day to day, so it stays
    flatteringly high (76-99%) even for models that lose to the naive
    baseline. Improvement_over_Naive_% (added by the horizon-aware wrappers
    that call this) is the metric that actually says whether a model adds
    value - see Phases/phase_9_notes.md.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "Target": target_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4),
    }


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
                }
            )

    return pd.DataFrame(per_stock_results)


def create_sequences(df, feature_columns, return_target_column, price_target_column,
                      close_column="Close", ticker_column="Ticker", window_size=WINDOW_SIZE):
    """
    Also returns the aligned Close and raw price-target arrays needed to
    convert a return prediction back to price space and score it against
    ground truth.

    close_column defaults to "Close" for backward compatibility, but when
    "Close" is itself one of feature_columns, the caller should pass a
    separate raw-close column instead (e.g. "Close_raw"). Reading raw Close
    from the same column used for the scaled feature would silently
    overwrite the standardized Close *feature* fed to the model with an
    unscaled one - see Phases/phase_7_notes.md.
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


def to_price(close_values, return_predictions):
    return np.asarray(close_values, dtype=float) * (1 + np.asarray(return_predictions, dtype=float))


def compute_backtested_error_distribution(y_true_price, y_pred_price):
    """
    Per stock, per target, per model, over the test set. Returns the error
    distribution needed to size a prediction interval, in price units (PKR).
    """
    errors = np.asarray(y_true_price, dtype=float) - np.asarray(y_pred_price, dtype=float)
    return {
        "std_error": float(errors.std()),
        "p10": float(np.percentile(errors, 10)),
        "p90": float(np.percentile(errors, 90)),
        "n": int(len(errors)),
    }


def predict_range_from_error_distribution(point_prediction, error_stats, z=Z_SCORE_80PCT):
    lower = point_prediction - z * error_stats["std_error"]
    upper = point_prediction + z * error_stats["std_error"]
    return lower, upper


def compute_ensemble_weights(model_mae_by_type):
    """Lower MAE -> higher weight (inverse-error weighting)."""
    inverse_errors = {m: 1 / e for m, e in model_mae_by_type.items() if e > 0}
    total = sum(inverse_errors.values())
    return {m: w / total for m, w in inverse_errors.items()}


def ensemble_predict(predictions_by_type, weights):
    return sum(predictions_by_type[m] * weights[m] for m in predictions_by_type if m in weights)


def build_error_distributions(tickers, y_true_price, preds_by_model_price, target_label, store):
    tickers = np.asarray(tickers)
    y_true_price = np.asarray(y_true_price, dtype=float)
    for model_name, preds in preds_by_model_price.items():
        preds = np.asarray(preds, dtype=float)
        for ticker in np.unique(tickers):
            mask = tickers == ticker
            stats = compute_backtested_error_distribution(y_true_price[mask], preds[mask])
            store.setdefault(ticker, {}).setdefault(target_label, {})[model_name] = stats


def _naive_relative_improvement(actual, close, preds):
    """Default metric_fn for bootstrap_paired_comparison: same naive-relative
    MAE-improvement-% reported elsewhere in this codebase as
    Improvement_over_Naive_%/Ensemble_Improvement_over_Naive_%."""
    naive_mae = mean_absolute_error(actual, close)
    pred_mae = mean_absolute_error(actual, preds)
    return 100 * (naive_mae - pred_mae) / naive_mae if naive_mae > 0 else float("nan")


def bootstrap_paired_comparison(tickers, actual, close, pred_a, pred_b,
                                 metric_fn=None, n_resamples=2000, seed=42):
    """
    Ticker-level (cluster) bootstrap for two paired prediction series that
    share the same tickers/actual/close rows (e.g. two phases' ensemble
    blends scored on the identical validate.csv rows). Resamples unique
    tickers with replacement - each draw pulls that ticker's whole row-block
    intact, so within-ticker serial correlation (worst for overlapping-
    horizon targets, e.g. 60d) is preserved exactly rather than modeled with
    an autocovariance estimator. Not valid for unpaired series - pred_a and
    pred_b must be scored against the same actual/close rows in the same
    order.

    metric_fn(actual, close, preds) -> float. Defaults to the same
    naive-relative MAE-improvement-% used elsewhere in this codebase, so
    bootstrap output is on the same scale as numbers already reported.

    Returns observed_a/observed_b/observed_diff (metric_fn(pred_b) -
    metric_fn(pred_a) on the real, non-resampled data), the full
    bootstrap_diffs array, pct_b_wins (share of resamples where b beat a),
    a two-sided p_raw, and a 95% percentile CI on the difference.
    """
    tickers = np.asarray(tickers)
    actual = np.asarray(actual, dtype=float)
    close = np.asarray(close, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    if metric_fn is None:
        metric_fn = _naive_relative_improvement

    unique_tickers = np.unique(tickers)
    ticker_to_idx = {t: np.where(tickers == t)[0] for t in unique_tickers}

    observed_a = metric_fn(actual, close, pred_a)
    observed_b = metric_fn(actual, close, pred_b)
    observed_diff = observed_b - observed_a

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled_tickers = rng.choice(unique_tickers, size=len(unique_tickers), replace=True)
        idx = np.concatenate([ticker_to_idx[t] for t in sampled_tickers])
        metric_a = metric_fn(actual[idx], close[idx], pred_a[idx])
        metric_b = metric_fn(actual[idx], close[idx], pred_b[idx])
        diffs[i] = metric_b - metric_a

    p_raw = min(2 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))), 1.0)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    return {
        "observed_a": observed_a,
        "observed_b": observed_b,
        "observed_diff": observed_diff,
        "bootstrap_diffs": diffs,
        "pct_b_wins": float(np.mean(diffs > 0)),
        "p_raw": p_raw,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }
