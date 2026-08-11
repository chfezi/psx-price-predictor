# Phase 6: aligning the training strategy with the commodity forecasting reference project

## Context

Phases 1 through 5 of this project cleaned 25 PSX stocks, engineered around 47 features, merged them into one master_dataset.csv with Ticker as a feature, trained one unified model per model type (XGBoost, Random Forest, LSTM, Linear Regression) across all 25 stocks together, and served results through a Streamlit dashboard.

A separate commodity forecasting project was reviewed for its training strategy. Several parts of that strategy are being adopted here. The architecture decision has already been made: this project keeps its unified single model across all 25 stocks, with Ticker as a categorical feature. It does not switch to a separate model per stock. Everything in this phase is designed to work inside that existing architecture and inside the existing master_dataset.csv, not to replace it.

Before writing any code, read the current project structure end to end: clean_data.py, feature_engineering.py, merge_and_split.py, the modeling script, apply_range_sanity_check(), predict_price_range(), and app.py. Confirm the actual column names, file paths, and function signatures against what is described below, since this document is a plan, not a byte-for-byte match to the existing code. Adjust names to match what is actually in the project.

## Scope of this phase

1. Redefine the prediction target as a return instead of a raw price level
2. Reformulate the raw-price-based features into stationary ratio and return form
3. Replace the ATR sanity check with a backtested-error-based range and confidence calculation
4. Add an ensemble step that combines model predictions per stock instead of picking one best model
5. Add SHAP-based explainability as a required part of the output

Multi-horizon forecasting (the reference project's 1 to 120 day horizon set) is not in scope for this phase. Adding it means training separate models per horizon on top of the existing 25-stock unified structure, which multiplies the model count significantly and is a large enough change to warrant its own phase after this one lands and is validated.

## Step 1: redefine the target as a return

Current targets are Target_High and Target_Low, both raw price levels. Add return-based versions alongside them, do not delete the price-based columns yet, since the existing models and the sanity-check comparison still need them for a before/after comparison.

```python
import pandas as pd
import numpy as np

def add_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Ticker', 'Date']).copy()

    df['Target_High_Return'] = (
        df.groupby('Ticker')['High'].shift(-1) - df['Close']
    ) / df['Close']
    df['Target_Low_Return'] = (
        df.groupby('Ticker')['Low'].shift(-1) - df['Close']
    ) / df['Close']

    return df
```

Two things need checking against the real data before this is trusted:

- Confirm the return columns are computed against Close (the reference project's own Target_Return_Nd formula divides by the same day's Close, not the prior day's High or Low), and that this matches what makes sense for a next-day High and Low corridor prediction here.
- Check the actual distribution of these returns across all 25 stocks before picking a clip range. The reference project used clip(-0.8, 1.5) for horizons up to 120 days on commodities, that range does not transfer directly to a next-day PSX return. Compute the 0.1 and 99.9 percentiles of Target_High_Return and Target_Low_Return across the full dataset and clip at those, or at a fixed value like plus or minus 15 percent if the percentile-based bounds look unstable. State whichever choice is made and why, in the phase's own notes.

## Step 2: reformulate features into stationary form

Go through the existing ~47 features and sort them into three groups: keep as is, convert to ratio or return form, and drop.

Keep as is (already stationary or bounded): RSI, rolling volatility or rolling standard deviation, returns, candle and intraday features (body size, wick ratios, that kind of thing, as long as they are already normalized by price or range), date features.

Convert:

```python
def add_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # raw SMA / EMA values -> ratio of Close to that average
    for window in [5, 10, 20, 50]:
        sma_col = f'SMA_{window}'
        if sma_col in df.columns:
            df[f'Close_to_SMA_{window}'] = df['Close'] / df[sma_col]

    for span in [10, 20]:
        ema_col = f'EMA_{span}'
        if ema_col in df.columns:
            df[f'Close_to_EMA_{span}'] = df['Close'] / df[ema_col]

    # raw price lags -> return lags
    df['Return'] = df.groupby('Ticker')['Close'].pct_change()
    for lag in [1, 2, 3, 5, 10]:
        df[f'Ret_Lag_{lag}'] = df.groupby('Ticker')['Return'].shift(lag)

    # MACD normalized by price, if not already
    if 'MACD' in df.columns and 'MACD_Pct' not in df.columns:
        df['MACD_Pct'] = df['MACD'] / df['Close']

    # ATR normalized by price
    if 'ATR' in df.columns:
        df['ATR_Pct'] = df['ATR'] / df['Close']

    # Bollinger Bands -> %B and bandwidth instead of raw band values
    if {'BB_Upper', 'BB_Lower', 'BB_Middle'}.issubset(df.columns):
        band_width = df['BB_Upper'] - df['BB_Lower']
        df['BB_PercentB'] = (df['Close'] - df['BB_Lower']) / band_width
        df['BB_Bandwidth'] = band_width / df['BB_Middle']

    return df
```

Drop after the above conversions exist and have been validated: Close_lag_1, Close_lag_2, Close_lag_3, EMA_12, any raw SMA or EMA column, and any raw Bollinger Band value column. These are exactly the features already flagged in the earlier leakage check as high-correlation with the target. Converting them to ratio and return form is what resolves that concern properly, rather than just explaining it away as expected correlation.

After this step, rerun the same leakage check that was done before (confirm Target_High and Target_Low still align correctly per Ticker with the shifted values) against the new feature set, and rerun the naive baseline comparison so the new feature set has a fresh reference point, not the old one.

## Step 3: replace the ATR sanity check with a backtested-error-based range

The current fix bounds an overly wide Linear Regression prediction using the stock's own recent ATR, applied after the model's prediction is already made. The reference project instead derives the range width directly from each model's own backtested error distribution for that specific stock and target, so the range comes from how wrong the model has actually been historically, not from a separate volatility measure bolted on afterward.

```python
def compute_backtested_error_distribution(y_true, y_pred):
    """
    Run this per stock, per target (High, Low), per model, over the
    validation set. Returns the error distribution needed to size a
    prediction interval.
    """
    errors = y_true - y_pred
    return {
        'std_error': errors.std(),
        'p10': np.percentile(errors, 10),
        'p90': np.percentile(errors, 90),
    }

def predict_range_from_error_distribution(point_prediction, error_stats, z=1.28):
    """
    z=1.28 gives roughly an 80 percent interval assuming errors are close
    to normal. Compare against the p10/p90 empirical bounds from
    compute_backtested_error_distribution and use whichever is more
    stable for that stock, thin-traded stocks with less validation data
    will have noisier empirical percentiles than the normal approximation.
    """
    lower = point_prediction - z * error_stats['std_error']
    upper = point_prediction + z * error_stats['std_error']
    return lower, upper
```

This needs to be computed once per stock per target per model on the validation set, then stored (a small lookup table keyed by Ticker and target is enough) and reused at inference time, not recomputed on every prediction. Confidence for the output can come from the same error distribution: a tighter std_error relative to the stock's own price level means higher confidence, a wider one means lower confidence.

Keep the old apply_range_sanity_check() function in the codebase during this phase rather than deleting it, and run both the old ATR-based range and the new error-distribution range side by side on the validation set so the two can be compared directly before the old one is retired.

## Step 4: ensemble instead of single-best-model selection

Right now the best model is picked per stock per target and used alone. Replace this with a weighted average across the model types that were trained (XGBoost, Random Forest, LSTM, Linear Regression), weighted by each model's own backtested accuracy for that specific stock and target, the same per-stock weighting principle as the range calculation in step 3.

```python
def compute_ensemble_weights(model_errors_by_type: dict) -> dict:
    """
    model_errors_by_type: {'xgboost': mae, 'random_forest': mae, 'lstm': mae, 'linear_regression': mae}
    Lower MAE should get a higher weight. Inverse-error weighting is a
    reasonable starting point.
    """
    inverse_errors = {m: 1 / e for m, e in model_errors_by_type.items()}
    total = sum(inverse_errors.values())
    return {m: w / total for m, w in inverse_errors.items()}

def ensemble_predict(predictions_by_type: dict, weights: dict) -> float:
    return sum(predictions_by_type[m] * weights[m] for m in predictions_by_type)
```

Keep the single-best-model prediction available as a comparison option in the Streamlit app's manual override dropdown, since that override feature already exists and is useful for demonstrating that the ensemble is actually doing better than the best individual model, not just averaging away the best one's edge.

## Step 5: SHAP-based explainability

Add SHAP values on the tree-based models (XGBoost and Random Forest, these support SHAP directly without extra setup) per stock per target, and surface the top three features as plain-language drivers alongside each prediction.

```python
import shap

def get_top_drivers(model, X_row, feature_names, top_n=3):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)

    contributions = list(zip(feature_names, shap_values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    drivers = []
    for name, value in contributions[:top_n]:
        direction = 'pushing up' if value > 0 else 'pushing down'
        drivers.append(f"{name}: {direction}")
    return drivers
```

LSTM was already excluded from the live Streamlit interface for complexity reasons, so it does not need a SHAP or Integrated Gradients equivalent for this phase. If it gets added back later, Integrated Gradients through Captum is the equivalent method the reference project uses for its neural network models.

Add the driver output to the Streamlit dashboard next to each stock's predicted range, the same place the reference project shows its "top drivers" on each forecast card.

## Deliverables for this phase

- Updated feature_engineering.py (or equivalent) producing the stationary feature set alongside the existing raw-price features, not replacing them yet
- A comparison table: old feature set plus price targets versus new feature set plus return targets, evaluated with the same MAE, RMSE, MAPE, R2, and Accuracy_% metrics already used, per stock
- Backtested error distributions computed and stored per stock per target per model
- Old ATR-based range versus new error-distribution-based range compared side by side on the validation set
- Ensemble predictions compared against the current single-best-model predictions, per stock per target
- SHAP driver output wired into the Streamlit app for at least the XGBoost and Random Forest models

## Validation checklist before calling this phase done

- Confirm Target_High_Return and Target_Low_Return align correctly per Ticker (no cross-stock shift bugs), the same check already done for the price-based targets
- Confirm the new stationary features do not reintroduce the same high-correlation issue the old Close_lag and EMA_12 features had
- Confirm the new error-distribution range is not systematically wider or narrower than the old ATR-based range in a way that is not backed by the model's actual validation error
- Confirm the ensemble beats the single-best-model approach on at least a majority of the 25 stocks, if it does not, that is a legitimate finding worth reporting, not something to force