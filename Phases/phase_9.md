# Phase 9: Open/Close Targets, GPU Training, Loss Logging

## Objective

Replace Target_High and Target_Low with Target_Open and Target_Close across
the entire pipeline. Keep the unified single-model architecture (25 stocks
together, Ticker as a feature) exactly as it is. Add GPU support for LSTM
and GRU. Extend the loss logging that already exists in Phase 7 to cover
this retrain. Fix the naive baseline for Open, since "today's Open held
constant" is not a fair comparison. Delete Phase 4 and Phase 6 outright,
which means pulling the handful of functions Phase 7 still imports from
Phase 6 into a shared module first, or the training script will not even
import.

## Why this needs a shared module first

train_models_phase7.py imports six functions from train_models_phase6.py
(evaluate_predictions, evaluate_per_stock, to_price,
compute_backtested_error_distribution, compute_ensemble_weights, and
create_sequences by extension through the horizon-aware wrappers).
train_models_phase6.py in turn imports FEATURE_COLUMNS from train_models.py
and loads five Phase 4 model files from disk in its Step 12 range
comparison. Deleting train_models.py and train_models_phase6.py without
moving these functions somewhere else breaks the import at the top of
whatever trains Phase 9's models.

None of these functions are High/Low-specific. They take price arrays and
model names as arguments and do generic evaluation or ensemble math. Moving
them once removes the dependency on files that are about to be deleted and
does not change any behavior.

Phase 6's Step 12 (old ATR-range vs new error-distribution-range
comparison) is not carried forward. It depended on loading Phase 4's model
files, which are being deleted, and the result of that comparison is
already recorded: the old range had 24 percent coverage, the new one had
83 percent. That comparison already happened once. Phase 9 does not need
to repeat it against a set of models that no longer exists.

## Step 1: model_utils.py

New file. Move these functions here, unchanged, from
train_models_phase6.py:

- evaluate_predictions
- evaluate_per_stock
- create_sequences
- to_price
- compute_backtested_error_distribution
- predict_range_from_error_distribution
- compute_ensemble_weights
- ensemble_predict
- build_error_distributions

Also move the WINDOW_SIZE, Z_SCORE_80PCT constants here, since
create_sequences and predict_range_from_error_distribution use them as
defaults.

train_models_phase7.py's horizon-aware wrappers (evaluate_predictions_h,
evaluate_per_stock_h, create_sequences_h, build_error_distributions_h) stay
where a Phase 9 training script defines them, since they are specific to
this project's horizon extension and were written for Phase 7 rather than
copied from anywhere reusable.

## Step 2: feature_engineering.py

Replace add_target_variables, add_return_targets, and add_horizon_targets
with Open/Close versions. Same shapes as before: raw shifted value, a
clipped return relative to that day's Close, and horizon variants for
n = 1, 5, 10, 20, 60.

```python
def add_target_variables(df):
    df["Target_Open"] = df["Open"].shift(-1)
    df["Target_Close"] = df["Close"].shift(-1)
    return df


def add_return_targets(df):
    """
    Same clip rationale as the High/Low version this replaces: PSX's daily
    circuit limits mean single-day moves past roughly 15 percent are data
    errors, not real prices, so the clip protects training from that
    handful of bad rows rather than reflecting a real return ceiling.
    """
    df["Target_Open_Return"] = (df["Target_Open"] - df["Close"]) / df["Close"]
    df["Target_Close_Return"] = (df["Target_Close"] - df["Close"]) / df["Close"]

    for col in ["Target_Open_Return", "Target_Close_Return"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).clip(-0.15, 0.15)

    return df


def add_horizon_targets(df, horizons=HORIZONS):
    for n in horizons:
        open_col = f"Target_Open_{n}d"
        close_col = f"Target_Close_{n}d"
        df[open_col] = df["Open"].shift(-n)
        df[close_col] = df["Close"].shift(-n)

        clip = 0.15 * np.sqrt(n)
        df[f"Target_Open_Return_{n}d"] = (
            (df[open_col] - df["Close"]) / df["Close"]
        ).replace([np.inf, -np.inf], np.nan).clip(-clip, clip)
        df[f"Target_Close_Return_{n}d"] = (
            (df[close_col] - df["Close"]) / df["Close"]
        ).replace([np.inf, -np.inf], np.nan).clip(-clip, clip)

    return df
```

check_no_leakage's target_column argument changes from "Target_High" to
"Target_Close" wherever main() calls it, plus the equivalent horizon-suffixed
names.

### Close-outside-[Low,High] fix

clean_data.py's WARNING at lines 67-71 already detects the 66 rows where
Close falls outside [Low, High] but never filters them, unlike the
negative-price and negative-volume checks below it which do filter. Since
Close is becoming a primary target instead of a supporting feature, add the
filter now:

```python
outside_range = (df["Close"] < df["Low"]) | (df["Close"] > df["High"])
if outside_range.any():
    print(f"WARNING: dropping {outside_range.sum()} rows where Close falls outside [Low, High]")
    df = df[~outside_range].reset_index(drop=True)
```

Place this filter alongside the existing negative-price/negative-volume
filters, not just the warning print, so it actually removes the rows this
time.

## Step 3: merge_and_split.py

TARGET_COLUMNS changes from ["Target_High", "Target_Low"] to
["Target_Open", "Target_Close"].

HORIZON_TARGET_COLUMNS's side list changes from ["High", "Low"] to
["Open", "Close"]. Everything else in this file, including the dropna
exclusion logic for horizon columns and the time-based split, stays as it
is.

## Step 4: train_models_phase9.py

New file, structured the same way as train_models_phase7.py: one unified
model per horizon per model type per target, trained on all 25 stocks
together. Same five horizons, same five model types. Imports from
model_utils.py instead of train_models_phase6.py.

### Naive baseline, asymmetric by target

```python
def naive_prediction(df, target, horizon):
    """
    Close: no change from today's Close is the standard random-walk
    baseline for a closing price and stays as-is.

    Open: today's Open held constant is not a fair baseline, since Open
    reflects the previous session's close plus whatever gap happened
    overnight. The fair comparison is tomorrow's Open equals today's
    Close, since that is what happens most of the time absent a gap.
    Using the weaker baseline would make the model's Improvement_over_Naive_%
    look better than it honestly is.
    """
    if target == "Close":
        return df["Close"].values
    return df["Close"].values  # Open's fair naive baseline is also today's Close
```

Both branches return the same column here, which looks redundant, but the
function stays split by target so the reasoning is visible at the call site
and so a future change to one baseline does not silently change the other.

### GPU support

Add device detection near the top of the file:

```python
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training LSTM/GRU on: {DEVICE}")
```

For GRU (PyTorch), move the model and every tensor to DEVICE:

```python
model = GRUModel(input_size).to(DEVICE)
X_train_t = torch.tensor(X_train_seq, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1).to(DEVICE)
X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)
y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1).to(DEVICE)
```

Inside the DataLoader loop, move each batch to DEVICE as well, since
DataLoader yields CPU tensors by default even when the dataset tensors were
already moved:

```python
for xb, yb in loader:
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    ...
```

When saving predictions back out, move tensors back to CPU before calling
.numpy(), since numpy cannot read directly from a CUDA tensor:

```python
test_pred = model(X_test_t).cpu().numpy().flatten()
```

For LSTM (Keras/TensorFlow), TensorFlow picks up an available GPU
automatically if the right CUDA/cuDNN libraries are installed, with no code
change needed in the model-building or fit() calls. The only addition
worth making is a startup print so it is visible which device is actually
being used, since a silent CPU fallback is easy to miss:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow sees {len(gpus)} GPU(s): {gpus}")
```

XGBoost, optionally, by passing device="cuda" at construction if a
compatible XGBoost version and driver are confirmed available:

```python
xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu",
)
```

Random Forest and Linear Regression are left as they are. Neither has a
meaningful GPU path in scikit-learn.

### Loss logging

Phase 7 already writes phase7_training_loss_history.csv with per-epoch
Train_Loss and Val_Loss for both LSTM and GRU. Keep the same structure,
same column names, new filename: phase9_training_loss_history.csv. No new
logging mechanism needed, this pattern already does what was asked for.

For a live view while training runs, add a SummaryWriter alongside the
existing print/append pattern in the GRU loop:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f"runs/phase9_gru_{target.lower()}_{horizon}d")
# inside the epoch loop, after computing train_loss and val_loss:
writer.add_scalar("Loss/train", train_loss, epoch)
writer.add_scalar("Loss/val", val_loss, epoch)
```

Running `tensorboard --logdir=runs` from the project root gives a live
browser view of every run's loss curves while training is in progress.
LSTM's Keras training can log to the same runs/ directory via a
TensorBoard callback passed alongside the existing EarlyStopping callback.

### Early stopping

Already implemented for both LSTM (Keras EarlyStopping, patience 10,
restore_best_weights True) and GRU (the manual patience-based loop already
in train_models_phase7.py, patience 10). Carry this logic over unchanged.
Epoch count is not a number to pick by hand here, it is already decided by
validation loss plateauing.

### Fresh scaler

phase6_feature_scaler.pkl is a live dependency of Phase 7's Linear
Regression, LSTM, and GRU models at inference time. Since Phase 6 is being
deleted, fit a new scaler in this file and save it as
phase9_feature_scaler.pkl, the same way train_models_phase6.py's Step 2
did.

### Evaluation

Same metrics as Phase 7: MAE, RMSE, MAPE, R2, Accuracy_%,
Directional_Accuracy_%, Improvement_over_Naive_%. All computed against
Open/Close instead of High/Low, no changes to the formulas.

Add one new metric: Open-to-Close spread accuracy, comparing predicted
(Close - Open) against actual (Close - Open). This takes over the role
High/Low's range coverage used to serve, now expressed as how well the
model captures a day's overall move rather than a range.

```python
def spread_accuracy(pred_open, pred_close, actual_open, actual_close):
    pred_spread = pred_close - pred_open
    actual_spread = actual_close - actual_open
    mae_spread = mean_absolute_error(actual_spread, pred_spread)
    return mae_spread
```

## Step 5: manage_model_storage.py

TARGETS changes from ["High", "Low"] to ["Open", "Close"].

PHASE4_FILES and PHASE6_FILES_TO_MOVE lists become irrelevant, since those
files are being deleted rather than moved into models_evaluation_only/.
Remove the two move loops for Phase 4 and Phase 6 files. Keep the
compute_needed_combos and unneeded-combo logic, since that still applies to
deciding which of Phase 9's own model files are worth keeping in the live
models/ directory versus setting aside.

Filename pattern in the unneeded-combo loop changes from
`phase7_{prefix}_{target.lower()}_{horizon}d` to
`phase9_{prefix}_{target.lower()}_{horizon}d`, matching whatever prefix
train_models_phase9.py actually saves under.

## Step 6: delete outright

- train_models.py
- train_models_phase6.py
- All Phase 4 model files (lr_high.pkl, lr_low.pkl, rf_high.pkl, rf_low.pkl,
  xgb_high.json, xgb_low.json, lstm_high.keras, lstm_low.keras)
- All Phase 6 model files (phase6_lr_high.pkl, phase6_lr_low.pkl,
  phase6_rf_high.pkl, phase6_rf_low.pkl, phase6_xgb_high.json,
  phase6_xgb_low.json, phase6_lstm_high.keras, phase6_lstm_low.keras,
  phase6_error_distributions.pkl, phase6_ensemble_weights.pkl)
- phase6_feature_scaler.pkl, once phase9_feature_scaler.pkl exists and
  train_models_phase9.py no longer references it
- best_model_lookup.pkl and feature_scaler.pkl (Phase 4's own scaler and
  lookup table, only used by Phase 6's Step 12 comparison, which is not
  being carried forward)
- models_evaluation_only/ in full
- All Phase 7 model files trained on High/Low (phase7_*_high_*d.*,
  phase7_*_low_*d.*), once Phase 9's Open/Close equivalents are trained
  and verified working

Before deleting, pull the final metrics from
phase6_model_comparison_overall.csv, phase6_feature_target_comparison.csv,
phase7_model_comparison_overall.csv, and phase7_model_comparison_per_stock.csv
into a plain results table if any of those numbers are wanted for the
before/after summary planned for the redone Phase 8. The CSVs themselves
are small and can be kept even after the model files they describe are
gone. The model files cannot be regenerated without retraining, the CSVs
can be copied in a few seconds.

## Not in this phase

SARIMAX and Prophet are not part of Phase 9. Both are single-series models
that would require 25 separate fits per model type, which breaks the
unified single-model architecture. Given PSX's documented low daily
volatility, they would likely land at or below the naive baseline the same
way the existing models sometimes do, which is a legitimate finding but not
one that needs 50 more models trained to make it. If added at all, this
would be a small separate comparison script run against a handful of
stocks, not part of the serving pipeline.

Phase 8's dashboard redesign happens after Phase 9 is complete and
verified, not alongside it. The cone-based UI concept was built around a
High-Low range and needs to become a point-estimate-with-interval concept
for Open and Close, which is a design decision, not a relabeling.

News and sentiment features are a later phase, after Phase 9 is stable.