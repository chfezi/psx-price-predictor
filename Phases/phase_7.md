# Phase 7: multi-horizon forecasting and GRU

## Context

Phase 6 replaced the raw-price targets with return-based targets, reformulated the raw-price features into stationary ratio and return form, replaced the ATR sanity check with a backtested-error-based range, added an ensemble step, and added SHAP-based explainability, all inside the existing unified single-model architecture (25 stocks trained together, Ticker as a feature).

Phase 7 adds a second dimension on top of that: multiple forecast horizons instead of only next-day, and one additional model type, GRU. The unified architecture from Phase 6 does not change. Each horizon still trains one model per model type across all 25 stocks together, not a separate model per stock, and not a separate model per commodity the way the reference project does it.

Before writing any code, confirm the actual state of the Phase 6 work in the codebase: the exact names of the return-target columns, the stationary feature columns, and the ensemble and range functions Phase 6 produced. This document assumes Phase 6 is finished and its output is the starting point for Phase 7.

## Scope of this phase

1. Add horizon-specific target columns
2. Handle the trailing-rows problem per stock
3. Train the existing model roster plus GRU across the chosen horizons
4. Extend the backtested error distribution, range calculation, and ensemble weighting to be horizon-aware
5. Extend the evaluation table with a horizon column

Frontend changes and the before-and-after comparison summary are Phase 8, not this phase.

## Step 1: choose the horizon set

The reference project used eight horizons, 1 through 120 trading days. That set was built for commodities with around 2,500 rows each and a single time series per asset. This project has 25 stocks, several of them thin-traded (PAKT, NESTLE, COLG, ILP have fewer rows to begin with), so a horizon of 90 or 120 trading days would leave very few valid training rows for those stocks once the trailing rows are dropped.

A smaller horizon set is the reasonable default here: 1-day (already exists from Phase 6), 5-day (about a week), 10-day (about two weeks), 20-day (about a month), 60-day (about a quarter). Check the actual row count that survives for the thin-traded stocks at 60 days before committing to it. If any stock ends up with too few rows to train on reliably at that horizon, drop that stock from the 60-day model rather than dropping the horizon for everyone, and note it in the results.

## Step 2: horizon-specific targets

Same target definition Phase 6 established (a return relative to that day's Close), just shifted further out per horizon, computed per Ticker so one stock's future values never leak into another's row.

```python
def add_horizon_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df.sort_values(['Ticker', 'Date']).copy()

    for n in horizons:
        df[f'Target_High_Return_{n}d'] = (
            df.groupby('Ticker')['High'].shift(-n) - df['Close']
        ) / df['Close']
        df[f'Target_Low_Return_{n}d'] = (
            df.groupby('Ticker')['Low'].shift(-n) - df['Close']
        ) / df['Close']

    return df
```

This predicts the High and Low of the specific trading day N days out, the same definition Phase 6 uses for next-day, just shifted further into the future. It is not a range across the whole N-day window between now and then.

## Step 3: trailing rows per horizon

The last N rows of each stock's history do not have a valid target for horizon N, since there is no future data that far out yet for that stock. Drop these per Ticker, per horizon, when building that horizon's training set. Do not drop them from the master dataset itself, since a row missing only the 60-day target can still be valid for the 1-day and 5-day targets.

```python
def get_training_rows_for_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_col = f'Target_High_Return_{horizon}d'
    return df.dropna(subset=[target_col, f'Target_Low_Return_{horizon}d'])
```

## Step 4: add GRU

GRU is a lighter alternative to LSTM, fewer parameters, and it reuses the same data preparation LSTM already has: scaled features reshaped into sequences. Reuse the existing scaler and sequence-building code from the LSTM pipeline rather than writing a second one.

```python
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)
```

Lock the hyperparameters (learning rate, sequence length, batch size, number of epochs or early stopping patience) identically across every horizon for GRU, the same way LSTM's hyperparameters should stay fixed across horizons. This is what makes it possible to say a difference in accuracy across horizons comes from the horizon itself, not from different tuning at each one.

Whether GRU gets served live in the Streamlit app, the same way LSTM currently does not, is a Phase 8 decision, not this one. Train and evaluate it here regardless.

## Step 5: train the roster per horizon

Every model type from Phase 6, plus GRU, gets trained once per horizon, still across all 25 stocks together with Ticker as a feature. Tree models and Linear Regression use the same reformulated feature table from Phase 6 directly. LSTM and GRU use the scaled, sequenced version of that same feature table.

```python
model_types = ['xgboost', 'random_forest', 'lstm', 'gru', 'linear_regression']
horizons = [1, 5, 10, 20, 60]
targets = ['High', 'Low']

for horizon in horizons:
    training_df = get_training_rows_for_horizon(master_df, horizon)
    for target in targets:
        target_col = f'Target_{target}_Return_{horizon}d'
        for model_type in model_types:
            # reuse the Phase 6 training function for this model type,
            # pointed at target_col instead of the next-day target
            train_and_save_model(
                model_type=model_type,
                training_df=training_df,
                target_col=target_col,
                horizon=horizon,
            )
```

With 5 horizons, 5 model types, and 2 targets, this is 50 models total, still far smaller than the reference project's count, since that project multiplies by 6 separate commodities and this project keeps one model per horizon per type across all 25 stocks.

## Step 6: horizon-aware range, confidence, and ensemble

Phase 6's backtested error distribution and ensemble weighting were computed per stock per target per model. Extend both to also be keyed by horizon, since a model's error at 60 days out is not the same as its error at 1 day out, and the range should reflect that.

```python
# error_distributions[(ticker, target, horizon, model_type)] = {'std_error': ..., 'p10': ..., 'p90': ...}
# ensemble_weights[(ticker, target, horizon)] = {'xgboost': w1, 'gru': w2, ...}
```

Once this is in place, check that range width actually widens as horizon increases for most stocks. This should come out of the backtested errors on its own. If a stock's range stays flat or gets narrower at a longer horizon, that is worth a second look before trusting that horizon's output for that stock.

## Step 7: extend the evaluation table

Add a Horizon column to whatever evaluation table Phase 6 produced. Alongside MAE, RMSE, MAPE, R2, and Accuracy_%, add Directional Accuracy (percentage of times the model correctly predicted whether the price moved up or down) and Improvement over Naive Baseline as a percentage, both per model, per stock, per target, now also per horizon. The reference project uses exactly these two for its own horizon comparisons, and they are the metrics that make a horizon-by-horizon table actually readable, since MAE and RMSE are on different scales at 1 day versus 60 days and are harder to compare across horizons at a glance. These can also be backfilled onto the existing 1-day results from Phase 6 if that was not already done.

## Deliverables for this phase

- Horizon-specific target columns for the chosen horizon set, added per Ticker
- GRU trained and evaluated across all horizons and both targets, using the same locked-hyperparameter discipline as the other models
- Full model roster (XGBoost, Random Forest, LSTM, GRU, Linear Regression) trained per horizon per target, 50 models total for the 5-horizon set
- Backtested error distributions, range calculation, and ensemble weights extended to be horizon-aware
- Evaluation table extended with a Horizon column, Directional Accuracy, and Improvement over Naive Baseline

## Validation checklist before calling this phase done

- Confirm the thin-traded stocks have enough rows left at the longest chosen horizon to train reliably; drop that stock from that horizon's model if not, rather than forcing it
- Confirm horizon targets align correctly per Ticker (no cross-stock shift bugs), the same check already run for the Phase 6 targets
- Confirm range width widens with horizon for most stocks, and flag any stock where it does not
- Confirm GRU's hyperparameters are identical across every horizon it was trained on