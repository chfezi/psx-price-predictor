# Phase 7 notes: decisions and results

Companion to `Phases/phase_7.md`. Documents the choices the doc explicitly asked
to be stated and why, plus the actual numbers produced by `train_models_phase7.py`
on 2026-08-10. Implementation: `feature_engineering.py` (horizon targets),
`merge_and_split.py` (dropna fix - see Decision 2), `train_models_phase7.py`
(training/evaluation/error-distributions/ensemble-weights). No `app.py` changes
this phase, per the doc's own scope note.

## Decision 1: horizon set is 1/5/10/20/60 trading days

Per the doc's own reasoning - fewer than the reference project's 8 horizons
because this project pools all 25 stocks into one model per horizon rather than
training per-asset.

## Decision 2: no stock needs to be dropped from the 60-day horizon - and a real bug this surfaced

Checked row counts directly before writing any training code: PAKT, the
thinnest-traded stock (1573 total rows), still has 1513 rows left after
dropping the trailing 60. Every stock keeps 1500+ rows at every horizon. The
doc's "drop the stock from that horizon" contingency never triggers here,
because the unified architecture pools all 25 stocks into one training set.

Implementing this checking surfaced a real bug worth recording. `merge_and_split.py`
originally called `.dropna()` on `train_df`/`test_df`/`validate_df` with no
column subset - i.e. across every column, master dataset included. Once
`feature_engineering.py` added horizon target columns with intentional
trailing-row NaNs (`Target_High_60d` etc, NaN in each ticker's last 60 rows),
that blanket dropna started deleting those rows from `train.csv`/`test.csv`
entirely - which also destroyed those rows' otherwise-valid 1d/5d/10d/20d
targets, silently shrinking every horizon's training set to the 60-day
horizon's intersection. This is exactly what `Phases/phase_7.md` Step 3 warns
against ("do not drop them from the master dataset itself"). Fixed by excluding
the new horizon-specific target columns from the dropna subset in
`merge_and_split.py` (`HORIZON_TARGET_COLUMNS`); confirmed the fix by rerunning
and checking `train.csv`/`test.csv`/`validate.csv` row counts matched Phase 6's
exactly (30463/6091/9893) again afterward.

That investigation also turned up a second, related finding: **`train.csv` and
`test.csv` have zero NaN in any horizon target, at any horizon.** Because
`add_horizon_targets` shifts each ticker's full continuous history (2018 through
mid-2026) *before* the time-based split, a horizon-60 lookup from the last row of
`train.csv` (Dec 2023) or `test.csv` (Dec 2024) lands on real data still inside
the dataset's full history, not a gap. Only `validate.csv` - covering the true
end of history - has genuine trailing NaNs (1475-1500 rows per horizon,
scaling with horizon length, matching `n=60`: `25 tickers x 60 rows = 1500`).
Since training uses `train.csv`/`test.csv` only (`validate.csv` stays untouched,
per Phase 6's convention), `get_training_rows_for_horizon()` in
`train_models_phase7.py` turns out to be a no-op safety net on the data actually
used for training here - correct and worth keeping for robustness, but not
doing any real work in this particular dataset.

## Decision 3: return-target clip scales with horizon as `+/-0.15 * sqrt(horizon)`

A flat +/-15% (Phase 6's 1-day clip) would be wrong at longer horizons: the real
1st/99th percentile of 60-day returns is already +/-40-76%, so a flat 15% clip
would cut off the *bulk* of genuine 60-day moves, not just bad-data tails - the
opposite of what clipping should do. Checked the actual effect of `sqrt(horizon)`
scaling before committing to it:

| Horizon | Clip | Rows beyond clip (High) | Rows beyond clip (Low) |
|---|---|---|---|
| 1d | +/-15.0% | (Phase 6: ~0.03-0.07%) | (Phase 6: ~0.03-0.07%) |
| 5d | +/-33.5% | 0.15% | 0.12% |
| 10d | +/-47.4% | 0.15% | 0.16% |
| 20d | +/-67.1% | 0.20% | 0.21% |
| 60d | +/-116.2% | 0.31% | 0.22% |

Consistently small and slowly growing with horizon - the clip only touches
bad-data tails at every horizon, the same property Phase 6's flat clip had at 1
day, not the bulk of the distribution.

## Decision 4: one shared feature scaler, reused from Phase 6, not refit

`models/phase6_feature_scaler.pkl` loaded directly - the feature table
(`NEW_FEATURE_COLUMNS`) doesn't change per horizon, only the target does.

## Decision 5: every horizon, including 1-day, retrained under the new pipeline

Per the doc's own model count (5 horizons x 5 model types x 2 targets = 50
models). Phase 6's `phase6_*` artifacts are untouched; the fresh 1-day run here
is a useful sanity check on its own (see "Fresh 1-day results" below).

## A bug found while building this, and since fixed in Phase 6 too

While reusing Phase 6's LSTM sequence-prep pattern (`train_df_scaled[NEW_FEATURE_COLUMNS]
= scaler.transform(...)`, then restoring raw `Close` for price conversion), it
became clear that restoring raw `Close` into the *same* column used for model
input would silently overwrite the scaled `Close` feature with an unscaled one -
`Close` is both one of `NEW_FEATURE_COLUMNS` (meant to be standardized for the
model) and the value `create_sequences` needs raw for price conversion. Feeding
the model an unscaled Close (ranging ~11 to ~9000 across the 25 stocks) alongside
42 properly standardized features would reintroduce exactly the kind of scale
mismatch Phase 6 was designed to eliminate. `train_models_phase6.py`'s actual
`create_sequences`/LSTM prep had this pattern too. Phase 7 avoids it from the
start via a separate `Close_raw` column in `create_sequences_h()`.

Update: at the user's request this was then fixed in `train_models_phase6.py`
itself (same `Close_raw` pattern, added a `close_column` parameter to
`create_sequences`) and the full Phase 6 pipeline was rerun. Effect was mild but
real - LSTM MAE improved slightly (High: 8.72 -> 8.52, Low: 7.47 -> 7.40); every
other Phase 6 headline number was unchanged. See the addendum at the top of
`Phases/phase_6_notes.md` for the full writeup.

## Fresh 1-day results (sanity check against Phase 6)

| Model | Target | Phase 6 MAE | Phase 7 (fresh) MAE |
|---|---|---|---|
| Random Forest | High | 8.58 | 8.58 |
| XGBoost | High | 8.54 | 8.54 |
| Linear Regression | High | 8.87 | 8.87 |
| LSTM | High | 8.72 | 8.72 |

RF/XGBoost/LR match Phase 6 exactly (deterministic, same data/hyperparameters).
LSTM matches almost exactly too (8.72 both times) despite being a fresh training
run with random initialization - a good sign the pipeline is behaving
consistently, and that the `Close_raw` fix above didn't change results here in
any obviously large way (LSTM training has run-to-run noise regardless of the
fix, so this one data point isn't proof either way, just a consistency check).

## Result: accuracy and directional accuracy degrade smoothly with horizon (as expected)

Average `Accuracy_%` across all 5 model types, both targets:

| Horizon | Avg Accuracy_% | Avg Directional_Accuracy_% |
|---|---|---|
| 1d | 98.61 | 89.88 |
| 5d | 95.59 | 62.74 |
| 10d | 93.00 | 56.52 |
| 20d | 89.07 | 52.19 |
| 60d | 78.62 | 53.39 |

Both degrade steadily as the horizon lengthens, which is the expected, honest
shape for a forecasting problem - not a bug. Directional accuracy at 20d/60d
sits close to 50% (a coin flip) for several models, which matters more than the
Accuracy_%/MAPE numbers alone: MAPE-based accuracy still reads as "89%" at 20
days mostly because stock prices don't move a huge % day-to-day even over a
month, not because the model is confidently right about direction.

## Result: most models barely beat, or lose to, the naive baseline beyond 1 day

`Improvement_over_Naive_%` (naive = today's High/Low held constant N days out):
strongly positive at 1d (+5% to +19% across the 5 model types), then mostly
**negative** from 5d onward - e.g. at 20d every model type shows -4% to -22%
improvement (i.e. *worse* than just guessing no change), and at 60d only Linear
Regression stays (barely) positive (+2.21% High, +1.77% Low); every other model
type is negative even at 60d. This is a legitimate, unflattering finding, not
smoothed over: multi-day/multi-week High/Low forecasting on these 25 PSX stocks,
with this feature set, is not clearly better than a naive no-change forecast
once the flattering MAPE-based Accuracy_% numbers are checked against a proper
baseline. Worth keeping in mind for Phase 8 (frontend/serving decisions) -
showing a confident-looking 89% "accuracy" for a 20-day forecast without this
context would be misleading.

## Result: GRU vs LSTM - GRU is consistently the weaker of the two here

Averaged across both targets, at every horizon, GRU has higher MAE and lower
Accuracy_% than LSTM:

| Horizon | LSTM avg MAE | GRU avg MAE | LSTM avg Accuracy_% | GRU avg Accuracy_% |
|---|---|---|---|---|
| 1d | 8.04 | 8.57 | 98.61 | 98.58 |
| 5d | 20.16 | 26.28 | 95.72 | 95.36 |
| 10d | 32.28 | 40.61 | 93.12 | 92.84 |
| 20d | 49.98 | 58.38 | 89.48 | 89.26 |
| 60d | 112.97 | 102.98 | 77.78 | 79.84 |

LSTM wins on both metrics at 1d/5d/10d/20d. At 60d the ranking flips both ways
at once: GRU has the lower MAE (102.98 vs LSTM's 112.97) *and* the higher
Accuracy_% (79.84 vs 77.78) - LSTM's 60d MAE is dragged up by a notably bad
High-target run (MAE 125.43, its worst result in the whole table), while GRU's
60d numbers are unremarkable rather than good.
Overall: GRU is the weaker model type at short-to-medium horizons and is not a
clear win at 60d either - "a lighter alternative to LSTM" per the doc, but not
a better one on this data, at these locked hyperparameters (hidden_size=32,
1 layer, batch_size=64, Adam lr=0.001, up to 50 epochs, patience 10 - identical
across all 5 horizons, by construction, since these are module-level constants
never varied inside the training loop).

## Result: range width widens with horizon for every ticker/target/model - no exceptions

250/250 (25 tickers x 2 targets x 5 model types) comparisons between horizon-1
and horizon-60 backtested `std_error` show the range widening (`std_60d >
std_1d`), a 100% widen rate with zero exceptions. This is exactly the behavior
the backtested-error-based range from Phase 6 is supposed to produce
automatically once it's keyed by horizon, and it held for every single
combination checked, not just "most" as the doc's validation checklist asks for.

## Deliverables produced

- `data/phase7_model_comparison_overall.csv`, `_per_stock.csv` - Horizon,
  Directional_Accuracy_%, and Improvement_over_Naive_% added to Phase 6's metric set
- `data/phase7_range_width_by_horizon.csv` - the 1d-vs-60d widening check above
- `models/phase7_error_distributions.pkl` - `{ticker: {target: {horizon: {model: stats}}}}`
- `models/phase7_ensemble_weights.pkl` - `{ticker: {target: {horizon: weights}}}`
- `models/phase7_{lr,rf,xgb,lstm,gru}_{high,low}_{1,5,10,20,60}d.*` - 50 trained models
