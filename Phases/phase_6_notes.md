# Phase 6 notes: decisions and results

Companion to `Phases/phase_6.md`. Documents the choices the doc explicitly asked
to be stated and why, plus the actual numbers produced by `train_models_phase6.py`
on 2026-08-10. Implementation: `feature_engineering.py` (return targets +
stationary features), `train_models_phase6.py` (training/comparison/range/
ensemble), `app.py` (live wiring).

## Addendum (2026-08-10): LSTM Close-scaling bug found and fixed

While building Phase 7's equivalent sequence-prep code, it became clear that
`train_models_phase6.py`'s LSTM prep had a bug: `"Close"` is one of
`NEW_FEATURE_COLUMNS` (meant to be standardized for model input), but the code
also restored raw `Close` into that same column afterward so `create_sequences`
could use it for price conversion - silently overwriting the scaled `Close`
*feature* fed to the LSTM with an unscaled one (ranging ~11 to ~9000 across the
25 stocks, mixed in among ~42 properly standardized features). Fixed by carrying
the raw close on a separate `Close_raw` column instead, and reran
`train_models_phase6.py` in full so every downstream artifact (LSTM models,
error distributions, ensemble weights, all comparison CSVs) stays consistent.

Effect: LSTM MAE improved slightly (High: 8.72 -> 8.52, Low: 7.47 -> 7.40) -
consistent with the fix, though modest, meaning the bug's practical impact here
was mild. Every other model type (LR/RF/XGBoost) was unaffected (they don't use
this scaled-sequence code path). None of the headline findings below changed:
old-vs-new avg accuracy is still 96.13 -> 98.62, the ensemble still wins only
16%/16% (High/Low) of the time, and the range comparison is essentially
unchanged (mean width 64.65 vs the original 65.42, mean coverage 0.8313 vs
0.8285). `app.py` needed no changes - it loads `models/phase6_*` files by name,
so the corrected models are picked up automatically.

## Decision 1: return-target clip range is a fixed +/-15%, not percentile-based

Computed `Target_High_Return`/`Target_Low_Return` across all 25 stocks (51,454
rows) before deciding anything. The 0.1/99.9 percentile tails are dominated by
a small number of clearly bad rows, not real tail risk:

- COLG has rows with `Low=0.0` and `Open=0.0` (e.g. 2018-06-12), producing a
  `Target_Low_Return` of exactly `-1.0`. Several other COLG dates also show this.
- MARI (2024-09-13), SYS (2025-05-27), LUCK (2025-04-18), UBL (2025-06-20) each
  have a single-day recorded move of -49% to -88%, which isn't plausible under
  PSX's daily circuit-breaker limits and looks like an unadjusted corporate
  action (split/bonus/rights issue) or a data entry error, not a real trading
  day.
- Only 13/51,454 rows exceed +/-15% for `Target_High_Return`; 34/51,454 for
  `Target_Low_Return`.

Because so few rows dominate the tail, a percentile-based clip (e.g. 0.1/99.9)
would be set by the bad data itself rather than by genuine return variance. Used
the doc's own stated fallback instead: a fixed +/-15% clip. This is a data-quality
finding, not just a modeling choice — worth a follow-up pass on the raw PSX
scraper/cleaning step to catch `Low=0`/`Open=0` rows and validate against PSX's
actual daily circuit limits, independent of this phase.

## Decision 2: "validation set" in the doc maps to this project's `test.csv`

This project already has a train/test/validate split (`merge_and_split.py`):
`train` (pre-2024), `test` (2024), `validate` (2025). Phase 4's `train_models.py`
uses `test_df` to compare models and pick best-per-stock, and touches
`validate_df` exactly once for a final honest check (see the comment in
`train_models.py`'s Step 11). That's functionally the same role the doc means by
"validation set" in Step 3-4. So: backtested error distributions, ensemble
weights, and the range/ensemble comparisons are all computed on `test_df`.
`validate_df` was not touched in this phase at all.

## Decision 3: everything downstream operates in price space

Models are trained on `Target_High_Return`/`Target_Low_Return`, but every
prediction is converted back immediately with `predicted_price = Close * (1 +
predicted_return)`, using the row's actual `Close`. Ground truth for evaluation
is the raw (unclipped) `Target_High`/`Target_Low`, not a value reconstructed
from the clipped return — so the reported metrics reflect real-world price
error, including on the handful of rows whose training target got clipped.

## Decision 4: SHAP shown from the higher-weighted tree model

The doc restricts SHAP to XGBoost and Random Forest (LSTM/Linear Regression
excluded). `app.py` shows the top-3 SHAP drivers from whichever of the two has
the higher ensemble weight for that stock+target, labeled explicitly (e.g.
"Top drivers (XGBoost)") so it's clear which model they come from.

## Result: old (price target, raw features) vs new (return target, stationary features)

Average `Accuracy_%` across all 25 stocks, both targets, LR/RF/XGBoost/LSTM
(from `data/phase6_feature_target_comparison.csv`):

| Pipeline | Avg Accuracy_% |
|---|---|
| Old (price target, raw features) | 96.13 |
| New (return target, stationary features) | 98.62 |

The leakage check (`feature_engineering.py`) confirms why: against
`Target_High` (old), `Close_lag_1/2/3` and `EMA_12` sit at 0.98-0.99
correlation across every one of the 25 stocks. Against `Target_High_Return`
(new), using the full stationary feature set, the highest correlation of any
feature is ~0.29 (`Range_Percentage`/`High_Low_Range`) — no leakage flagged for
any stock. The new pipeline isn't just more interpretable, it's measurably more
accurate too.

## Result: old ATR-based range vs new error-distribution range

From `data/phase6_range_comparison.csv`, averaged across all 25 stocks:

| Approach | Avg width (PKR) | Coverage (realized High & Low both inside band) |
|---|---|---|
| Old (ATR x3 clamp) | 15.84 | 24.0% |
| New (error-distribution, z=1.28) | 65.42 | 82.9% |

The old ATR-based range was badly miscalibrated: it looked tight, but it only
actually contained the realized next-day High/Low 24% of the time — false
confidence from a range that was too narrow, not evidence the model was
unusually accurate. The new range is wider but its ~83% coverage is close to
its ~80% design target (z=1.28 on each side), which is what "the range comes
from how wrong the model has actually been" is supposed to produce. This is
the clearest validation in this phase that the range redesign in Step 3 was
necessary, independent of anything else in this phase.

## Result: ensemble vs single-best model — ensemble does NOT win, reported honestly

From `data/phase6_ensemble_comparison.csv`, inverse-MAE weighted ensemble
(XGBoost/Random Forest/LSTM/Linear Regression) vs the single best-performing
model per stock+target, both scored on the same rows:

| Target | Ensemble beats single-best |
|---|---|
| High | 4/25 stocks (16%) |
| Low | 4/25 stocks (16%) |

This is a legitimate negative finding, not something to force. Looking at
`phase6_ensemble_comparison.csv`, the four models' MAE per stock are usually
within a few percent of each other (e.g. BAHL High: best single 1.21 vs
ensemble 1.23; FFC High: 3.21 vs 3.22) — the four model types are making
highly correlated errors on the same rows, so averaging them dilutes the best
model's edge instead of diversifying away independent error, and simple
inverse-MAE weighting on the winner's own (partly luck-driven, on a one-year
test window) MAE isn't enough to overcome that. `app.py` keeps the
single-best-model path available as an explicit toggle for exactly this
reason — the ensemble is not a strict improvement and users should be able to
compare live.

For reference, single-best-model win distribution across the 25 stocks (both
targets combined, `phase6_model_comparison_per_stock.csv`): LSTM 16, Linear
Regression 14, Random Forest 13, XGBoost 7 — no single model type dominates,
which is itself a decent argument for keeping all four in the picture even
though naive averaging isn't the way to combine them.
