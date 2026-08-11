# Frontend notes: decisions and results

Companion to `Phases/frontend.md`. Documents the two gaps the doc's own
"Wiring it into the pipeline" section left open, plus one path adjustment
made while wiring them up.

## Gap 1: `COMPANY_NAMES` for all 25 tickers

`master_dataset.csv` has no name column to look up against (checked its
header directly), so `COMPANY_NAMES` in `app.py` is a static dict, extended
from the doc's original six entries to all 25 tickers in the dataset
(`BAHL`, `COLG`, `DGKC`, `EFERT`, `ENGROH`, `FFC`, `HBL`, `HUBC`, `ILP`,
`INDU`, `LUCK`, `MARI`, `MCB`, `MEBL`, `MLCF`, `NESTLE`, `NETSOL`, `OGDC`,
`PAKT`, `PPL`, `PSO`, `SYS`, `TRG`, `UBL`, `UNITY`). The doc's sample
`COMPANY_NAMES` used `ENGRO` and `OGDC`; the live dataset uses `ENGROH`
(Engro Holdings, the post-2023 rename of Engro Corporation) instead of
`ENGRO`, so `sample_predictions()`'s four rows were updated to match the
tickers `app.py` actually needs to resolve names for.

## Gap 2: `generate_predictions.py`, the missing assembly script

New file, living alongside `train_models_phase9.py` per the doc's own
suggestion. Loads `data/master_dataset.csv` and, for each of the 25
tickers, runs the latest feature row through the right Phase 9 model for
every (target, horizon) cell, writing one row per stock to
`data/phase9_predictions.csv` in the exact schema `frontend.md` specifies.

"Right model" is decided by the same rule `manage_model_storage.py`'s
`compute_needed_combos()` and the old Phase 8 `app.py`'s
`compute_serving_lookup()` already used: whichever model type has the best
real `Improvement_over_Naive_%` for that (ticker, target, horizon) in
`data/phase9_model_comparison_per_stock.csv`, falling back to the naive
baseline (today's Close, per `train_models_phase9.naive_prediction` - both
Open and Close share this baseline) when nothing beat it. Reuses
`FEATURE_COLUMNS`, `GRUModel`, `HORIZONS`, and `TARGETS` from
`train_models_phase9.py` and `WINDOW_SIZE` from `model_utils.py` rather than
redefining them, so the inference-time feature list can't drift from what
each model was actually trained on.

Checks `models_evaluation_only/` as well as `models/` when resolving a model
file, since `manage_model_storage.py` already moved the 6 combos that no
stock ever selects out of the live directory - `generate_predictions.py`
still needs those 6 for whichever ticker did *not* pick them, i.e. it never
needs a combo that isn't the serving choice for at least one ticker, but it
can't assume every serving choice still lives in `models/`.

Requires the `.venv` interpreter (has the CUDA build of `torch`, plus
`tensorflow`/`xgboost`/`scikit-learn`, per Decision 1 in
`Phases/phase_9_notes.md`), not the global one:

```
.venv/Scripts/python.exe generate_predictions.py
```

Ran successfully for all 25 tickers on 2026-08-11, producing
`data/phase9_predictions.csv`.

## Path adjustment: `phase9_predictions.csv` lives under `data/`, not the repo root

`frontend.md`'s `load_predictions()` defaults to the bare filename
`"phase9_predictions.csv"`, which would resolve relative to whatever
directory `streamlit run` is launched from. Every other data artifact in
this project (`master_dataset.csv`, all `phase*_*.csv` comparison files)
lives in `data/`, so `app.py`'s `load_predictions()` defaults to
`DATA_DIR / "phase9_predictions.csv"` (`DATA_DIR = BASE_DIR / "data"`,
`BASE_DIR` resolved from `__file__`) instead, matching that convention and
working regardless of the process's current working directory.

## Follow-up: predicted date, data staleness, and the all-zero Close % complaint

Raised after the first pass: cards had no date on them at all (just "Kal"/
yesterday, in `frontend.md`'s original wording), and predicted Close showed
`+0.0%` on many cards, which read as a bug.

Both trace back to the same root cause. `data/master_dataset.csv`'s latest
row is **2026-08-06** for every ticker (checked directly - `df["Date"].max()`
across the whole file, and per-ticker, is the same date), five calendar days
before this was investigated (2026-08-11), so "Kal" (yesterday) was already
wrong - the dashboard's idea of "yesterday" is actually the dataset's last
trading day, not literally the day before whenever someone opens the page.

The `+0.0%` on Close is real model output, not a display bug: per
`Phases/phase_9_notes.md`'s own headline finding, only 2 of 50
(Model, Target, Horizon) combinations ever beat the naive baseline, so most
(ticker, Close, horizon) cells fall back to "today's Close held constant" -
an exact 0% change by construction. Checked across all 25 tickers:
1d Close alone is naive-baseline (all-zero) for 14/25 tickers.

Fixed by making both facts visible on the page instead of silently
mislabeling or looking broken:

- `generate_predictions.py` now writes a `Data_Date` column (the actual
  latest date used, per ticker - currently identical across all 25 since
  `master_dataset.csv` is updated in one batch) and a `Model_{target}_{h}d`
  column per cell recording which model served it, or `"naive_baseline"`.
- `app.py`'s top caption reads "Data as of `<Data_Date>` (latest trading day
  in the dataset)", separate from "dashboard generated `<now>`" - these are
  two different timestamps and conflating them was the original bug.
- Each card's "Kal" line is now the actual `Data_Date`, and a new
  "Predicted for `<date>`" line converts the selected horizon into a real
  calendar date via `advance_trading_days()` (skips weekends and the same
  `PSX_HOLIDAYS_2026` list the old cone-based `app.py` used).
- Each metric shows a small "flat: no model beat naive here" caption
  whenever its `Model_*` column reads `naive_baseline`, so a 0% cell reads
  as an honest model result instead of a broken one.

`generate_predictions.py` needs rerunning (`.venv/Scripts/python.exe
generate_predictions.py`) whenever `master_dataset.csv` gets a new trading
day appended, since nothing here refreshes it automatically - see the
"stale data" caveat this note exists to surface, not fix.

### Reverted: the "flat: no model beat naive here" caption

Added above, then explicitly rejected by the user on the same day: the % on
each metric should just be the predicted price compared against yesterday's
actual price, full stop - no separate naive-baseline framing layered on top.
Removed `open_is_naive`/`close_is_naive` and their captions from
`render_card()`; `pct_change()` (unchanged throughout) was already exactly
"predicted vs yesterday", so no calculation changed, only the extra
annotation was pulled back out. `generate_predictions.py`'s `Model_*`
columns are left in `phase9_predictions.csv` - harmless, and available again
if a future request wants the naive/model distinction surfaced some other
way - but `app.py` no longer reads them.

## Follow-up: best-performing model shown under each card

Requested right after the revert above: show which model is actually
serving each stock's prediction. Added a caption at the bottom of every
card - `Best model (<horizon>) - Open: <model> | Close: <model>` - reading
straight from the `Model_Open_{h}d`/`Model_Close_{h}d` columns
`generate_predictions.py` already writes (the same ones the naive-flag
caption used before it was reverted, now put to a use the user actually
asked for). Updates live when the horizon selector changes, since it reads
the same per-row columns the price metrics do. `naive_baseline` is relabeled
"Naive baseline" for readability; a missing column (the `sample_predictions()`
fallback path) shows "N/A" via `format_model_name()`'s `pd.isna` check
rather than crashing.

## Follow-up: serving now always picks the best-of-5 model, not naive on a tie/loss

Explicit design change from the user: `Improvement_over_Naive_%` was meant
as a diagnostic (is a model overfitting, does it actually respond to
input?), not a gate on what gets shown. `compute_serving_lookup()` in
`generate_predictions.py` no longer filters to `Improvement_over_Naive_% > 0`
before picking a winner - it now always takes the model with the single
best `Improvement_over_Naive_%` (equivalently, lowest MAE against the same
naive baseline) among the 5 trained types for that (ticker, target,
horizon), full stop. `naive_baseline` can no longer come out of this
function; the `model_type == "naive_baseline"` branch in `main()` is now
dead code, kept only as a defensive no-op rather than removed. Regenerated
`data/phase9_predictions.csv` - confirmed 0 `naive_baseline` values across
all 250 `Model_*` cells post-change (was the majority of Close cells
before, per the "Result: the naive baseline is very hard to beat" finding
in `Phases/phase_9_notes.md` - that finding is unchanged, it's just no
longer what the dashboard serves). `manage_model_storage.py`'s eval-only
split still matters here even though its own selection rule differs
slightly (positive-improvement-only) - `model_path()` already checked
`models_evaluation_only/` as a fallback before this change, which turned
out to matter more now, since a model can be this dashboard's best-of-5 pick
without ever having had a positive improvement anywhere.

## Follow-up: serving switched from single-best-model to the ensemble blend

After discussing whether the single-model results (mostly losing to naive,
per `Phases/phase_9_notes.md`) could be improved: `train_models_phase9.py`
already computed per-ticker/target/horizon inverse-MAE ensemble weights
(`model_utils.compute_ensemble_weights`) into
`models/phase9_ensemble_weights.pkl`, but nothing downstream - not the old
Phase 8 `app.py`, not this dashboard's serving lookup - ever read that file.
Picking a single "best" model by one train/test split's
`Improvement_over_Naive_%` throws away the other 4 models' information and
is sensitive to that split's own sampling noise; blending was free (already
computed) and a standard variance-reduction move, so it's what's wired in
now instead of picking a winner.

`generate_predictions.py` rewritten: `compute_serving_lookup()` and its
`phase9_model_comparison_per_stock.csv` dependency are gone. For every
(ticker, target, horizon) it now runs **all 5** model types
(`predict_ensemble()`), blends them with `model_utils.ensemble_predict()`
using that ticker's own weights from `phase9_ensemble_weights.pkl`, and
writes the blended price. Two columns per cell instead of one:
`Model_{target}_{h}d` is now always the literal string `"Ensemble"`, and a
new `Top_Model_{target}_{h}d` records whichever single model type carries
the most weight in that cell's blend, kept for display purposes (nothing
downstream branches on it). This needs every one of the 50 model files
reachable, not just whichever one ever won a stock's serving slot, since
every model type feeds every blend now - `model_path()`'s existing check of
`models_evaluation_only/` as a fallback after `models/` already covered
this without changes.

`app.py`'s per-card caption changed from "Serving (`<h>`) - Open: `<model>`
| Close: `<model>`" to "Ensemble (`<h>`) - Open top: `<model>` | Close top:
`<model>`", reading the new `Top_Model_*` column instead of `Model_*`,
since `Model_*` no longer varies (always "Ensemble") and showing that alone
would be uninformative.

Not yet measured: whether the blend actually beats naive baseline more
often than the single best model did per horizon/ticker - that would need
rerunning the Improvement_over_Naive_% comparison against blended
predictions on the held-out test set, which `generate_predictions.py`
doesn't do (it only scores the live/latest row, there's no ground truth for
it yet). Worth doing as a follow-up if the ensemble's real-world value is
in question.

## Follow-up: does the ensemble blend actually help? Measured on test.csv, then validate.csv

Two new one-off scripts, not part of the serving pipeline:
`evaluate_ensemble.py` (scores the ensemble blend against
`data/test.csv`, the same set `train_models_phase9.py` used) and
`evaluate_ensemble_validate.py` (same logic, against `data/validate.csv`,
which nothing in this project has ever touched for training, weighting, or
model selection - see `Phases/phase_9_notes.md`). Both reuse
`evaluate_horizon_target()` and mirror `train_models_phase9.py`'s own
methodology (tabular models score every row, LSTM/GRU only rows with a full
30-day lookback within the set being scored) so results are directly
comparable to the existing comparison CSVs. Outputs:
`data/phase9_ensemble_vs_single_overall{,_validate}.csv` (25-ticker
aggregate, per horizon/target) and `data/phase9_ensemble_comparison{,_validate}.csv`
(per-ticker, following the `Ticker/Target/N_Rows/Best_Single_Model/
Best_Single_MAE/Ensemble_MAE/Ensemble_Beats_Single` schema `data/phase6_ensemble_comparison.csv`
already established in an earlier phase, extended with `Horizon` and both
sides' `Improvement_over_Naive_%`).

**Why two runs, not one:** the ensemble weights and this script's own
"best single model per ticker" pick are both derived from `test.csv`'s own
errors, so scoring the ensemble against `test.csv` again is partly
circular - and the per-ticker "best single model" number in particular is
a hindsight pick (argmin MAE on the exact set it's then scored against),
an advantage no real deployment has in advance. `validate.csv` is the
first genuinely out-of-sample check in this project.

**Aggregate result (25 tickers combined), test.csv vs validate.csv:**

| Target/Horizon | test.csv: Ensemble vs Naive | test.csv: Best Single vs Naive | validate.csv: Ensemble vs Naive | validate.csv: Best Single vs Naive |
|---|---|---|---|---|
| Close 60d | **+2.14%** | +0.01% | **+1.83%** | -2.99% |
| Open 60d | +1.40% | +1.57% | **+1.88%** | -2.11% |
| all other horizons/targets | negative | negative (mostly) | negative | negative (mostly) |

The single-model "win" at 60d that looked real on `test.csv` (Linear
Regression, roughly break-even to slightly positive) reverses to clearly
negative on `validate.csv` - direct evidence that picking a single winner
per ticker/horizon by lowest test-set error doesn't generalize, it's an
artifact of that one split. The ensemble's edge at 60d is the one result
that **holds up on data neither the weights nor any model selection ever
saw** - the most credible finding in this whole exercise.

**Per-ticker (250 cells), noisier, included for completeness:**
Ensemble beats the hindsight-picked single model 18.8% of the time on
`test.csv`, rising to 28.0% on `validate.csv` (the single-model pick's edge
shrinks once it's off the data it was chosen on, consistent with the
aggregate finding above). Ensemble beats naive in 51.2% / 46.0% of cells;
the hindsight single-model pick beats naive in 64.0% / 65.2% of cells - a
number that looks good but is inflated by the same selection-bias problem,
not a fair comparison to a strategy that has to commit to one model type in
advance.

**Conclusion, and why the ensemble stays wired into `generate_predictions.py`
as-is**: it doesn't reliably beat picking a winner in hindsight on the exact
data being measured (nothing would - that's what hindsight selection is
for), but it's the one approach whose 60d edge over naive actually survives
contact with unseen data. Given this project's own headline finding
(`Phases/phase_9_notes.md`: only 2 of 50 combos ever beat naive at all), a
strategy that holds up out-of-sample at the one horizon where a real edge
exists is worth more than one that wins mostly by fitting the test set.

## Follow-up: fixed a real train/inference window mismatch in generate_predictions.py

Found while explaining why some models rank differently by MAE vs
Directional_Accuracy_%, then confirmed as worth fixing regardless of the
news/sentiment decision: `model_utils.create_sequences` builds every
training sequence as `features[i - window_size:i]` - the 30 days strictly
**before** anchor day `i`, never including day `i` itself (day `i` is where
the target/price/close values come from). `generate_predictions.py`'s
`predict_price()` used `ticker_df.tail(WINDOW_SIZE)` for LSTM/GRU inference,
which takes the last 30 rows **including** today (the anchor day being
predicted from) as the sequence's final step - a genuine train/inference
mismatch, not a stylistic difference. It didn't affect any of the
backtested comparison numbers in this whole conversation (those were all
computed via `create_sequences_h`, the correct construction, both in
`train_models_phase9.py` and in `evaluate_ensemble.py`) - it only affected
the live predictions this script actually writes to
`data/phase9_predictions.csv` for the dashboard.

Fixed: `ticker_df.iloc[-(WINDOW_SIZE + 1):-1]` - today's row excluded, the
30 rows before it kept, matching training exactly. Confirmed the fix has a
real, non-trivial effect by diffing predictions before/after: mean absolute
change of 0.12-0.17 PKR at 1d, 1.26-1.47 PKR at 60d (up to 6.2 PKR for one
ticker) - small but real, concentrated in whichever share of the ensemble
blend LSTM/GRU carry for that cell (LR/RF/XGBoost predictions are
unaffected, they were never sequence-based). Regenerated
`data/phase9_predictions.csv` with the fix; `app.py` needed no changes.

## Follow-up: mid-conversation experiment - do macro features help? (Phase 10, unplanned)

Prompted by a question about whether news/sentiment (planned separately,
not part of this project yet) would be justified given the current
technical-only feature set's weak results. Before committing to news
scraping, tested a much cheaper hypothesis first: two market-wide macro
features, sourced for real (no fabricated data):

- **KSE-100 index**: Yahoo Finance (`^KSE`) for 2018-01-02 through
  2021-09-30 (Yahoo's feed goes mostly `None` after that), spliced with
  PSX's own `dps.psx.com.pk/timeseries/eod/KSE100` API from 2021-08-12
  onward - the two sources overlap for ~7 weeks with no gap, cross-checked
  against each other before splicing.
- **USD/PKR**: Yahoo Finance (`PKR=X`), complete 2018-01-01 to present,
  only 6 missing days (forward-filled).

Built by `build_macro_data.py` into `data/macro_data.csv` (`KSE100_Return_1d`,
`USDPKR_Return_1d`, one row per calendar date). `add_macro_features.py`
merges these onto `data/train.csv`/`test.csv`/`validate.csv` by Date -
same rows, same split, two new columns - producing `_macro` variants
without touching the originals.

**Phase 10, not a Phase 9 rewrite**: `train_models_phase10.py` is
`train_models_phase9.py` with `FEATURE_COLUMNS` extended by the two macro
columns and every output (`models/phase10_*`, `data/phase10_*`) under its
own prefix - nothing from Phase 9 was overwritten, so the two are directly
comparable rather than one replacing the other. Trained the full 50 combos
(5 horizons x 2 targets x 5 model types) plus its own ensemble weights, same
methodology as Phase 9 throughout.

`evaluate_phase10_vs_phase9.py` re-ran the same ensemble-blend evaluation
(`evaluate_ensemble.py`'s methodology, generalized to take a phase config)
against both `test.csv`/`test_macro.csv` and `validate.csv`/`validate_macro.csv`,
so Phase 9 and Phase 10 get scored identically. Full results in
`data/phase10_vs_phase9_comparison.csv`.

**Result on `validate.csv` (the trustworthy, never-touched-by-training set):**
macro features **hurt** at 1d and 5d for both targets (-0.5 to -1% worse
each), and **help** at 10d-60d - most notably **60d Open's edge nearly
doubled, +1.88% -> +3.72%** over naive, with 20d and 10d Open also
improving. This exactly matches the mechanism argued for beforehand: macro
variables move too slowly to explain single-day dynamics (adds noise
there), but have real time to matter over 2 weeks to 3 months, concentrated
at exactly the horizon (60d) that already had the strongest signal.
`test.csv`'s version of this comparison shows a noisier, less consistent
version of the same broad pattern (short horizons flat-to-worse, several
long-horizon cells improving) - expected, since `test.csv` comparisons in
this project are already known to be the less trustworthy of the two (see
the ensemble-vs-single-model follow-up above).

**Not yet done, worth flagging**: `generate_predictions.py` (the live
dashboard's assembly script) still reads Phase 9's models exclusively - it
was not repointed at Phase 10, since this was framed as an experiment to
answer a question, not yet a decision to change what's served. If the
macro features are kept, `generate_predictions.py` needs the same kind of
per-horizon awareness this finding implies (macro helps 10d+, hurts 1d/5d)
rather than a blanket switch to Phase 10's models everywhere.

## Follow-up: is the Phase 9 vs Phase 10 gap real, or noise? (statistical significance added)

The user asked whether there's a better way to compare Phase 9 vs Phase 10
than raw MAE - a good instinct, since the earlier verdict ("Phase 10 wins on
Close@20d/60d and Open@10d/20d/60d") was based on point estimates alone with
no sense of whether those gaps were distinguishable from noise.

Added to `evaluate_phase10_vs_phase9.py` (validate.csv only - test.csv
comparisons are already known to be partly circular in this project since
ensemble weights were fit on test.csv's own errors):

- **RMSE** alongside MAE for the ensemble blend (mirrors the existing
  pattern in `model_utils.evaluate_predictions`).
- **A ticker-level (cluster) bootstrap**, added as `model_utils.bootstrap_paired_comparison`
  (generic, reusable - takes two paired prediction series over shared
  tickers/actual/close rows, resamples the 25 tickers with replacement 2000x,
  keeping each ticker's whole row-block intact so within-ticker serial
  correlation - worst at 60d, where each day's target window overlaps ~59/60
  with its neighbor - is preserved exactly rather than modeled). Chosen over
  Diebold-Mariano because DM+HAC is built for a single time series, not a
  25-ticker panel, and the standard HAC bandwidth at horizon 60 (lag≈59) is
  itself unreliable with only ~330-390 usable rows per ticker.
- **Wilcoxon signed-rank test** on the 25 paired per-ticker improvement
  deltas, as a cheap complementary check (`scipy.stats.wilcoxon`,
  `statsmodels` isn't installed in this environment).
- **Holm-Bonferroni correction** across the 10 cells tested (2 targets x 5
  horizons) - hand-rolled since `statsmodels` isn't available - since 10
  independent tests at raw p<0.05 would give a ~40% false-positive rate
  before correction.
- **RMSE-winner and Directional-Accuracy-winner** columns alongside the
  existing MAE-winner, to see whether "who wins" depends on which metric
  you look at.

Verified `data/validate.csv`/`validate_macro.csv` are row-for-row identical
in Ticker/Date/Close/Open/Target_* (only the 2 macro columns differ) before
relying on this - confirms Phase 9's and Phase 10's per-row arrays for a
given cell are genuinely paired, which is what makes the tighter paired
bootstrap/Wilcoxon design valid here instead of a weaker unpaired test.
Output: `data/phase10_vs_phase9_significance_validate.csv` (10 rows, one per
Target x Horizon cell).

### Result: only 1 of 10 cells survives correction - and it favors Phase 9

**Close@1d is the only statistically significant cell after Holm correction**
(`Bootstrap_p_holm` = 0.01, Wilcoxon p = 0.0125) - and it favors **Phase 9
(no macro)**, matching its point estimate. Every other cell, including all
of the "Phase 10 wins" cells reported earlier (Open@10d/20d/60d,
Close@20d/60d), comes back **not significant** after correction
(`Bootstrap_p_holm` = 1.00 in every one of those 9 cells) - the point
estimates lean the direction already reported, but with only 25 tickers to
resample from, none of those gaps clear the bar to call them real effects
rather than noise. RMSE agrees with MAE on the winner in all 10 cells (no
new information there), but Directional_Accuracy_% disagrees with the
MAE/RMSE winner in 4 of 10 cells (Close@1d, Close@5d, Close@20d, Open@20d) -
"which phase is better" genuinely depends on whether magnitude accuracy or
directional correctness is what matters for a given cell.

**Corrected bottom line, superseding the earlier "Phase 10 wins at
10d/20d/60d" framing**: there is not enough statistical evidence in this
25-ticker sample to justify switching *any* cell to Phase 10 over Phase 9.
The one significant result says the opposite - avoid Phase 10 specifically
at Close@1d. The macro features' apparent long-horizon advantage is a real,
honestly-reported point estimate, but not yet a confirmed effect - worth
re-testing as more validate-period data accumulates rather than acting on
now. This is exactly the kind of finding this project's own established
practice says to report plainly rather than smooth over (see the naive-
baseline result in `Phases/phase_9_notes.md`).

## Follow-up: naive baseline isn't a model, stop labeling it like one

Caught immediately: `naive_baseline` labeled "Naive baseline" under a
"Best model" caption implied it had won a model comparison, but it's not a
trained model at all - it's what gets served when nothing beat "hold
yesterday's Close". `format_model_name()` now returns "none (naive held)"
for that case instead, and the caption prefix changed from "Best model" to
"Serving", so the line reads "Serving (1d) - Open: Random Forest | Close:
none (naive held)" - accurate for both the real-model and no-model-won
cases without implying naive baseline competed as a model type.

## Verification

`app.py` has no runtime dependency on `torch`/`tensorflow`/`xgboost` - it
only reads the assembled CSV, matching `frontend.md`'s intent that the
dashboard not load model files directly. Verified with
`streamlit.testing.v1.AppTest` (headless script-run, no browser available
in this environment): 25 cards render, 0 exceptions, 53 `st.metric` widgets
(3 summary + 2 per card x 25), 26 `st.markdown` calls (1 title + 25 ticker
labels). Spot-checked BAHL's card: Kal Open/Close Rs 175.00/177.33,
predicted 1d Open Rs 177.59 (+1.5%), predicted 1d Close Rs 177.33 (+0.0%,
BAHL's 1d Close cell fell back to naive baseline) - both figures trace back
to `data/phase9_predictions.csv` correctly.
