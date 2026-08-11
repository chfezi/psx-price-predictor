# Phase 8 notes: decisions and results

Companion to `Phases/phase_8.md`. Documents the choices the doc explicitly asked
to be stated and why, plus what was found building and testing the actual
dashboard on 2026-08-10. Implementation: `manage_model_storage.py` (storage
cleanup), `app.py` (full rebuild).

## Two things the doc references that don't exist in this repo

Confirmed directly with the user before starting: "the card and cone design
already agreed on" / "the SVG cone structure already prototyped" and "an
earlier project summary document that only covers phases 1 through 5" don't
exist anywhere in this repo, its git history, or any other phase doc. The cone
was designed fresh from `phase_8.md`'s written description (Decision 5/6/7
below); `before_and_after_summary.md`'s phases-1-to-5 section is synthesized
directly from `Phases/phase_1.md` through `phase_5.md`, the only source
material that actually exists.

## Decision 1: serving-model selection uses Phase 7's data exclusively

Ran the doc's `pick_serving_model` (best `Improvement_over_Naive_%` among the
5 model types, naive-baseline fallback if none clear zero) against
`data/phase7_model_comparison_per_stock.csv`. Phase 7 already retrained every
horizon *including 1-day* under one consistent pipeline (5 model types, same
metrics), so it's the single coherent source for all 5 horizon tabs. Phase 6's
narrower 4-model, 1-day-only artifacts are retired from live serving. This
also moves the app from Phase 6's ensemble-blend approach to single-best-by-
improvement per cell - consistent with Phase 6/7's own finding that the
ensemble rarely won (16%). `models/phase7_ensemble_weights.pkl` is therefore
no longer read by `app.py` at all (kept on disk, unused - Step 1 picks one
model, it doesn't blend).

## Decision 2: naive-baseline fallback rate by horizon (computed for real)

| Horizon | Fallback rate (of 50 ticker/target cells) |
|---|---|
| 1d | 0% |
| 5d | 4% |
| 10d | 28% |
| 20d | 26% |
| 60d | 18% |

Substantial at 10d/20d/60d - exactly the honest finding `phase_8.md` says must
be surfaced plainly. Verified live in the browser: NESTLE's card correctly
shows "1d: below naive baseline" instead of a fabricated confidence number
(NESTLE happens to be naive at 1d specifically, not just at the longer
horizons where fallback is more common); DGKC's cone visibly pinches to a
point at 10d specifically because both its High and Low targets fall back to
naive baseline at that horizon while using real models at 5d/20d/60d - the
cone's "bowtie" shape at 10d for that stock is the naive-pinch design (Decision
7) working correctly, not a bug, confirmed by cross-checking the raw CSV.

## Decision 3: storage cleanup - the real win is retiring Phase 4/6, not trimming Phase 7

Checked which of the 50 Phase 7 `(horizon, target, model_type)` combos are
*never* the serving choice for any of the 25 stocks: only 4 - `(10d, Low,
GRU)`, `(20d, High, RF)`, `(20d, Low, RF)`, `(60d, Low, XGBoost)` - because
with 25 stocks independently picking their own best model, nearly every model
type wins somewhere. `manage_model_storage.py` moved those 4 plus all Phase 4
originals and all Phase 6 models (except `phase6_feature_scaler.pkl`, still
needed live - Phase 7's LR/LSTM/GRU models were trained against it directly)
into `models_evaluation_only/`. Actual result: `models/` 1135MB -> 560MB (46
files remain, matching the computed needed set exactly); `models_evaluation_only/`
holds 575MB.

## Decision 4: direction badge neutral band is +/-0.5%

Grounded in the real 1-day return distribution: median `|Return_1d|` is 1.02%,
25th percentile is 0.42%. +/-0.5% sits just above that 25th percentile.
Verified live: MLCF showed a green "Bullish" badge with a matching green cone,
confirming both the classifier and the direction-based coloring work together.

## Decision 5: cone rendering - hand-rolled inline SVG, shared mini/full

25 individual Plotly charts in a grid would be real per-card JS overhead and
would look like 25 charts, not 25 small decorations. One Python function
(`build_cone_svg`) builds a polygon (two polylines fanning from the
current-price origin, filled between them), shared by both the card and detail
contexts.

## Decision 6: horizon "click" is via `st.tabs`, not SVG hit-testing

With no prototype to match and Streamlit's sandboxed custom components making
raw SVG click handling fragile, horizon selection uses `st.tabs`, and the cone
SVG is regenerated per tab with the marker line at that tab's position.
Verified live: switching tabs moves the marker instantly with no server round
trip - Streamlit renders all 5 tabs' content in one script run and only
toggles visibility client-side, so this is actually *faster* than a real
click-driven redraw would be, not a compromise.

## Decision 7: naive-baseline cells get no fabricated confidence or width

Per the doc's own Step 2 wording ("a small neutral-toned label instead of a
confidence percentage"). A naive horizon is drawn as a pinch point (the
polygon's upper and lower boundary both collapse to the current-price line at
that horizon) with a small dashed circle marker, rather than a shaded band,
since Phase 7 never computed a backtested error distribution for the naive
baseline itself (only used it as the improvement denominator).

## Decision 8: the overview grid's mini cone doesn't run live inference at 5/10/20/60d

Not part of the original plan - added after reconsidering performance. The
doc's requirement is that the mini cone's *width* reflect "each horizon's
backtested range width for that stock" - that's `std_error`, already
precomputed in `phase7_error_distributions.pkl`, needing no model loading at
all. Only the 1-day point prediction (for the direction badge and 1-day
confidence number) needs live inference. So `compute_card_data` anchors the
mini cone's center at the current price and fans the width from backtested
error alone; only the detail view (opened on click, one stock at a time) does
live per-horizon inference across all 5 horizons. This keeps the overview grid
from having to load models for horizons the user hasn't asked to see yet.

## A real bug found while testing in the browser: SVG `<text>` gets stripped

`build_cone_svg` originally drew horizon labels ("1d", "5d", ...) as SVG
`<text>` elements under the cone. In the browser this rendered as the cone
shape correctly (lines/polygon/circles all survived) but the labels appeared
as a single cramped, unstyled, unpositioned string ("1d5d10d20d60d") below the
image rather than as spaced-out labels - caught by zooming into the actual
screenshot, not just eyeballing it at normal size. Streamlit's
`unsafe_allow_html` sanitizer strips `<text>` tags but leaves their inner text
content behind as bare, unpositioned text nodes. Fixed by removing `<text>`
from the SVG entirely and rendering the horizon labels as a native
`st.columns` row directly under the cone image instead - confirmed working
(evenly spaced "1d 5d 10d 20d 60d" aligned under the chart) after the fix.

## Validation checklist results

- All 25 stocks render with no missing data, including thin-traded ones
  (spot-checked NESTLE and PAKT directly in the browser).
- Naive-fallback label verified against real data (NESTLE at 1d, DGKC at 10d).
- Cone widths visibly differ between stocks and are not a fixed shape - e.g.
  MLCF's cone is visibly narrower/differently shaped than DGKC's bowtie.
- Card click -> detail view -> horizon tab -> marker movement all verified
  working in the browser, including the SHAP section (only appearing for
  XGBoost/Random Forest cells, correctly absent for COLG's 10d tab where both
  targets serve from Linear Regression) and the secondary chart/table section.
- No console errors at any point during testing.
