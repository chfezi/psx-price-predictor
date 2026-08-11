# Phase 8: dashboard redesign

## Context

Phases 6 and 7 changed what the model predicts and how well it predicts it. This phase changes how the results are shown. The current app.py uses a single overview table, a stock selector dropdown, and a flat range bar per stock. That layout is replaced, not extended, with the card and cone design already agreed on: an overview grid of stock cards, each with a small cone shape showing how the predicted range widens with horizon, and a detail view per stock with a full interactive cone across the five horizons from Phase 7.

The Plotly price chart and the model comparison table from Phase 5 are useful and stay, but move into a secondary, collapsible section under the new detail view rather than sitting on the main page.

Before writing any code, look at the current app.py, the error distribution and ensemble weight lookups Phase 6 and Phase 7 produced, and the Directional Accuracy and Improvement-over-Naive numbers Phase 7 computed, since this phase surfaces those numbers in the UI rather than computing anything new.

## An honest finding from Phase 7 that shapes this phase

Beyond 1 day, most model types barely beat, or lose to, a naive "no change" baseline. Only Linear Regression stays slightly ahead of naive at 60 days, every other model type is negative past the short horizons. This is not something to hide behind a nicer UI. The dashboard needs to say so plainly wherever it applies, the same honesty principle already used for the naive-baseline row in the Phase 5 comparison table.

## Scope of this phase

1. Pick which model serves each stock, target, and horizon combination, using Improvement-over-Naive rather than raw accuracy
2. Build the overview grid of stock cards with a mini cone
3. Build the detail view with the full interactive cone and horizon tabs
4. Surface Directional Accuracy, Improvement-over-Naive, and an honest low-confidence label where it applies
5. Keep the SHAP top-3 drivers from Phase 6, and the Plotly chart and comparison table from Phase 5, in a secondary section
6. Manage model storage, since models/ is now about 1.1GB
7. Write a before-and-after summary document covering the whole journey from phase 5 to phase 8

## Step 1: model selection per card

For each stock, target, and horizon, pick the model to actually serve based on Improvement-over-Naive on the validation set, not just lowest error. If every model type is negative for a given stock, target, and horizon (loses to naive), serve the naive baseline itself for that specific cell and mark it accordingly, rather than serving a model that is quietly worse than doing nothing.

```python
def pick_serving_model(model_results: dict) -> dict:
    """
    model_results: {model_type: {'imp_pct': float, 'mae': float, ...}}
    Returns the model to serve, or a naive-baseline marker if nothing clears zero.
    """
    positive_models = {m: r for m, r in model_results.items() if r['imp_pct'] > 0}
    if not positive_models:
        return {'model': 'naive_baseline', 'imp_pct': 0.0}

    best = max(positive_models.items(), key=lambda kv: kv[1]['imp_pct'])
    return {'model': best[0], 'imp_pct': best[1]['imp_pct']}
```

This also directly addresses the storage problem in step 6, since a model that never gets picked for any stock, target, or horizon does not need to ship in the live app.

## Step 2: overview grid

One card per stock, 25 total, laid out in a responsive grid. Each card shows:

- Ticker and current price
- A direction badge, bullish, bearish, or neutral, based on whether the 1-day predicted midpoint sits above, below, or near the current price
- A small cone shape, narrow near the origin and widening toward the right edge, built from each horizon's backtested range width for that stock, not just a decorative shape
- The 1-day confidence percentage
- If the 1-day model for that stock did not clear naive, a small neutral-toned label instead of a confidence percentage, something like "below naive baseline", not hidden and not styled as an error, just stated

Reuse the SVG cone structure already prototyped, a polygon between two polylines fanning out from a single origin point, colored by direction (green family for bullish, red family for bearish, amber or gray for neutral).

## Step 3: detail view

Opens when a card is clicked. Contains:

- Ticker, current price, direction badge
- Horizon tabs, 1d, 5d, 10d, 20d, 60d
- The full interactive cone, a vertical marker line moves to the selected horizon's position on click, same mechanic as the prototype
- A range readout (low to high) and a confidence percentage for the selected horizon
- Directional Accuracy and Improvement-over-Naive for the selected horizon, shown as two small numbers next to the confidence percentage, not buried in a separate table
- A visible, plainly worded caution label when the selected horizon's serving model is the naive baseline itself or has single-digit Improvement-over-Naive, so a user selecting 60 days sees honestly that the forecast is close to a coin flip, not just a smaller confidence number
- Which model produced the number (XGBoost, Random Forest, LSTM, GRU, Linear Regression, or naive baseline)
- The SHAP top-3 drivers from Phase 6, only shown when the serving model supports SHAP (tree models); when the serving model is naive baseline, skip this section entirely rather than showing empty or fabricated drivers

## Step 4: secondary section

Below the detail view's main content, in a collapsed or lower-priority section: the Phase 5 Plotly price chart with the predicted range marked, and the full model comparison table (all model types, all metrics, for the selected stock and target). This is where someone checking the underlying work looks, it does not need the same visual polish as the cards above it.

## Step 5: manage model storage

Using the serving-model decision from step 1, only the models actually selected for at least one stock, target, and horizon combination need to ship with the live app. Move the rest to a separate folder, something like models_evaluation_only/, or keep them out of what gets deployed, and note in the app which models are evaluation-only in case someone asks why a stronger-on-paper model is not being served for a particular stock.

## Step 6: write a before-and-after summary

There is an earlier project summary document that only covers phases 1 through 5. Write a new file, before_and_after_summary.md, that covers what changed from that earlier state through phase 6, 7, and 8, using the real numbers and decisions from Phases/phase_6_notes.md, Phases/phase_7_notes.md, and this phase's own work, not approximate or placeholder figures.

Write it for a reader who does not already know the codebase. No code, no library names, no unexplained technical terms, plain sentences throughout. Where a technical idea has to be mentioned (a percentage-change target instead of a raw price, a model beating or losing to a naive guess, a range that widens the further out it looks), explain it in one plain sentence at the point it comes up rather than assuming the reader already knows it.

Cover, in order: what the project does in one paragraph, what it looked like before (phases 1 to 5), what changed in phase 6 and why, what changed in phase 7 and why, what changed in phase 8 and why, and a short before-and-after table at the end summarizing the main points. Include the honest findings as they actually came out, the leakage numbers, the range coverage improvement, the ensemble test that did not win, the naive-baseline comparison across horizons, and GRU's mixed result, stated plainly rather than smoothed over.

## Deliverables for this phase

- app.py rebuilt around the card-and-cone layout described above, the old single-table layout is retired
- Model serving decisions computed per stock, target, and horizon using Improvement-over-Naive, with naive-baseline fallback where nothing clears zero
- Directional Accuracy and Improvement-over-Naive visible in the detail view for the selected horizon
- An honest low-confidence label wired in wherever the serving model is naive baseline or barely ahead of it
- Phase 5's chart and comparison table preserved in a secondary section
- A short note on which models actually ship versus which are evaluation-only, and the resulting size of the deployed models/ folder
- before_and_after_summary.md, written in plain language, covering phases 5 through 8 with real numbers

## Validation checklist before calling this phase done

- Confirm every one of the 25 stocks renders a card with no missing data, including the thin-traded ones
- Confirm the low-confidence label actually appears for stock, target, horizon combinations where Phase 7's numbers say it should, spot check a handful against the Phase 7 notes
- Confirm the cone's visual width actually reflects each stock's own backtested range, not a fixed decorative shape reused across every card
- Confirm the app still runs and loads within a reasonable time now that model storage has been trimmed to only the serving models