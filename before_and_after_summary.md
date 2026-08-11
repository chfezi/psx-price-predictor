# Fung Forge PSX Price Predictor: what this project does, and how it changed

## What the project does

This project looks at the daily trading history of 25 stocks on the Pakistan Stock
Exchange - going back to 2018 - and uses that history to guess how far a stock's
price might move on an upcoming trading day, or over the next week, two weeks,
month, or three months. Instead of trying to guess one exact future price (which
is close to impossible for any stock), it predicts a range: a low and a high that
the stock's price is likely to fall between. A dashboard shows this range for
every stock at a glance, and lets you dig into any one stock for more detail.

## What it looked like before this round of changes (through Phase 5)

The first version of the project worked like this: for each stock, four different
prediction methods (four "models," each one a different way of learning patterns
from the historical data) were trained to guess tomorrow's highest and lowest
price. The four methods were then compared against each other on data they hadn't
seen during training, and whichever one did best for a given stock was the one
actually used to make live predictions. The results were shown on a single
dashboard page: one table listing every stock's predicted range for the next
trading day, a dropdown to pick a stock and see more detail, and a price chart.

This version worked, but it had a real weakness that was already visible by the
end of that phase: it predicted a raw price level (for example, "this stock will
trade between 523,209 and 761,667 rupees tomorrow" for a stock that was actually
only worth around 500 rupees). One of the four prediction methods occasionally
produced wildly unrealistic numbers like this, because it was trying to learn
directly from raw prices - and raw prices vary hugely from one stock to the next
(one stock might trade around 11 rupees, another around 7,600), which makes it
hard for a single shared method to learn good general rules. A patch was added
to clamp the predicted range to something more realistic using each stock's
recent typical daily swing, but that was treating the symptom, not the cause.

## What changed in Phase 6, and why

Phase 6 addressed that root cause directly: instead of predicting a raw price,
the models were switched to predicting a percentage change from today's price
(for example, "expect a move of about +1.5%" instead of "expect a price of
1,274"). A percentage change means the same thing regardless of whether a stock
is worth 11 rupees or 7,600 rupees, so one shared set of models can learn much
more consistent, general rules across all 25 stocks. The features fed into the
models (the various signals calculated from price history that the models
actually learn from - moving averages, recent volatility, and so on) were
reworked the same way, turning raw price-based numbers into ratios and
percentages wherever possible.

This turned out to matter more than expected. A routine check for "data
leakage" (features that accidentally let a model see something close to the
answer it's supposed to be predicting, which makes it look far more accurate
than it really is) found that several of the old raw-price features were
doing exactly that - correlated with the outcome at a 98-99% level across
every one of the 25 stocks. Under the new percentage-based approach, that
same check found nothing above roughly 29%, and overall accuracy measured on
data the models hadn't seen actually went up, from 96.1% to 98.6%.

The old "clamp the range using recent typical swing" patch was also replaced
with something more principled: each model's range is now sized from how
wrong that specific model has actually been in the past, for that specific
stock, rather than from a separate rule bolted on afterward. Tested against
real outcomes, the old patch's range only actually contained the real
next-day outcome about 24% of the time, despite looking tight and confident.
The new approach's range contained the real outcome about 83% of the time -
close to its intended 80% target. The old approach wasn't just imprecise, it
was actively misleading: a narrow-looking range that's wrong three times out
of four is worse than a wider one that's honest about the uncertainty.

One more idea was tried and openly did not pay off: combining all four
prediction methods into a single blended guess (an "ensemble"), on the theory
that averaging several imperfect guesses often beats picking just one. Tested
directly, the blended guess only beat the single best method about 16% of the
time. That's reported as a real, negative result rather than left out or
glossed over - the four methods turned out to make very similar mistakes on
the same stocks, so averaging them didn't add the diversity that would have
made it worth doing.

## What changed in Phase 7, and why

Phase 6 only predicted the next single trading day. Phase 7 extended that to
four more time horizons - about a week, two weeks, a month, and three months
out - so the dashboard could show not just "tomorrow" but a fuller picture of
how the forecast changes the further out you look. A fifth prediction method
was also added (a lighter, simpler cousin of one of the four already in use)
purely to see how it compared.

Extending the target that far out required rethinking one thing from Phase 6:
the percentage-change cutoff used to keep a small number of clearly bad data
rows (things like a recorded price of literally zero) from distorting model
training. A flat cutoff that made sense for one day out was far too tight for
three months out, since real prices genuinely move much further over three
months than over one day - so the cutoff was scaled up with the length of the
forecast instead of staying fixed.

The most important finding from this phase was an honest and slightly
uncomfortable one: the further out the forecast, the less it actually beats
the simplest possible guess - "assume nothing changes." Measured against that
simple guess, the prediction methods clearly added value for next-day
forecasts (roughly 5-19% better than assuming no change, depending on the
method). By three months out, only one of the five methods was still even
slightly ahead of that simple guess; the other four were doing no better, or
worse. This isn't a flaw introduced by this phase - it's a realistic property
of trying to forecast stock prices that far ahead, and it's exactly the kind
of result that needs to be shown plainly rather than hidden behind an
impressive-looking accuracy percentage. (Percentage-based accuracy numbers
still look fairly high even for weak forecasts, simply because stock prices
don't move a huge amount in percentage terms most days - which is part of why
comparing against the "assume nothing changes" guess matters.) The new,
lighter fifth prediction method came out a bit behind the more established
one it was compared to at most horizons, though it edged ahead at the very
longest one.

## What changed in Phase 8, and why

With five time horizons and five prediction methods now in play, the old
single-table dashboard from Phase 5 no longer fit the information well - a
flat table listing "best model" once per stock doesn't show how a forecast's
confidence changes across five different horizons. Phase 8 replaced it with a
card for each of the 25 stocks, each showing a small fan-shaped graphic that
visually widens from left to right - narrow near "today" and progressively
wider further out, since the honest, backtested uncertainty genuinely grows
the further into the future the forecast reaches. Clicking a stock's card
opens a bigger version of the same graphic with a tab for each of the five
horizons, showing the predicted range, how confident that forecast actually
is, and which prediction method produced it.

Phase 8 also changed how the "which method do we actually show" decision gets
made: instead of picking whichever method scored the lowest raw error, it now
picks whichever method showed the best real improvement over the "assume
nothing changes" baseline from Phase 7 - and if none of the five methods
actually beat that baseline for a given stock, horizon, and direction, the
dashboard says so plainly instead of quietly showing a number that lost to
doing nothing. That turned out to matter in practice: for the one-day
forecast, every stock had at least one method that beat the simple baseline,
but by two weeks out, close to 28% of the stock-and-direction combinations had
no method that cleared it, and the dashboard now shows a clear label for those
rather than a fabricated confidence number.

Because the dashboard now only ever shows predictions from the specific
combination of method, stock, and time horizon that actually gets picked by
that rule, a lot of trained-but-never-used prediction files could be safely
set aside rather than kept loaded and ready. That trimmed the working set of
files the live dashboard needs from about 1.1 gigabytes down to about 560
megabytes, with the rest kept on hand for reference rather than deleted.

## Before and after

| | Before (through Phase 5) | After (Phase 6-8) |
|---|---|---|
| What's predicted | A raw price level | A percentage change from today's price |
| Time horizons shown | Next trading day only | Next day, 1 week, 2 weeks, 1 month, 3 months |
| Prediction methods | 4 | 5 |
| How the shown method is picked | Lowest raw error on held-out data | Best real improvement over a "no change" baseline; falls back to an honest label if nothing clears it |
| Leakage in the features | Several features 98-99% correlated with the answer | Nothing above ~29% |
| Accuracy on unseen data | 96.1% | 98.6% |
| How wide the predicted range is | A fixed rule based on recent typical swing | Sized from each method's own real track record, per stock |
| How often the range actually contained the real outcome | ~24% of the time | ~83% of the time |
| Combining methods into one blended guess | Not tried | Tried, and openly reported as not working (beat the single best method only ~16% of the time) |
| Dashboard layout | One table + a dropdown for detail | A card per stock with a widening range graphic, click-through detail view with a tab per time horizon |
| Live prediction files kept ready to serve | ~1.1 GB | ~560 MB |
