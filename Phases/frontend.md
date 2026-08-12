# Stock detail view design

This is a companion to `frontend.md`. That file covers the main dashboard: a
single-page grid of all 25 stocks with the summary stat cards at the top.
This file covers a second piece, a detail view for one stock at a time,
styled after the FundForge dashboard's Bitcoin detail screen. It only
appears when a user selects a stock, so the main page stays as simple as the
supervisor asked for.

## Layout

**Header.** Ticker and company name sit on the left. The right side shows
yesterday's actual open and close, in large text, with the date directly
below in smaller muted text. This is deliberately the previous day's real
prices, not a prediction, so the user has a fixed reference point before
looking at anything the model produced.

**Horizon selector.** Five pills: 1d, 5d, 10d, 20d, 60d. Selecting one
updates the chart and every stat below it.

**Chart.** A solid line shows price history up to today. A dashed vertical
line marks today. From that point, the chart splits into three dashed lines:
an upper bound, a lower bound, and a center line running to the predicted
close at the selected horizon. The area between the upper and lower bound is
filled at low opacity, so the band widens visibly as the horizon grows. This
mirrors what your Phase 6 and 7 work already produces: the error-distribution
range around each prediction, not a flat point estimate.

**Stat row.** Three cards:

1. Predicted open and close for the selected horizon.
2. Improvement over the naive baseline, as a percentage.
3. Directional accuracy, as a percentage.

The 100 minus MAPE accuracy score used in the Phase 4 through 7 dashboards is
left out on purpose. Your own naive baseline test found it scores close to
the model itself for most PSX stocks (HBL's XGBoost model was ahead of naive
by only 0.5 to 0.7 points), because daily price moves are small enough that
"no change from yesterday" already looks accurate on that metric. Improvement
over naive and directional accuracy are the two numbers that actually say
whether the model is doing something naive cannot.

**Day over day change.** A line below the stat row showing the percentage
change from yesterday's actual open to the predicted open, and from
yesterday's actual close to the predicted close.

**Top drivers.** Three small tags showing the top SHAP features for this
prediction, each with an up or down arrow and colored green or red depending
on whether that feature pushed the prediction up or down. This reuses the
SHAP top-3 output already wired into the Phase 6 app, just displayed as tags
instead of a table.

## Where each number comes from

| Element | Source |
|---|---|
| Yesterday's open and close | Last row of the stock's data before the prediction date, in the Phase 9 dataset |
| Predicted open and close | `phase9_predictions.csv` (or the equivalent assembly script output), keyed by ticker and horizon |
| Confidence band bounds | `compute_backtested_error_distribution()` in `model_utils.py`, at the selected horizon |
| Improvement over naive | `Improvement_over_Naive_%` column from the Phase 7 eval table, per ticker and horizon |
| Directional accuracy | `Directional Accuracy` column from the same eval table |
| Top drivers | Existing SHAP output already computed in the Phase 6 app |

None of these need to be computed live in the Streamlit app. They should all
be precomputed and read from the saved CSVs and pickled eval tables, the same
way the rest of the dashboard already works.

## Fitting this into the existing page

The main page from `frontend.md` stays the default view: the summary stat
cards and the grid or list of all 25 stocks. Add a click handler (or a
`st.selectbox` as a simpler alternative) that sets a ticker in
`st.session_state`. When a ticker is selected, render this detail view below
or beside the main grid. No new page and no navigation bar, the whole thing
stays on one screen.

## Streamlit implementation

```python
import streamlit as st
import plotly.graph_objects as go

HORIZON_DAYS = {"1d": 1, "5d": 5, "10d": 10, "20d": 20, "60d": 60}


def render_stock_detail(ticker, data, horizon="5d"):
    """
    data keys:
        company_name, yesterday_open, yesterday_close, date,
        predicted_open, predicted_close,
        improvement_vs_naive, directional_accuracy,
        top_drivers: list of (feature_name, "up" or "down"),
        history_prices: list of recent closes ending at today,
        cone_upper_end, cone_lower_end: predicted band bounds at the horizon
    """
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### {ticker}")
        st.caption(data["company_name"])
    with col2:
        oc1, oc2 = st.columns(2)
        oc1.metric("Open", f"{data['yesterday_open']:.2f}")
        oc2.metric("Close", f"{data['yesterday_close']:.2f}")
        st.caption(data["date"])

    horizon = st.segmented_control(
        "Horizon",
        list(HORIZON_DAYS.keys()),
        default=horizon,
        key=f"horizon_{ticker}",
    )
    horizon_days = HORIZON_DAYS[horizon]

    history = data["history_prices"]
    n = len(history)
    future_x = [n - 1, n - 1 + horizon_days]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=history, mode="lines",
        line=dict(color="#5F5E5A", width=2), name="History",
    ))
    fig.add_trace(go.Scatter(
        x=future_x + future_x[::-1],
        y=[history[-1], data["cone_upper_end"],
           data["cone_lower_end"], history[-1]],
        fill="toself", fillcolor="rgba(29,158,117,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=future_x, y=[history[-1], data["predicted_close"]],
        mode="lines+markers",
        line=dict(color="#1D9E75", width=2, dash="dash"),
        name="Prediction",
    ))
    fig.add_vline(x=n - 1, line_dash="dot", line_color="gray",
                  annotation_text="Today")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                       showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted open / close",
              f"{data['predicted_open']:.2f} / {data['predicted_close']:.2f}")
    m2.metric("Improvement vs naive",
              f"{data['improvement_vs_naive']:+.1f}%")
    m3.metric("Directional accuracy",
              f"{data['directional_accuracy']:.0f}%")

    open_chg = (data["predicted_open"] - data["yesterday_open"]) \
        / data["yesterday_open"] * 100
    close_chg = (data["predicted_close"] - data["yesterday_close"]) \
        / data["yesterday_close"] * 100
    st.caption(
        f"Open vs yesterday: {open_chg:+.2f}%    "
        f"Close vs yesterday: {close_chg:+.2f}%"
    )

    st.caption("Top drivers")
    tags = []
    for feature, direction in data["top_drivers"]:
        bg = "#EAF3DE" if direction == "up" else "#FCEBEB"
        fg = "#27500A" if direction == "up" else "#791F1F"
        arrow = "up" if direction == "up" else "down"
        tags.append(
            f'<span style="background:{bg}; color:{fg}; padding:3px 10px; '
            f'border-radius:999px; font-size:12px; margin-right:6px;">'
            f'{arrow} {feature}</span>'
        )
    st.markdown(" ".join(tags), unsafe_allow_html=True)
```

## Selecting a stock from the main grid

```python
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

st.session_state.selected_ticker = st.selectbox(
    "View detail for", options=all_tickers,
    index=None, placeholder="Select a stock",
)

if st.session_state.selected_ticker:
    render_stock_detail(
        st.session_state.selected_ticker,
        get_stock_data(st.session_state.selected_ticker),
    )
```

Replace the `st.selectbox` with a click handler on the grid cards if you want
selection from the grid itself rather than a separate dropdown. Either way,
`get_stock_data()` should pull from the sources listed in the table above,
not recompute anything on the fly.