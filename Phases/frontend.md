# Frontend (Phase 8): PSX Predictor dashboard

This document covers the Phase 8 dashboard for the PSX stock prediction project, built after the Phase 9 migration to Open and Close prediction targets. The dashboard is a single Streamlit page meant for a demo audience that includes non-technical viewers, so it stays on one screen and keeps interaction to a minimum.

## Design decisions

The page has no sidebar, no multi-step flow, and no separate detail view. Everything a viewer needs sits on one scrollable page.

Each stock gets a card. A card holds yesterday's actual Open and Close, a horizon selector, the predicted Open and Close for the selected horizon with a percentage change against yesterday's actual price, and a trend line for the predicted Close across all five horizons.

The horizon selector (1d, 5d, 10d, 20d, 60d) is the one interactive control on the page. It was added after the earlier no-button design proved too limited: without it, a viewer had no way to see the predicted price at a specific horizon, only the overall trend line. Everything else on the page is visible without a click.

The trend line plots Close only, not Open. Two overlapping lines per card made the grid harder to read, and Close is the figure most people already associate with a stock's daily price.

A summary row at the top gives three numbers: how many stocks are tracked, how many are predicted to rise from yesterday's Close by the next trading day, and how many are predicted to fall.

Color follows a plain rule throughout: green for a predicted increase, red for a predicted decrease. Streamlit's `st.metric` widget applies this automatically from the sign of the delta, so the badges do not need custom styling.

## Data the dashboard expects

The dashboard reads a CSV with one row per stock and these columns:

```
Ticker, Yesterday_Open, Yesterday_Close,
Pred_Open_1d, Pred_Close_1d,
Pred_Open_5d, Pred_Close_5d,
Pred_Open_10d, Pred_Close_10d,
Pred_Open_20d, Pred_Close_20d,
Pred_Open_60d, Pred_Close_60d
```

`manage_model_storage.py` already produces Open and Close model files per horizon after Phase 9. This dashboard does not load those model files directly. It expects a separate assembly step that runs each stock's latest feature row through the right model for each horizon and target, then writes the results into the file above. If that assembly script does not exist yet, `load_predictions()` below falls back to sample data for four stocks so the dashboard can be built and reviewed on its own.

## Full code

```python
"""
Phase 8 dashboard for the PSX stock prediction project.
Single page, built for a demo audience that includes non-technical
viewers. The horizon selector is the only interactive control on
the page. Everything else is visible without a click.

Requires streamlit >= 1.36 for st.segmented_control. On an older
version, swap that call for st.radio(horizontal=True).
"""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="PSX Predictor", layout="wide")

HORIZONS = ["1d", "5d", "10d", "20d", "60d"]

COMPANY_NAMES = {
    "HBL": "Habib Bank",
    "ENGRO": "Engro Corp",
    "LUCK": "Lucky Cement",
    "FFC": "Fauji Fertilizer",
    "OGDC": "Oil and Gas Dev",
    "MEBL": "Meezan Bank",
}


def load_predictions(path: str = "phase9_predictions.csv") -> pd.DataFrame:
    """
    Loads the Phase 9 predictions file. See the column list in
    frontend.md for the expected schema. Falls back to sample_predictions()
    if the file is not found, so the dashboard can be tested before the
    prediction assembly step is wired up.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.warning(f"{path} not found. Showing sample data.")
        return sample_predictions()


def sample_predictions() -> pd.DataFrame:
    """Sample data for four stocks, used only when the real predictions
    file is missing."""
    rows = [
        dict(Ticker="HBL", Yesterday_Open=146.80, Yesterday_Close=148.20,
             Pred_Open_1d=149.50, Pred_Close_1d=151.30,
             Pred_Open_5d=151.10, Pred_Close_5d=153.20,
             Pred_Open_10d=153.10, Pred_Close_10d=155.80,
             Pred_Open_20d=155.30, Pred_Close_20d=158.30,
             Pred_Open_60d=157.80, Pred_Close_60d=161.20),
        dict(Ticker="ENGRO", Yesterday_Open=306.90, Yesterday_Close=305.50,
             Pred_Open_1d=304.20, Pred_Close_1d=302.80,
             Pred_Open_5d=302.30, Pred_Close_5d=299.10,
             Pred_Open_10d=299.20, Pred_Close_10d=294.50,
             Pred_Open_20d=296.40, Pred_Close_20d=290.80,
             Pred_Open_60d=295.30, Pred_Close_60d=288.90),
        dict(Ticker="LUCK", Yesterday_Open=810.20, Yesterday_Close=812.75,
             Pred_Open_1d=815.40, Pred_Close_1d=818.20,
             Pred_Open_5d=821.10, Pred_Close_5d=825.60,
             Pred_Open_10d=829.20, Pred_Close_10d=835.10,
             Pred_Open_20d=835.20, Pred_Close_20d=842.30,
             Pred_Open_60d=842.10, Pred_Close_60d=850.40),
        dict(Ticker="FFC", Yesterday_Open=131.10, Yesterday_Close=132.40,
             Pred_Open_1d=133.20, Pred_Close_1d=133.80,
             Pred_Open_5d=134.30, Pred_Close_5d=135.90,
             Pred_Open_10d=136.10, Pred_Close_10d=138.20,
             Pred_Open_20d=138.00, Pred_Close_20d=140.50,
             Pred_Open_60d=139.80, Pred_Close_60d=142.60),
    ]
    return pd.DataFrame(rows)


def pct_change(new_value, old_value):
    return (new_value - old_value) / old_value * 100


def render_card(row: pd.Series):
    ticker = row["Ticker"]
    name = COMPANY_NAMES.get(ticker, ticker)

    with st.container(border=True):
        top_left, top_right = st.columns([2, 3])
        with top_left:
            st.markdown(f"**{ticker}**")
            st.caption(name)
        with top_right:
            horizon = st.segmented_control(
                "Horizon", HORIZONS, default="1d",
                key=f"horizon_{ticker}", label_visibility="collapsed",
            )
        if horizon is None:
            horizon = "1d"

        st.caption(
            f"Kal: Open Rs {row['Yesterday_Open']:.2f}, "
            f"Close Rs {row['Yesterday_Close']:.2f}"
        )

        open_val = row[f"Pred_Open_{horizon}"]
        close_val = row[f"Pred_Close_{horizon}"]
        open_pct = pct_change(open_val, row["Yesterday_Open"])
        close_pct = pct_change(close_val, row["Yesterday_Close"])

        col1, col2 = st.columns(2)
        col1.metric("Predicted Open", f"Rs {open_val:.2f}", f"{open_pct:+.1f}%")
        col2.metric("Predicted Close", f"Rs {close_val:.2f}", f"{close_pct:+.1f}%")

        trend = pd.DataFrame(
            {"Close": [row[f"Pred_Close_{h}"] for h in HORIZONS]},
            index=HORIZONS,
        )
        st.line_chart(trend, height=120)


def main():
    df = load_predictions()

    st.markdown("### PSX Predictor")
    st.caption(f"Last updated {datetime.now().strftime('%b %d, %Y, %I:%M %p')}")

    gainers = (df["Pred_Close_1d"] > df["Yesterday_Close"]).sum()
    losers = (df["Pred_Close_1d"] < df["Yesterday_Close"]).sum()

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Stocks tracked", len(df))
    stat2.metric("Gainers today", int(gainers))
    stat3.metric("Losers today", int(losers))

    cols_per_row = 3
    chunks = [df.iloc[i:i + cols_per_row] for i in range(0, len(df), cols_per_row)]
    for chunk in chunks:
        cols = st.columns(cols_per_row)
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                render_card(row)


if __name__ == "__main__":
    main()
```

## Wiring it into the pipeline

Two things stand between this file and a working demo. First, `COMPANY_NAMES` only covers the six tickers used while testing the layout in this document. It needs an entry for all 25 stocks in the dataset, or a lookup against whatever name field already exists in `master_dataset.csv`. Second, and more important, `phase9_predictions.csv` does not exist yet. It needs a script that loads each of the 50 Phase 9 models, runs the latest feature row for each stock through the right model for each horizon and target, and writes one row per stock into the schema shown above. That script can live alongside `train_models_phase9.py` or as a new `generate_predictions.py`, whichever fits the rest of the Phase 9 file layout better.

## Known limits

A 3-column grid across 25 stocks runs to roughly nine rows, so the page requires scrolling. That is expected for a single-page layout at this stock count and does not conflict with the no-navigation goal, since scrolling is not a control the viewer has to learn.

`st.segmented_control` needs Streamlit 1.36 or newer. If the deployment environment pins an older version, `st.radio(horizontal=True)` is a direct substitute with the same one-selection-per-card behavior, though it renders as radio buttons rather than pills.