"""
Phase 8 dashboard for the PSX stock prediction project, per Phases/frontend.md.
Single page, built for a demo audience that includes non-technical
viewers. The horizon selector is the only interactive control on
the page. Everything else is visible without a click.

Predictions are read from data/phase9_predictions.csv, assembled by
generate_predictions.py from the Phase 9 Open/Close models - see
Phases/frontend.md "Data the dashboard expects" and "Wiring it into the
pipeline".

Requires streamlit >= 1.36 for st.segmented_control. On an older
version, swap that call for st.radio(horizontal=True).
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import Z_SCORE_80PCT, predict_range_from_error_distribution

st.set_page_config(page_title="FundForge: PSX Predictor", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HORIZONS = ["1d", "5d", "10d", "20d", "60d"]
HORIZON_DAYS = {"1d": 1, "5d": 5, "10d": 10, "20d": 20, "60d": 60}

# PSX market holidays, checked against as trading days are advanced past
# them - keeps the "predicted for" date from landing on a day the market
# was actually closed.
PSX_HOLIDAYS_2026 = [
    "2026-02-05", "2026-03-23", "2026-05-01", "2026-08-14", "2026-12-25",
]

# All 97 tickers in data/master_dataset.csv (KSE-100 expansion,
# Phases/kse_100_expand.md - DHPL excluded, see that doc's "DHPL excluded
# after Stage 4"). No name field exists in that file to look up instead,
# so this is a static lookup - see Phases/frontend.md "Wiring it into the
# pipeline".
COMPANY_NAMES = {
    "BAHL": "Bank Al Habib",
    "COLG": "Colgate-Palmolive Pakistan",
    "DGKC": "D.G. Khan Cement",
    "EFERT": "Engro Fertilizers",
    "ENGROH": "Engro Holdings",
    "FFC": "Fauji Fertilizer",
    "HBL": "Habib Bank",
    "HUBC": "Hub Power Company",
    "ILP": "Interloop",
    "INDU": "Indus Motor Company",
    "LUCK": "Lucky Cement",
    "MARI": "Mari Petroleum",
    "MCB": "MCB Bank",
    "MEBL": "Meezan Bank",
    "MLCF": "Maple Leaf Cement",
    "NESTLE": "Nestle Pakistan",
    "NETSOL": "NetSol Technologies",
    "OGDC": "Oil and Gas Dev",
    "PAKT": "Pakistan Tobacco",
    "PPL": "Pakistan Petroleum",
    "PSO": "Pakistan State Oil",
    "SYS": "Systems Limited",
    "TRG": "TRG Pakistan",
    "UBL": "United Bank",
    "UNITY": "Unity Foods",
    "PGLC": "Pak-Gulf Leasing Company",
    "PSX": "Pakistan Stock Exchange",
    "BAFL": "Bank Alfalah",
    "BOP": "The Bank of Punjab",
    "FABL": "Faysal Bank",
    "HMB": "Habib Metropolitan Bank",
    "NBP": "National Bank of Pakistan",
    "SCBPL": "Standard Chartered Bank Pakistan",
    "AKBL": "Askari Bank",
    "ABL": "Allied Bank",
    "AICL": "Adamjee Insurance",
    "GADT": "Gadoon Textile Mills",
    "YOUW": "Yousaf Weaving Mills",
    "NML": "Nishat Mills",
    "MEHT": "Mehmood Textile Mills",
    "KTML": "Kohinoor Textile Mills",
    "BNWM": "Bannu Woollen Mills",
    "IBFL": "Ibrahim Fibres",
    "JDWS": "JDW Sugar Mills",
    "BWCL": "Bestway Cement",
    "FCCL": "Fauji Cement Company",
    "KOHC": "Kohat Cement Company",
    "CHCC": "Cherat Cement Company",
    "POWER": "Power Cement",
    "PIOC": "Pioneer Cement",
    "ATRL": "Attock Refinery",
    "CNERGY": "Cnergyico PK",
    "KAPCO": "Kot Addu Power Company",
    "KEL": "K-Electric",
    "APL": "Attock Petroleum",
    "SNGP": "Sui Northern Gas Pipelines",
    "SSGC": "Sui Southern Gas Company",
    "POL": "Pakistan Oilfields",
    "ISL": "International Steels",
    "INIL": "International Industries",
    "MTL": "Millat Tractors",
    "SAZEW": "Sazgar Engineering Works",
    "ATLH": "Atlas Honda",
    "HCAR": "Honda Atlas Cars Pakistan",
    "GAL": "Ghandhara Automobiles",
    "GHNI": "Ghandhara Industries",
    "THALL": "Thal",
    "PAEL": "Pak Elektron",
    "PIBTL": "Pakistan International Bulk Terminal",
    "AIRLINK": "Air Link Communication",
    "PTC": "Pakistan Telecommunication Company",
    "HUMNL": "Hum Network",
    "AHCL": "Arif Habib Corporation",
    "FATIMA": "Fatima Fertilizer Company",
    "SEARL": "The Searle Company",
    "AGP": "AGP",
    "GLAXO": "GlaxoSmithKline Pakistan",
    "ABOT": "Abbott Laboratories Pakistan",
    "CPHL": "Citi Pharma",
    "HALEON": "Haleon Pakistan",
    "HINOON": "Highnoon Laboratories",
    "LCI": "Lucky Core Industries",
    "LOTCHEM": "Lotte Chemical Pakistan",
    "PKGS": "Packages",
    "SSOM": "S.S. Oil Mills",
    "SRVI": "Service Industries",
    "FFL": "Fauji Foods",
    "MUREB": "Murree Brewery Company",
    "NATF": "National Foods",
    "RMPL": "Rafhan Maize Products",
    "UPFL": "Unilever Pakistan Foods",
    "TGL": "Tariq Glass Industries",
    "GHGL": "Ghani Glass",
    "PSEL": "Pakistan Services",
    "PABC": "Pakistan Aluminium Beverage Cans",
    "SHFA": "Shifa International Hospitals",
    "JVDC": "Javedan Corporation",
}


def load_predictions(path: Path = DATA_DIR / "phase9_predictions.csv") -> pd.DataFrame:
    """
    Loads the Phase 9 predictions file assembled by generate_predictions.py.
    See the column list in frontend.md for the expected schema. Falls back
    to sample_predictions() if the file is not found, so the dashboard can
    be tested before the prediction assembly step is wired up.
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
        dict(Ticker="ENGROH", Yesterday_Open=306.90, Yesterday_Close=305.50,
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
    df = pd.DataFrame(rows)
    df["Data_Date"] = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    return df


@st.cache_data
def load_master_data(path: str = str(DATA_DIR / "master_dataset.csv")) -> pd.DataFrame:
    """Per-ticker price history for the detail view's chart. Cached since
    this file is far larger than the other CSVs the dashboard reads and the
    detail view re-reads it on every horizon-pill click."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def pct_change(new_value, old_value):
    return (new_value - old_value) / old_value * 100


def format_model_name(model_type) -> str:
    """naive_baseline isn't a trained model - it's what gets served when
    nothing beat "no change", so it's worded differently from an actual
    model name here rather than listed as if it were one."""
    if pd.isna(model_type):
        return "N/A"
    if model_type == "naive_baseline":
        return "none (naive held)"
    return model_type


def advance_trading_date(start_date, n: int, holidays=None):
    """Start date -> the actual date the n-trading-day-out prediction is
    for, skipping weekends and PSX holidays. Works on datetime/Timestamp
    alike, so the detail chart can feed it a real date straight from
    master_dataset.csv."""
    holiday_dates = {datetime.strptime(h, "%Y-%m-%d").date() for h in (holidays or [])}
    date = start_date
    reached = 0
    while reached < n:
        date += timedelta(days=1)
        if date.weekday() < 5 and date.date() not in holiday_dates:
            reached += 1
    return date


def advance_trading_days(data_date: str, n: int, holidays=None) -> str:
    """String-label wrapper around advance_trading_date(), for captions."""
    start = datetime.strptime(data_date, "%Y-%m-%d")
    return advance_trading_date(start, n, holidays).strftime("%b %d, %Y")


def render_card(row: pd.Series, data_date: str):
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

        data_date_label = datetime.strptime(data_date, "%Y-%m-%d").strftime("%b %d, %Y")
        st.caption(
            f"{data_date_label}: Open Rs {row['Yesterday_Open']:.2f}, "
            f"Close Rs {row['Yesterday_Close']:.2f}"
        )

        predicted_date = advance_trading_days(data_date, HORIZON_DAYS[horizon], PSX_HOLIDAYS_2026)
        st.caption(f"Predicted for {predicted_date}")

        open_val = row[f"Pred_Open_{horizon}"]
        close_val = row[f"Pred_Close_{horizon}"]
        open_pct = pct_change(open_val, row["Yesterday_Open"])
        close_pct = pct_change(close_val, row["Yesterday_Close"])

        col1, col2 = st.columns(2)
        col1.metric("Predicted Open", f"Rs {open_val:.2f}", f"{open_pct:+.1f}%")
        col2.metric("Predicted Close", f"Rs {close_val:.2f}", f"{close_pct:+.1f}%")

        # Index must be the numeric day count, not the "1d"/"5d" label -
        # st.line_chart sorts a string index alphabetically ("10d" < "1d"
        # < "20d" < "5d" < "60d"), which scrambles the horizon order. A
        # numeric index sorts ascending by value instead, matching HORIZONS.
        trend = pd.DataFrame(
            {"Close": [row[f"Pred_Close_{h}"] for h in HORIZONS]},
            index=[HORIZON_DAYS[h] for h in HORIZONS],
        )
        st.line_chart(trend, height=120)

        open_model = format_model_name(row.get(f"Model_Open_{horizon}"))
        close_model = format_model_name(row.get(f"Model_Close_{horizon}"))
        st.caption(f"Serving ({horizon}) - Open: {open_model} | Close: {close_model}")

        st.button(
            "View detail", key=f"detail_btn_{ticker}", use_container_width=True,
            on_click=_select_ticker, args=(ticker,),
        )


def _select_ticker(ticker):
    st.session_state.selected_ticker = ticker


def get_history_prices(master_df: pd.DataFrame, ticker: str, n: int = 250):
    """Last n trading days of (Date, Close) for the detail chart's solid
    history line, ending at the dataset's latest row. Returns real dates,
    not a bare row index, so the chart's x-axis and hover tooltip read as
    actual calendar dates. n defaults to roughly one trading year (~250
    days) - long enough to show a real trend, short enough that the last
    ~8 years of history (master_dataset.csv goes back to 2018) doesn't
    compress the recent, most relevant moves into an unreadable smear."""
    ticker_df = master_df[master_df["Ticker"] == ticker].sort_values("Date").tail(n)
    return ticker_df["Date"].tolist(), ticker_df["Close"].tolist()


def render_stock_detail(ticker: str, preds_row: pd.Series, master_df: pd.DataFrame,
                         data_date: str):
    """Phases/frontend.md's stock detail view. Close is the target behind
    the chart's band and the improvement/directional-accuracy stats - Open
    only appears in the predicted open/close stat card - matching the
    single center line the cone design is built around. Accuracy_%
    (100 - MAPE) is left out on purpose, per the doc: it scores close to
    naive for most PSX stocks, so improvement-over-naive and directional
    accuracy are the two numbers that actually say whether the model beats
    naive.
    """
    name = COMPANY_NAMES.get(ticker, ticker)
    data_date_label = datetime.strptime(data_date, "%Y-%m-%d").strftime("%b %d, %Y")

    with st.container(border=True):
        top1, top2 = st.columns([2, 1])
        with top1:
            st.markdown(f"### {ticker}")
            st.caption(name)
        with top2:
            oc1, oc2 = st.columns(2)
            oc1.metric("Open", f"Rs {preds_row['Yesterday_Open']:.2f}")
            oc2.metric("Close", f"Rs {preds_row['Yesterday_Close']:.2f}")
            st.caption(data_date_label)

        horizon = st.segmented_control(
            "Horizon", HORIZONS, default="5d", key=f"detail_horizon_{ticker}",
        )
        if horizon is None:
            horizon = "5d"
        horizon_days = HORIZON_DAYS[horizon]

        predicted_open = float(preds_row[f"Pred_Open_{horizon}"])
        predicted_close = float(preds_row[f"Pred_Close_{horizon}"])
        # The chosen model's own backtested numbers (see
        # generate_predictions.py's choose_best_model()) - not an ensemble's,
        # since a single best model per cell is what's actually served.
        improvement = preds_row.get(f"Improvement_Close_{horizon}")
        directional_accuracy = preds_row.get(f"Directional_Accuracy_Close_{horizon}")
        std_error = preds_row.get(f"Std_Error_Close_{horizon}")

        history_dates, history = get_history_prices(master_df, ticker)
        today_date = history_dates[-1]
        future_date = advance_trading_date(today_date, horizon_days, PSX_HOLIDAYS_2026)
        future_x = [today_date, future_date]

        if pd.notna(std_error):
            lower, upper = predict_range_from_error_distribution(
                predicted_close, {"std_error": float(std_error)}, Z_SCORE_80PCT)
        else:
            lower = upper = predicted_close

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_dates, y=history, mode="lines",
            line=dict(color="#5F5E5A", width=2), name="History",
        ))
        fig.add_trace(go.Scatter(
            x=future_x + future_x[::-1],
            y=[history[-1], upper, lower, history[-1]],
            fill="toself", fillcolor="rgba(29,158,117,0.12)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=future_x, y=[history[-1], predicted_close],
            mode="lines+markers",
            line=dict(color="#1D9E75", width=2, dash="dash"),
            name="Prediction",
        ))
        fig.add_vline(x=today_date, line_dash="dot", line_color="gray", annotation_text="Today")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key=f"detail_chart_{ticker}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted open / close", f"Rs {predicted_open:.2f} / {predicted_close:.2f}")
        m2.metric("Improvement vs naive", f"{improvement:+.1f}%" if pd.notna(improvement) else "N/A")
        m3.metric("Directional accuracy", f"{directional_accuracy:.0f}%" if pd.notna(directional_accuracy) else "N/A")

        open_chg = pct_change(predicted_open, preds_row["Yesterday_Open"])
        close_chg = pct_change(predicted_close, preds_row["Yesterday_Close"])
        st.caption(f"Open vs yesterday: {open_chg:+.2f}%    Close vs yesterday: {close_chg:+.2f}%")


def main():
    df = load_predictions()
    data_date = str(df["Data_Date"].iloc[0])
    data_date_label = datetime.strptime(data_date, "%Y-%m-%d").strftime("%b %d, %Y")

    st.markdown("### FundForge: PSX Predictor")
    st.caption(
        f"Data as of {data_date_label} (latest trading day in the dataset) - "
        f"dashboard generated {datetime.now().strftime('%b %d, %Y, %I:%M %p')}"
    )

    gainers = (df["Pred_Close_1d"] > df["Yesterday_Close"]).sum()
    losers = (df["Pred_Close_1d"] < df["Yesterday_Close"]).sum()

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Stocks tracked", len(df))
    stat2.metric("Gainers today", int(gainers))
    stat3.metric("Losers today", int(losers))

    if "selected_ticker" not in st.session_state:
        # Defaults to the first ticker (rather than None) so the detail
        # view is visible immediately on load, not hidden behind a click -
        # a stock is always "selected" from the dashboard's point of view.
        st.session_state.selected_ticker = sorted(df["Ticker"].unique())[0]

    st.selectbox(
        "View detail for", options=sorted(df["Ticker"].unique()),
        placeholder="Select a stock", key="selected_ticker",
    )

    # Rendered right after the selector, above the grid, so it's the first
    # thing visible rather than something a user has to scroll past all 25
    # cards to find.
    selected_row = df[df["Ticker"] == st.session_state.selected_ticker].iloc[0]
    render_stock_detail(
        st.session_state.selected_ticker, selected_row,
        load_master_data(), data_date,
    )
    st.divider()

    cols_per_row = 3
    chunks = [df.iloc[i:i + cols_per_row] for i in range(0, len(df), cols_per_row)]
    for chunk in chunks:
        cols = st.columns(cols_per_row)
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                render_card(row, data_date)


if __name__ == "__main__":
    main()
