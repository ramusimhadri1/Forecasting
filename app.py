import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from datetime import timedelta

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# =========================================================
# OPTIONAL LSTM
# =========================================================
try:

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

    TENSORFLOW_AVAILABLE = True

except:

    TENSORFLOW_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="End-to-End Time Series Forecasting System",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# PROFESSIONAL ENTERPRISE UI
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* ======================================================
BACKGROUND
====================================================== */
.stApp {

    background:
    linear-gradient(
        135deg,
        #020617 0%,
        #0F172A 45%,
        #111827 100%
    );

    color: #F8FAFC;
}

/* ======================================================
SIDEBAR
====================================================== */
section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #111827,
        #1E293B
    );

    border-right:
    1px solid rgba(255,255,255,0.08);
}

/* ======================================================
HEADERS
====================================================== */
h1 {

    font-size: 3.2rem !important;

    font-weight: 700 !important;

    background:
    linear-gradient(
        90deg,
        #38BDF8,
        #818CF8,
        #C084FC
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

h2, h3, h4 {

    color: #F8FAFC;
}

/* ======================================================
METRIC CARDS
====================================================== */
[data-testid="metric-container"] {

    background:
    linear-gradient(
        145deg,
        rgba(30,41,59,0.95),
        rgba(15,23,42,0.95)
    );

    border:
    1px solid rgba(255,255,255,0.08);

    padding: 20px;

    border-radius: 18px;

    box-shadow:
    0px 8px 25px rgba(0,0,0,0.35);

    transition: 0.3s;
}

[data-testid="metric-container"]:hover {

    transform: translateY(-5px);

    border:
    1px solid #3B82F6;
}

/* ======================================================
BUTTONS
====================================================== */
.stButton>button {

    background:
    linear-gradient(
        90deg,
        #2563EB,
        #7C3AED
    );

    color: white;

    border-radius: 14px;

    height: 3.2em;

    font-size: 17px;

    font-weight: 600;

    border: none;

    width: 100%;
}

.stButton>button:hover {

    background:
    linear-gradient(
        90deg,
        #7C3AED,
        #2563EB
    );

    transform: scale(1.02);
}

/* ======================================================
DOWNLOAD BUTTON
====================================================== */
.stDownloadButton>button {

    background:
    linear-gradient(
        90deg,
        #059669,
        #10B981
    );

    color: white;

    border-radius: 14px;

    height: 3em;

    font-size: 16px;

    font-weight: 600;

    border: none;
}

/* ======================================================
TABS
====================================================== */
.stTabs [data-baseweb="tab-list"] {

    gap: 18px;
}

.stTabs [data-baseweb="tab"] {

    background:
    rgba(255,255,255,0.05);

    border-radius: 12px;

    padding: 12px 22px;

    color: white;

    font-weight: 600;
}

.stTabs [aria-selected="true"] {

    background:
    linear-gradient(
        90deg,
        #2563EB,
        #7C3AED
    ) !important;
}

/* ======================================================
DATAFRAME
====================================================== */
[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;

    border:
    1px solid rgba(255,255,255,0.06);
}

/* ======================================================
FOOTER
====================================================== */
.footer {

    text-align: center;

    padding: 25px;

    margin-top: 30px;

    border-radius: 20px;

    background:
    linear-gradient(
        90deg,
        rgba(30,41,59,0.8),
        rgba(15,23,42,0.8)
    );

    border:
    1px solid rgba(255,255,255,0.06);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<h1 style='text-align:center;'>

🚀 End-to-End Time Series Forecasting System

</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='
text-align:center;
font-size:22px;
color:#CBD5E1;
margin-bottom:25px;
'>

Production Ready Forecasting Platform
using ARIMA • SARIMA • Prophet • XGBoost • LSTM

</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv("Forecasting.csv")

    return df

df = load_data()

# =========================================================
# COLUMN SETTINGS
# =========================================================
DATE_COLUMN = "Date"
STATE_COLUMN = "State"
SALES_COLUMN = "Total"

# =========================================================
# DATA CLEANING
# =========================================================
df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(subset=[DATE_COLUMN])

df[SALES_COLUMN] = (
    df[SALES_COLUMN]
    .astype(str)
    .str.replace(",", "")
)

df[SALES_COLUMN] = pd.to_numeric(
    df[SALES_COLUMN],
    errors="coerce"
)

df[SALES_COLUMN] = df[SALES_COLUMN].ffill()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Forecast Controls")

selected_state = st.sidebar.selectbox(
    "📍 Select State",
    sorted(df[STATE_COLUMN].unique())
)

min_date = df[DATE_COLUMN].min()
max_date = df[DATE_COLUMN].max()

from_date = st.sidebar.date_input(
    "📅 From Date",
    min_date
)

to_date = st.sidebar.date_input(
    "📅 To Date",
    max_date
)

forecast_weeks = st.sidebar.slider(
    "📈 Forecast Weeks",
    1,
    8,
    8
)

generate_forecast = st.sidebar.button(
    "🚀 Generate Forecast",
    use_container_width=True
)

# =========================================================
# FILTER DATA
# =========================================================
filtered_df = df[

    (df[STATE_COLUMN] == selected_state)

    &

    (df[DATE_COLUMN] >= pd.to_datetime(from_date))

    &

    (df[DATE_COLUMN] <= pd.to_datetime(to_date))

].sort_values(DATE_COLUMN).copy()

# =========================================================
# FEATURE ENGINEERING
# =========================================================
filtered_df["lag_1"] = filtered_df[SALES_COLUMN].shift(1)

filtered_df["lag_7"] = filtered_df[SALES_COLUMN].shift(7)

filtered_df["lag_30"] = filtered_df[SALES_COLUMN].shift(30)

filtered_df["rolling_mean_7"] = (
    filtered_df[SALES_COLUMN]
    .rolling(7)
    .mean()
)

filtered_df["rolling_std_7"] = (
    filtered_df[SALES_COLUMN]
    .rolling(7)
    .std()
)

filtered_df["day_of_week"] = (
    filtered_df[DATE_COLUMN]
    .dt.dayofweek
)

filtered_df["month"] = (
    filtered_df[DATE_COLUMN]
    .dt.month
)

filtered_df["holiday_flag"] = np.where(
    filtered_df["day_of_week"].isin([5,6]),
    1,
    0
)

filtered_df = filtered_df.dropna()

# =========================================================
# KPI SECTION
# =========================================================
total_sales = filtered_df[SALES_COLUMN].sum()

avg_sales = filtered_df[SALES_COLUMN].mean()

max_sales = filtered_df[SALES_COLUMN].max()

growth_rate = (
    (
        filtered_df[SALES_COLUMN].iloc[-1]
        -
        filtered_df[SALES_COLUMN].iloc[0]
    )
    /
    filtered_df[SALES_COLUMN].iloc[0]
) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Total Revenue",
    f"${total_sales:,.0f}"
)

col2.metric(
    "📊 Average Sales",
    f"${avg_sales:,.0f}"
)

col3.metric(
    "🚀 Growth Rate",
    f"{growth_rate:.2f}%"
)

col4.metric(
    "🔥 Peak Sales",
    f"${max_sales:,.0f}"
)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "📊 Dashboard",

    "🤖 Models",

    "🔮 Forecast",

    "🌐 API",

    "📂 Data"

])

# =========================================================
# DASHBOARD TAB
# =========================================================
with tab1:

    st.subheader("📈 Historical Sales Trend")

    fig = px.line(
        filtered_df,
        x=DATE_COLUMN,
        y=SALES_COLUMN,
        template="plotly_dark"
    )

    fig.update_traces(
        line=dict(
            color="#38BDF8",
            width=4
        )
    )

    fig.update_layout(
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
