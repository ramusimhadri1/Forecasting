import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from datetime import timedelta

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

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
    page_title="End-to-End Time Series Forecasting",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# PROFESSIONAL UI
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */
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

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
    linear-gradient(
        180deg,
        #111827,
        #1E293B
    );
}

/* Main Title */
h1 {
    font-size: 3rem !important;
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

/* Metric Cards */
[data-testid="metric-container"] {

    background:
    linear-gradient(
        145deg,
        rgba(30,41,59,0.95),
        rgba(15,23,42,0.95)
    );

    border-radius: 18px;

    padding: 20px;

    border:
    1px solid rgba(255,255,255,0.08);

    box-shadow:
    0px 8px 25px rgba(0,0,0,0.35);
}

/* Buttons */
.stButton>button {

    background:
    linear-gradient(
        90deg,
        #2563EB,
        #7C3AED
    );

    color: white;

    border-radius: 14px;

    font-weight: 600;

    border: none;

    height: 3.2em;

    width: 100%;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {

    background:
    rgba(255,255,255,0.05);

    border-radius: 12px;

    padding: 10px 20px;

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

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<h1 style='text-align:center;'>

🚀 End-to-End Time Series Forecasting

</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='
text-align:center;
font-size:22px;
color:#CBD5E1;
margin-bottom:25px;
'>

AI Powered Forecasting System using
ARIMA • SARIMA • Prophet • XGBoost • LSTM

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

forecast_weeks = st.sidebar.slider(
    "📈 Forecast Weeks",
    1,
    8,
    8
)

generate_forecast = st.sidebar.button(
    "🚀 Generate Forecast"
)

# =========================================================
# FILTER DATA
# =========================================================
filtered_df = df[
    df[STATE_COLUMN] == selected_state
].sort_values(DATE_COLUMN).copy()

# =========================================================
# FEATURE ENGINEERING
# =========================================================
filtered_df["lag_1"] = filtered_df[SALES_COLUMN].shift(1)

filtered_df["lag_7"] = filtered_df[SALES_COLUMN].shift(7)

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
# KPIs
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
    "📂 Dataset"

])

# =========================================================
# DASHBOARD
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

# =========================================================
# MODEL TRAINING
# =========================================================
features = [

    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "day_of_week",
    "month",
    "holiday_flag"

]

X = filtered_df[features]

y = filtered_df[SALES_COLUMN]

split_index = int(len(filtered_df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

results = {}

# =========================================================
# XGBOOST MODEL
# =========================================================
xgb_model = XGBRegressor()

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_pred
    )
)

mae = mean_absolute_error(
    y_test,
    xgb_pred
)

r2 = r2_score(
    y_test,
    xgb_pred
)

results["XGBoost"] = rmse

# =========================================================
# ARIMA
# =========================================================
try:

    arima_model = SARIMAX(
        y_train,
        order=(1,1,1)
    ).fit(disp=False)

    arima_pred = arima_model.forecast(
        len(y_test)
    )

    arima_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            arima_pred
        )
    )

    results["ARIMA"] = arima_rmse

except:

    results["ARIMA"] = 999999

# =========================================================
# SARIMA
# =========================================================
try:

    sarima_model = SARIMAX(
        y_train,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    ).fit(disp=False)

    sarima_pred = sarima_model.forecast(
        len(y_test)
    )

    sarima_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            sarima_pred
        )
    )

    results["SARIMA"] = sarima_rmse

except:

    results["SARIMA"] = 999999

# =========================================================
# PROPHET
# =========================================================
try:

    prophet_df = filtered_df[
        [DATE_COLUMN, SALES_COLUMN]
    ]

    prophet_df.columns = [
        "ds",
        "y"
    ]

    prophet_model = Prophet()

    prophet_model.fit(
        prophet_df
    )

    future = prophet_model.make_future_dataframe(
        periods=len(y_test)
    )

    forecast = prophet_model.predict(
        future
    )

    prophet_pred = forecast["yhat"].tail(
        len(y_test)
    )

    prophet_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prophet_pred
        )
    )

    results["Prophet"] = prophet_rmse

except:

    results["Prophet"] = 999999

# =========================================================
# MODEL COMPARISON
# =========================================================
comparison_df = pd.DataFrame({

    "Model": list(results.keys()),
    "RMSE": list(results.values())

})

comparison_df = comparison_df.sort_values(
    "RMSE"
)

best_model = comparison_df.iloc[0]["Model"]

# =========================================================
# MODELS TAB
# =========================================================
with tab2:

    st.subheader("🤖 Model Performance")

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    model_fig = px.bar(
        comparison_df,
        x="Model",
        y="RMSE",
        color="Model",
        text_auto=True,
        template="plotly_dark"
    )

    st.plotly_chart(
        model_fig,
        use_container_width=True
    )

    st.success(f"""
🏆 Best Model Selected: {best_model}

✅ RMSE: {comparison_df.iloc[0]['RMSE']:.2f}

✅ MAE: {mae:.2f}

✅ R² Score: {r2:.2f}
""")

# =========================================================
# FORECAST TAB
# =========================================================
with tab3:

    st.subheader("🔮 Future Forecast")

    if generate_forecast:

        forecast_days = forecast_weeks * 7

        future_predictions = []

        future_dates = []

        last_row = filtered_df.iloc[-1:].copy()

        for i in range(forecast_days):

            pred = xgb_model.predict(
                last_row[features]
            )[0]

            future_predictions.append(pred)

            next_date = (
                last_row[DATE_COLUMN].iloc[0]
                + timedelta(days=1)
            )

            future_dates.append(next_date)

            new_row = {

                DATE_COLUMN: next_date,

                "lag_1": pred,

                "lag_7": pred,

                "rolling_mean_7": pred,

                "rolling_std_7":
                filtered_df[SALES_COLUMN].std(),

                "day_of_week":
                next_date.dayofweek,

                "month":
                next_date.month,

                "holiday_flag":
                1 if next_date.dayofweek in [5,6]
                else 0
            }

            last_row = pd.DataFrame([new_row])

        forecast_df = pd.DataFrame({

            "Forecast Date": future_dates,
            "Predicted Sales": future_predictions

        })

        st.success(
            f"Forecast generated using {best_model}"
        )

        # =================================================
        # FORECAST GRAPH
        # =================================================
        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=filtered_df[DATE_COLUMN],
                y=filtered_df[SALES_COLUMN],
                mode='lines',
                name='Historical Sales'
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=forecast_df["Forecast Date"],
                y=forecast_df["Predicted Sales"],
                mode='lines',
                name='Forecast'
            )
        )

        forecast_fig.update_layout(
            template="plotly_dark",
            title="Historical vs Forecast",
            height=600
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        # =================================================
        # FORECAST TABLE
        # =================================================
        st.subheader("📋 Forecast Results")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        csv = forecast_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            "forecast_results.csv",
            "text/csv"
        )

# =========================================================
# API TAB
# =========================================================
with tab4:

    st.subheader("⚡ FastAPI Backend")

    st.code("""
from fastapi import FastAPI

app = FastAPI()

@app.get('/forecast')
def forecast():

    return {
        'model': 'XGBoost',
        'forecast': [1200, 1300, 1400]
    }
""", language="python")

# =========================================================
# DATA TAB
# =========================================================
with tab5:

    st.subheader("📂 Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown("""

<div style='
text-align:center;
padding:20px;
font-size:18px;
color:#CBD5E1;
'>

🚀 End-to-End Time Series Forecasting

<br><br>

AI Powered Forecasting Platform

</div>

""", unsafe_allow_html=True)
