import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("best_model.pkl")

st.title("Forecast Prediction App")

st.write("Enter input values")

lag_1 = st.number_input("lag_1", value=200)
lag_7 = st.number_input("lag_7", value=180)
lag_30 = st.number_input("lag_30", value=150)
rolling_mean_7 = st.number_input("rolling_mean_7", value=190)
rolling_std_7 = st.number_input("rolling_std_7", value=12)
day_of_week = st.number_input("day_of_week", value=2)
month = st.number_input("month", value=5)
holiday_flag = st.number_input("holiday_flag", value=0)

if st.button("Predict"):

    sample_data = pd.DataFrame({
        'lag_1': [lag_1],
        'lag_7': [lag_7],
        'lag_30': [lag_30],
        'rolling_mean_7': [rolling_mean_7],
        'rolling_std_7': [rolling_std_7],
        'day_of_week': [day_of_week],
        'month': [month],
        'holiday_flag': [holiday_flag]
    })

    prediction = model.predict(sample_data)

    st.success(f"Forecast Prediction: {prediction[0]}")
