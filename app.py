from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("best_model.pkl")

@app.get("/")
def home():

    return {
        "message": "Forecast API Running"
    }

@app.get("/forecast")
def forecast():

    sample_data = pd.DataFrame({

        'lag_1': [200],
        'lag_7': [180],
        'lag_30': [150],
        'rolling_mean_7': [190],
        'rolling_std_7': [12],
        'day_of_week': [2],
        'month': [5],
        'holiday_flag': [0]

    })

    prediction = model.predict(sample_data)

    return {
        "forecast": prediction.tolist()
    }