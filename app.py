import joblib
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel


# ==========================================
# Load Trained Model
# ==========================================

MODEL_PATH = Path("models/bank_marketing_model.pkl")

model = joblib.load(MODEL_PATH)


# ==========================================
# Create FastAPI App
# ==========================================

app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for predicting whether a customer will subscribe to a term deposit.",
    version="1.0.0"
)


# ==========================================
# Input Data Model
# ==========================================

class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str


# ==========================================
# Home Endpoint
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Bank Marketing Prediction API is running",
        "status": "success"
    }


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
def predict(data: CustomerData):

    input_data = [[
        data.age,
        data.job,
        data.marital,
        data.education,
        data.default,
        data.balance,
        data.housing,
        data.loan,
        data.contact,
        data.day,
        data.month,
        data.duration,
        data.campaign,
        data.pdays,
        data.previous,
        data.poutcome
    ]]

    columns = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome"
    ]

    import pandas as pd

    input_df = pd.DataFrame(
        input_data,
        columns=columns
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        result = "yes"
    else:
        result = "no"

    return {
        "prediction": int(prediction),
        "subscription": result
    }