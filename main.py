from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# Initialize the App
app = FastAPI(
    title="Indian Telecom Churn Prediction System",
    description="Predicts retention risk for Jio, Airtel, Vi, and BSNL",
    version="1.0"
)

# Load the trained Pipeline
# This contains both the data cleaning logic and the ML model
model = joblib.load('india_churn_model.joblib')

# Define the Input Format (The "Bouncer")
class SubscriberData(BaseModel):
    telecom_partner: str  # e.g. "Reliance Jio"
    state: str            # e.g. "Maharashtra"
    age: int
    gender: str           # "M" or "F"
    data_used: int        # GB per month
    calls_made: int       # Minutes
    sms_sent: int

@app.post("/predict_retention")
def predict_churn(data: SubscriberData):
    # 1. Convert JSON input to DataFrame
    input_df = pd.DataFrame([data.model_dump()])
    
    # 2. Get Prediction (0 or 1)
    prediction = model.predict(input_df)[0]
    
    # 3. Get Probability (Confidence Score, e.g., 0.85)
    probability = model.predict_proba(input_df)[0][1]
    
    # 4. Business Logic (Retention Strategy)
    status = "Safe"
    action = "No Action"
    
    if probability > 0.8:
        status = "CRITICAL RISK"
        action = "Call immediately & Offer Free 5G Upgrade"
    elif probability > 0.5:
        status = "High Risk"
        action = "Send SMS with 15% Recharge Discount"
        
    # 5. Return JSON Response
    return {
        "churn_prediction": int(prediction),
        "risk_score": round(probability, 2),
        "status": status,
        "recommended_action": action
    }

# Health Check Endpoint
@app.get("/")
def home():
    return {"message": "Telecom Churn API is Online"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)