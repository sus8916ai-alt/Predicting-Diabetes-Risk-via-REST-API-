from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="Predicts diabetes risk using a trained Machine Learning model",
    version="1.0"
)

# Load trained model
model = joblib.load("diabetes_model.pkl")


# Input Schema
class HealthData(BaseModel):
    BMI: float
    Age: int
    BloodPressure: float
    Glucose: float


# Home Route
@app.get("/")
def home():
    return {
        "message": "Diabetes Risk Prediction API is running",
        "docs": "/docs"
    }


# Health Check Route
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Prediction Route
@app.post("/predict")
def predict(data: HealthData):

    # Convert input into model format
    features = np.array([
        [
            data.BMI,
            data.Age,
            data.BloodPressure,
            data.Glucose
        ]
    ])

    # Make predictions
    prediction = int(model.predict(features)[0])

    # Get probability if supported by model
    try:
        probability = float(model.predict_proba(features)[0][1])
    except AttributeError:
        probability = None

    return {
        "risk_class": prediction,
        "risk_probability": probability,
        "message": "High Risk" if prediction == 1 else "Low Risk"
    }
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np

# # Load trained model
# model = joblib.load("diabetes_model.pkl")

# # Define input schema
# class HealthData(BaseModel):
#     BMI: float
#     Age: int
#     BloodPressure: float
#     Glucose: float

# app = FastAPI()

# @app.post("/predict")
# def predict(data: HealthData):
#     # Convert input to numpy array
#     features = np.array([[data.BMI, data.Age, data.BloodPressure, data.Glucose]])
    
#     # Predict probability
#     prob = model.predict_proba(features)[0][1]  # probability of diabetes
#     prediction = int(model.predict(features)[0])  # 0 or 1
    
#     return {
#         "risk_probability": prob,
#         "risk_class": prediction
#     }
