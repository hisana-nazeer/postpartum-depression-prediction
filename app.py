from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def train_model():
    df = pd.read_csv('ppd_clean.csv')
    df = df.drop(columns=['PPD', 'total_score'])
    
    target_cols = ['Feeling sad or Tearful', 'Feeling anxious', 'Feeling of guilt']
    df['total_score'] = df[target_cols].sum(axis=1)
    df['PPD'] = (df['total_score'] >= 4).astype(int)
    
    X = df.drop(columns=['PPD', 'total_score', 'Feeling sad or Tearful',
                          'Feeling anxious', 'Feeling of guilt'])
    y = df['PPD']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    return model, scaler

model, scaler = train_model()

class PatientData(BaseModel):
    Age: float
    Irritable_towards_baby_and_partner: int
    Trouble_sleeping_at_night: int
    Problems_concentrating_or_making_decision: int
    Overeating_or_loss_of_appetite: int
    Overeating_or_loss_of_appetite: int
    Problems_of_bonding_with_baby: int
    Suicide_attempt: int

@app.get("/")
def home():
    return {"message": "PPD Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[
        data.Age,
        data.Irritable_towards_baby_and_partner,
        data.Trouble_sleeping_at_night,
        data.Problems_concentrating_or_making_decision,
        data.Overeating_or_loss_of_appetite,
        data.Problems_of_bonding_with_baby,
        data.Suicide_attempt
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Clinical override
    if data.Suicide_attempt == 1:
        prediction = 1
        probability = max(float(probability), 0.90)
    
    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 3),
        "result": "PPD Positive" if prediction == 1 else "PPD Negative"
    }
