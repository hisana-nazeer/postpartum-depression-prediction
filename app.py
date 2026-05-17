from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

#load model and scaler

model= joblib.load('ppd_model.pkl')
scaler=joblib.load('ppd_scaler.pkl')

#define app

app= FastAPI()



# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



#defien input structure

class PatientData(BaseModel):
    Age:float
    Irritable_towards_baby_and_partner : int
    Trouble_sleeping_at_night:int
    Problems_concentrating_or_making_decision: int
    Overeating_or_loss_of_appetite: int
    Problems_of_bonding_with_baby: int
    Suicide_attempt: int
        
      
#define prediction end point

@app.get("/")
def home():
    return {"message":"PPD Prediction API is running"}

@app.post("/predict")
def predict(data:PatientData):
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
    
    return {
        "prediction":int(prediction),
        "probability":round(float(probability), 3),
        "result": "PPD Positive" if prediction == 1 else "PPD Negative"
    }


#1. class PatientData(BaseModel)
#BaseModel comes from a library called Pydantic.
#It automatically:

#Validates incoming data types
#Rejects wrong inputs with clear errors
#Converts JSON from the request into a Python object

#So when someone sends this to your API:
#json{"Age": 37.5, "Irritable_towards_baby_and_partner": 2}
#Pydantic converts it into a PatientData object your function can use.
#Think of it as a form validator — ensures data is correct before reaching your model.
    
    
                

                   