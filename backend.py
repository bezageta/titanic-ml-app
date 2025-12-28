
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load models
dt_model = joblib.load('decision_tree_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')

app = FastAPI()

# Define input features
class Passenger(BaseModel):
    pclass: int
    age: float
    sibsp: int
    parch: int
    fare: float
    sex_male: int
    embarked_Q: int
    embarked_S: int

@app.post("/predict")
def predict(passenger: Passenger):
    features = [[
        passenger.pclass, passenger.age, passenger.sibsp,
        passenger.parch, passenger.fare,
        passenger.sex_male, passenger.embarked_Q, passenger.embarked_S
    ]]
    
    dt_prediction = dt_model.predict(features)[0]
    lr_prediction = lr_model.predict(features)[0]
    
    return {
        "Decision_Tree_Prediction": int(dt_prediction),
        "Logistic_Regression_Prediction": int(lr_prediction)
    }
