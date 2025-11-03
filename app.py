import pickle
from Schema.pydentic_model import UserInput
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import pandas as pd

with open("Model/model.pkl","rb") as f:
    models = pickle.load(f)

std = models['Standard_Scaler']
model = models['Logistic_Regression']

app = FastAPI(title="Diabetes Prediction System using Logistic Regression.")

@app.get("/")
def default():
    return {"message":"Welcome to Diabetes Prediction System using Logistic Regression projects Fast APIs end Points","For Prediction the Diabetes using FAST APIS":"Copy This Given url into your browser tab.'http://127.0.0.1:8000/docs'"}

@app.post("/predict")
def prediction(predict:UserInput):
    df = pd.DataFrame([{
        'Pregnancies':predict.Pregnancies,
        'Glucose':predict.Glucose,
        'BloodPressure':predict.BloodPressure,
        'SkinThickness':predict.SkinThickness,
        'Insulin':predict.Insulin,
        'BMI':predict.BMI,
        'DiabetesPedigreeFunction':predict.DiabetesPedigreeFunction,
        'Age':predict.Age
    }])

    df = std.transform(df)

    prediction_values = model.predict(df)

    temp = "You have a risk of Diabetes." if prediction_values == 1 else "You don't have Diabetes."

    return JSONResponse(status_code=200,content={
        "Prediction Answer is:":f"{temp}"
    })