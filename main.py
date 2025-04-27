from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/")
def read_root():
    return {"message": "ML Model is live!"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([np.array(features)])
    return {"prediction": prediction.tolist()}
