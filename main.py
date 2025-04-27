from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model/model.pkl")

class Features(BaseModel):
    features: list[float]



@app.get("/")
def read_root():
    return {"message": "ML Model is live!"}


@app.post("/predict")
def predict(data: Features):
    prediction = model.predict([np.array(data.features)])
    return {"prediction": prediction.tolist()}