from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

from src.model.trainer import ModelTrainer
import logging

app = FastAPI(title="ML Model Serving API")
prediction_counter = Counter('model_predictions_total', 'Total number of predictions made')


class PredictionRequest(BaseModel):
    text: str


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest().decode('utf-8'), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        model = ModelTrainer.load_model()
        prediction = model.predict([request.text])
        prediction_counter.inc()
        
        return {
            "prediction": prediction.tolist()[0],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))