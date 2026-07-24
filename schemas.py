from pydantic import BaseModel

class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str
    alert_triggered: bool