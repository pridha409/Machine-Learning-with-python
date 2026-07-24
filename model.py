import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class FraudDetector:
    def __init__(self, model_path="models/"):
        self.model = joblib.load(Path(model_path) / "fraud_model.pkl")
        self.scaler = joblib.load(Path(model_path) / "scaler.pkl")
        self.feature_columns = joblib.load(Path(model_path) / "feature_columns.pkl")
    
    def preprocess(self, raw_dict: dict) -> np.ndarray:
        df = pd.DataFrame([raw_dict])
        df = df.reindex(columns=self.feature_columns, fill_value=0)
        df = df.fillna(0)
        return self.scaler.transform(df)
    
    def predict(self, transaction: dict) -> dict:
        X = self.preprocess(transaction)
        prob = self.model.predict_proba(X)[0][1]
        pred = int(prob >= 0.5)
        return {
            "fraud_probability": round(prob, 4),
            "prediction": "FRAUD" if pred == 1 else "NORMAL",
            "alert_triggered": pred == 1
        }