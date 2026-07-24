from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.model import FraudDetector
from app.schemas import PredictionResponse
from app.logger import log_prediction
import time

app = FastAPI(title="Fraud Detection API")
detector = FraudDetector()

@app.get("/", response_class=HTMLResponse)
async def form():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            input, button { margin: 8px 0; padding: 8px; }
            button { background: #007bff; color: white; border: none; cursor: pointer; }
            .result { margin-top: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Ethereum Fraud Detector</h1>
        <form id="txForm">
            <label>Sent transactions:</label>
            <input type="number" name="Sent_tnx" value="10"><br>
            <label>Received transactions:</label>
            <input type="number" name="Received_Tnx" value="5"><br>
            <label>Avg minutes between sent tx:</label>
            <input type="number" name="Avg_min_between_sent_tnx" value="100"><br>
            <button type="submit">Check Fraud</button>
        </form>
        <div class="result" id="result"></div>
        <script>
            const form = document.getElementById('txForm');
            form.onsubmit = async (e) => {
                e.preventDefault();
                const data = Object.fromEntries(new FormData(form).entries());
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await res.json();
                document.getElementById('result').innerHTML = 
                    `Fraud probability: ${result.fraud_probability} – ${result.prediction} (Alert: ${result.alert_triggered})`;
            };
        </script>
    </body>
    </html>
    """)

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: dict):
    try:
        start = time.time()
        result = detector.predict(transaction)
        latency = (time.time() - start) * 1000
        log_prediction(
            address=transaction.get("Address", "web_form"),
            probability=result["fraud_probability"],
            prediction=result["prediction"],
            latency_ms=latency
        )
        print(f"Prediction done in {latency:.2f} ms")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}