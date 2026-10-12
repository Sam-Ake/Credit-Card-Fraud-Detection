from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_bundle.joblib"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
threshold = bundle["threshold"]
features = bundle["features"]

app = FastAPI(title="Credit Card Fraud Detection API")


class TransactionRequest(BaseModel):
    transactions: list[dict[str, float]]


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }


@app.post("/detect")
def detect_fraud(request: TransactionRequest):
    if not request.transactions:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one transaction."
        )

    data = pd.DataFrame(request.transactions)

    missing = [col for col in features if col not in data.columns]
    extra = [col for col in data.columns if col not in features]

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_columns": missing,
                "unexpected_columns": extra
            }
        )

    data = data[features]

    try:
        data = data.astype(float)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="All transaction features must be numeric."
        )

    if not np.isfinite(data.to_numpy()).all():
        raise HTTPException(
            status_code=400,
            detail="Transaction data contains missing or non-finite values."
        )

    scaled = scaler.transform(data)
    scores = -model.score_samples(scaled)

    results = []

    for score in scores:
        results.append({
            "anomaly_score": float(score),
            "threshold": float(threshold),
            "flagged_as_suspicious": bool(score >= threshold)
        })

    return {"results": results}