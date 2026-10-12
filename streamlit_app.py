from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_bundle.joblib"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("Credit Card Fraud Detection Dashboard")
st.write(
    "Upload a CSV containing the transaction features "
    "required by the trained model."
)

if not MODEL_PATH.exists():
    st.error("Model bundle not found. Train and save the model first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
default_threshold = float(bundle["threshold"])

threshold = st.slider(
    "Fraud detection threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(np.clip(default_threshold, 0.0, 1.0)),
    step=0.01
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    missing = [col for col in features if col not in data.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    if len(data) > 100000:
        st.error("Please upload no more than 100,000 rows at a time.")
        st.stop()

    original_data = data.copy()
    X_input = data[features].copy()

    try:
        X_input = X_input.astype(float)
    except (ValueError, TypeError):
        st.error("All model features must be numeric.")
        st.stop()

    if not np.isfinite(X_input.to_numpy()).all():
        st.error("The data contains missing or non-finite values.")
        st.stop()

    scaled = scaler.transform(X_input)
    scores = -model.score_samples(scaled)

    results = original_data.copy()
    results["Anomaly Score"] = scores
    results["Flagged"] = scores >= threshold

    total = len(results)
    flagged = int(results["Flagged"].sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Transactions analysed", f"{total:,}")
    col2.metric("Flagged transactions", f"{flagged:,}")
    col3.metric(
        "Flagged percentage",
        f"{(flagged / total * 100) if total else 0:.2f}%"
    )

    st.subheader("Anomaly Score Distribution")

    score_chart = px.histogram(
        results,
        x="Anomaly Score",
        color="Flagged",
        nbins=50,
        title="Distribution of Transaction Anomaly Scores"
    )

    st.plotly_chart(score_chart, use_container_width=True)

    st.subheader("Transaction Results")

    st.dataframe(results, use_container_width=True)

    flagged_rows = results[results["Flagged"]]

    st.subheader("Flagged Transactions")
    st.dataframe(flagged_rows, use_container_width=True)

    csv_output = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results CSV",
        data=csv_output,
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )