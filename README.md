# Credit Card Fraud Detection

## Overview

This project explores credit card fraud detection using anomaly detection and machine learning.

## Objectives

* Explore transaction data and investigate class imbalance.
* Visualise transaction patterns and use PCA for dimensionality reduction.
* Train an Isolation Forest model using legitimate training transactions.
* Evaluate anomaly scores using precision-recall metrics.
* Expose the model through a FastAPI endpoint.
* Provide an interactive Streamlit dashboard.

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download the CSV separately and place it inside the `data/` directory. The raw dataset is not included in this repository.

## Methodology

1. Inspect data quality and explore transaction distributions.
2. Split the dataset into training, validation and test sets.
3. Fit feature scaling on legitimate training transactions.
4. Train an Isolation Forest anomaly detector.
5. Select a decision threshold using validation data.
6. Evaluate the final model on the held-out test set.

## Run Locally

Install the dependencies:

`pip install -r requirements.txt`

Run the API:

`uvicorn app:app --reload`

Run the dashboard:

`streamlit run streamlit_app.py`

## Results

Add the actual validation and test Average Precision, precision, recall, F1-score, and confusion matrix after running the notebook.

## Limitations

An anomaly score does not represent a calibrated probability of fraud. Unusual transactions may be legitimate, and fraudulent transactions may not always be detected. This project is educational and is not a production banking fraud detection system.

## Author

Add your name, GitHub profile, and deployed application link here.
