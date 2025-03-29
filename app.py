# app.py - Updated for 1-hour BERA/USD log-return prediction

from flask import Flask, jsonify
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import joblib

app = Flask(__name__)

# BERA model configuration
MODEL_PATH_BERA = os.getenv("MODEL_PATH", "data/model_bera.pkl")
FEATURES_PATH_BERA = os.getenv("FEATURES_PATH", "data/features_bera.csv")

@app.route("/inference/BERA", methods=["GET"])
def apiAdapter():
    try:
        model = joblib.load(MODEL_PATH_BERA)
        df = pd.read_csv(FEATURES_PATH_BERA)
        X = df.drop(columns=["target_BERAUSDT"])
        y_pred = model.predict(X)
        return jsonify({"log_return_prediction": float(y_pred[-1])})  # Return latest 1-hour log-return prediction
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Original BTC endpoint preserved as deprecated
@app.route("/inference/BTC", methods=["GET"])
def btcAdapter():
    return jsonify({"message": "BTC/USD endpoint is deprecated or unused in BERA context."})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
