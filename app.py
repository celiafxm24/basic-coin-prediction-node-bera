# Updated app.py for BERA/USD log-return prediction
from flask import Flask, jsonify
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import joblib

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "data/model_bera.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/features_bera.csv")

@app.route("/inference/BERA", methods=["GET"])
def apiAdapter():
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(FEATURES_PATH)
        X = df.drop(columns=["target_BERAUSDT"])
        y_pred = model.predict(X)
        return jsonify({"log_return_prediction": float(y_pred[-1])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Original BTC endpoint (not deleted)
@app.route("/inference/BTC", methods=["GET"])
def btcAdapter():
    return jsonify({"message": "BTC/USD endpoint is deprecated or unused in BERA context."})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
