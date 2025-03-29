import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify
import joblib  # Added import for joblib
from data import fetch_data, fetch_live_data
from model import format_data, format_live_data, train_model

app = Flask(__name__)

binance_data_path = "./data/binance"
training_price_data_path = "./data/price_data.csv"
model_path = "./data/model_{coin}.pkl"
scaler_path = "./data/scaler.pkl"
DATA_PROVIDER = "binance"
TIMEFRAME = "1h"

os.makedirs("./data", exist_ok=True)
os.makedirs(binance_data_path, exist_ok=True)

def update_data():
    print("Starting data update process...")
    files_btc, files_bera = fetch_data(binance_data_path)
    format_data(files_btc, files_bera, DATA_PROVIDER)
    train_model(training_price_data_path, model_path.format(coin="bera"), scaler_path)
    print("Data update and training completed.")

@app.route("/inference/<coin>", methods=["GET"])
def inference(coin):
    if coin not in ["BTC", "BERA"]:
        return jsonify({"error": "Unsupported coin"}), 400

    model = joblib.load(model_path.format(coin=coin.lower()))
    scaler = joblib.load(scaler_path)

    live_data = fetch_live_data(["BTCUSDT", "BERAUSDT"])
    live_df = format_live_data(live_data)

    live_X = live_df.drop(columns=["target_BERAUSDT"])
    live_X_scaled = scaler.transform(live_X)
    log_return_prediction = model.predict(live_X_scaled)[-1]

    latest_price = live_df["close_BERAUSDT"].iloc[-1]
    predicted_price = latest_price * np.exp(log_return_prediction)

    print(f"Predicted 1h BERA/USD Log Return: {log_return_prediction:.6f}")
    print(f"Latest BERA Price: {latest_price:.3f}")
    print(f"Predicted BERA Price in 1h: {predicted_price:.3f}")

    return jsonify({"log_return_prediction": log_return_prediction})

if __name__ == "__main__":
    update_data()
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
