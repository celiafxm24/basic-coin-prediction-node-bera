import json
import os
import time
import numpy as np
import pandas as pd
from flask import Flask, Response
import joblib
from model import download_data, format_data, train_model, get_inference, format_live_data
from config import model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER

app = Flask(__name__)

def update_data():
    print("Starting data update process...")
    data_dir = os.path.join(os.getcwd(), "data", "binance")
    price_data_file = os.path.join(os.getcwd(), "data", "price_data.csv")
    model_file = model_file_path
    scaler_file = scaler_file_path
    for path in [data_dir, price_data_file, model_file, scaler_file]:
        if os.path.exists(path):
            if os.path.isdir(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
            else:
                os.remove(path)
            print(f"Cleared {path}")
    
    print("Downloading BTC data...")
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    print("Downloading BERA data...")
    files_bera = download_data("BERA", TRAINING_DAYS, REGION, DATA_PROVIDER)
    if not files_btc or not files_bera:
        print("No data files downloaded. Skipping format_data and training.")
        return
    print("Formatting data...")
    format_data(files_btc, files_bera, DATA_PROVIDER)
    print("Training model...")
    train_model(TIMEFRAME)
    print("Data update and training completed.")

@app.route("/inference/<string:token>")
def generate_inference(token):
    if not token or token.upper() != TOKEN:
        error_msg = "Token is required" if not token else f"Token {token} not supported, expected {TOKEN}"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')
    try:
        if not os.path.exists(model_file_path):
            raise FileNotFoundError("Model file not found. Please run update first to train the model.")
        
        # Load model and scaler
        model = joblib.load(model_file_path)
        scaler = joblib.load(scaler_file_path)

        # Get and format live data
        live_data = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        live_df = format_live_data(live_data)
        
        # Prepare features
        print(f"live_df shape: {live_df.shape}")
        live_X = live_df.drop(columns=["target_BERAUSDT"])
        print(f"live_X shape: {live_X.shape}")
        live_X_scaled = scaler.transform(live_X)
        print(f"live_X_scaled shape: {live_X_scaled.shape}")

        # Predict on the latest row and convert to Python float
        log_return_prediction = float(model.predict(live_X_scaled[-1].reshape(1, -1))[0])
        
        # Calculate predicted price
        latest_price = live_df["close_BERAUSDT"].iloc[-1]
        predicted_price = latest_price * np.exp(log_return_prediction)

        # Log with 3 decimal places
        print(f"Predicted 1h BERA/USD Log Return: {log_return_prediction:.6f}")
        print(f"Latest BERA Price: {latest_price:.3f}")
        print(f"Predicted BERA Price in 1h: {predicted_price:.3f}")

        # Return JSON response
        return Response(json.dumps({"log_return_prediction": log_return_prediction}), status=200, mimetype='application/json')
    except Exception as e:
        print(f"Inference error: {str(e)}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    try:
        update_data()
        return "0"
    except Exception as e:
        print(f"Update failed: {str(e)}")
        return "1"

if __name__ == "__main__":
    update_data()
    while not os.path.exists(model_file_path) or not os.path.exists(scaler_file_path):
        print("Waiting for model and scaler files to be generated...")
        time.sleep(5)
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
