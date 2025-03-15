import json
import os
import time
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER

app = Flask(__name__)

def update_data():
    print("Starting data update process...")
    # Clear all data to force refresh
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
        else:
            print(f"Path not found, no need to clear: {path}")
    
    print("Downloading BTC data...")
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    if not files_btc:
        raise RuntimeError("Failed to download BTC data.")
    print(f"Downloaded {len(files_btc)} BTC files.")
    
    print("Downloading ETH data...")
    files_eth = download_data("ETH", TRAINING_DAYS, REGION, DATA_PROVIDER)
    if not files_eth:
        raise RuntimeError("Failed to download ETH data.")
    print(f"Downloaded {len(files_eth)} ETH files.")
    
    print("Formatting data...")
    format_data(files_btc, files_eth, DATA_PROVIDER)
    if not os.path.exists(price_data_file):
        raise RuntimeError("Failed to format data: price_data.csv not created.")
    print("Data formatted successfully.")
    
    print("Training model...")
    train_model(TIMEFRAME)
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        raise RuntimeError("Training failed: model.pkl or scaler.pkl not created.")
    print("Data update and training completed.")

@app.route("/inference/<string:token>")
def generate_inference(token):
    if not token or token.upper() != TOKEN:
        error_msg = "Token is required" if not token else f"Token {token} not supported, expected {TOKEN}"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')
    try:
        if not os.path.exists(model_file_path):
            raise FileNotFoundError("Model file not found. Please run update first to train the model.")
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(str(inference), status=200, mimetype='text/plain')
    except Exception as e:
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
    # Wait until training completes
    while not (os.path.exists(model_file_path) and os.path.exists(scaler_file_path)):
        print("Waiting for model and scaler files to be generated...")
        time.sleep(5)
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
