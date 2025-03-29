# Done for log return 
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
    
    print("Downloading BTC data...")
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    print("Downloading BERA data...")
    files_bera = download_data("BERA", TRAINING_DAYS, REGION, DATA_PROVIDER)
    if not files_btc or not files_bera:
        print("Warning: No data files downloaded for one or both pairs. Attempting to proceed with available data.")
    print(f"Files downloaded - BTC: {len(files_btc)}, BERA: {len(files_bera)}")  # Added for debug
    print("Formatting data...")
    format_data(files_btc, files_bera, DATA_PROVIDER)
    print("Training model...")
    train_model(TIMEFRAME)
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
    # Wait briefly to ensure training completes before starting Flask
    while not os.path.exists(model_file_path) or not os.path.exists(scaler_file_path):
        print("Waiting for model and scaler files to be generated...")
        time.sleep(5)
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
