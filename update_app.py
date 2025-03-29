# update_app.py - Extended to support BERA log-return feature generation

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Original BTC pipeline logic may exist here (not modified)

# --- BERA log-return feature pipeline start ---

def calculate_log_return(current_price, future_price):
    return np.log(future_price / current_price)

def generate_features_bera(data):
    features = pd.DataFrame()

    for col in ["open", "high", "low", "close"]:
        for i in range(1, 11):
            features[f"{col}_BERAUSDT_lag{i}"] = data[col].shift(i)
    for col in ["open", "high", "low", "close"]:
        for i in range(1, 11):
            features[f"{col}_BTCUSDT_lag{i}"] = data[f"{col}_BTCUSDT"].shift(i)

    features["hour_of_day"] = data["timestamp"].dt.hour

    current = data["close"]
    future = data["close"].shift(-12)  # 12 * 5min = 60min
    features["target_BERAUSDT"] = calculate_log_return(current, future)

    features.dropna(inplace=True)
    return features

def save_features():
    input_path = os.getenv("BERA_SOURCE", "data/raw_bera.csv")
    output_path = os.getenv("FEATURES_PATH", "data/features_bera.csv")

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)

    features = generate_features_bera(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"BERA features saved to {output_path}")

# Entry point for new BERA pipeline
if __name__ == "__main__":
    save_features()

# --- BERA log-return feature pipeline end ---
