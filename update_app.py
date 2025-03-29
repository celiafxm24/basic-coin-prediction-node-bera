# update_app.py - Updated for 1-hour BERA/USD log-return prediction

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def calculate_log_return(current_price, future_price):
    return np.log(future_price / current_price)

def generate_features_bera(data):
    # Resample to 1-hour intervals
    data_1h = data.resample("1h", on="timestamp").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    features = pd.DataFrame(index=data_1h.index)
    
    # Generate lagged features for BERA/USD
    for col in ["open", "high", "low", "close"]:
        for i in range(1, 11):
            features[f"{col}_BERAUSDT_lag{i}"] = data_1h[col].shift(i)
    
    # Add hour of day as a feature
    features["hour_of_day"] = data_1h.index.hour
    
    # Target: 1-hour log-return
    current = data_1h["close"]
    future = data_1h["close"].shift(-1)  # Next 1-hour period
    features["target_BERAUSDT"] = calculate_log_return(current, future)
    
    # Drop rows with NaN values
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

# Entry point
if __name__ == "__main__":
    save_features()
