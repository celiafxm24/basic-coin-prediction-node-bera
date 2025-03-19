import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, MODEL

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")
scaler_file_path = os.path.join(data_base_path, "scaler.pkl")

# [Previous download_data_binance, download_data_coingecko, download_data functions remain unchanged]

def format_data(files_btc, files_eth, data_provider):
    # [Previous initial checks and file filtering remain unchanged until data processing]
    
    # [Previous BTC processing loop remains largely unchanged]
    # [Previous ETH processing loop remains largely unchanged]

    if price_df_btc.empty or price_df_eth.empty:
        print("No data processed for BTCUSDT or ETHUSDT")
        print(f"BTC DataFrame rows: {len(price_df_btc)}, ETH DataFrame rows: {len(price_df_eth)}")
        return

    print(f"Skipped files due to errors: {skipped_files}")
    
    price_df_btc = price_df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
    price_df_eth = price_df_eth.rename(columns=lambda x: f"{x}_ETHUSDT")
    price_df = pd.concat([price_df_btc, price_df_eth], axis=1)

    if TIMEFRAME != "1m":
        price_df = price_df.resample(TIMEFRAME).agg({
            f"{metric}_{pair}": "last" 
            for pair in ["ETHUSDT", "BTCUSDT"] 
            for metric in ["open", "high", "low", "close"]
        })

    # Modified: Calculate log returns instead of price change
    for pair in ["ETHUSDT", "BTCUSDT"]:
        price_df[f"log_return_{pair}"] = np.log(price_df[f"close_{pair}"].shift(-1) / price_df[f"close_{pair}"])
        for metric in ["open", "high", "low", "close"]:
            for lag in range(1, 11):
                price_df[f"{metric}_{pair}_lag{lag}"] = price_df[f"{metric}_{pair}"].shift(lag)

    price_df["hour_of_day"] = price_df.index.hour
    price_df["target_ETHUSDT"] = price_df["log_return_ETHUSDT"]  # Modified: Use log return as target
    price_df = price_df.dropna()
    
    if len(price_df) == 0:
        print("No data remains after preprocessing. Check data availability or timeframe.")
        return

    price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Data saved to {training_price_data_path}")

def load_frame(file_path, timeframe):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file {file_path} does not exist. Run data update first.")
    
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    if df.empty:
        raise ValueError(f"Training data file {file_path} is empty.")
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    features = [
        f"{metric}_{pair}_lag{lag}" 
        for pair in ["ETHUSDT", "BTCUSDT"]
        for metric in ["open", "high", "low", "close"]
        for lag in range(1, 11)
    ] + ["hour_of_day"]
    
    X = df[features]
    y = df["target_ETHUSDT"]  # Now this is log returns
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(len(X) * 0.8)
    if split_idx == 0:
        raise ValueError("Not enough data to split into train and test sets.")
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

# [preprocess_live_data function remains largely unchanged]

def train_model(timeframe, file_path=training_price_data_path):
    # [Previous code remains unchanged until after model training]
    
    # Note: The model now predicts log returns instead of price changes
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Convert predictions back to price space for metrics (optional, for interpretation)
    train_mae = mean_absolute_error(y_train, train_pred)  # MAE in log return space
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    print(f"Training MAE (log returns): {train_mae:.6f}")
    print(f"Training RMSE (log returns): {train_rmse:.6f}")
    print(f"Training R²: {r2:.6f}")

    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    r2 = r2_score(y_test, test_pred)
    print(f"Test MAE (log returns): {mae:.6f}")
    print(f"Test RMSE (log returns): {rmse:.6f}")
    print(f"Test R²: {r2:.6f}")
    
    # [Model saving code remains unchanged]
    return model, scaler

def get_inference(token, timeframe, region, data_provider):
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    # [Data download code remains unchanged]
    
    ticker_url = f'https://api.binance.{region}/api/v3/ticker/price?symbol=ETHUSDT'
    response = requests.get(ticker_url)
    response.raise_for_status()
    latest_price = float(response.json()['price'])
    
    X_new = preprocess_live_data(df_btc, df_eth)
    log_return_pred = loaded_model.predict(X_new[-1].reshape(1, -1))[0]
    
    # Modified: Convert log return prediction to absolute price
    predicted_price = latest_price * np.exp(log_return_pred)
    
    print(f"Predicted 6h ETH/USD Log Return: {log_return_pred:.6f}")
    print(f"Latest ETH Price: {latest_price:.2f}")
    print(f"Predicted ETH Price in 6h: {predicted_price:.2f}")
    return predicted_price

# [Rest of the code remains unchanged]
