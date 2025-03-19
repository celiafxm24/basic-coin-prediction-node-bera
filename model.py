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
# [Previous format_data, load_frame, preprocess_live_data, train_model functions remain unchanged]

def get_inference(token, timeframe, region, data_provider):
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    if data_provider == "coingecko":
        df_btc = download_coingecko_current_day_data("BTC", CG_API_KEY)
        df_eth = download_coingecko_current_day_data("ETH", CG_API_KEY)
    else:
        df_btc = download_binance_current_day_data("BTCUSDT", region)
        df_eth = download_binance_current_day_data("ETHUSDT", region)
    
    ticker_url = f'https://api.binance.{region}/api/v3/ticker/price?symbol=ETHUSDT'
    response = requests.get(ticker_url)
    response.raise_for_status()
    latest_price = float(response.json()['price'])
    
    X_new = preprocess_live_data(df_btc, df_eth)
    log_return_pred = loaded_model.predict(X_new[-1].reshape(1, -1))[0]
    
    # Calculate predicted price for logging purposes, but don't return it
    predicted_price = latest_price * np.exp(log_return_pred)
    
    print(f"Predicted 6h ETH/USD Log Return: {log_return_pred:.6f}")
    print(f"Latest ETH Price: {latest_price:.2f}")
    print(f"Predicted ETH Price in 6h: {predicted_price:.2f}")
    return log_return_pred  # Return log return instead of predicted price
