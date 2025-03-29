import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

def calculate_log_return(current_price, new_price):
    """
    Calculate Log-Return = ln(new_price/current_price)
    """
    return np.log(new_price / current_price)

# ---------------------------
# Original BTC model training
def train_btc_model(data_path, model_save_path):
    # Placeholder for the original BTC volatility prediction training code.
    # (Original source code remains unchanged.)
    pass

# ---------------------------
# New function: BERA/USD Log-Return Prediction training
def train_bera_model(data_path, model_save_path):
    """
    Trains an XGBoost model using 81 input features and the target BERA/USD log-return.
    Expected features:
      - open_BERAUSDT_lag1 ... open_BERAUSDT_lag10
      - high_BERAUSDT_lag1 ... high_BERAUSDT_lag10
      - low_BERAUSDT_lag1 ... low_BERAUSDT_lag10
      - close_BERAUSDT_lag1 ... close_BERAUSDT_lag10
      - open_BTCUSDT_lag1 ... open_BTCUSDT_lag10
      - high_BTCUSDT_lag1 ... high_BTCUSDT_lag10
      - low_BTCUSDT_lag1 ... low_BTCUSDT_lag10
      - close_BTCUSDT_lag1 ... close_BTCUSDT_lag10
      - hour_of_day
    The target variable 'target_BERAUSDT' is computed as:
        ln(new_price/current_price)
    where current_price and new_price are assumed to be available in the dataset
    (if not, they should be provided or computed beforehand).
    """
    # Load dataset
    data = pd.read_csv(data_path)
    
    # If target_BERAUSDT is not present, calculate it using provided price columns.
    # Assumes columns 'current_price_BERA' and 'new_price_BERA' exist.
    if 'target_BERAUSDT' not in data.columns:
        data['target_BERAUSDT'] = calculate_log_return(data['current_price_BERA'], data['new_price_BERA'])
    
    # Construct feature columns
    bera_features = [f"{metric}_BERAUSDT_lag{lag}" for metric in ['open', 'high', 'low', 'close'] for lag in range(1, 11)]
    btc_features = [f"{metric}_BTCUSDT_lag{lag}" for metric in ['open', 'high', 'low', 'close'] for lag in range(1, 11)]
    feature_columns = bera_features + btc_features + ['hour_of_day']
    
    X = data[feature_columns]
    y = data['target_BERAUSDT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }
    num_round = 100
    bst = xgb.train(params, dtrain, num_round, evals=[(dtest, 'eval')])
    
    # Save the trained model
    joblib.dump(bst, model_save_path)

if __name__ == '__main__':
    # Uncomment one of the lines below to train the desired model.
    # For BTC prediction:
    # train_btc_model('btc_training_data.csv', 'model.pkl')
    
    # For BERA prediction (1-hour Log-Return prediction):
    train_bera_model('bera_training_data.csv', 'model_bera.pkl')
