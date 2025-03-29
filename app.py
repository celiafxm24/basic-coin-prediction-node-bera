from flask import Flask, request, jsonify
import os
import joblib
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# Use environment variable TOKEN (default to BTC if not provided)
TOKEN = os.environ.get('TOKEN', 'BTC').upper()

# Load the appropriate model based on TOKEN
if TOKEN == 'BERA':
    model = joblib.load('model_bera.pkl')  # New BERA model
else:
    model = joblib.load('model.pkl')         # Original BTC model

@app.route('/inference/<token>', methods=['POST'])
def inference(token):
    token = token.upper()
    data = request.get_json(force=True)
    
    # Convert input JSON into a DataFrame.
    # For BERA predictions, we expect 81 features:
    # 40 features for BERA (open, high, low, close lags 1-10),
    # 40 features for BTC (open, high, low, close lags 1-10), plus 'hour_of_day'
    df = pd.DataFrame([data])
    
    # Create a DMatrix for XGBoost prediction
    dmatrix = xgb.DMatrix(df)
    pred = model.predict(dmatrix)
    
    return jsonify({"prediction": float(pred[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
