# model.py - Model training for 1-hour BERA/USD Log-Return Prediction

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# Constants
DATA_PATH = os.getenv("TRAINING_DATA", "data/training_bera.csv")
MODEL_SAVE_PATH = os.getenv("MODEL_PATH", "data/model_bera.pkl")

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Extract features and target
X = df.drop(columns=["target_BERAUSDT"])
y = df["target_BERAUSDT"]

# Train the model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective="reg:squarederror",
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

# Save the model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(model, MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
