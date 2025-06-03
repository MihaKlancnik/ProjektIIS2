import os
import sys
import joblib
import pandas as pd
from train_models import load_data, load_fear_greed, create_features

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def run_model(crypto_name, use_fng=True):
    print(f"\nRunning model for {crypto_name} (use_fng={use_fng})")

    # Load data
    df = load_data(crypto_name)

    if use_fng:
        fng_df = load_fear_greed()
        df = pd.merge(df, fng_df[['date', 'fng_value']], on='date', how='left')
        df['fng_value'] = df['fng_value'].fillna(method='ffill').fillna(method='bfill').fillna(df['fng_value'].mean())

    # Create features
    df_features = create_features(df)

    # Use only the latest row of features
    latest_row = df_features.iloc[-1:]

    # Load saved model files
    suffix = "_with_fng" if use_fng else "_nofng"
    model_path = f"models/{crypto_name.lower()}_model{suffix}.pkl"
    scaler_path = f"models/{crypto_name.lower()}_scaler{suffix}.pkl"
    imputer_path = f"models/{crypto_name.lower()}_imputer{suffix}.pkl"
    features_path = f"models/{crypto_name.lower()}_features{suffix}.txt"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, imputer_path, features_path]):
        print("❌ Missing model files. Please train models first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)

    with open(features_path, "r") as f:
        feature_cols = f.read().splitlines()

    # Prepare input
    X = latest_row[feature_cols]
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    # Predict
    prediction = model.predict(X_scaled)[0]  # 1D array of 5 predictions
    for i, price in enumerate(prediction, start=1):
        print(f"Prediction for +{i} hour(s): {price:.2f}")

if __name__ == "__main__":
    # Example usage:
    run_model("bitcoin", use_fng=True)
    run_model("ethereum", use_fng=True)
    run_model("solana", use_fng=True)
