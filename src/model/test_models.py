import os
import sys
import joblib
import pandas as pd
import numpy as np  # Added numpy
import csv
import datetime
from train_models import load_data, load_fear_greed, create_features
from tensorflow.keras.models import Sequential, load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def save_predictions_to_csv(crypto_name, predictions, use_fng):
    suffix = "_with_fng" if use_fng else "_nofng"
    file_path = f"predictions/{crypto_name.lower()}_prediction{suffix}.csv"


    os.makedirs("predictions", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [[timestamp, predictions[0]]] 

    file_exists = os.path.exists(file_path)
    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "price"])
        writer.writerows(rows)

def run_model(crypto_name, use_fng=True):
    print(f"\nRunning model for {crypto_name} (use_fng={use_fng})")

    SEQUENCE_LENGTH = 24 


    df = load_data(crypto_name)

    if use_fng:
        fng_df = load_fear_greed()
        fng_df['date'] = pd.to_datetime(fng_df['date'], format='%d-%m-%Y')
        df = pd.merge(df, fng_df[['date', 'fng_value']], on='date', how='left')
        df['fng_value'] = df['fng_value'].ffill().bfill().fillna(df['fng_value'].mean())


    df_features = create_features(df)

    if df_features.empty:
        print(f"❌ Feature dataframe is empty for {crypto_name}. Skipping testing.")
        return
    
    if len(df_features) < SEQUENCE_LENGTH:
        print(f"❌ Not enough data in df_features to form a sequence of length {SEQUENCE_LENGTH}. Available: {len(df_features)}. Skipping testing for {crypto_name}.")
        return


    suffix = "_with_fng" if use_fng else "_nofng"
    model_path = f"models/{crypto_name.lower()}_model{suffix}.h5"
    feature_scaler_path = f"models/{crypto_name.lower()}_feature_scaler{suffix}.pkl"
    target_scaler_path = f"models/{crypto_name.lower()}_target_scaler{suffix}.pkl"
    imputer_path = f"models/{crypto_name.lower()}_imputer{suffix}.pkl"
    features_path = f"models/{crypto_name.lower()}_features{suffix}.txt"

    required_files = [model_path, feature_scaler_path, target_scaler_path, imputer_path, features_path]
    if not all(os.path.exists(p) for p in required_files):
        print(f"❌ Missing one or more model files for {crypto_name}{suffix}. Please train models first.")
        for p in required_files:
            if not os.path.exists(p):
                print(f"   Missing: {p}")
        return

    model = load_model(model_path)
    feature_scaler = joblib.load(feature_scaler_path) 
    target_scaler = joblib.load(target_scaler_path)
    imputer = joblib.load(imputer_path)

    with open(features_path, "r") as f:
        feature_cols = f.read().splitlines()

    input_df_for_sequence = df_features[feature_cols].iloc[-SEQUENCE_LENGTH:]

    X_imputed = imputer.transform(input_df_for_sequence)
    X_scaled = feature_scaler.transform(X_imputed)


    X_reshaped = X_scaled.reshape(1, SEQUENCE_LENGTH, X_scaled.shape[1])
    scaled_prediction = model.predict(X_reshaped)[0]  


    final_prediction = target_scaler.inverse_transform(scaled_prediction.reshape(1, -1))[0] 

    print(f"Predictions for {crypto_name} (use_fng={use_fng}):")
    for i, price in enumerate(final_prediction, start=1):
        print(f"Prediction for +{i} hour(s): {price:.2f}")


    save_predictions_to_csv(crypto_name, final_prediction, use_fng)

if __name__ == "__main__":
    print("Starting model testing process...")

    run_model("bitcoin", use_fng=False)
    run_model("ethereum", use_fng=False)
    run_model("solana", use_fng=False)

    run_model("bitcoin", use_fng=True)
    run_model("ethereum", use_fng=True)
    run_model("solana", use_fng=True)

    print("\nAll models tested successfully!")
