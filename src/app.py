from flask import Flask, render_template, request
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from model.train_models import load_data, load_fear_greed, create_features
from tensorflow.keras.models import Sequential, load_model

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/preprocessed/price')

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    cryptos = ['bitcoin', 'ethereum', 'solana']
    use_fng = request.form.get('use_fng', 'with_fng') == 'with_fng'
    suffix = '_with_fng' if use_fng else '_nofng'
    graphs = {}

    for crypto_name in cryptos:
        model_path = os.path.join(MODEL_DIR, f'{crypto_name}_model{suffix}.h5')
        scaler_path = os.path.join(MODEL_DIR, f'{crypto_name}_scaler{suffix}.pkl')
        imputer_path = os.path.join(MODEL_DIR, f'{crypto_name}_imputer{suffix}.pkl')
        features_path = os.path.join(MODEL_DIR, f'{crypto_name}_features{suffix}.txt')

        if not all(os.path.exists(p) for p in [model_path, scaler_path, imputer_path, features_path]):
            predictions[crypto_name] = ['Missing model files']
            continue

        # Update the model loading logic to use the new neural network model
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)

        with open(features_path, 'r') as f:
            feature_cols = f.read().splitlines()

        # Load data and create features
        df = load_data(crypto_name)
        fng_df = load_fear_greed()
        df = pd.merge(df, fng_df[['date', 'fng_value']], on='date', how='left')
        df['fng_value'] = df['fng_value'].fillna(method='ffill').fillna(method='bfill').fillna(df['fng_value'].mean())
        df_features = create_features(df)

        # Use only the latest row of features
        latest_row = df_features.iloc[-1:]
        X = latest_row[feature_cols]
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        # Predict
        prediction = model.predict(X_scaled)
        predictions[crypto_name] = [f'{price:.2f}' for price in prediction]

        # Load last day's prices
        price_file = os.path.join(DATA_DIR, f'{crypto_name}.csv')
        if os.path.exists(price_file):
            price_data = pd.read_csv(price_file)
            last_day_prices = price_data.tail(24)['price'].tolist()
        else:
            last_day_prices = []

        combined_prices = last_day_prices + [float(p) for p in predictions[crypto_name]]

        # Create graph
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(last_day_prices)), last_day_prices, marker='o', label='Last Day Prices', color='blue')
        plt.plot(range(len(last_day_prices), len(combined_prices)), combined_prices[len(last_day_prices):], marker='o', label='Predicted Prices', color='red')
        plt.title(f'{crypto_name.capitalize()} Prices and Predictions')
        plt.xlabel('Time (Hours)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save graph to base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graphs[crypto_name] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

    return render_template('index.html', predictions=predictions, graphs=graphs, use_fng=use_fng)

if __name__ == '__main__':
    app.run(debug=True)