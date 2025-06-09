from flask import Flask, render_template, request, send_file
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

SEQUENCE_LENGTH = 24 # Define sequence length matching training

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    cryptos = ['bitcoin', 'ethereum', 'solana']
    use_fng = request.form.get('use_fng', 'with_fng') == 'with_fng'
    suffix = '_with_fng' if use_fng else '_nofng'
    graphs = {}
    comparison_graphs = {}

    for crypto_name in cryptos:
        model_path = os.path.join(MODEL_DIR, f'{crypto_name}_model{suffix}.h5')
        # Corrected scaler path to feature_scaler and added target_scaler path
        feature_scaler_path = os.path.join(MODEL_DIR, f'{crypto_name}_feature_scaler{suffix}.pkl')
        target_scaler_path = os.path.join(MODEL_DIR, f'{crypto_name}_target_scaler{suffix}.pkl')
        imputer_path = os.path.join(MODEL_DIR, f'{crypto_name}_imputer{suffix}.pkl')
        features_path = os.path.join(MODEL_DIR, f'{crypto_name}_features{suffix}.txt')

        if not all(os.path.exists(p) for p in [model_path, feature_scaler_path, target_scaler_path, imputer_path, features_path]):
            predictions[crypto_name] = ['Missing model files']
            continue

        # Update the model loading logic to use the new neural network model
        model = load_model(model_path)
        feature_scaler = joblib.load(feature_scaler_path) # Load feature_scaler
        target_scaler = joblib.load(target_scaler_path) # Load target_scaler
        imputer = joblib.load(imputer_path)

        with open(features_path, 'r') as f:
            feature_cols = f.read().splitlines()

        # Load data and create features
        df = load_data(crypto_name)
        fng_df = load_fear_greed()
        df = pd.merge(df, fng_df[['date', 'fng_value']], on='date', how='left')
        # Updated fillna to use .ffill().bfill()
        df['fng_value'] = df['fng_value'].ffill().bfill().fillna(df['fng_value'].mean())
        df_features = create_features(df)

        # Add check for sufficient data for a sequence
        if len(df_features) < SEQUENCE_LENGTH:
            predictions[crypto_name] = [f'Not enough data to form a sequence of {SEQUENCE_LENGTH}']
            graphs[crypto_name] = None # Or a placeholder image/message
            comparison_graphs[crypto_name] = None
            continue

        # Prepare input sequence
        # Select the last SEQUENCE_LENGTH rows for the input sequence
        input_df_for_sequence = df_features[feature_cols].iloc[-SEQUENCE_LENGTH:]


        X_imputed = imputer.transform(input_df_for_sequence)
        X_scaled = feature_scaler.transform(X_imputed) # Shape: (SEQUENCE_LENGTH, num_features)

        # Reshape for LSTM: (1, SEQUENCE_LENGTH, num_features)
        X_reshaped = X_scaled.reshape(1, SEQUENCE_LENGTH, X_scaled.shape[1])

        # Predict
        scaled_prediction = model.predict(X_reshaped)[0] # Shape: (5,)

        # Inverse transform the predictions
        final_prediction = target_scaler.inverse_transform(scaled_prediction.reshape(1, -1))[0] # Shape: (5,)
        predictions[crypto_name] = [f'{float(price):.2f}' for price in final_prediction]

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

        # Load prediction data
        prediction_file = os.path.join(os.path.dirname(__file__), f'../predictions/{crypto_name}_prediction{suffix}.csv')
        if not os.path.exists(prediction_file):
            comparison_graphs[crypto_name] = None
            continue

        prediction_data = pd.read_csv(prediction_file)
        prediction_data['timestamp'] = pd.to_datetime(prediction_data['timestamp'])

        # Load actual price data
        price_file = os.path.join(DATA_DIR, f'{crypto_name}.csv')
        if not os.path.exists(price_file):
            comparison_graphs[crypto_name] = None
            continue

        price_data = pd.read_csv(price_file)
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

        # Ensure timestamps are in the same format and timezone, and round to the nearest hour
        prediction_data['timestamp'] = pd.to_datetime(prediction_data['timestamp']).dt.round('h') # Corrected 'H' to 'h'
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp']).dt.round('h') # Corrected 'H' to 'h'

        # Filter actual price data to match prediction timestamps
        merged_data = pd.merge(prediction_data, price_data, on='timestamp', how='inner', suffixes=('_predicted', '_actual'))

        # Filter data for the last two days
        two_days_ago = pd.Timestamp.now() - pd.Timedelta(days=2)
        prediction_data = prediction_data[prediction_data['timestamp'] >= two_days_ago]
        price_data = price_data[price_data['timestamp'] >= two_days_ago]

        # Merge filtered data
        merged_data = pd.merge(prediction_data, price_data, on='timestamp', how='inner', suffixes=('_predicted', '_actual'))

        # Create comparison graph
        plt.figure(figsize=(10, 5))
        plt.plot(merged_data['timestamp'], merged_data['price_actual'], marker='o', label='Actual Prices', color='blue')
        plt.plot(merged_data['timestamp'], merged_data['price_predicted'], marker='o', label='Predicted Prices', color='red')
        
        
        if not merged_data.empty:
            min_val = min(merged_data['price_actual'].min(), merged_data['price_predicted'].min())
            max_val = max(merged_data['price_actual'].max(), merged_data['price_predicted'].max())
            padding = (max_val - min_val) * 0.1 # 10% padding
            plt.ylim(min_val - padding, max_val + padding)
            
        plt.title(f'{crypto_name.capitalize()} Actual vs Predicted Prices')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save graph to base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        comparison_graphs[crypto_name] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

    return render_template('index.html', predictions=predictions, graphs=graphs, comparison_graphs=comparison_graphs, use_fng=use_fng)

@app.route('/reports/<report_name>')
def get_report(report_name):
    return send_file(f'reports/{report_name}')

@app.route('/validations/<validation_suite>')
def get_validation(validation_suite):
    return send_file(f'gx/uncommitted/data_docs/local_site/validations/{validation_suite}')

if __name__ == '__main__':
    app.run(debug=True)