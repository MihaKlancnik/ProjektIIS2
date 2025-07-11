from flask import Flask, render_template, request, send_file
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from model.train_models import load_data, load_fear_greed, create_features
from tensorflow.keras.models import Sequential, load_model
import src.data.cleanup

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/preprocessed/price')

SEQUENCE_LENGTH = 24 # Define sequence length matching training

# Run cleanup before starting the app
src.data.cleanup.cleanup_data_folder(os.path.join(os.path.dirname(__file__), "../data"), max_age_days=70)

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
        feature_scaler_path = os.path.join(MODEL_DIR, f'{crypto_name}_feature_scaler{suffix}.pkl')
        target_scaler_path = os.path.join(MODEL_DIR, f'{crypto_name}_target_scaler{suffix}.pkl')
        imputer_path = os.path.join(MODEL_DIR, f'{crypto_name}_imputer{suffix}.pkl')
        features_path = os.path.join(MODEL_DIR, f'{crypto_name}_features{suffix}.txt')

        if not all(os.path.exists(p) for p in [model_path, feature_scaler_path, target_scaler_path, imputer_path, features_path]):
            predictions[crypto_name] = ['Missing model files']
            continue

        model = load_model(model_path)
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        imputer = joblib.load(imputer_path)

        with open(features_path, 'r') as f:
            feature_cols = f.read().splitlines()

        df = load_data(crypto_name)
        fng_df = load_fear_greed()
        df = pd.merge(df, fng_df[['date', 'fng_value']], on='date', how='left')
        df['fng_value'] = df['fng_value'].ffill().bfill().fillna(df['fng_value'].mean())
        df_features = create_features(df)

        if len(df_features) < SEQUENCE_LENGTH:
            predictions[crypto_name] = [f'Not enough data to form a sequence of {SEQUENCE_LENGTH}']
            graphs[crypto_name] = None
            comparison_graphs[crypto_name] = None
            continue

        input_df_for_sequence = df_features[feature_cols].iloc[-SEQUENCE_LENGTH:]

        X_imputed = imputer.transform(input_df_for_sequence)
        X_scaled = feature_scaler.transform(X_imputed)
        X_reshaped = X_scaled.reshape(1, SEQUENCE_LENGTH, X_scaled.shape[1])

        scaled_prediction = model.predict(X_reshaped)[0]

        final_prediction = target_scaler.inverse_transform(scaled_prediction.reshape(1, -1))[0]
        predictions[crypto_name] = [f'{float(price):.2f}' for price in final_prediction]

        # ---- Price Graph ----
        price_file = os.path.join(DATA_DIR, f'{crypto_name}.csv')
        if os.path.exists(price_file):
            price_data = pd.read_csv(price_file)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            last_19_hours_data = price_data.sort_values('timestamp').tail(19)
            last_day_prices = last_19_hours_data['price'].tolist()
            last_day_timestamps = last_19_hours_data['timestamp'].tolist()
        else:
            last_day_prices = []
            last_day_timestamps = []

        
        if last_day_timestamps:
            last_timestamp = last_day_timestamps[-1]
        else:
            last_timestamp = pd.Timestamp.now()

        future_timestamps = [last_timestamp + pd.Timedelta(hours=i) for i in range(1, 6)]

        plt.figure(figsize=(10, 5))
        plt.plot(last_day_timestamps, last_day_prices, marker='o', label='Last 19 Hours Prices', color='blue')
        plt.plot(future_timestamps, [float(p) for p in predictions[crypto_name]], marker='o', label='Next 5 Hours Predictions', color='red')

        plt.title(f'{crypto_name.capitalize()} Prices and Predictions')
        plt.xlabel('Time (Hourly)')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs[crypto_name] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Actual vs Predicted 
        prediction_file = os.path.join(os.path.dirname(__file__), f'../predictions/{crypto_name}_prediction{suffix}.csv')
        if not os.path.exists(prediction_file):
            comparison_graphs[crypto_name] = None
            continue

        prediction_data = pd.read_csv(prediction_file)
        prediction_data['timestamp'] = pd.to_datetime(prediction_data['timestamp'])

        price_file = os.path.join(DATA_DIR, f'{crypto_name}.csv')
        if not os.path.exists(price_file):
            comparison_graphs[crypto_name] = None
            continue

        price_data = pd.read_csv(price_file)
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

        prediction_data['timestamp'] = pd.to_datetime(prediction_data['timestamp']).dt.round('h')
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp']).dt.round('h')

        merged_data = pd.merge(prediction_data, price_data, on='timestamp', how='inner', suffixes=('_predicted', '_actual'))

        two_days_ago = pd.Timestamp.now() - pd.Timedelta(days=2)
        prediction_data = prediction_data[prediction_data['timestamp'] >= two_days_ago]
        price_data = price_data[price_data['timestamp'] >= two_days_ago]

        merged_data = pd.merge(prediction_data, price_data, on='timestamp', how='inner', suffixes=('_predicted', '_actual'))

        plt.figure(figsize=(10, 5))
        plt.plot(merged_data['timestamp'], merged_data['price_actual'], marker='o', label='Actual Prices', color='blue')
        plt.plot(merged_data['timestamp'], merged_data['price_predicted'], marker='o', label='Predicted Prices', color='red')

        if not merged_data.empty:
            min_val = min(merged_data['price_actual'].min(), merged_data['price_predicted'].min())
            max_val = max(merged_data['price_actual'].max(), merged_data['price_predicted'].max())
            padding = (max_val - min_val) * 0.1
            plt.ylim(min_val - padding, max_val + padding)

        plt.title(f'{crypto_name.capitalize()} Actual vs Predicted Prices')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
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

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    try:
        essential_dirs = ['models', 'data/preprocessed/price', 'predictions']
        missing_dirs = []
        
        for dir_path in essential_dirs:
            full_path = os.path.join(os.path.dirname(__file__), '..', dir_path)
            if not os.path.exists(full_path):
                missing_dirs.append(dir_path)
        
        model_dir = os.path.join(MODEL_DIR)
        model_files = []
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        
        status = {
            'status': 'healthy' if not missing_dirs and model_files else 'degraded',
            'timestamp': pd.Timestamp.now().isoformat(),
            'checks': {
                'directories': {
                    'status': 'ok' if not missing_dirs else 'warning',
                    'missing': missing_dirs
                },
                'models': {
                    'status': 'ok' if model_files else 'warning',
                    'count': len(model_files)
                }
            }
        }
        
        return status, 200 if status['status'] == 'healthy' else 503
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'timestamp': pd.Timestamp.now().isoformat(),
            'error': str(e)
        }, 503

@app.route('/metrics')
def metrics():
    """Basic metrics endpoint for monitoring."""
    try:
        model_dir = os.path.join(MODEL_DIR)
        model_count = 0
        if os.path.exists(model_dir):
            model_count = len([f for f in os.listdir(model_dir) if f.endswith('.h5')])
        
        predictions_dir = os.path.join(os.path.dirname(__file__), '..', 'predictions')
        prediction_files = 0
        if os.path.exists(predictions_dir):
            prediction_files = len([f for f in os.listdir(predictions_dir) if f.endswith('.csv')])
        
        return {
            'models_available': model_count,
            'prediction_files': prediction_files,
            'supported_cryptos': ['bitcoin', 'ethereum', 'solana'],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)