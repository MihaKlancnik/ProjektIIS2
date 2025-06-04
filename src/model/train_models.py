import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import joblib
import mlflow
import mlflow.tensorflow
import mlflow.sklearn

# Setup MLflow tracking URI safely for all OS types
mlruns_dir = os.path.abspath(os.path.join(os.getcwd(), "mlruns"))
if not os.path.exists(mlruns_dir):
    os.makedirs(mlruns_dir)

tracking_uri = f"file://{mlruns_dir}"
print(f"Setting MLflow tracking URI to: {tracking_uri}")  # Debug print
mlflow.set_tracking_uri(tracking_uri)

def load_data(crypto_name):
    """Load cryptocurrency price data."""
    try:
        crypto_paths = {
            'bitcoin': 'data/preprocessed/price/bitcoin.csv',
            'ethereum': 'data/preprocessed/price/ethereum.csv',
            'solana': 'data/preprocessed/price/solana.csv'
        }
        crypto_key = crypto_name.lower()
        if crypto_key in crypto_paths:
            path = crypto_paths[crypto_key]
        else:
            raise FileNotFoundError(f"No data path specified for {crypto_name}")
        if os.path.exists(path):
            print(f"Loading data from {path}")
            df = pd.read_csv(path)
        else:
            raise FileNotFoundError(f"Could not find price data for {crypto_name} at the specified path: {path}")

        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                break
        if timestamp_col is None and len(df.columns) >= 2:
            timestamp_col = df.columns[0]

        price_col = None
        for col in df.columns:
            if 'price' in col.lower():
                price_col = col
                break
        if price_col is None and len(df.columns) >= 2:
            if timestamp_col == df.columns[0]:
                price_col = df.columns[1]
            else:
                price_col = df.columns[0]

        clean_df = pd.DataFrame()
        clean_df['timestamp'] = pd.to_datetime(df[timestamp_col])
        clean_df['price'] = pd.to_numeric(df[price_col], errors='coerce')
        clean_df = clean_df.sort_values('timestamp')
        clean_df['price'] = clean_df['price'].fillna(method='ffill').fillna(method='bfill')
        clean_df = clean_df.dropna(subset=['price'])
        clean_df['date'] = pd.to_datetime(clean_df['timestamp'].dt.date)
        clean_df['hour'] = clean_df['timestamp'].dt.hour
        clean_df['day_of_week'] = clean_df['timestamp'].dt.dayofweek

        return clean_df

    except Exception as e:
        print(f"Error loading {crypto_name} data: {e}")
        raise

def load_fear_greed():
    """Load fear and greed index data."""
    try:
        path = "data/preprocessed/fear_greed/fear_greed_index.csv"
        if os.path.exists(path):
            print(f"Loading fear and greed data from {path}")
            df = pd.read_csv(path)
        else:
            raise FileNotFoundError("Could not find fear and greed data file at the specified path")

        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        value_col = next((col for col in df.columns if 'value' in col.lower() or 'index' in col.lower()), None)

        if not date_col or not value_col:
            raise ValueError("Required columns not found in Fear and Greed data.")

        clean_df = pd.DataFrame()
        clean_df['date'] = pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='coerce')
        clean_df['fng_value'] = pd.to_numeric(df[value_col], errors='coerce')
        clean_df['fng_value'] = clean_df['fng_value'].ffill().bfill()
        clean_df = clean_df.dropna()

        return clean_df

    except Exception as e:
        print(f"Error loading Fear and Greed data: {e}")
        return pd.DataFrame(columns=['date', 'fng_value'])

def create_features(df, lookback=24, target_col='price'):
    """Create features for price prediction model."""
    min_rows_required = lookback + 12 + 5
    if len(df) < min_rows_required:
        print(f"❌ Not enough rows for feature creation. Required: {min_rows_required}, Available: {len(df)}")
        return pd.DataFrame()

    df_features = df.copy()
    for i in range(1, lookback + 1):
        df_features[f'price_lag_{i}'] = df_features[target_col].shift(i)
    df_features['price_rolling_mean_24'] = df_features[target_col].rolling(window=24).mean()
    df_features['price_rolling_std_24'] = df_features[target_col].rolling(window=24).std()
    for i in range(1, 6):
        df_features[f'price_next_{i}'] = df_features[target_col].shift(-i)
    df_features = df_features.dropna()

    return df_features

def train_and_save_model(crypto_name, use_fng=False):
    """Train model for cryptocurrency price prediction."""
    print(f"\nTraining model for {crypto_name}...")
    suffix = '_with_fng' if use_fng else '_nofng'

    with mlflow.start_run(run_name=f"train_{crypto_name}{suffix}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id} for {crypto_name}{suffix}")
        mlflow.log_param("crypto_name", crypto_name)
        mlflow.log_param("use_fng", use_fng)

        df = load_data(crypto_name)
        print(f"Data shape after loading: {df.shape}")

        if use_fng:
            print("Including fear and greed index in model...")
            fng_df = load_fear_greed()
            print(f"Fear and Greed data shape: {fng_df.shape}")

            if not fng_df.empty:
                df = pd.merge(df, fng_df, on='date', how='left')
                print(f"Data shape after merging with Fear and Greed Index: {df.shape}")
                df['fng_value'] = df['fng_value'].ffill().bfill().fillna(df['fng_value'].mean())

        df_features = create_features(df)
        print(f"Feature dataframe shape: {df_features.shape}")

        if df_features.empty:
            print(f"❌ Feature dataframe is empty for {crypto_name}. Skipping training.")
            return

        feature_cols = [col for col in df_features.columns if 'lag' in col or 'rolling' in col or 'hour' in col or 'day_of_week' in col]
        if use_fng and 'fng_value' in df_features.columns:
            feature_cols.append('fng_value')

        target_cols = [f'price_next_{i}' for i in range(1, 6)]
        X = df_features[feature_cols]
        y = df_features[target_cols]

        if X.empty or y.empty:
            print(f"❌ Feature matrix or target matrix is empty for {crypto_name}. Skipping training.")
            return

        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        mlflow.log_param("feature_count", X_scaled.shape[1])
        mlflow.log_param("lookback_period", 24)

        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(target_cols))
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        mlflow.log_param("optimizer_name", "adam")
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("loss_function", "mse")

        epochs = 50
        batch_size = 32
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        print(f"Starting model training for {crypto_name}{suffix}...")
        history = model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, verbose=1)
        print(f"Model training finished for {crypto_name}{suffix}.")

        final_train_loss = history.history['loss'][-1]
        mlflow.log_metric("final_train_loss", final_train_loss)
        print(f"Logged final_train_loss: {final_train_loss} for {crypto_name}{suffix}")

        model_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, f'{crypto_name}_model{suffix}.keras')
        scaler_path = os.path.join(model_dir, f'{crypto_name}_scaler{suffix}.pkl')
        imputer_path = os.path.join(model_dir, f'{crypto_name}_imputer{suffix}.pkl')
        features_path = os.path.join(model_dir, f'{crypto_name}_features{suffix}.txt')

        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(imputer, imputer_path)
        with open(features_path, 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")

        print(f"Saved model and artifacts locally for {crypto_name}{suffix}.")

        example_input = np.random.rand(1, X_scaled.shape[1])
        mlflow.tensorflow.log_model(
            model,
            artifact_path=os.path.join("artifacts", f"{crypto_name}{suffix}_tf_model"),
            input_example=example_input
        )
        mlflow.sklearn.log_model(scaler, artifact_path=os.path.join("artifacts", f"{crypto_name}{suffix}_scaler_model"))
        mlflow.sklearn.log_model(imputer, artifact_path=os.path.join("artifacts", f"{crypto_name}{suffix}_imputer_model"))
        mlflow.log_text("\n".join(feature_cols), artifact_file=os.path.join("artifacts", f"{crypto_name}{suffix}_features.txt"))
        print(f"Logged model and artifacts to MLflow for {crypto_name}{suffix}.")

    print(f"Finished training and logging for {crypto_name}{suffix}.")
    return

if __name__ == "__main__":
    print("Starting model training process...")

    train_and_save_model("bitcoin", use_fng=False)
    train_and_save_model("ethereum", use_fng=False)
    train_and_save_model("solana", use_fng=False)

    train_and_save_model("bitcoin", use_fng=True)
    train_and_save_model("ethereum", use_fng=True)
    train_and_save_model("solana", use_fng=True)

    print("\nAll models trained and saved successfully!")
