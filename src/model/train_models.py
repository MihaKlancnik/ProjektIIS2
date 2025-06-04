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

def load_data(crypto_name):
    """Load cryptocurrency price data."""
    try:
        # Use only the specified file paths for each crypto
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
        
        # Identify timestamp column
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                break
        
        if timestamp_col is None and len(df.columns) >= 2:
            # Assume first column is timestamp
            timestamp_col = df.columns[0]
        
        # Identify price column
        price_col = None
        for col in df.columns:
            if 'price' in col.lower():
                price_col = col
                break
        
        if price_col is None and len(df.columns) >= 2:
            # Assume second column is price if we have timestamp as first
            if timestamp_col == df.columns[0]:
                price_col = df.columns[1]
            else:
                # Otherwise assume first column is price
                price_col = df.columns[0]
        
        # Create a clean dataframe with standard column names
        clean_df = pd.DataFrame()
        clean_df['timestamp'] = pd.to_datetime(df[timestamp_col])
        clean_df['price'] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Sort by timestamp
        clean_df = clean_df.sort_values('timestamp')
        
        # Handle missing values
        clean_df['price'] = clean_df['price'].fillna(method='ffill').fillna(method='bfill')
        clean_df = clean_df.dropna(subset=['price'])
        
        # Add date column for merging with fear and greed data
        clean_df['date'] = clean_df['timestamp'].dt.date
        clean_df['date'] = pd.to_datetime(clean_df['date'])
        
        # Add basic features
        clean_df['hour'] = clean_df['timestamp'].dt.hour
        clean_df['day_of_week'] = clean_df['timestamp'].dt.dayofweek
        
        return clean_df
    
    except Exception as e:
        print(f"Error loading {crypto_name} data: {e}")
        raise

def load_fear_greed():
    """Load fear and greed index data."""
    try:
        # Use only the specified file path
        path = "data/preprocessed/fear_greed/fear_greed_index.csv"
        if os.path.exists(path):
            print(f"Loading fear and greed data from {path}")
            df = pd.read_csv(path)
        else:
            raise FileNotFoundError("Could not find fear and greed data file at the specified path")

        # Identify columns
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        value_col = next((col for col in df.columns if 'value' in col.lower() or 'index' in col.lower()), None)

        if not date_col or not value_col:
            raise ValueError("Required columns not found in Fear and Greed data.")

        clean_df = pd.DataFrame()
        # Convert date format to match price data
        clean_df['date'] = pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='coerce')
        clean_df['fng_value'] = pd.to_numeric(df[value_col], errors='coerce')

        clean_df['fng_value'] = clean_df['fng_value'].ffill().bfill()
        clean_df = clean_df.dropna()

        return clean_df

    except Exception as e:
        print(f"Error loading Fear and Greed data: {e}")
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=['date', 'fng_value'])

def create_features(df, lookback=24, target_col='price'):
    """Create features for price prediction model."""
    # Ensure there are enough rows for feature creation
    min_rows_required = lookback + 12 + 5  # Lookback + rolling window + target shifts
    if len(df) < min_rows_required:
        print(f"❌ Not enough rows for feature creation. Required: {min_rows_required}, Available: {len(df)}")
        return pd.DataFrame()

    df_features = df.copy()

    # Create lagged features
    for i in range(1, lookback + 1):
        df_features[f'price_lag_{i}'] = df_features[target_col].shift(i)

    # Create rolling window features
    df_features['price_rolling_mean_24'] = df_features[target_col].rolling(window=24).mean()
    df_features['price_rolling_std_24'] = df_features[target_col].rolling(window=24).std()

    # Create target variables for next 5 hours
    for i in range(1, 6):
        df_features[f'price_next_{i}'] = df_features[target_col].shift(-i)

    # Drop rows with NaN values
    df_features = df_features.dropna()

    return df_features

def train_and_save_model(crypto_name, use_fng=False):
    """Train model for cryptocurrency price prediction."""
    print(f"\nTraining model for {crypto_name}...")

    # Load price data
    df = load_data(crypto_name)
    print(f"Data shape after loading: {df.shape}")

    # Add fear and greed index if specified
    if use_fng:
        print("Including fear and greed index in model...")
        fng_df = load_fear_greed()
        print(f"Fear and Greed data shape: {fng_df.shape}")

        if not fng_df.empty:
            # Merge on date
            df = pd.merge(df, fng_df, on='date', how='left')
            print(f"Data shape after merging with Fear and Greed Index: {df.shape}")

            # Forward fill missing values
            df['fng_value'] = df['fng_value'].ffill()

            # If there are still NaN values, use backward fill
            if df['fng_value'].isna().any():
                df['fng_value'] = df['fng_value'].bfill()

            # If there are STILL NaN values, use mean
            if df['fng_value'].isna().any():
                df['fng_value'] = df['fng_value'].fillna(df['fng_value'].mean())

    # Create features
    df_features = create_features(df)
    print(f"Feature dataframe shape: {df_features.shape}")

    # Check if feature dataframe is empty
    if df_features.empty:
        print(f"❌ Feature dataframe is empty for {crypto_name}. Skipping training.")
        return

    # Define features and target
    feature_cols = [col for col in df_features.columns if 'lag' in col or 'rolling' in col or 'hour' in col or 'day_of_week' in col]
    if use_fng and 'fng_value' in df_features.columns:
        feature_cols.append('fng_value')

    target_cols = [f'price_next_{i}' for i in range(1, 6)]

    X = df_features[feature_cols]
    y = df_features[target_cols]

    # Check if X or y is empty
    if X.empty or y.empty:
        print(f"❌ Feature matrix or target matrix is empty for {crypto_name}. Skipping training.")
        return

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = Sequential([
        Dense(128, activation='relu', input_dim=X_scaled.shape[1]),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    model.fit(X_scaled, y, epochs=30, batch_size=64, verbose=1)

    # Save model and preprocessing artifacts
    os.makedirs('models', exist_ok=True)
    suffix = '_with_fng' if use_fng else '_nofng'
    model.save(f'models/{crypto_name.lower()}_model{suffix}.h5')
    joblib.dump(scaler, f'models/{crypto_name.lower()}_scaler{suffix}.pkl')
    joblib.dump(imputer, f'models/{crypto_name.lower()}_imputer{suffix}.pkl')

    with open(f'models/{crypto_name.lower()}_features{suffix}.txt', 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"Model for {crypto_name} saved successfully.")

def train_and_log_model(crypto_name, use_fng=False):
    """Train model and log metrics to MLflow."""
    mlflow.set_experiment("Crypto Price Prediction")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("crypto_name", crypto_name)
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("use_fng", use_fng)

        # Load price data
        df = load_data(crypto_name)
        print(f"Data shape after loading: {df.shape}")

        # Add fear and greed index if specified
        if use_fng:
            print("Including fear and greed index in model...")
            fng_df = load_fear_greed()
            print(f"Fear and Greed data shape: {fng_df.shape}")

            if not fng_df.empty:
                # Merge on date
                df = pd.merge(df, fng_df, on='date', how='left')
                print(f"Data shape after merging with Fear and Greed Index: {df.shape}")

                # Forward fill missing values
                df['fng_value'] = df['fng_value'].ffill()

                # If there are still NaN values, use backward fill
                if df['fng_value'].isna().any():
                    df['fng_value'] = df['fng_value'].bfill()

                # If there are STILL NaN values, use mean
                if df['fng_value'].isna().any():
                    df['fng_value'] = df['fng_value'].fillna(df['fng_value'].mean())

        # Create features
        df_features = create_features(df)
        print(f"Feature dataframe shape: {df_features.shape}")

        # Check if feature dataframe is empty
        if df_features.empty:
            print(f"Feature dataframe is empty for {crypto_name}. Skipping training.")
            return

        # Define features and target
        feature_cols = [col for col in df_features.columns if 'lag' in col or 'rolling' in col or 'hour' in col or 'day_of_week' in col]
        if use_fng and 'fng_value' in df_features.columns:
            feature_cols.append('fng_value')

        target_cols = [f'price_next_{i}' for i in range(1, 6)]

        X = df_features[feature_cols]
        y = df_features[target_cols]

        # Check if X or y is empty
        if X.empty or y.empty:
            print(f"Feature matrix or target matrix is empty for {crypto_name}. Skipping training.")
            return

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = Sequential([
            Dense(128, activation='relu', input_dim=X_scaled.shape[1]),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(y.shape[1], activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
        model.fit(X_scaled, y, epochs=30, batch_size=64, verbose=1)

        # Log metrics
        y_pred = model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        mlflow.log_metric("mean_squared_error", mse)

        # Save model and preprocessing artifacts
        os.makedirs('models', exist_ok=True)
        suffix = '_with_fng' if use_fng else '_nofng'
        model_path = f'models/{crypto_name.lower()}_model{suffix}.h5'
        model.save(model_path)
        mlflow.keras.log_model(model, "model")

        scaler_path = f'models/{crypto_name.lower()}_scaler{suffix}.pkl'
        joblib.dump(scaler, scaler_path)

        imputer_path = f'models/{crypto_name.lower()}_imputer{suffix}.pkl'
        joblib.dump(imputer, imputer_path)

        with open(f'models/{crypto_name.lower()}_features{suffix}.txt', 'w') as f:
            f.write('\n'.join(feature_cols))

        print(f"Model for {crypto_name} saved successfully.")

if __name__ == "__main__":
    print("Starting model training process...")

    # Train models WITHOUT fear and greed index
    train_and_log_model("bitcoin", use_fng=False)
    train_and_log_model("ethereum", use_fng=False)
    train_and_log_model("solana", use_fng=False)

    # Train models WITH fear and greed index
    train_and_log_model("bitcoin", use_fng=True)
    train_and_log_model("ethereum", use_fng=True)
    train_and_log_model("solana", use_fng=True)

    print("\nAll models trained and saved successfully!")


# poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
#poetry run python src/model/train_models.py
#poetry run mlflow ui
#poetry run mlflow ui