import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM # Added LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os
import joblib
import mlflow
import mlflow.tensorflow
import mlflow.sklearn

if os.getenv("DAGSHUB_USER") and os.getenv("DAGSHUB_REPO"):
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{os.getenv('DAGSHUB_USER')}/{os.getenv('DAGSHUB_REPO')}.mlflow"
    print(f"MLFLOW_TRACKING_URI set to: {os.environ['MLFLOW_TRACKING_URI']}")
else:
    print("DAGSHUB_USER or DAGSHUB_REPO not set. MLflow will run locally or use default URI.")

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
        # Round timestamp to the nearest hour
        clean_df['timestamp'] = clean_df['timestamp'].dt.round('h') # Changed 'H' to 'h'
        clean_df['price'] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Sort by timestamp
        clean_df = clean_df.sort_values('timestamp')
        
        # Aggregate data by the rounded hour, taking the mean price for that hour
        # This helps to avoid issues with duplicate timestamps after rounding if original data was more granular
        # and ensures a single price point per hour.
        clean_df = clean_df.groupby('timestamp').agg(
            price=('price', 'mean'),
            # Keep other relevant columns if they exist and should be aggregated
            # For example, if 'volume' was a column: volume=('volume', 'sum')
        ).reset_index()

        # Handle missing values that might arise from aggregation or were already present
        clean_df['price'] = clean_df['price'].ffill().bfill() # Changed from fillna(method=...)
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
    print(f"\\nTraining model for {crypto_name}...")
    suffix = '_with_fng' if use_fng else '_nofng'

    # Start MLflow run
    with mlflow.start_run(run_name=f"train_{crypto_name}{suffix}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id} for {crypto_name}{suffix}")
        mlflow.log_param("crypto_name", crypto_name)
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
            print(f"❌ Feature dataframe is empty for {crypto_name}. Skipping training.")
            return

        # Define features and target
        feature_cols = [col for col in df_features.columns if 'lag' in col or 'rolling' in col or 'hour' in col or 'day_of_week' in col]
        if use_fng and 'fng_value' in df_features.columns:
            feature_cols.append('fng_value')

        target_cols = [f'price_next_{i}' for i in range(1, 6)]

        X_original = df_features[feature_cols]
        y_original = df_features[target_cols]

        # Check if X or y is empty
        if X_original.empty or y_original.empty:
            print(f"❌ Feature matrix or target matrix is empty for {crypto_name}. Skipping training.")
            return

        # Handle missing values in features
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_original)

        # Scale features
        feature_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X_imputed)

        # Scale target variable y
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y_original)

        # Create sequences for LSTM
        sequence_length = 24  # Number of past timesteps to look at
        X_list_for_sequences = []
        y_list_for_sequences = []

        # Number of sequences we can create
        # Each sequence X uses `sequence_length` from X_scaled.
        # The corresponding y is taken from y_scaled at the end of that X sequence.
        num_samples_for_sequences = len(X_scaled) - sequence_length + 1

        if num_samples_for_sequences > 0:
            for i in range(num_samples_for_sequences):
                # X sequence is from index i to i + sequence_length - 1
                X_list_for_sequences.append(X_scaled[i : i + sequence_length])
                # y target corresponds to the data at index i + sequence_length - 1
                # This index refers to y_scaled, which has the multi-step ahead targets
                y_list_for_sequences.append(y_scaled[i + sequence_length - 1])
        else:
            print(f"❌ Not enough data to create sequences of length {sequence_length} for {crypto_name}{suffix}. Available X_scaled: {len(X_scaled)}. Skipping training.")
            # Log a parameter to MLflow indicating why training was skipped for this run.
            mlflow.log_param("training_status", "skipped_insufficient_data_for_sequences")
            return

        X_sequenced = np.array(X_list_for_sequences)
        y_sequenced = np.array(y_list_for_sequences)


        # Log feature-related parameters
        mlflow.log_param("feature_count", X_sequenced.shape[2]) # Features per timestep
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("lookback_period", 24) # Assuming 24 from create_features default

        # Define model (RNN - LSTM based)
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X_sequenced.shape[1], X_sequenced.shape[2])), # LSTM layer
            Dense(25, activation='relu'), # Intermediate Dense layer
            Dense(y_sequenced.shape[1] if len(y_sequenced.shape) > 1 else 1)  # Output layer, matches number of targets
        ])

        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        mlflow.log_param("optimizer_name", "adam")
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("loss_function", "MeanSquaredError")

        # Train model
        epochs = 20 # Keeping epochs relatively low to manage training time
        batch_size = 32
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        print(f"Starting model training for {crypto_name}{suffix} with {X_sequenced.shape[0]} sequences...")
        history = model.fit(X_sequenced, y_sequenced, epochs=epochs, batch_size=batch_size, verbose=1) # Use X_sequenced, y_sequenced
        print(f"Model training finished for {crypto_name}{suffix}.")

        # Log metrics
        final_train_loss = history.history['loss'][-1]
        mlflow.log_metric("final_train_loss", final_train_loss)
        print(f"Logged final_train_loss: {final_train_loss} for {crypto_name}{suffix}")

        # Save model, scaler, imputer, and features
        model_dir = 'models' # Ensure this directory exists or is created
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, f'{crypto_name}_model{suffix}.h5')
        feature_scaler_path = os.path.join(model_dir, f'{crypto_name}_feature_scaler{suffix}.pkl') # Renamed for clarity
        target_scaler_path = os.path.join(model_dir, f'{crypto_name}_target_scaler{suffix}.pkl') # New target scaler
        imputer_path = os.path.join(model_dir, f'{crypto_name}_imputer{suffix}.pkl')
        features_path = os.path.join(model_dir, f'{crypto_name}_features{suffix}.txt')

        model.save(model_path)
        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path) # Save the target scaler
        joblib.dump(imputer, imputer_path)
        with open(features_path, 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        print(f"Saved model and artifacts locally for {crypto_name}{suffix}.")

        # Log model and artifacts to MLflow
        # Provide an input example for the Keras model
        input_example_keras = X_sequenced[:5] if len(X_sequenced) >=5 else X_sequenced # Take a small sample
        if input_example_keras.shape[0] > 0: # Ensure example is not empty
            mlflow.tensorflow.log_model(model, artifact_path=f"{crypto_name}{suffix}_tf_model", input_example=input_example_keras)
        else:
            print("Warning: Keras input example is empty, not logging model with input example.")
            mlflow.tensorflow.log_model(model, artifact_path=f"{crypto_name}{suffix}_tf_model")


        # Provide an input example for the scikit-learn feature_scaler
        # Input to feature_scaler is X_imputed
        input_example_feature_scaler = X_imputed[:5] if len(X_imputed) >= 5 else X_imputed
        if input_example_feature_scaler.shape[0] > 0:
             mlflow.sklearn.log_model(feature_scaler, artifact_path=f"{crypto_name}{suffix}_feature_scaler_model", input_example=input_example_feature_scaler)
        else:
            print("Warning: Feature scaler input example is empty, not logging with input example.")
            mlflow.sklearn.log_model(feature_scaler, artifact_path=f"{crypto_name}{suffix}_feature_scaler_model")

        # Provide an input example for the scikit-learn target_scaler
        # Input to target_scaler is y_original
        input_example_target_scaler = y_original.iloc[:5].to_numpy() if len(y_original) >= 5 else y_original.to_numpy()
        if input_example_target_scaler.shape[0] > 0:
            mlflow.sklearn.log_model(target_scaler, artifact_path=f"{crypto_name}{suffix}_target_scaler_model", input_example=input_example_target_scaler)
        else:
            print("Warning: Target scaler input example is empty, not logging with input example.")
            mlflow.sklearn.log_model(target_scaler, artifact_path=f"{crypto_name}{suffix}_target_scaler_model")


        # Provide an input example for the scikit-learn imputer
        # Input to imputer is X_original
        input_example_imputer = X_original.iloc[:5].to_numpy() if len(X_original) >=5 else X_original.to_numpy()
        if input_example_imputer.shape[0] > 0:
            mlflow.sklearn.log_model(imputer, artifact_path=f"{crypto_name}{suffix}_imputer_model", input_example=input_example_imputer)
        else:
            print("Warning: Imputer input example is empty, not logging with input example.")
            mlflow.sklearn.log_model(imputer, artifact_path=f"{crypto_name}{suffix}_imputer_model")

        mlflow.log_text("\\\\n".join(feature_cols), artifact_file=f"{crypto_name}{suffix}_features.txt")
        print(f"Logged model and artifacts to MLflow for {crypto_name}{suffix}.")

    print(f"Finished training and logging for {crypto_name}{suffix}.")
    return

if __name__ == "__main__":
    print("Starting model training process...")

    # Train models WITHOUT fear and greed index
    train_and_save_model("bitcoin", use_fng=False)
    train_and_save_model("ethereum", use_fng=False)
    train_and_save_model("solana", use_fng=False)

    # Train models WITH fear and greed index
    train_and_save_model("bitcoin", use_fng=True)
    train_and_save_model("ethereum", use_fng=True)
    train_and_save_model("solana", use_fng=True)

    print("\nAll models trained and saved successfully!")

        #shranit si mors vse napovedi ko ti jih da v frontendu pa pol fertchas podatke ko so svezi 
    #tkda dodaj v frontend kok uspesno si napovedu pol ko ze dobis podatke