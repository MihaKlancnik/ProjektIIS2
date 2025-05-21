import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib
import os
from datetime import datetime, timedelta

def load_data(crypto_name):
    """Load cryptocurrency price data."""
    try:
        # Try different potential file path patterns
        potential_paths = [
            f"data/preprocessed/price/{crypto_name.lower()}.csv",
            f"data/{crypto_name.lower()}_prices.csv",
            f"data/{crypto_name.lower()}.csv",
            f"data/{crypto_name.lower()}_price.csv"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                print(f"Loading data from {path}")
                df = pd.read_csv(path)
                break
        else:
            # If no file is found, try the current directory
            for path in [p.split('/')[-1] for p in potential_paths]:
                if os.path.exists(path):
                    print(f"Loading data from {path}")
                    df = pd.read_csv(path)
                    break
            else:
                raise FileNotFoundError(f"Could not find price data for {crypto_name}")
        
        # Identify timestamp column
        timestamp_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                timestamp_col = col
                break
        
        if timestamp_col is None and len(df.columns) >= 2:
            # Assume first column is timestamp
            timestamp_col = df.columns[0]
        
        # Identify price column
        price_col = None
        for col in df.columns:
            if 'price' in col.lower() or 'value' in col.lower():
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
        clean_df['price'] = clean_df['price'].fillna(method='ffill')
        
        if clean_df['price'].isna().any():
            # If there are still NaN values at the beginning, use bfill
            clean_df['price'] = clean_df['price'].fillna(method='bfill')
        
        # If there are STILL NaN values, drop them
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
        # Try different potential file paths
        potential_paths = [
            "data/preprocessed/price/fear_greed_index.csv",
            "data/preprocessed/fear_greed/fear_greed.csv",
            "data/fear_greed.csv",
            "src/data/fear_greed.csv",
            "fear_greed.csv"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                print(f"Loading fear and greed data from {path}")
                df = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError("Could not find fear and greed data file")
        
        # Check for expected columns
        date_col = None
        value_col = None
        
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
            elif 'value' in col.lower() or 'index' in col.lower() or 'fng' in col.lower():
                value_col = col
        
        # If columns not found, make some assumptions
        if date_col is None and len(df.columns) >= 2:
            # Try the last column as date
            date_col = df.columns[-1]
        
        if value_col is None and len(df.columns) >= 1:
            # Try the first column as value
            value_col = df.columns[0]
        
        # Create clean dataframe
        clean_df = pd.DataFrame()
        
        # Try to parse date with different formats
        try:
            clean_df['date'] = pd.to_datetime(df[date_col])
        except:
            try:
                clean_df['date'] = pd.to_datetime(df[date_col], format='%d-%m-%Y')
            except:
                try:
                    clean_df['date'] = pd.to_datetime(df[date_col], format='%m-%d-%Y')
                except Exception as e:
                    print(f"Warning: Could not parse date column: {e}")
                    # Use row index as date as last resort
                    clean_df['date'] = pd.date_range(start='2025-05-01', periods=len(df), freq='D')
        
        # Get value column
        clean_df['fng_value'] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Handle missing values
        clean_df['fng_value'] = clean_df['fng_value'].fillna(method='ffill').fillna(method='bfill')
        
        # Sort by date
        clean_df = clean_df.sort_values('date')
        
        return clean_df
    
    except Exception as e:
        print(f"Error loading fear and greed data: {e}")
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=['date', 'fng_value'])

def create_features(df, lookback=24, target_col='price'):
    """Create features for price prediction model."""
    df_features = df.copy()
    
    # Create lagged features
    for i in range(1, lookback + 1):
        df_features[f'price_lag_{i}'] = df_features[target_col].shift(i)
    
    # Create rolling window features
    df_features['price_rolling_mean_12'] = df_features[target_col].rolling(window=12).mean()
    df_features['price_rolling_std_12'] = df_features[target_col].rolling(window=12).std()
    
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
    
    # Add fear and greed index if specified
    if use_fng:
        print("Including fear and greed index in model...")
        fng_df = load_fear_greed()
        
        if not fng_df.empty:
            # Merge on date
            df = pd.merge(df, fng_df[['date', 'fng_value']], on='date', how='left')
            
            # Forward fill missing values
            df['fng_value'] = df['fng_value'].fillna(method='ffill')
            
            # If there are still NaN values, use backward fill
            if df['fng_value'].isna().any():
                df['fng_value'] = df['fng_value'].fillna(method='bfill')
            
            # If there are STILL NaN values, use mean
            if df['fng_value'].isna().any():
                df['fng_value'] = df['fng_value'].fillna(df['fng_value'].mean())
    
    # Create features
    df_features = create_features(df)
    
    # Define features and target
    feature_cols = [col for col in df_features.columns if 'lag' in col or 'rolling' in col or 'hour' in col or 'day_of_week' in col]
    if use_fng and 'fng_value' in df_features.columns:
        feature_cols.append('fng_value')
    
    target_cols = [f'price_next_{i}' for i in range(1, 6)]
    
    X = df_features[feature_cols]
    y = df_features[target_cols]
    
    # Handle any remaining missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    # Use RandomForestRegressor as it's more robust
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Evaluate model
    y_pred = model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    print(f"{crypto_name} model MSE: {mse:.2f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and associated objects
    model_filename = f"models/{crypto_name.lower()}_model.pkl"
    scaler_filename = f"models/{crypto_name.lower()}_scaler.pkl"
    imputer_filename = f"models/{crypto_name.lower()}_imputer.pkl"
    features_filename = f"models/{crypto_name.lower()}_features.txt"
    
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(imputer, imputer_filename)
    
    # Save feature names
    with open(features_filename, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    
    print(f"Model saved to {model_filename}")
    return model, scaler, imputer, feature_cols

if __name__ == "__main__":
    print("Starting model training process...")
    
    # Train Bitcoin model with fear and greed index
    train_and_save_model("bitcoin", use_fng=True)
    
    # Train Ethereum model
    train_and_save_model("ethereum", use_fng=False)
    
    # Train Solana model
    train_and_save_model("solana", use_fng=False)
    
    print("\nAll models trained and saved successfully!")