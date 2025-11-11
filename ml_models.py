import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class MLModels:
    """Class containing various machine learning models for Bitcoin price prediction."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
        
    def prepare_data(self, data, target_column='Close', lookback_days=60):
        """
        Prepare data for machine learning models.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            target_column (str): Column to predict
            lookback_days (int): Number of days to look back for features
        
        Returns:
            dict: Prepared data for training
        """
        # Create feature columns
        features_df = data.copy()
        
        # Price-based features
        features_df['SMA_10'] = features_df['Close'].rolling(window=10).mean()
        features_df['SMA_30'] = features_df['Close'].rolling(window=30).mean()
        features_df['EMA_12'] = features_df['Close'].ewm(span=12).mean()
        features_df['EMA_26'] = features_df['Close'].ewm(span=26).mean()
        
        # Volatility features
        features_df['Volatility'] = features_df['Close'].rolling(window=30).std()
        features_df['Price_Range'] = features_df['High'] - features_df['Low']
        features_df['Price_Position'] = (features_df['Close'] - features_df['Low']) / (features_df['High'] - features_df['Low'])
        
        # Volume features
        features_df['Volume_SMA'] = features_df['Volume'].rolling(window=10).mean()
        features_df['Volume_Ratio'] = features_df['Volume'] / features_df['Volume_SMA']
        
        # Momentum features
        features_df['Momentum_5'] = features_df['Close'] / features_df['Close'].shift(5)
        features_df['Momentum_10'] = features_df['Close'] / features_df['Close'].shift(10)
        features_df['Rate_of_Change'] = features_df['Close'].pct_change(periods=10)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features_df[f'Close_Lag_{lag}'] = features_df['Close'].shift(lag)
            features_df[f'Volume_Lag_{lag}'] = features_df['Volume'].shift(lag)
        
        # Drop NaN values
        features_df = features_df.dropna()
        
        # Select feature columns (exclude target and original OHLCV)
        self.feature_columns = [col for col in features_df.columns 
                               if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
        
        X = features_df[self.feature_columns]
        y = features_df[target_column]
        
        return {
            'X': X,
            'y': y,
            'features_df': features_df,
            'feature_columns': self.feature_columns
        }
    
    def train_linear_regression(self, data):
        """
        Train a Linear Regression model.
        
        Args:
            data (dict): Prepared data
            
        Returns:
            tuple: (trained_model, metrics)
        """
        X, y = data['X'], data['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy
        actual_direction = np.sign(y_test.diff().dropna())
        pred_direction = np.sign(pd.Series(y_pred).diff().dropna())
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': directional_accuracy
        }
        
        # Store scaler for future predictions
        self.scalers['linear'] = scaler
        
        return model, metrics
    
    def train_random_forest(self, data):
        """
        Train a Random Forest model.
        
        Args:
            data (dict): Prepared data
            
        Returns:
            tuple: (trained_model, metrics)
        """
        X, y = data['X'], data['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy
        actual_direction = np.sign(y_test.diff().dropna())
        pred_direction = np.sign(pd.Series(y_pred).diff().dropna())
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': directional_accuracy
        }
        
        return model, metrics
    
    def train_lstm(self, data, sequence_length=60):
        """
        Train an LSTM model.
        
        Args:
            data (dict): Prepared data
            sequence_length (int): Length of input sequences
            
        Returns:
            tuple: (trained_model, metrics)
        """
        features_df = data['features_df']
        
        # Prepare data for LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(features_df[['Close']])
        
        # Create sequences
        X_lstm, y_lstm = [], []
        for i in range(sequence_length, len(scaled_data)):
            X_lstm.append(scaled_data[i-sequence_length:i, 0])
            y_lstm.append(scaled_data[i, 0])
        
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
        
        # Split data
        split_index = int(0.8 * len(X_lstm))
        X_train, X_test = X_lstm[:split_index], X_lstm[split_index:]
        y_train, y_test = y_lstm[:split_index], y_lstm[split_index:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Train model
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_split=0.1,
            verbose=0
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_actual = scaler.inverse_transform(y_pred).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        r2 = r2_score(y_test_actual, y_pred_actual)
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(y_test_actual))
        pred_direction = np.sign(np.diff(y_pred_actual))
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': directional_accuracy
        }
        
        # Store scaler for future predictions
        self.scalers['lstm'] = scaler
        
        return model, metrics
    
    def predict_linear_regression(self, model, data, prediction_days):
        """
        Generate predictions using Linear Regression model.
        
        Args:
            model: Trained Linear Regression model
            data (pd.DataFrame): Historical data
            prediction_days (int): Number of days to predict
            
        Returns:
            tuple: (predictions, confidence_intervals)
        """
        # Prepare the most recent data
        recent_data = self.prepare_data(data)
        X_recent = recent_data['X'].tail(1)
        
        predictions = []
        confidence_intervals = []
        
        # Current price for relative predictions
        current_price = data['Close'].iloc[-1]
        
        # Generate predictions
        for day in range(prediction_days):
            # Scale the features
            X_scaled = self.scalers['linear'].transform(X_recent)
            
            # Make prediction
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
            
            # Calculate confidence interval (simplified approach)
            # In practice, you'd use prediction intervals from the model
            confidence_interval = abs(pred - current_price) * 0.1  # 10% confidence range
            confidence_intervals.append(confidence_interval)
            
            # Update features for next prediction (simplified)
            # This would normally involve updating all features based on the prediction
            # For simplicity, we'll use the current features
        
        return predictions, confidence_intervals
    
    def predict_random_forest(self, model, data, prediction_days):
        """
        Generate predictions using Random Forest model.
        
        Args:
            model: Trained Random Forest model
            data (pd.DataFrame): Historical data
            prediction_days (int): Number of days to predict
            
        Returns:
            tuple: (predictions, confidence_intervals)
        """
        # Prepare the most recent data
        recent_data = self.prepare_data(data)
        X_recent = recent_data['X'].tail(1)
        
        predictions = []
        confidence_intervals = []
        
        # Current price for relative predictions
        current_price = data['Close'].iloc[-1]
        
        # Generate predictions
        for day in range(prediction_days):
            # Make prediction
            pred = model.predict(X_recent)[0]
            predictions.append(pred)
            
            # Calculate confidence interval using prediction std from trees
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X_recent)[0] for tree in model.estimators_])
            confidence_interval = np.std(tree_predictions) * 1.96  # 95% confidence
            confidence_intervals.append(confidence_interval)
        
        return predictions, confidence_intervals
    
    def predict_lstm(self, model, data, prediction_days, sequence_length=60):
        """
        Generate predictions using LSTM model.
        
        Args:
            model: Trained LSTM model
            data (pd.DataFrame): Historical data
            prediction_days (int): Number of days to predict
            sequence_length (int): Length of input sequences
            
        Returns:
            tuple: (predictions, confidence_intervals)
        """
        # Get the last sequence_length days of data
        recent_data = data['Close'].tail(sequence_length).values
        
        # Scale the data
        scaler = self.scalers['lstm']
        scaled_data = scaler.transform(recent_data.reshape(-1, 1))
        
        predictions = []
        confidence_intervals = []
        
        # Generate predictions
        current_sequence = scaled_data.flatten()
        
        for day in range(prediction_days):
            # Prepare input for prediction
            X_input = current_sequence[-sequence_length:].reshape(1, sequence_length, 1)
            
            # Make prediction
            pred_scaled = model.predict(X_input, verbose=0)[0, 0]
            
            # Inverse transform prediction
            pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(pred)
            
            # Calculate confidence interval (simplified)
            # In practice, you'd use Monte Carlo dropout or ensemble methods
            confidence_interval = abs(pred - data['Close'].iloc[-1]) * 0.15  # 15% confidence range
            confidence_intervals.append(confidence_interval)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence, pred_scaled)
        
        return predictions, confidence_intervals
