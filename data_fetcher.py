import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import logging

class DataFetcher:
    """Class to handle Bitcoin data fetching from various sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_bitcoin_data(_self, symbol="BTC-USD", period="1y", interval="1d"):
        """
        Fetch Bitcoin historical data using yfinance.
        
        Args:
            symbol (str): Bitcoin symbol (default: BTC-USD)
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pd.DataFrame: Bitcoin price data with OHLCV columns
        """
        try:
            # Create yfinance ticker
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                _self.logger.error(f"No data returned for {symbol}")
                return None
            
            # Clean and prepare data
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                _self.logger.error(f"Missing required columns in data")
                return None
            
            # Add additional calculated columns
            data['Price_Change'] = data['Close'].pct_change()
            data['Price_Change_Abs'] = data['Close'].diff()
            data['High_Low_Spread'] = data['High'] - data['Low']
            data['OHLC_Average'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
            
            _self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            _self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            st.error(f"Failed to fetch Bitcoin data: {str(e)}")
            return None
    
    def get_latest_price(self, symbol="BTC-USD"):
        """
        Get the latest Bitcoin price.
        
        Args:
            symbol (str): Bitcoin symbol
            
        Returns:
            float: Latest Bitcoin price
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                return data['Close'].iloc[-1]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching latest price: {str(e)}")
            return None
    
    def validate_data(self, data):
        """
        Validate the fetched data for quality and completeness.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            
        Returns:
            tuple: (is_valid, issues_list)
        """
        issues = []
        
        if data is None or data.empty:
            issues.append("Data is empty or None")
            return False, issues
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # Check for data consistency
        if (data['High'] < data['Low']).any():
            issues.append("High prices are lower than low prices in some records")
        
        if (data['High'] < data['Close']).any():
            issues.append("High prices are lower than close prices in some records")
        
        if (data['Low'] > data['Close']).any():
            issues.append("Low prices are higher than close prices in some records")
        
        # Check for negative values
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (data[col] <= 0).any():
                issues.append(f"Negative or zero values found in {col}")
        
        # Check for missing values
        if data.isnull().any().any():
            null_columns = data.columns[data.isnull().any()].tolist()
            issues.append(f"Missing values found in columns: {null_columns}")
        
        # Check data recency
        if not data.empty:
            latest_date = data.index[-1]
            days_old = (datetime.now() - latest_date.replace(tzinfo=None)).days
            if days_old > 7:
                issues.append(f"Data is {days_old} days old")
        
        return len(issues) == 0, issues
    
    def get_data_summary(self, data):
        """
        Get a summary of the fetched data.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            
        Returns:
            dict: Summary statistics
        """
        if data is None or data.empty:
            return {}
        
        summary = {
            'total_records': len(data),
            'date_range': {
                'start': data.index[0].strftime('%Y-%m-%d'),
                'end': data.index[-1].strftime('%Y-%m-%d')
            },
            'price_stats': {
                'current_price': data['Close'].iloc[-1],
                'min_price': data['Close'].min(),
                'max_price': data['Close'].max(),
                'avg_price': data['Close'].mean(),
                'total_return': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            },
            'volume_stats': {
                'avg_volume': data['Volume'].mean(),
                'max_volume': data['Volume'].max(),
                'min_volume': data['Volume'].min()
            }
        }
        
        return summary
