import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def format_currency(amount, currency='USD', decimal_places=2):
    """
    Format a number as currency.
    
    Args:
        amount (float): Amount to format
        currency (str): Currency symbol
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted currency string
    """
    if currency == 'USD':
        return f"${amount:,.{decimal_places}f}"
    else:
        return f"{amount:,.{decimal_places}f} {currency}"

def calculate_percentage_change(old_value, new_value):
    """
    Calculate percentage change between two values.
    
    Args:
        old_value (float): Original value
        new_value (float): New value
        
    Returns:
        float: Percentage change
    """
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100

def calculate_returns(prices):
    """
    Calculate returns from price series.
    
    Args:
        prices (pd.Series): Price series
        
    Returns:
        pd.Series: Returns series
    """
    return prices.pct_change()

def calculate_volatility(prices, window=30):
    """
    Calculate rolling volatility.
    
    Args:
        prices (pd.Series): Price series
        window (int): Rolling window size
        
    Returns:
        pd.Series: Volatility series
    """
    returns = calculate_returns(prices)
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio.
    
    Args:
        returns (pd.Series): Returns series
        risk_free_rate (float): Risk-free rate
        
    Returns:
        float: Sharpe ratio
    """
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown.
    
    Args:
        prices (pd.Series): Price series
        
    Returns:
        float: Maximum drawdown as percentage
    """
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()

def validate_data_quality(data):
    """
    Validate data quality and return issues.
    
    Args:
        data (pd.DataFrame): Data to validate
        
    Returns:
        dict: Validation results
    """
    issues = []
    
    # Check for missing values
    if data.isnull().any().any():
        null_cols = data.columns[data.isnull().any()].tolist()
        issues.append(f"Missing values in columns: {null_cols}")
    
    # Check for negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in data.columns and (data[col] <= 0).any():
            issues.append(f"Negative or zero values in {col}")
    
    # Check for data consistency
    if 'High' in data.columns and 'Low' in data.columns:
        if (data['High'] < data['Low']).any():
            issues.append("High prices lower than low prices")
    
    # Check for outliers (prices that change by more than 50% in a day)
    if 'Close' in data.columns:
        price_changes = data['Close'].pct_change().abs()
        if (price_changes > 0.5).any():
            issues.append("Extreme price changes detected (>50% in one day)")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'total_records': len(data),
        'date_range': {
            'start': data.index[0] if not data.empty else None,
            'end': data.index[-1] if not data.empty else None
        }
    }

def create_date_range(start_date, end_date, freq='D'):
    """
    Create a date range.
    
    Args:
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        freq (str): Frequency
        
    Returns:
        pd.DatetimeIndex: Date range
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def resample_data(data, freq='D', agg_func='last'):
    """
    Resample data to different frequency.
    
    Args:
        data (pd.DataFrame): Data to resample
        freq (str): Target frequency
        agg_func (str or dict): Aggregation function
        
    Returns:
        pd.DataFrame: Resampled data
    """
    if agg_func == 'ohlc':
        # For OHLC data
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        return data.resample(freq).agg(agg_dict)
    else:
        return data.resample(freq).agg(agg_func)

def calculate_correlation_matrix(data, method='pearson'):
    """
    Calculate correlation matrix for numerical columns.
    
    Args:
        data (pd.DataFrame): Input data
        method (str): Correlation method
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    return data[numeric_cols].corr(method=method)

def detect_outliers(data, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a column.
    
    Args:
        data (pd.DataFrame): Input data
        column (str): Column to check
        method (str): Detection method ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        return z_scores > threshold
    
    return pd.Series([False] * len(data), index=data.index)

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator (float): Numerator
        denominator (float): Denominator
        default (float): Default value if division by zero
        
    Returns:
        float: Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator

def format_large_number(number):
    """
    Format large numbers with appropriate suffixes.
    
    Args:
        number (float): Number to format
        
    Returns:
        str: Formatted number string
    """
    if number >= 1e12:
        return f"{number/1e12:.2f}T"
    elif number >= 1e9:
        return f"{number/1e9:.2f}B"
    elif number >= 1e6:
        return f"{number/1e6:.2f}M"
    elif number >= 1e3:
        return f"{number/1e3:.2f}K"
    else:
        return f"{number:.2f}"

def get_market_hours():
    """
    Get market hours information.
    
    Returns:
        dict: Market hours information
    """
    return {
        'crypto': {
            'open': '24/7',
            'timezone': 'UTC',
            'description': 'Cryptocurrency markets are open 24/7'
        },
        'stock': {
            'open': '09:30',
            'close': '16:00',
            'timezone': 'EST',
            'description': 'Stock markets are open 9:30 AM - 4:00 PM EST'
        }
    }

def is_market_open(market_type='crypto'):
    """
    Check if market is currently open.
    
    Args:
        market_type (str): Type of market ('crypto' or 'stock')
        
    Returns:
        bool: True if market is open
    """
    if market_type == 'crypto':
        return True  # Crypto markets are always open
    
    # For stock markets, check if it's a weekday and within trading hours
    now = datetime.now()
    if now.weekday() >= 5:  # Weekend
        return False
    
    # Check if within trading hours (simplified)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close
