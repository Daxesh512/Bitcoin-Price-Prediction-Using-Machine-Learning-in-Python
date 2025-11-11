import pandas as pd
import numpy as np
import pandas_ta as ta

class TechnicalIndicators:
    """Class for calculating various technical indicators for Bitcoin price analysis."""
    
    def __init__(self):
        pass
    
    def add_all_indicators(self, data):
        """
        Add all technical indicators to the dataframe.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            
        Returns:
            pd.DataFrame: Data with technical indicators added
        """
        df = data.copy()
        
        # Moving Averages
        df = self.add_moving_averages(df)
        
        # Momentum Indicators
        df = self.add_momentum_indicators(df)
        
        # Volatility Indicators
        df = self.add_volatility_indicators(df)
        
        # Volume Indicators
        df = self.add_volume_indicators(df)
        
        # Trend Indicators
        df = self.add_trend_indicators(df)
        
        return df
    
    def add_moving_averages(self, data):
        """
        Add various moving averages.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            
        Returns:
            pd.DataFrame: Data with moving averages
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = ta.sma(df['Close'], length=5)
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_100'] = ta.sma(df['Close'], length=100)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        
        # Exponential Moving Averages
        df['EMA_5'] = ta.ema(df['Close'], length=5)
        df['EMA_10'] = ta.ema(df['Close'], length=10)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        
        # Weighted Moving Average
        df['WMA_10'] = ta.wma(df['Close'], length=10)
        df['WMA_20'] = ta.wma(df['Close'], length=20)
        
        return df
    
    def add_momentum_indicators(self, data):
        """
        Add momentum indicators.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            
        Returns:
            pd.DataFrame: Data with momentum indicators
        """
        df = data.copy()
        
        # RSI (Relative Strength Index)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['RSI_30'] = ta.rsi(df['Close'], length=30)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Histogram'] = macd['MACDh_12_26_9']
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch['STOCHk_14_3_3']
        df['Stoch_D'] = stoch['STOCHd_14_3_3']
        
        # Williams %R
        df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'])
        
        # Rate of Change
        df['ROC'] = ta.roc(df['Close'], length=10)
        df['ROC_20'] = ta.roc(df['Close'], length=20)
        
        # Momentum
        df['MOM'] = ta.mom(df['Close'], length=10)
        df['MOM_20'] = ta.mom(df['Close'], length=20)
        
        return df
    
    def add_volatility_indicators(self, data):
        """
        Add volatility indicators.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            
        Returns:
            pd.DataFrame: Data with volatility indicators
        """
        df = data.copy()
        
        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20)
        df['BB_Lower'] = bb['BBL_20_2.0']
        df['BB_Middle'] = bb['BBM_20_2.0']
        df['BB_Upper'] = bb['BBU_20_2.0']
        df['BB_Width'] = bb['BBB_20_2.0']
        df['BB_Percent'] = bb['BBP_20_2.0']
        
        # Average True Range
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_20'] = ta.atr(df['High'], df['Low'], df['Close'], length=20)
        
        # Keltner Channel
        kc = ta.kc(df['High'], df['Low'], df['Close'])
        df['KC_Lower'] = kc['KCLe_20_2']
        df['KC_Middle'] = kc['KCBe_20_2']
        df['KC_Upper'] = kc['KCUe_20_2']
        
        # Donchian Channel
        dc = ta.donchian(df['High'], df['Low'])
        df['DC_Lower'] = dc['DCL_20_20']
        df['DC_Middle'] = dc['DCM_20_20']
        df['DC_Upper'] = dc['DCU_20_20']
        
        return df
    
    def add_volume_indicators(self, data):
        """
        Add volume indicators.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            
        Returns:
            pd.DataFrame: Data with volume indicators
        """
        df = data.copy()
        
        # Volume Moving Averages
        df['Volume_SMA_10'] = ta.sma(df['Volume'], length=10)
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['Volume_EMA_10'] = ta.ema(df['Volume'], length=10)
        
        # On-Balance Volume
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        
        # Accumulation/Distribution Line
        df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Chaikin Money Flow
        df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume Price Trend
        df['VPT'] = ta.vpt(df['Close'], df['Volume'])
        
        # Money Flow Index
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return df
    
    def add_trend_indicators(self, data):
        """
        Add trend indicators.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            
        Returns:
            pd.DataFrame: Data with trend indicators
        """
        df = data.copy()
        
        # Parabolic SAR
        df['PSAR'] = ta.psar(df['High'], df['Low'])
        
        # Average Directional Index
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx['ADX_14']
        df['DMP'] = adx['DMP_14']
        df['DMN'] = adx['DMN_14']
        
        # Commodity Channel Index
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
        
        # Aroon
        aroon = ta.aroon(df['High'], df['Low'])
        df['Aroon_Up'] = aroon['AROONU_14']
        df['Aroon_Down'] = aroon['AROOND_14']
        df['Aroon_Osc'] = aroon['AROONOSC_14']
        
        # Supertrend
        supertrend = ta.supertrend(df['High'], df['Low'], df['Close'])
        df['Supertrend'] = supertrend['SUPERT_7_3.0']
        df['Supertrend_Direction'] = supertrend['SUPERTd_7_3.0']
        
        return df
    
    def calculate_custom_indicators(self, data):
        """
        Calculate custom indicators specific to Bitcoin analysis.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            
        Returns:
            pd.DataFrame: Data with custom indicators
        """
        df = data.copy()
        
        # Price position within the day's range
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Intraday intensity
        df['Intraday_Intensity'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        
        # True Range
        df['True_Range'] = np.maximum(df['High'] - df['Low'], 
                                     np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                               abs(df['Low'] - df['Close'].shift(1))))
        
        # Gap analysis
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Percent'] = (df['Gap'] / df['Close'].shift(1)) * 100
        
        # Volatility measures
        df['Price_Volatility'] = df['Close'].rolling(window=20).std()
        df['Volume_Volatility'] = df['Volume'].rolling(window=20).std()
        
        # Support and resistance levels (simplified)
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        # Price momentum
        df['Price_Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Price_Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Price_Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volume momentum
        df['Volume_Momentum'] = df['Volume'] / df['Volume'].rolling(window=10).mean()
        
        return df
    
    def get_signal_summary(self, data):
        """
        Get a summary of buy/sell signals from various indicators.
        
        Args:
            data (pd.DataFrame): Bitcoin price data with indicators
            
        Returns:
            dict: Summary of signals
        """
        if data.empty:
            return {}
        
        latest = data.iloc[-1]
        signals = {}
        
        # RSI signals
        if 'RSI' in data.columns:
            if latest['RSI'] > 70:
                signals['RSI'] = 'Overbought (Sell)'
            elif latest['RSI'] < 30:
                signals['RSI'] = 'Oversold (Buy)'
            else:
                signals['RSI'] = 'Neutral'
        
        # MACD signals
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                signals['MACD'] = 'Bullish'
            else:
                signals['MACD'] = 'Bearish'
        
        # Bollinger Bands signals
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
            if latest['Close'] > latest['BB_Upper']:
                signals['Bollinger_Bands'] = 'Overbought'
            elif latest['Close'] < latest['BB_Lower']:
                signals['Bollinger_Bands'] = 'Oversold'
            else:
                signals['Bollinger_Bands'] = 'Normal'
        
        # Moving average signals
        if all(col in data.columns for col in ['Close', 'SMA_50', 'SMA_200']):
            if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
                signals['MA_Trend'] = 'Strong Bullish'
            elif latest['Close'] > latest['SMA_50']:
                signals['MA_Trend'] = 'Bullish'
            elif latest['Close'] < latest['SMA_50']:
                signals['MA_Trend'] = 'Bearish'
        
        return signals
