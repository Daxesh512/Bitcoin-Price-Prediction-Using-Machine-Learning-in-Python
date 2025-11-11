import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import time
import math

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bitcoin theme
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
    }
    .stMetric {
        background-color: #2E2E2E;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #F7931A;
    }
    .stButton > button {
        background-color: #F7931A;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #E6851A;
    }
    .bitcoin-price {
        font-size: 2rem;
        font-weight: bold;
        color: #F7931A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(45deg, #2E2E2E, #3E3E3E);
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #F7931A;
    }
    .prediction-card {
        background: linear-gradient(135deg, #2E2E2E, #1E1E1E);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4285F4;
        margin: 1rem 0;
    }
    .sidebar .stSelectbox > div > div {
        background-color: #2E2E2E;
    }
    .price-history {
        background-color: #2E2E2E;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #4285F4;
        font-family: monospace;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bitcoin_price' not in st.session_state:
    st.session_state.bitcoin_price = None
if 'price_history' not in st.session_state:
    st.session_state.price_history = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'price_timestamps' not in st.session_state:
    st.session_state.price_timestamps = []
if 'real_time_enabled' not in st.session_state:
    st.session_state.real_time_enabled = False
if 'update_counter' not in st.session_state:
    st.session_state.update_counter = 0

def fetch_bitcoin_price():
    """Fetch current Bitcoin price from CoinGecko API"""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            bitcoin_data = data.get('bitcoin', {})
            
            current_price = bitcoin_data.get('usd')
            price_change_24h = bitcoin_data.get('usd_24h_change', 0)
            last_updated = bitcoin_data.get('last_updated_at')
            
            return {
                'price': current_price,
                'change_24h': price_change_24h,
                'timestamp': datetime.now(),
                'last_updated': last_updated
            }
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error fetching Bitcoin price: {str(e)}")
        return None

def simple_moving_average(prices, window):
    """Calculate simple moving average"""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window

def linear_trend_prediction(prices, days_ahead):
    """Simple linear trend prediction"""
    if len(prices) < 2:
        return None
    
    # Calculate average daily change over last 7 days
    recent_prices = prices[-7:] if len(prices) >= 7 else prices
    
    if len(recent_prices) < 2:
        return None
    
    # Simple linear regression slope
    n = len(recent_prices)
    x_values = list(range(n))
    y_values = recent_prices
    
    # Calculate slope
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    
    numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
    denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    
    # Predict future prices
    last_price = prices[-1]
    predictions = []
    
    for day in range(1, days_ahead + 1):
        predicted_price = last_price + (slope * day)
        predictions.append(max(predicted_price, 0))  # Ensure non-negative
    
    return predictions

def momentum_prediction(prices, days_ahead):
    """Momentum-based prediction"""
    if len(prices) < 10:
        return None
    
    # Calculate momentum over different periods
    momentum_3d = (prices[-1] / prices[-4] - 1) if len(prices) >= 4 else 0
    momentum_7d = (prices[-1] / prices[-8] - 1) if len(prices) >= 8 else 0
    
    # Average momentum
    avg_momentum = (momentum_3d + momentum_7d) / 2
    
    # Apply momentum to future predictions with decay
    predictions = []
    last_price = prices[-1]
    
    for day in range(1, days_ahead + 1):
        # Decay momentum over time
        momentum_factor = avg_momentum * (0.95 ** day)
        predicted_price = last_price * (1 + momentum_factor)
        predictions.append(max(predicted_price, 0))
    
    return predictions

def create_price_chart_ascii(prices, width=50):
    """Create ASCII chart for price visualization"""
    if len(prices) < 2:
        return "Not enough data for chart"
    
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price
    
    if price_range == 0:
        return "Price unchanged"
    
    chart = []
    for price in prices:
        normalized = (price - min_price) / price_range
        bar_length = int(normalized * width)
        bar = "█" * bar_length + "░" * (width - bar_length)
        chart.append(f"${price:8,.0f} |{bar}|")
    
    return "\n".join(chart)

def create_simple_chart_data(prices, timestamps=None):
    """Create simple chart data that Streamlit can handle"""
    if not prices:
        return {}
    
    # Create simple data structure for charts
    chart_data = {}
    
    if timestamps:
        # Use timestamps if provided
        for i, (price, timestamp) in enumerate(zip(prices, timestamps)):
            chart_data[timestamp.strftime('%H:%M:%S')] = price
    else:
        # Use simple indexing
        for i, price in enumerate(prices):
            chart_data[f"Update {i+1}"] = price
    
    return chart_data

# Main title and header
st.title("₿ Bitcoin Price Prediction Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Prediction parameters
    st.subheader("Prediction Settings")
    prediction_days = st.slider(
        "Days to Predict",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to predict into the future"
    )
    
    # Model selection
    st.subheader("Prediction Models")
    use_linear = st.checkbox("Linear Trend", value=True)
    use_momentum = st.checkbox("Momentum Model", value=True)
    
    # Real-time controls
    st.subheader("Real-Time Updates")
    real_time_mode = st.checkbox("Enable Real-Time Mode", value=st.session_state.real_time_enabled)
    
    if real_time_mode != st.session_state.real_time_enabled:
        st.session_state.real_time_enabled = real_time_mode
        if real_time_mode:
            st.success("Real-time mode enabled!")
        else:
            st.info("Real-time mode disabled")
    
    refresh_interval = st.selectbox(
        "Update Interval",
        options=[10, 15, 30, 60],
        index=2,
        format_func=lambda x: f"{x} seconds",
        help="How often to fetch new Bitcoin prices"
    )
    
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    
    # Manual refresh
    if st.button("🔄 Refresh Data", type="primary"):
        with st.spinner("Fetching latest Bitcoin price..."):
            price_data = fetch_bitcoin_price()
            if price_data:
                st.session_state.bitcoin_price = price_data
                st.session_state.price_history.append(price_data['price'])
                st.session_state.price_timestamps.append(price_data['timestamp'])
                st.session_state.update_counter += 1
                
                # Keep only last 50 price points for real-time charts
                if len(st.session_state.price_history) > 50:
                    st.session_state.price_history = st.session_state.price_history[-50:]
                    st.session_state.price_timestamps = st.session_state.price_timestamps[-50:]
                
                st.session_state.last_update = datetime.now()
                st.success("Data updated successfully!")

# Real-time auto-refresh logic
if auto_refresh or st.session_state.real_time_enabled:
    refresh_time = refresh_interval if st.session_state.real_time_enabled else 30
    
    if (st.session_state.last_update is None or 
        (datetime.now() - st.session_state.last_update).seconds >= refresh_time):
        
        with st.spinner("🔄 Updating Bitcoin price..."):
            price_data = fetch_bitcoin_price()
            if price_data:
                st.session_state.bitcoin_price = price_data
                st.session_state.price_history.append(price_data['price'])
                st.session_state.price_timestamps.append(price_data['timestamp'])
                st.session_state.update_counter += 1
                
                # Keep only last 50 price points for real-time charts
                if len(st.session_state.price_history) > 50:
                    st.session_state.price_history = st.session_state.price_history[-50:]
                    st.session_state.price_timestamps = st.session_state.price_timestamps[-50:]
                
                st.session_state.last_update = datetime.now()
                st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Current price display
    if st.session_state.bitcoin_price:
        price_data = st.session_state.bitcoin_price
        
        # Real-time status indicator
        real_time_status = "🔴 LIVE" if st.session_state.real_time_enabled else "⚪ STATIC"
        status_color = "#00D4AA" if st.session_state.real_time_enabled else "#808080"
        
        st.markdown(f"""
        <div class="bitcoin-price">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 1.2rem; color: #FFFFFF;">
                    Current Bitcoin Price
                </div>
                <div style="font-size: 0.9rem; color: {status_color}; font-weight: bold;">
                    {real_time_status} | Update #{st.session_state.update_counter}
                </div>
            </div>
            <div style="font-size: 3rem; font-weight: bold; color: #F7931A;">
                ${price_data['price']:,.2f}
            </div>
            <div style="font-size: 1rem; color: {'#00D4AA' if price_data['change_24h'] >= 0 else '#FF6B6B'}; margin-top: 0.5rem;">
                24h Change: {price_data['change_24h']:+.2f}%
            </div>
            <div style="font-size: 0.9rem; color: #808080; margin-top: 0.5rem;">
                Last updated: {price_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Real-time price charts section
        if len(st.session_state.price_history) > 1:
            st.subheader("📈 Real-Time Price Charts")
            
            # Create tabs for different chart views
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["🔴 Live Chart", "📊 Full History", "📈 ASCII View"])
            
            with chart_tab1:
                # Real-time live chart (last 15 data points for responsiveness)
                live_prices = st.session_state.price_history[-15:]
                
                # Add real-time chart header
                col_live1, col_live2 = st.columns([3, 1])
                
                with col_live1:
                    st.write("**Live Bitcoin Price Movement**")
                
                with col_live2:
                    if st.session_state.real_time_enabled:
                        st.markdown(f"""
                        <div style="background-color: #FF4B4B; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; text-align: center; font-size: 0.8rem; font-weight: bold;">
                            🔴 LIVE
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #808080; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; text-align: center; font-size: 0.8rem; font-weight: bold;">
                            ⚪ PAUSED
                        </div>
                        """, unsafe_allow_html=True)
                
                # Live chart
                st.line_chart(
                    data=live_prices,
                    height=350,
                    use_container_width=True
                )
                
                # Real-time metrics
                if len(live_prices) >= 2:
                    price_change = live_prices[-1] - live_prices[-2]
                    change_pct = (price_change / live_prices[-2]) * 100
                    session_change = live_prices[-1] - live_prices[0] if len(live_prices) > 1 else 0
                    session_change_pct = (session_change / live_prices[0]) * 100 if len(live_prices) > 1 else 0
                    
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                    
                    with col_metric1:
                        st.metric("Last Tick", f"${price_change:+,.2f}", f"{change_pct:+.2f}%")
                    
                    with col_metric2:
                        trend_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
                        trend_text = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"
                        st.metric("Direction", f"{trend_emoji} {trend_text}")
                    
                    with col_metric3:
                        st.metric("Session Change", f"${session_change:+,.2f}", f"{session_change_pct:+.2f}%")
                    
                    with col_metric4:
                        volatility = abs(price_change / live_prices[-2]) * 100 if live_prices[-2] != 0 else 0
                        vol_level = "HIGH" if volatility > 1 else "MED" if volatility > 0.1 else "LOW"
                        vol_color = "#FF6B6B" if volatility > 1 else "#F7931A" if volatility > 0.1 else "#00D4AA"
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #808080;">Volatility</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: {vol_color};">{vol_level}</div>
                            <div style="font-size: 0.8rem; color: #808080;">{volatility:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with chart_tab2:
                # Full history chart (all available data)
                st.write("**Complete Price History**")
                
                all_prices = st.session_state.price_history
                st.line_chart(
                    data=all_prices,
                    height=400,
                    use_container_width=True
                )
                
                # Historical statistics
                if len(all_prices) >= 2:
                    col_hist1, col_hist2, col_hist3 = st.columns(3)
                    
                    with col_hist1:
                        high_price = max(all_prices)
                        low_price = min(all_prices)
                        st.metric("Session High", f"${high_price:,.2f}")
                        st.metric("Session Low", f"${low_price:,.2f}")
                    
                    with col_hist2:
                        avg_price = sum(all_prices) / len(all_prices)
                        total_updates = len(all_prices)
                        st.metric("Average Price", f"${avg_price:,.2f}")
                        st.metric("Total Updates", f"{total_updates}")
                    
                    with col_hist3:
                        price_range = high_price - low_price
                        range_pct = (price_range / low_price) * 100 if low_price > 0 else 0
                        st.metric("Price Range", f"${price_range:,.2f}")
                        st.metric("Range %", f"{range_pct:.2f}%")
            
            with chart_tab3:
                # ASCII chart for text-based visualization
                recent_prices_ascii = st.session_state.price_history[-12:]  # Last 12 data points
                chart = create_price_chart_ascii(recent_prices_ascii, 35)
                
                st.markdown(f"""
                <div class="price-history">
                    <strong>Real-Time ASCII Chart (Last 12 Updates):</strong><br><br>
                    <pre>{chart}</pre>
                </div>
                """, unsafe_allow_html=True)
                
                # ASCII chart legend
                st.markdown("""
                **Chart Legend:**
                - Each line represents one price update
                - Bars show relative price within the range
                - Longer bars = higher prices
                - Updates flow from top (oldest) to bottom (newest)
                """)
            
            # Price statistics
            st.subheader("📊 Price Statistics")
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                high_price = max(st.session_state.price_history)
                st.metric("Highest", f"${high_price:,.2f}")
            
            with col_stats2:
                low_price = min(st.session_state.price_history)
                st.metric("Lowest", f"${low_price:,.2f}")
            
            with col_stats3:
                avg_price = sum(st.session_state.price_history) / len(st.session_state.price_history)
                st.metric("Average", f"${avg_price:,.2f}")
    else:
        st.info("Click 'Refresh Data' to load the current Bitcoin price")

with col2:
    # Predictions section
    if (st.session_state.bitcoin_price and 
        len(st.session_state.price_history) >= 2 and 
        (use_linear or use_momentum)):
        
        st.subheader("🔮 Price Predictions")
        
        predictions = {}
        
        # Linear trend prediction
        if use_linear:
            linear_pred = linear_trend_prediction(st.session_state.price_history, prediction_days)
            if linear_pred:
                predictions['Linear Trend'] = linear_pred
        
        # Momentum prediction
        if use_momentum:
            momentum_pred = momentum_prediction(st.session_state.price_history, prediction_days)
            if momentum_pred:
                predictions['Momentum'] = momentum_pred
        
        # Create prediction visualization tabs
        pred_tab1, pred_tab2 = st.tabs(["📊 Prediction Charts", "📋 Summary"])
        
        with pred_tab1:
            # Combined prediction chart
            st.subheader("🔮 Future Price Predictions")
            
            # Combine historical and prediction data for visualization
            historical_recent = st.session_state.price_history[-10:]  # Last 10 historical points
            
            # Create combined data for each model
            for model_name, pred_prices in predictions.items():
                if pred_prices:
                    # Combine historical + predictions
                    combined_data = historical_recent + pred_prices
                    
                    # Create labels for the combined data
                    historical_labels = [f"H-{len(historical_recent)-i}" for i in range(len(historical_recent))]
                    prediction_labels = [f"P+{i+1}" for i in range(len(pred_prices))]
                    all_labels = historical_labels + prediction_labels
                    
                    st.write(f"**{model_name} Model**")
                    st.line_chart(combined_data, height=250)
                    
                    # Add separator line indicator
                    st.markdown(f"""
                    <div style="background-color: #2E2E2E; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; font-size: 0.8rem; color: #808080;">
                        📍 Historical data ends at point {len(historical_recent)} | Predictions start at point {len(historical_recent)+1}
                    </div>
                    """, unsafe_allow_html=True)
        
        with pred_tab2:
            # Display prediction summaries
            for model_name, pred_prices in predictions.items():
                if pred_prices:
                    final_price = pred_prices[-1]
                    current_price = st.session_state.bitcoin_price['price']
                    change_pct = ((final_price - current_price) / current_price) * 100
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h4 style="color: #4285F4; margin-bottom: 1rem;">{model_name}</h4>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #F7931A; margin-bottom: 0.5rem;">
                            ${final_price:,.2f}
                        </div>
                        <div style="color: {'#00D4AA' if change_pct >= 0 else '#FF6B6B'}; font-size: 1rem;">
                            {change_pct:+.2f}% in {prediction_days} days
                        </div>
                        <div style="color: #808080; font-size: 0.9rem; margin-top: 0.5rem;">
                            Prediction range: ${min(pred_prices):,.2f} - ${max(pred_prices):,.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Technical indicators with charts
        st.subheader("📊 Technical Indicators")
        
        # Create tabs for technical analysis
        tech_tab1, tech_tab2 = st.tabs(["📈 Moving Averages", "📊 Volatility Analysis"])
        
        with tech_tab1:
            if len(st.session_state.price_history) >= 5:
                # Calculate multiple moving averages
                prices = st.session_state.price_history
                current_price = st.session_state.bitcoin_price['price']
                
                # Calculate different MA periods
                ma_data = {}
                if len(prices) >= 3:
                    ma_3 = simple_moving_average(prices, 3)
                    ma_data['MA-3'] = ma_3
                if len(prices) >= 5:
                    ma_5 = simple_moving_average(prices, 5)
                    ma_data['MA-5'] = ma_5
                if len(prices) >= 10:
                    ma_10 = simple_moving_average(prices, 10)
                    ma_data['MA-10'] = ma_10
                
                # Display MA values and trends
                col_ma1, col_ma2, col_ma3 = st.columns(3)
                
                with col_ma1:
                    if 'MA-3' in ma_data:
                        trend = "Bullish" if current_price > ma_data['MA-3'] else "Bearish"
                        trend_color = "#00D4AA" if trend == "Bullish" else "#FF6B6B"
                        st.markdown(f"""
                        <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 8px;">
                            <div style="color: #FFFFFF; margin-bottom: 0.5rem;">3-Period MA</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #F7931A;">${ma_data['MA-3']:,.2f}</div>
                            <div style="color: {trend_color}; font-size: 0.9rem;">{trend}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_ma2:
                    if 'MA-5' in ma_data:
                        trend = "Bullish" if current_price > ma_data['MA-5'] else "Bearish"
                        trend_color = "#00D4AA" if trend == "Bullish" else "#FF6B6B"
                        st.markdown(f"""
                        <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 8px;">
                            <div style="color: #FFFFFF; margin-bottom: 0.5rem;">5-Period MA</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #F7931A;">${ma_data['MA-5']:,.2f}</div>
                            <div style="color: {trend_color}; font-size: 0.9rem;">{trend}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_ma3:
                    if 'MA-10' in ma_data:
                        trend = "Bullish" if current_price > ma_data['MA-10'] else "Bearish"
                        trend_color = "#00D4AA" if trend == "Bullish" else "#FF6B6B"
                        st.markdown(f"""
                        <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 8px;">
                            <div style="color: #FFFFFF; margin-bottom: 0.5rem;">10-Period MA</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #F7931A;">${ma_data['MA-10']:,.2f}</div>
                            <div style="color: {trend_color}; font-size: 0.9rem;">{trend}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Chart showing price vs moving averages
                if len(prices) >= 10:
                    st.write("**Price vs Moving Averages (Last 15 periods)**")
                    recent_prices = prices[-15:]
                    
                    # Calculate MAs for the recent period
                    ma_chart_data = []
                    for i in range(len(recent_prices)):
                        if i >= 2:  # MA-3 needs at least 3 points
                            ma_chart_data.append(simple_moving_average(recent_prices[:i+1], min(3, i+1)))
                        else:
                            ma_chart_data.append(recent_prices[i])
                    
                    # Display as line chart
                    st.line_chart([recent_prices, ma_chart_data], height=200)
        
        with tech_tab2:
            # Volatility analysis with charts
            if len(st.session_state.price_history) >= 7:
                recent_prices = st.session_state.price_history[-15:]  # Last 15 periods
                
                # Calculate volatility over time
                volatility_data = []
                for i in range(3, len(recent_prices)):  # Need at least 3 points for volatility
                    price_subset = recent_prices[max(0, i-6):i+1]  # 7-day window
                    if len(price_subset) >= 2:
                        price_changes = [abs(price_subset[j] - price_subset[j-1])/price_subset[j-1] 
                                       for j in range(1, len(price_subset))]
                        volatility = sum(price_changes) / len(price_changes) * 100
                        volatility_data.append(volatility)
                
                if volatility_data:
                    # Current volatility metrics
                    current_volatility = volatility_data[-1] if volatility_data else 0
                    avg_volatility = sum(volatility_data) / len(volatility_data) if volatility_data else 0
                    
                    volatility_level = "High" if current_volatility > 5 else "Medium" if current_volatility > 2 else "Low"
                    volatility_color = "#FF6B6B" if volatility_level == "High" else "#F7931A" if volatility_level == "Medium" else "#00D4AA"
                    
                    # Display volatility metrics
                    col_vol1, col_vol2 = st.columns(2)
                    
                    with col_vol1:
                        st.markdown(f"""
                        <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 8px;">
                            <div style="color: #FFFFFF; margin-bottom: 0.5rem;">Current Volatility</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #F7931A;">{current_volatility:.2f}%</div>
                            <div style="color: {volatility_color}; font-size: 0.9rem;">Level: {volatility_level}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_vol2:
                        st.markdown(f"""
                        <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 8px;">
                            <div style="color: #FFFFFF; margin-bottom: 0.5rem;">Average Volatility</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #F7931A;">{avg_volatility:.2f}%</div>
                            <div style="color: #808080; font-size: 0.9rem;">Historical Average</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Volatility chart
                    st.write("**Volatility Trend**")
                    st.line_chart(volatility_data, height=200)
                    
                    # Volatility interpretation
                    if current_volatility > avg_volatility * 1.5:
                        volatility_msg = "🔥 Highly volatile period - Exercise caution"
                        volatility_msg_color = "#FF6B6B"
                    elif current_volatility < avg_volatility * 0.5:
                        volatility_msg = "😴 Low volatility period - Market consolidation"
                        volatility_msg_color = "#00D4AA"
                    else:
                        volatility_msg = "⚖️ Normal volatility levels"
                        volatility_msg_color = "#F7931A"
                    
                    st.markdown(f"""
                    <div style="background-color: #2E2E2E; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <div style="color: {volatility_msg_color}; font-weight: bold;">{volatility_msg}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.info("Load price data and select prediction models to see forecasts")

# Market information section
st.markdown("---")
st.subheader("ℹ️ Market Information")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("""
    **Market Status**
    - 24/7 Trading
    - Global Market
    - High Volatility Asset
    """)

with col_info2:
    st.markdown("""
    **Data Source**
    - CoinGecko API
    - Real-time Updates
    - USD Pricing
    """)

with col_info3:
    st.markdown("""
    **Prediction Models**
    - Linear Trend Analysis
    - Momentum Indicators
    - Technical Analysis
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #808080; font-size: 0.9rem; padding: 1rem;">
    <strong>Disclaimer:</strong> This application is for educational purposes only. 
    Cryptocurrency investments carry significant risk. Always do your own research 
    and consult with financial advisors before making investment decisions.
</div>
""", unsafe_allow_html=True)

# Real-time refresh status and countdown
if st.session_state.real_time_enabled or auto_refresh:
    # Calculate time until next refresh
    if st.session_state.last_update:
        refresh_time = refresh_interval if st.session_state.real_time_enabled else 30
        time_since_update = (datetime.now() - st.session_state.last_update).seconds
        time_until_refresh = max(0, refresh_time - time_since_update)
        
        # Display countdown in sidebar
        with st.sidebar:
            if time_until_refresh > 0:
                st.markdown(f"""
                <div style="background-color: #2E2E2E; padding: 0.8rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
                    <div style="color: #F7931A; font-weight: bold; font-size: 0.9rem;">
                        Next update in: {time_until_refresh}s
                    </div>
                    <div style="background-color: #1E1E1E; height: 4px; border-radius: 2px; margin: 0.5rem 0;">
                        <div style="background-color: #F7931A; height: 100%; width: {((refresh_time - time_until_refresh) / refresh_time) * 100}%; border-radius: 2px; transition: width 0.3s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #2E2E2E; padding: 0.8rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
                    <div style="color: #00D4AA; font-weight: bold; font-size: 0.9rem;">
                        Updating now...
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Auto-refresh timer with shorter sleep for smoother updates
    time.sleep(1)
    st.rerun()