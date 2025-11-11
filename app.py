import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_fetcher import DataFetcher
from ml_models import MLModels
from visualization import Visualizer
from technical_indicators import TechnicalIndicators
from utils import format_currency, calculate_percentage_change

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'bitcoin_data' not in st.session_state:
    st.session_state.bitcoin_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Initialize classes
@st.cache_resource
def initialize_components():
    data_fetcher = DataFetcher()
    ml_models = MLModels()
    visualizer = Visualizer()
    tech_indicators = TechnicalIndicators()
    return data_fetcher, ml_models, visualizer, tech_indicators

data_fetcher, ml_models, visualizer, tech_indicators = initialize_components()

# Main title and header
st.title("₿ Bitcoin Price Prediction Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data fetching parameters
    st.subheader("Data Parameters")
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Maximum": "max"
    }
    
    selected_period = st.selectbox(
        "Historical Data Period",
        options=list(period_options.keys()),
        index=3  # Default to 1 year
    )
    
    # Prediction parameters
    st.subheader("Prediction Parameters")
    prediction_days = st.slider(
        "Days to Predict",
        min_value=1,
        max_value=365,
        value=30,
        help="Number of days to predict into the future"
    )
    
    # Model selection
    st.subheader("Model Selection")
    model_options = {
        "Linear Regression": "linear",
        "Random Forest": "random_forest",
        "LSTM Neural Network": "lstm"
    }
    
    selected_models = st.multiselect(
        "Choose Models",
        options=list(model_options.keys()),
        default=["Linear Regression", "Random Forest"],
        help="Select one or more models for prediction"
    )
    
    # Action buttons
    st.subheader("Actions")
    load_data_btn = st.button("📊 Load Bitcoin Data", type="primary")
    train_models_btn = st.button("🤖 Train Models", disabled=not st.session_state.data_loaded)
    predict_btn = st.button("🔮 Generate Predictions", disabled=not st.session_state.models_trained)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Data loading section
    if load_data_btn or st.session_state.data_loaded:
        with st.spinner("Loading Bitcoin data..."):
            try:
                period = period_options[selected_period]
                bitcoin_data = data_fetcher.fetch_bitcoin_data(period=period)
                
                if bitcoin_data is not None and not bitcoin_data.empty:
                    st.session_state.bitcoin_data = bitcoin_data
                    st.session_state.data_loaded = True
                    
                    # Add technical indicators
                    bitcoin_data_with_indicators = tech_indicators.add_all_indicators(bitcoin_data)
                    st.session_state.bitcoin_data = bitcoin_data_with_indicators
                    
                    st.success(f"✅ Successfully loaded {len(bitcoin_data)} days of Bitcoin data")
                    
                    # Display current price info
                    current_price = bitcoin_data['Close'].iloc[-1]
                    prev_price = bitcoin_data['Close'].iloc[-2]
                    price_change = calculate_percentage_change(prev_price, current_price)
                    
                    st.metric(
                        label="Current Bitcoin Price",
                        value=format_currency(current_price),
                        delta=f"{price_change:+.2f}%"
                    )
                    
                else:
                    st.error("❌ Failed to load Bitcoin data. Please try again.")
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    # Data visualization
    if st.session_state.data_loaded and st.session_state.bitcoin_data is not None:
        st.subheader("📈 Historical Bitcoin Price Chart")
        
        # Chart timeframe selector
        chart_timeframe = st.selectbox(
            "Chart Timeframe",
            options=["1D", "7D", "30D", "90D", "1Y", "All"],
            index=4,
            key="chart_timeframe"
        )
        
        # Filter data based on timeframe
        if chart_timeframe != "All":
            days_map = {"1D": 1, "7D": 7, "30D": 30, "90D": 90, "1Y": 365}
            days = days_map[chart_timeframe]
            chart_data = st.session_state.bitcoin_data.tail(days)
        else:
            chart_data = st.session_state.bitcoin_data
        
        # Create price chart
        price_chart = visualizer.create_price_chart(chart_data)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Technical indicators chart
        st.subheader("📊 Technical Indicators")
        indicators_chart = visualizer.create_indicators_chart(chart_data)
        st.plotly_chart(indicators_chart, use_container_width=True)

with col2:
    # Model training section
    if train_models_btn or st.session_state.models_trained:
        if st.session_state.data_loaded and len(selected_models) > 0:
            with st.spinner("Training machine learning models..."):
                try:
                    # Prepare data for training
                    training_data = ml_models.prepare_data(st.session_state.bitcoin_data)
                    
                    # Train selected models
                    model_results = {}
                    for model_name in selected_models:
                        model_key = model_options[model_name]
                        
                        if model_key == "linear":
                            model, metrics = ml_models.train_linear_regression(training_data)
                        elif model_key == "random_forest":
                            model, metrics = ml_models.train_random_forest(training_data)
                        elif model_key == "lstm":
                            model, metrics = ml_models.train_lstm(training_data)
                        
                        model_results[model_name] = {
                            'model': model,
                            'metrics': metrics
                        }
                    
                    st.session_state.model_results = model_results
                    st.session_state.models_trained = True
                    
                    st.success("✅ Models trained successfully!")
                    
                    # Display model performance
                    st.subheader("🎯 Model Performance")
                    for model_name, results in model_results.items():
                        with st.expander(f"{model_name} Metrics"):
                            metrics = results['metrics']
                            st.metric("Mean Absolute Error", f"${metrics['mae']:.2f}")
                            st.metric("Root Mean Square Error", f"${metrics['rmse']:.2f}")
                            if 'r2' in metrics:
                                st.metric("R² Score", f"{metrics['r2']:.4f}")
                            if 'accuracy' in metrics:
                                st.metric("Directional Accuracy", f"{metrics['accuracy']:.2%}")
                    
                except Exception as e:
                    st.error(f"❌ Error training models: {str(e)}")
        else:
            st.warning("⚠️ Please load data and select models first.")
    
    # Prediction section
    if predict_btn and st.session_state.models_trained:
        with st.spinner("Generating predictions..."):
            try:
                predictions = {}
                
                for model_name, results in st.session_state.model_results.items():
                    model = results['model']
                    model_key = model_options[model_name]
                    
                    if model_key == "linear":
                        pred_prices, confidence_intervals = ml_models.predict_linear_regression(
                            model, st.session_state.bitcoin_data, prediction_days
                        )
                    elif model_key == "random_forest":
                        pred_prices, confidence_intervals = ml_models.predict_random_forest(
                            model, st.session_state.bitcoin_data, prediction_days
                        )
                    elif model_key == "lstm":
                        pred_prices, confidence_intervals = ml_models.predict_lstm(
                            model, st.session_state.bitcoin_data, prediction_days
                        )
                    
                    predictions[model_name] = {
                        'prices': pred_prices,
                        'confidence_intervals': confidence_intervals
                    }
                
                st.session_state.predictions = predictions
                st.success("✅ Predictions generated successfully!")
                
                # Display predictions summary
                st.subheader("🔮 Prediction Summary")
                current_price = st.session_state.bitcoin_data['Close'].iloc[-1]
                
                for model_name, pred_data in predictions.items():
                    final_price = pred_data['prices'][-1]
                    price_change = calculate_percentage_change(current_price, final_price)
                    
                    st.metric(
                        label=f"{model_name} ({prediction_days}d)",
                        value=format_currency(final_price),
                        delta=f"{price_change:+.2f}%"
                    )
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")

# Predictions visualization
if st.session_state.predictions:
    st.subheader("📊 Price Predictions")
    
    # Create prediction chart
    prediction_chart = visualizer.create_prediction_chart(
        st.session_state.bitcoin_data,
        st.session_state.predictions,
        prediction_days
    )
    st.plotly_chart(prediction_chart, use_container_width=True)
    
    # Model comparison
    st.subheader("🏆 Model Comparison")
    
    # Create comparison metrics
    comparison_data = []
    current_price = st.session_state.bitcoin_data['Close'].iloc[-1]
    
    for model_name, pred_data in st.session_state.predictions.items():
        final_price = pred_data['prices'][-1]
        price_change = calculate_percentage_change(current_price, final_price)
        
        comparison_data.append({
            'Model': model_name,
            f'Predicted Price ({prediction_days}d)': format_currency(final_price),
            'Price Change': f"{price_change:+.2f}%",
            'Confidence Range': f"±{pred_data['confidence_intervals'][-1]:.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

# Data table
if st.session_state.data_loaded and st.session_state.bitcoin_data is not None:
    with st.expander("📋 Raw Data Table"):
        st.dataframe(
            st.session_state.bitcoin_data.tail(100),
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    "**Disclaimer:** This application is for educational purposes only. "
    "Cryptocurrency investments carry significant risk. Always do your own research "
    "and consult with financial advisors before making investment decisions."
)
