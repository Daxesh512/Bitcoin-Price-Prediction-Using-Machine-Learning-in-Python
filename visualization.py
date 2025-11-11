import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Visualizer:
    """Class for creating interactive visualizations of Bitcoin data and predictions."""
    
    def __init__(self):
        # Bitcoin orange color scheme
        self.colors = {
            'bitcoin_orange': '#F7931A',
            'trust_blue': '#4285F4',
            'success_green': '#00D4AA',
            'alert_red': '#FF6B6B',
            'dark_bg': '#1E1E1E',
            'text_white': '#FFFFFF',
            'gray': '#808080'
        }
    
    def create_price_chart(self, data, show_volume=True):
        """
        Create an interactive price chart with candlesticks and volume.
        
        Args:
            data (pd.DataFrame): Bitcoin price data
            show_volume (bool): Whether to show volume subplot
            
        Returns:
            plotly.graph_objects.Figure: Interactive price chart
        """
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Bitcoin Price (USD)', 'Trading Volume'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='BTC Price',
            increasing_line_color=self.colors['success_green'],
            decreasing_line_color=self.colors['alert_red']
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add moving averages if available
        if 'SMA_10' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_10'],
                    mode='lines',
                    name='SMA 10',
                    line=dict(color=self.colors['bitcoin_orange'], width=1),
                    opacity=0.7
                ),
                row=1, col=1 if show_volume else None
            )
        
        if 'SMA_30' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_30'],
                    mode='lines',
                    name='SMA 30',
                    line=dict(color=self.colors['trust_blue'], width=1),
                    opacity=0.7
                ),
                row=1, col=1 if show_volume else None
            )
        
        # Add volume bars
        if show_volume:
            volume_colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] 
                           else 'green' for i in range(len(data))]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=volume_colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Bitcoin Price Analysis',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            showlegend=True,
            height=600 if show_volume else 400,
            hovermode='x unified'
        )
        
        # Remove rangeslider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        return fig
    
    def create_indicators_chart(self, data):
        """
        Create a chart showing technical indicators.
        
        Args:
            data (pd.DataFrame): Bitcoin price data with indicators
            
        Returns:
            plotly.graph_objects.Figure: Technical indicators chart
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('RSI', 'MACD', 'Bollinger Bands'),
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.colors['bitcoin_orange'])
                ),
                row=1, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color=self.colors['trust_blue'])
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color=self.colors['alert_red'])
                ),
                row=2, col=1
            )
            
            # MACD histogram
            if 'MACD_Histogram' in data.columns:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color=self.colors['gray'],
                        opacity=0.6
                    ),
                    row=2, col=1
                )
        
        # Bollinger Bands with price
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=self.colors['bitcoin_orange'])
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color=self.colors['trust_blue'], dash='dash'),
                    opacity=0.7
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color=self.colors['trust_blue'], dash='dash'),
                    opacity=0.7,
                    fill='tonexty',
                    fillcolor='rgba(66, 133, 244, 0.1)'
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color=self.colors['gray'], dash='dot'),
                    opacity=0.7
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Technical Indicators',
            template='plotly_dark',
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_prediction_chart(self, historical_data, predictions, prediction_days):
        """
        Create a chart showing historical data and predictions.
        
        Args:
            historical_data (pd.DataFrame): Historical Bitcoin data
            predictions (dict): Dictionary of model predictions
            prediction_days (int): Number of days predicted
            
        Returns:
            plotly.graph_objects.Figure: Prediction chart
        """
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color=self.colors['bitcoin_orange'], width=2)
            )
        )
        
        # Create future dates
        last_date = historical_data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
        
        # Add predictions for each model
        colors = [self.colors['trust_blue'], self.colors['success_green'], 
                 self.colors['alert_red'], '#9C27B0', '#FF9800']
        
        for i, (model_name, pred_data) in enumerate(predictions.items()):
            color = colors[i % len(colors)]
            
            # Add prediction line
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=pred_data['prices'],
                    mode='lines+markers',
                    name=f'{model_name} Prediction',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=4)
                )
            )
            
            # Add confidence intervals
            upper_bound = [price + ci for price, ci in 
                          zip(pred_data['prices'], pred_data['confidence_intervals'])]
            lower_bound = [price - ci for price, ci in 
                          zip(pred_data['prices'], pred_data['confidence_intervals'])]
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    mode='lines',
                    name=f'{model_name} Upper CI',
                    line=dict(color=color, width=0),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    mode='lines',
                    name=f'{model_name} Lower CI',
                    line=dict(color=color, width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        # Add vertical line to separate historical and predicted data
        fig.add_vline(x=last_date, line_dash="dash", line_color="gray", 
                     annotation_text="Prediction Start")
        
        # Update layout
        fig.update_layout(
            title=f'Bitcoin Price Predictions ({prediction_days} days)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            showlegend=True,
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def create_model_comparison_chart(self, model_metrics):
        """
        Create a chart comparing model performance metrics.
        
        Args:
            model_metrics (dict): Dictionary of model metrics
            
        Returns:
            plotly.graph_objects.Figure: Model comparison chart
        """
        models = list(model_metrics.keys())
        metrics = ['MAE', 'RMSE', 'R²', 'Accuracy']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # MAE
        mae_values = [model_metrics[model]['mae'] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=mae_values, name='MAE', 
                  marker_color=self.colors['bitcoin_orange']),
            row=1, col=1
        )
        
        # RMSE
        rmse_values = [model_metrics[model]['rmse'] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', 
                  marker_color=self.colors['trust_blue']),
            row=1, col=2
        )
        
        # R²
        r2_values = [model_metrics[model].get('r2', 0) for model in models]
        fig.add_trace(
            go.Bar(x=models, y=r2_values, name='R²', 
                  marker_color=self.colors['success_green']),
            row=2, col=1
        )
        
        # Accuracy
        accuracy_values = [model_metrics[model].get('accuracy', 0) for model in models]
        fig.add_trace(
            go.Bar(x=models, y=accuracy_values, name='Accuracy', 
                  marker_color=self.colors['alert_red']),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Model Performance Comparison',
            template='plotly_dark',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_feature_importance_chart(self, model, feature_names):
        """
        Create a chart showing feature importance (for tree-based models).
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (list): List of feature names
            
        Returns:
            plotly.graph_objects.Figure: Feature importance chart
        """
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=[feature_names[i] for i in indices],
                y=[importances[i] for i in indices],
                marker_color=self.colors['bitcoin_orange'],
                name='Feature Importance'
            )
        )
        
        fig.update_layout(
            title='Top 20 Feature Importances',
            xaxis_title='Features',
            yaxis_title='Importance',
            template='plotly_dark',
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
