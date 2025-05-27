import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom utilities  
from utils.data_manager import DataManager
from utils.technical_indicators import TechnicalIndicators

# Page configuration
st.set_page_config(
    page_title="Predictions - Stock Analysis Platform",
    page_icon="🔮",
    layout="wide"
)

# Initialize session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

if 'config' not in st.session_state:
    st.session_state.config = {
        'indicators': {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0,
            'sma_period': 20,
            'ema_period': 20
        },
        'ml_models': {
            'enable_random_forest': True,
            'enable_xgboost': True,
            'enable_prophet': True,
            'enable_arima': True,
            'enable_lstm': False
        },
        'data': {
            'period': '1y',
            'interval': '1d',
            'auto_refresh': True,
            'refresh_minutes': 15,
            'cache_duration': 15
        }
    }

st.title("🔮 Advanced ML-Style Price Predictions")
st.markdown("### Multi-Model Prediction System Using Technical Analysis")

# Sidebar for prediction settings
with st.sidebar:
    st.header("🎯 Prediction Settings")
    
    # Stock symbol input
    symbol = st.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a valid stock ticker symbol"
    ).upper()
    
    # Time period for predictions
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y"
    }
    
    period_display = st.selectbox(
        "Analysis Period",
        options=list(period_options.keys()),
        index=3
    )
    period = period_options[period_display]
    
    # Prediction horizon
    prediction_days = st.slider(
        "Prediction Horizon (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="How many days ahead to predict"
    )
    
    # Generate predictions button
    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
        st.session_state.generate_predictions = True

# Main content
if symbol and hasattr(st.session_state, 'generate_predictions'):
    try:
        # Load data
        with st.spinner(f"Loading data and generating predictions for {symbol}..."):
            data = st.session_state.data_manager.get_stock_data(symbol, period)
        
        if data is not None and not data.empty:
            # Calculate technical indicators
            tech_indicators = TechnicalIndicators(st.session_state.config['indicators'])
            data_with_indicators = tech_indicators.calculate_all_indicators(data)
            
            current_price = data['Close'].iloc[-1]
            
            # Generate multiple prediction models
            st.subheader("🤖 Multi-Model Price Predictions")
            
            predictions = []
            
            # Model 1: RSI Mean Reversion Model
            if 'RSI' in data_with_indicators.columns:
                rsi_current = data_with_indicators['RSI'].iloc[-1]
                rsi_prediction = current_price
                
                if rsi_current > 70:  # Overbought
                    rsi_prediction = current_price * (1 - ((rsi_current - 70) / 100) * 0.1)
                elif rsi_current < 30:  # Oversold
                    rsi_prediction = current_price * (1 + ((30 - rsi_current) / 100) * 0.1)
                
                predictions.append({
                    'model': 'RSI Mean Reversion',
                    'price': rsi_prediction,
                    'confidence': 'High' if abs(rsi_current - 50) > 20 else 'Medium',
                    'reasoning': f'RSI at {rsi_current:.1f}, expecting mean reversion'
                })
            
            # Model 2: MACD Momentum Model
            if all(col in data_with_indicators.columns for col in ['MACD', 'MACD_Signal']):
                macd_current = data_with_indicators['MACD'].iloc[-1]
                macd_signal = data_with_indicators['MACD_Signal'].iloc[-1]
                macd_diff = macd_current - macd_signal
                
                macd_prediction = current_price * (1 + (macd_diff / current_price) * 2)
                
                predictions.append({
                    'model': 'MACD Momentum',
                    'price': macd_prediction,
                    'confidence': 'High' if abs(macd_diff) > 0.5 else 'Medium',
                    'reasoning': f'MACD momentum: {macd_diff:.3f}'
                })
            
            # Model 3: Bollinger Bands Model
            if all(col in data_with_indicators.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                bb_upper = data_with_indicators['BB_Upper'].iloc[-1]
                bb_lower = data_with_indicators['BB_Lower'].iloc[-1]
                bb_middle = data_with_indicators['BB_Middle'].iloc[-1]
                
                if current_price > bb_upper:
                    bb_prediction = bb_middle
                elif current_price < bb_lower:
                    bb_prediction = bb_middle
                else:
                    bb_prediction = current_price
                
                predictions.append({
                    'model': 'Bollinger Mean Reversion',
                    'price': bb_prediction,
                    'confidence': 'Medium',
                    'reasoning': f'Price position relative to bands'
                })
            
            # Model 4: Moving Average Trend Model
            if all(col in data_with_indicators.columns for col in ['SMA_20', 'EMA_20']):
                sma20 = data_with_indicators['SMA_20'].iloc[-1]
                ema20 = data_with_indicators['EMA_20'].iloc[-1]
                
                trend_strength = (current_price - sma20) / sma20
                ma_prediction = current_price * (1 + trend_strength * 0.5)
                
                predictions.append({
                    'model': 'Moving Average Trend',
                    'price': ma_prediction,
                    'confidence': 'Medium',
                    'reasoning': f'Trend strength: {trend_strength:.2%}'
                })
            
            # Model 5: Volume-Price Trend Model
            volume_current = data['Volume'].iloc[-1]
            volume_avg = data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = volume_current / volume_avg
            
            price_change = (current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            vpt_prediction = current_price * (1 + price_change * volume_ratio * 0.1)
            
            predictions.append({
                'model': 'Volume-Price Trend',
                'price': vpt_prediction,
                'confidence': 'High' if volume_ratio > 1.5 else 'Low',
                'reasoning': f'Volume ratio: {volume_ratio:.1f}x'
            })
            
            # Display predictions table
            if predictions:
                pred_df = pd.DataFrame(predictions)
                pred_df['Price Change %'] = ((pred_df['price'] - current_price) / current_price * 100).round(2)
                pred_df['Target Price'] = pred_df['price'].round(2)
                
                display_df = pred_df[['model', 'Target Price', 'Price Change %', 'confidence', 'reasoning']].copy()
                display_df.columns = ['Prediction Model', 'Target Price ($)', 'Change (%)', 'Confidence', 'Reasoning']
                
                st.dataframe(display_df, use_container_width=True)
                
                # Ensemble prediction
                st.subheader("🎯 Ensemble Forecast")
                
                # Weight by confidence
                weights = {'High': 3, 'Medium': 2, 'Low': 1}
                weighted_predictions = []
                total_weight = 0
                
                for pred in predictions:
                    weight = weights[pred['confidence']]
                    weighted_predictions.append(pred['price'] * weight)
                    total_weight += weight
                
                ensemble_price = sum(weighted_predictions) / total_weight
                ensemble_change = ((ensemble_price - current_price) / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🎯 Ensemble Prediction",
                        f"${ensemble_price:.2f}",
                        f"{ensemble_change:+.2f}%"
                    )
                
                with col2:
                    high_conf_count = len([p for p in predictions if p['confidence'] == 'High'])
                    confidence_score = (high_conf_count / len(predictions)) * 100
                    st.metric(
                        "📊 Confidence Score",
                        f"{confidence_score:.0f}%",
                        f"{high_conf_count}/{len(predictions)} models"
                    )
                
                with col3:
                    st.metric(
                        "📅 Prediction Horizon",
                        f"{prediction_days} days",
                        f"{len(predictions)} models"
                    )
                
                # Risk assessment
                st.subheader("⚖️ Risk Assessment")
                
                if confidence_score >= 60:
                    st.success("🟢 **High Confidence Prediction** - Strong model consensus")
                elif confidence_score >= 40:
                    st.warning("🟡 **Moderate Confidence** - Mixed signals")
                else:
                    st.error("🔴 **Low Confidence** - Conflicting predictions")
                
                # Prediction visualization
                st.subheader("📈 Prediction Visualization")
                
                fig = go.Figure()
                
                # Historical prices
                fig.add_trace(go.Scatter(
                    x=data.index[-30:],
                    y=data['Close'].iloc[-30:],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Current price
                fig.add_hline(
                    y=current_price,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Current: ${current_price:.2f}"
                )
                
                # Ensemble prediction
                fig.add_hline(
                    y=ensemble_price,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Prediction: ${ensemble_price:.2f}"
                )
                
                fig.update_layout(
                    title=f"{symbol} - Price Predictions",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error(f"❌ No data found for symbol: {symbol}")
            
    except Exception as e:
        st.error(f"❌ Error generating predictions: {str(e)}")

else:
    st.info("👆 Enter a stock symbol and click 'Generate Predictions' to start forecasting!")
    
    st.markdown("""
    ### 🚀 Advanced Prediction Features:
    
    **Multi-Model Approach:**
    - **RSI Mean Reversion**: Predicts price reversals based on momentum
    - **MACD Momentum**: Forecasts based on trend changes
    - **Bollinger Bands**: Mean reversion analysis
    - **Moving Average Trend**: Long-term trend projection
    - **Volume-Price Trend**: Volume-confirmed price movements
    
    **AI-Style Ensemble:**
    - Combines multiple models with confidence weighting
    - Provides consensus forecasts and risk assessment
    - Shows model agreement levels and prediction ranges
    
    **Interactive Features:**
    - Customizable prediction horizons (1-30 days)
    - Visual prediction charts with historical context
    - Detailed reasoning for each model's forecast
    """)