import streamlit as st
import pandas as pd
import numpy as np

# Import chart modules
from charts.price_charts import create_price_chart, create_price_with_predictions
from charts.technical_charts import create_rsi_chart, create_macd_chart, create_multi_indicator_dashboard
from charts.prediction_charts import create_prediction_comparison_chart, create_prediction_confidence_chart
from charts.dashboard_charts import create_comprehensive_dashboard, create_performance_summary_chart

# Import utilities
from utils.data_manager import DataManager
from utils.technical_indicators import TechnicalIndicators

# Page configuration
st.set_page_config(
    page_title="Charts - Technical Analysis Platform",
    page_icon="📊",
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
        'data': {
            'period': '1y',
            'interval': '1d',
            'auto_refresh': True,
            'refresh_minutes': 15,
            'cache_duration': 15
        }
    }

st.title("📊 Advanced Trading Charts")
st.markdown("### Comprehensive Visual Analysis Dashboard")

# Sidebar for chart controls
with st.sidebar:
    st.header("📈 Chart Controls")
    
    # Stock symbol input
    symbol = st.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a valid stock ticker symbol"
    ).upper()
    
    # Time period selection
    period_options = {
        "5 Days": "5d",
        "15 Days": "15d", 
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y"
    }
    
    period_display = st.selectbox(
        "Time Period",
        options=list(period_options.keys()),
        index=5  # Default to 1 Year
    )
    period = period_options[period_display]
    
    # Chart type selection
    st.markdown("---")
    st.subheader("📊 Chart Options")
    
    show_candlestick = st.checkbox("Candlestick Chart", value=True)
    show_volume = st.checkbox("Volume Analysis", value=True)
    show_technical = st.checkbox("Technical Indicators", value=True)
    show_predictions = st.checkbox("Prediction Charts", value=True)
    show_dashboard = st.checkbox("Trading Dashboard", value=False)
    
    # Generate charts button
    generate_charts = st.button("🚀 Generate Charts", type="primary", use_container_width=True)

# Main content
if symbol and generate_charts:
    try:
        # Load data
        with st.spinner(f"Loading data and generating charts for {symbol}..."):
            data = st.session_state.data_manager.get_stock_data(symbol, period)
        
        if data is not None and not data.empty:
            # Calculate technical indicators
            tech_indicators = TechnicalIndicators(st.session_state.config['indicators'])
            data_with_indicators = tech_indicators.calculate_all_indicators(data)
            
            current_price = data['Close'].iloc[-1]
            
            # Generate predictions for chart overlay
            predictions = []
            
            # Simple prediction generation for charts
            if 'RSI' in data_with_indicators.columns:
                rsi_current = data_with_indicators['RSI'].iloc[-1]
                if rsi_current > 70:
                    predictions.append({
                        'indicator': 'RSI Mean Reversion',
                        'target': current_price * 0.97,
                        'confidence': 'High'
                    })
                elif rsi_current < 30:
                    predictions.append({
                        'indicator': 'RSI Bounce',
                        'target': current_price * 1.03,
                        'confidence': 'High'
                    })
            
            if 'MACD' in data_with_indicators.columns and 'MACD_Signal' in data_with_indicators.columns:
                macd_current = data_with_indicators['MACD'].iloc[-1]
                macd_signal = data_with_indicators['MACD_Signal'].iloc[-1]
                if macd_current > macd_signal:
                    predictions.append({
                        'indicator': 'MACD Bullish',
                        'target': current_price * 1.02,
                        'confidence': 'Medium'
                    })
                else:
                    predictions.append({
                        'indicator': 'MACD Bearish',
                        'target': current_price * 0.98,
                        'confidence': 'Medium'
                    })
            
            # Display current price info
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📍 Current Price", f"${current_price:.2f}")
            
            with col2:
                daily_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                st.metric("📈 Daily Change", f"{daily_change:+.2f}%")
            
            with col3:
                volume_current = data['Volume'].iloc[-1]
                st.metric("📊 Volume", f"{volume_current:,.0f}")
            
            with col4:
                if predictions:
                    avg_prediction = np.mean([p['target'] for p in predictions])
                    pred_change = ((avg_prediction - current_price) / current_price) * 100
                    st.metric("🎯 Avg Prediction", f"${avg_prediction:.2f}", f"{pred_change:+.2f}%")
            
            st.markdown("---")
            
            # 1. Price Charts Section
            if show_candlestick:
                st.subheader("📈 Price Analysis")
                
                # Create tabs for different price views
                price_tab1, price_tab2 = st.tabs(["📊 Full Analysis", "🎯 With Predictions"])
                
                with price_tab1:
                    price_fig = create_price_chart(data_with_indicators, symbol)
                    if price_fig:
                        st.plotly_chart(price_fig, use_container_width=True)
                
                with price_tab2:
                    pred_price_fig = create_price_with_predictions(data, symbol, predictions, current_price)
                    if pred_price_fig:
                        st.plotly_chart(pred_price_fig, use_container_width=True)
            
            # 2. Technical Indicators Section
            if show_technical:
                st.subheader("🔧 Technical Indicators")
                
                # Create tabs for different indicators
                tech_tab1, tech_tab2, tech_tab3 = st.tabs(["📊 Dashboard", "📈 RSI", "📉 MACD"])
                
                with tech_tab1:
                    dashboard_fig = create_multi_indicator_dashboard(data_with_indicators)
                    if dashboard_fig:
                        st.plotly_chart(dashboard_fig, use_container_width=True)
                
                with tech_tab2:
                    rsi_fig = create_rsi_chart(data_with_indicators)
                    if rsi_fig:
                        st.plotly_chart(rsi_fig, use_container_width=True)
                    else:
                        st.info("RSI data not available for this timeframe")
                
                with tech_tab3:
                    macd_fig = create_macd_chart(data_with_indicators)
                    if macd_fig:
                        st.plotly_chart(macd_fig, use_container_width=True)
                    else:
                        st.info("MACD data not available for this timeframe")
            
            # 3. Prediction Charts Section
            if show_predictions and predictions:
                st.subheader("🎯 Prediction Analysis")
                
                pred_tab1, pred_tab2 = st.tabs(["📊 Comparison", "📈 Confidence"])
                
                with pred_tab1:
                    comparison_fig = create_prediction_comparison_chart(data, symbol, predictions, current_price)
                    if comparison_fig:
                        st.plotly_chart(comparison_fig, use_container_width=True)
                
                with pred_tab2:
                    confidence_fig = create_prediction_confidence_chart(predictions, current_price)
                    if confidence_fig:
                        st.plotly_chart(confidence_fig, use_container_width=True)
            
            # 4. Comprehensive Dashboard
            if show_dashboard:
                st.subheader("🏠 Trading Dashboard")
                
                dashboard_fig = create_comprehensive_dashboard(data_with_indicators, symbol, predictions, current_price)
                if dashboard_fig:
                    st.plotly_chart(dashboard_fig, use_container_width=True)
                
                # Performance summary
                st.subheader("📊 Performance Summary")
                perf_fig = create_performance_summary_chart(data, symbol)
                if perf_fig:
                    st.plotly_chart(perf_fig, use_container_width=True)
            
            # 5. Chart Summary
            st.markdown("---")
            st.subheader("📋 Chart Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **📊 Data Summary for {symbol}:**
                - **Time Period**: {period_display}
                - **Data Points**: {len(data)} days
                - **Current Price**: ${current_price:.2f}
                - **Technical Indicators**: {len([col for col in data_with_indicators.columns if col not in data.columns])} calculated
                """)
            
            with col2:
                if predictions:
                    st.markdown(f"""
                    **🎯 Prediction Summary:**
                    - **Total Predictions**: {len(predictions)}
                    - **Bullish Signals**: {len([p for p in predictions if p['target'] > current_price])}
                    - **Bearish Signals**: {len([p for p in predictions if p['target'] < current_price])}
                    - **High Confidence**: {len([p for p in predictions if p['confidence'] == 'High'])}
                    """)
                else:
                    st.markdown("**🎯 No predictions available for this timeframe**")
        
        else:
            st.error(f"❌ No data found for symbol: {symbol}")
            st.info("💡 Try a different stock symbol or check if the market is open.")
            
    except Exception as e:
        st.error(f"❌ Error generating charts: {str(e)}")
        st.info("💡 Please try again or contact support if the issue persists.")

else:
    # Welcome screen
    st.info("👆 Enter a stock symbol and click 'Generate Charts' to start your visual analysis!")
    
    st.markdown("""
    ## 🚀 Available Chart Types
    
    ### 📈 **Price Charts**
    - **Candlestick Charts**: Complete OHLC analysis with volume
    - **Moving Averages**: SMA 20, SMA 50 with trend analysis
    - **Bollinger Bands**: Volatility and mean reversion analysis
    - **Price Predictions**: Visual overlay of ML model targets
    
    ### 🔧 **Technical Indicators**
    - **RSI Chart**: Momentum analysis with overbought/oversold zones
    - **MACD Analysis**: Trend changes with signal lines and histogram
    - **Stochastic Oscillator**: %K and %D momentum indicators
    - **Multi-Indicator Dashboard**: Combined view of all indicators
    
    ### 🎯 **Prediction Visualizations**
    - **Model Comparison**: Compare all prediction targets visually
    - **Confidence Analysis**: Visual confidence levels for each model
    - **Timeline Projections**: Future price range predictions
    - **Risk Assessment**: Visual risk metrics and analysis
    
    ### 🏠 **Trading Dashboard**
    - **Comprehensive View**: 6-panel trading dashboard
    - **Performance Metrics**: Returns, volatility, and trend analysis
    - **Risk Analysis**: Support/resistance levels and risk metrics
    - **Volume Analysis**: Trading volume patterns and trends
    
    ---
    
    ### 💡 **How to Use**
    1. **Enter Stock Symbol**: Any valid ticker (AAPL, GOOGL, TSLA, etc.)
    2. **Select Time Period**: Choose from 5 days to 1 year
    3. **Choose Chart Types**: Select which visualizations to display
    4. **Generate Charts**: Click the button to create your analysis
    
    ### 🎨 **Interactive Features**
    - **Zoom & Pan**: Full chart interactivity
    - **Hover Details**: Detailed information on mouse hover
    - **Tab Navigation**: Organized chart sections
    - **Export Options**: Download charts as images
    
    **Ready to analyze? Enter a stock symbol above and explore comprehensive visual insights!**
    """)