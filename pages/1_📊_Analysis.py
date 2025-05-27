import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import custom utilities
from utils.data_manager import DataManager
from utils.technical_indicators import TechnicalIndicators
from utils.explanation_engine import ExplanationEngine

# Page configuration
st.set_page_config(
    page_title="Technical Analysis - Stock Analysis Platform",
    page_icon="📊",
    layout="wide"
)

# Initialize components
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
            'bb_std': 2,
            'sma_period': 20,
            'ema_period': 20
        }
    }

# Page header
st.title("📊 Technical Analysis")
st.markdown("### Deep dive into technical indicators and chart patterns")

# Sidebar for analysis controls
with st.sidebar:
    st.header("📈 Analysis Controls")
    
    # Stock symbol input
    symbol = st.text_input(
        "Stock Symbol",
        value=st.session_state.get('selected_symbol', 'AAPL'),
        help="Enter a valid stock ticker symbol"
    ).upper()
    
    st.session_state.selected_symbol = symbol
    
    # Time period selection
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    
    period_display = st.selectbox(
        "Time Period",
        options=list(period_options.keys()),
        index=3
    )
    period = period_options[period_display]
    
    # Chart type selection
    st.subheader("📊 Chart Settings")
    
    chart_type = st.selectbox(
        "Chart Type",
        ["Candlestick", "OHLC", "Line", "Area"],
        index=0
    )
    
    # Indicator selection
    st.subheader("📈 Technical Indicators")
    
    show_volume = st.checkbox("Volume", value=True)
    show_ma = st.checkbox("Moving Averages", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_stoch = st.checkbox("Stochastic", value=False)
    show_adx = st.checkbox("ADX", value=False)
    show_support_resistance = st.checkbox("Support/Resistance", value=False)

# Main content
if symbol:
    try:
        # Load data
        with st.spinner(f"Loading data for {symbol}..."):
            data = st.session_state.data_manager.get_stock_data(symbol, period)
        
        if data is not None and not data.empty:
            # Initialize technical indicators
            tech_indicators = TechnicalIndicators(st.session_state.config['indicators'])
            
            # Calculate all indicators
            data_with_indicators = tech_indicators.calculate_all_indicators(data)
            
            # Display key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change:.2f} ({price_change_pct:.2f}%)"
                )
            
            with col2:
                high_52w = data['High'].rolling(252).max().iloc[-1]
                low_52w = data['Low'].rolling(252).min().iloc[-1]
                st.metric("52W High", f"${high_52w:.2f}")
                st.metric("52W Low", f"${low_52w:.2f}")
            
            with col3:
                volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                volume_change = ((volume - avg_volume) / avg_volume) * 100
                st.metric(
                    "Volume",
                    f"{volume:,.0f}",
                    f"{volume_change:.1f}% vs avg"
                )
            
            with col4:
                # Calculate volatility
                returns = data['Close'].pct_change()
                volatility = returns.std() * np.sqrt(252) * 100
                st.metric("Volatility (Annual)", f"{volatility:.1f}%")
            
            with col5:
                # Market cap placeholder
                market_cap = "N/A"
                st.metric("Market Cap", market_cap)
                last_update = datetime.now().strftime("%H:%M:%S")
                st.caption(f"Updated: {last_update}")
            
            # Create comprehensive chart
            st.subheader("📈 Interactive Price Chart")
            
            # Determine number of subplots
            subplot_count = 1  # Price chart
            if show_volume:
                subplot_count += 1
            if show_rsi:
                subplot_count += 1
            if show_macd:
                subplot_count += 1
            if show_stoch:
                subplot_count += 1
            
            # Create subplot heights
            if subplot_count == 1:
                row_heights = [1.0]
            elif subplot_count == 2:
                row_heights = [0.7, 0.3]
            elif subplot_count == 3:
                row_heights = [0.6, 0.2, 0.2]
            elif subplot_count == 4:
                row_heights = [0.5, 0.2, 0.15, 0.15]
            else:
                row_heights = [0.4] + [0.15] * (subplot_count - 1)
            
            # Create subplots
            subplot_titles = ['Price Chart']
            if show_volume:
                subplot_titles.append('Volume')
            if show_rsi:
                subplot_titles.append('RSI')
            if show_macd:
                subplot_titles.append('MACD')
            if show_stoch:
                subplot_titles.append('Stochastic')
            
            fig = make_subplots(
                rows=subplot_count,
                cols=1,
                shared_xaxes=True,
                row_heights=row_heights,
                vertical_spacing=0.02,
                subplot_titles=subplot_titles
            )
            
            current_row = 1
            
            # Price chart
            if chart_type == "Candlestick":
                fig.add_trace(
                    go.Candlestick(
                        x=data_with_indicators.index,
                        open=data_with_indicators['Open'],
                        high=data_with_indicators['High'],
                        low=data_with_indicators['Low'],
                        close=data_with_indicators['Close'],
                        name="Price"
                    ),
                    row=current_row, col=1
                )
            elif chart_type == "OHLC":
                fig.add_trace(
                    go.Ohlc(
                        x=data_with_indicators.index,
                        open=data_with_indicators['Open'],
                        high=data_with_indicators['High'],
                        low=data_with_indicators['Low'],
                        close=data_with_indicators['Close'],
                        name="Price"
                    ),
                    row=current_row, col=1
                )
            elif chart_type == "Line":
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['Close'],
                        mode='lines',
                        name="Close Price",
                        line=dict(width=2)
                    ),
                    row=current_row, col=1
                )
            elif chart_type == "Area":
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['Close'],
                        fill='tonexty',
                        mode='lines',
                        name="Close Price",
                        line=dict(width=1)
                    ),
                    row=current_row, col=1
                )
            
            # Moving Averages
            if show_ma:
                if 'SMA_20' in data_with_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['SMA_20'],
                            name="SMA 20",
                            line=dict(color='orange', width=1)
                        ),
                        row=current_row, col=1
                    )
                
                if 'EMA_20' in data_with_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['EMA_20'],
                            name="EMA 20",
                            line=dict(color='blue', width=1)
                        ),
                        row=current_row, col=1
                    )
                
                if 'SMA_50' in data_with_indicators.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['SMA_50'],
                            name="SMA 50",
                            line=dict(color='red', width=1)
                        ),
                        row=current_row, col=1
                    )
            
            # Bollinger Bands
            if show_bb and all(col in data_with_indicators.columns for col in ['BB_Upper', 'BB_Lower']):
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['BB_Upper'],
                        name="BB Upper",
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['BB_Lower'],
                        name="BB Lower",
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ),
                    row=current_row, col=1
                )
            
            # Support and Resistance
            if show_support_resistance and 'R1' in data_with_indicators.columns:
                r1 = data_with_indicators['R1'].iloc[-1]
                s1 = data_with_indicators['S1'].iloc[-1]
                
                fig.add_hline(y=r1, line_dash="dot", line_color="red", 
                             annotation_text=f"R1: ${r1:.2f}", row=current_row, col=1)
                fig.add_hline(y=s1, line_dash="dot", line_color="green", 
                             annotation_text=f"S1: ${s1:.2f}", row=current_row, col=1)
            
            current_row += 1
            
            # Volume
            if show_volume:
                colors = ['red' if data_with_indicators['Close'].iloc[i] < data_with_indicators['Open'].iloc[i] 
                         else 'green' for i in range(len(data_with_indicators))]
                
                fig.add_trace(
                    go.Bar(
                        x=data_with_indicators.index,
                        y=data_with_indicators['Volume'],
                        name="Volume",
                        marker_color=colors
                    ),
                    row=current_row, col=1
                )
                current_row += 1
            
            # RSI
            if show_rsi and 'RSI' in data_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['RSI'],
                        name="RSI",
                        line=dict(color='purple')
                    ),
                    row=current_row, col=1
                )
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.3, row=current_row, col=1)
                
                current_row += 1
            
            # MACD
            if show_macd and all(col in data_with_indicators.columns for col in ['MACD', 'MACD_Signal']):
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['MACD'],
                        name="MACD",
                        line=dict(color='blue')
                    ),
                    row=current_row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['MACD_Signal'],
                        name="MACD Signal",
                        line=dict(color='red')
                    ),
                    row=current_row, col=1
                )
                
                if 'MACD_Histogram' in data_with_indicators.columns:
                    colors = ['green' if val >= 0 else 'red' for val in data_with_indicators['MACD_Histogram']]
                    fig.add_trace(
                        go.Bar(
                            x=data_with_indicators.index,
                            y=data_with_indicators['MACD_Histogram'],
                            name="MACD Histogram",
                            marker_color=colors,
                            opacity=0.7
                        ),
                        row=current_row, col=1
                    )
                
                current_row += 1
            
            # Stochastic
            if show_stoch and all(col in data_with_indicators.columns for col in ['Stoch_K', 'Stoch_D']):
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['Stoch_K'],
                        name="Stoch %K",
                        line=dict(color='blue')
                    ),
                    row=current_row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['Stoch_D'],
                        name="Stoch %D",
                        line=dict(color='red')
                    ),
                    row=current_row, col=1
                )
                
                # Stochastic levels
                fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
                
                current_row += 1
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} - Technical Analysis ({period_display})",
                xaxis_rangeslider_visible=False,
                height=200 + (subplot_count * 150),
                showlegend=True,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Analysis Summary
            st.subheader("📊 Technical Analysis Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Current Signals:**")
                
                # Get trading signals
                signals = tech_indicators.get_indicator_signals(data_with_indicators)
                
                if signals:
                    for indicator, signal_data in signals.items():
                        signal = signal_data['signal']
                        strength = signal_data['strength']
                        value = signal_data.get('value', 'N/A')
                        
                        if signal == 'BUY':
                            st.success(f"🟢 {indicator}: {signal} ({strength}) - {value}")
                        elif signal == 'SELL':
                            st.error(f"🔴 {indicator}: {signal} ({strength}) - {value}")
                        else:
                            st.info(f"🟡 {indicator}: {signal} ({strength}) - {value}")
                else:
                    st.info("No clear signals detected")
            
            with col2:
                st.markdown("**📊 Key Metrics:**")
                
                # Display key indicator values
                if 'RSI' in data_with_indicators.columns:
                    rsi_current = data_with_indicators['RSI'].iloc[-1]
                    st.write(f"RSI: {rsi_current:.1f}")
                
                if 'ADX' in data_with_indicators.columns:
                    adx_current = data_with_indicators['ADX'].iloc[-1]
                    st.write(f"ADX: {adx_current:.1f}")
                
                if 'BB_Position' in data_with_indicators.columns:
                    bb_pos = data_with_indicators['BB_Position'].iloc[-1]
                    st.write(f"BB Position: {bb_pos:.1f}%")
                
                if 'Volume' in data_with_indicators.columns:
                    vol_ratio = volume / avg_volume
                    st.write(f"Volume Ratio: {vol_ratio:.1f}x")
            
            # Pattern Recognition
            st.subheader("🔍 Pattern Analysis")
            
            # Simple pattern detection
            patterns = []
            
            # Doji detection (simplified)
            latest_candle = data_with_indicators.iloc[-1]
            body_size = abs(latest_candle['Close'] - latest_candle['Open'])
            candle_range = latest_candle['High'] - latest_candle['Low']
            
            if body_size < (candle_range * 0.1) and candle_range > 0:
                patterns.append("🕯️ Doji Pattern - Indecision")
            
            # Gap detection
            if len(data_with_indicators) > 1:
                prev_candle = data_with_indicators.iloc[-2]
                if latest_candle['Open'] > prev_candle['High']:
                    patterns.append("📈 Gap Up - Bullish")
                elif latest_candle['Open'] < prev_candle['Low']:
                    patterns.append("📉 Gap Down - Bearish")
            
            # Volume spike
            if volume > avg_volume * 2:
                patterns.append("📊 Volume Spike - Increased Interest")
            
            # Bollinger Band squeeze
            if 'BB_Width' in data_with_indicators.columns:
                bb_width = data_with_indicators['BB_Width'].iloc[-1]
                bb_width_avg = data_with_indicators['BB_Width'].rolling(20).mean().iloc[-1]
                if bb_width < bb_width_avg * 0.5:
                    patterns.append("🎯 Bollinger Band Squeeze - Breakout Expected")
            
            if patterns:
                for pattern in patterns:
                    st.write(f"• {pattern}")
            else:
                st.info("No significant patterns detected")
            
            # Export options
            st.subheader("📥 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Technical Data"):
                    csv = data_with_indicators.to_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{symbol}_technical_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📈 Download Signals"):
                    signals_df = pd.DataFrame.from_dict(signals, orient='index')
                    if not signals_df.empty:
                        csv = signals_df.to_csv()
                        st.download_button(
                            label="Download Signals CSV",
                            data=csv,
                            file_name=f"{symbol}_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
        
        else:
            st.error(f"❌ No data found for symbol: {symbol}")
            st.info("Please check the stock symbol and try again.")
    
    except Exception as e:
        st.error(f"❌ Error in analysis: {str(e)}")
        st.info("Please try again or contact support if the issue persists.")

else:
    st.info("👆 Please enter a stock symbol in the sidebar to begin analysis.")

# Help section
with st.expander("ℹ️ How to Use This Page"):
    st.markdown("""
    **Technical Analysis Page Guide:**
    
    1. **Stock Selection**: Enter any valid stock ticker symbol in the sidebar
    2. **Time Period**: Choose your analysis timeframe (1 month to 5 years)
    3. **Chart Type**: Select between Candlestick, OHLC, Line, or Area charts
    4. **Indicators**: Toggle various technical indicators on/off
    5. **Analysis**: Review the automated signals and pattern detection
    6. **Export**: Download the data and signals for further analysis
    
    **Key Features:**
    - Interactive charts with zoom and pan capabilities
    - Real-time indicator calculations
    - Automated signal generation
    - Pattern recognition
    - Data export functionality
    
    **Indicator Explanations:**
    - **RSI**: Momentum oscillator (0-100), overbought >70, oversold <30
    - **MACD**: Trend following momentum indicator
    - **Bollinger Bands**: Volatility bands around moving average
    - **Stochastic**: Momentum oscillator comparing closing price to price range
    - **ADX**: Trend strength indicator, >25 indicates strong trend
    """)
