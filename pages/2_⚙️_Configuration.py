import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Configuration - Stock Analysis Platform",
    page_icon="⚙️",
    layout="wide"
)

# Initialize session state with proper defaults
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
            'ema_period': 20,
            'adx_period': 14,
            'cci_period': 20,
            'williams_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'atr_period': 14,
            'mfi_period': 14
        },
        'ml_models': {
            'enable_random_forest': True,
            'enable_xgboost': True,
            'enable_prophet': True,
            'enable_arima': True,
            'enable_lstm': False,  # Disabled due to compatibility issues
            'rf_n_estimators': 100,
            'rf_max_depth': 10,
            'xgb_n_estimators': 100,
            'xgb_max_depth': 6,
            'prophet_periods': 30,
            'arima_order': [1, 1, 1]
        },
        'data': {
            'period': '1y',
            'interval': '1d',
            'auto_refresh': True,
            'refresh_minutes': 15,
            'cache_duration': 15
        }
    }

st.title("⚙️ Configuration")
st.markdown("### Customize Technical Indicators and ML Models")

# Create tabs for different configuration sections
tab1, tab2, tab3 = st.tabs(["📊 Technical Indicators", "🤖 ML Models", "💾 Data Settings"])

# Technical Indicators Configuration
with tab1:
    st.header("Technical Indicators Settings")
    
    # RSI Configuration
    st.subheader("🔄 RSI (Relative Strength Index)")
    col1, col2 = st.columns(2)
    
    with col1:
        rsi_period = st.slider(
            "RSI Period",
            min_value=5,
            max_value=50,
            value=int(st.session_state.config['indicators']['rsi_period']),
            step=1
        )
        st.session_state.config['indicators']['rsi_period'] = rsi_period
        
    with col2:
        st.info(f"Current RSI Period: {rsi_period} days")
        st.caption("Standard: 14 days. Lower values = more sensitive")
    
    # MACD Configuration
    st.subheader("📈 MACD (Moving Average Convergence Divergence)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        macd_fast = st.slider(
            "MACD Fast Period",
            min_value=5,
            max_value=30,
            value=int(st.session_state.config['indicators']['macd_fast']),
            step=1
        )
        st.session_state.config['indicators']['macd_fast'] = macd_fast
        
    with col2:
        macd_slow = st.slider(
            "MACD Slow Period",
            min_value=15,
            max_value=50,
            value=int(st.session_state.config['indicators']['macd_slow']),
            step=1
        )
        st.session_state.config['indicators']['macd_slow'] = macd_slow
        
    with col3:
        macd_signal = st.slider(
            "MACD Signal Period",
            min_value=5,
            max_value=20,
            value=int(st.session_state.config['indicators']['macd_signal']),
            step=1
        )
        st.session_state.config['indicators']['macd_signal'] = macd_signal
    
    # Bollinger Bands Configuration
    st.subheader("📊 Bollinger Bands")
    col1, col2 = st.columns(2)
    
    with col1:
        bb_period = st.slider(
            "BB Period",
            min_value=10,
            max_value=50,
            value=int(st.session_state.config['indicators']['bb_period']),
            step=1
        )
        st.session_state.config['indicators']['bb_period'] = bb_period
    
    with col2:
        bb_std = st.slider(
            "BB Standard Deviation",
            min_value=1.0,
            max_value=3.0,
            value=float(st.session_state.config['indicators']['bb_std']),
            step=0.1
        )
        st.session_state.config['indicators']['bb_std'] = bb_std
    
    # Moving Averages Configuration
    st.subheader("📉 Moving Averages")
    col1, col2 = st.columns(2)
    
    with col1:
        sma_period = st.slider(
            "SMA Period",
            min_value=5,
            max_value=100,
            value=int(st.session_state.config['indicators']['sma_period']),
            step=1
        )
        st.session_state.config['indicators']['sma_period'] = sma_period
        
    with col2:
        ema_period = st.slider(
            "EMA Period",
            min_value=5,
            max_value=100,
            value=int(st.session_state.config['indicators']['ema_period']),
            step=1
        )
        st.session_state.config['indicators']['ema_period'] = ema_period

# ML Models Configuration
with tab2:
    st.header("Machine Learning Models")
    
    st.subheader("🎯 Model Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tree-Based Models**")
        # Ensure ml_models config exists
        if 'ml_models' not in st.session_state.config:
            st.session_state.config['ml_models'] = {
                'enable_random_forest': True,
                'enable_xgboost': True,
                'enable_prophet': True,
                'enable_arima': True,
                'enable_lstm': False,
                'rf_n_estimators': 100,
                'rf_max_depth': 10,
                'xgb_n_estimators': 100,
                'xgb_max_depth': 6,
                'prophet_periods': 30,
                'arima_order': [1, 1, 1]
            }
        
        enable_rf = st.checkbox(
            "Random Forest",
            value=st.session_state.config['ml_models']['enable_random_forest']
        )
        st.session_state.config['ml_models']['enable_random_forest'] = enable_rf
        
        enable_xgb = st.checkbox(
            "XGBoost",
            value=st.session_state.config['ml_models']['enable_xgboost']
        )
        st.session_state.config['ml_models']['enable_xgboost'] = enable_xgb
        
    with col2:
        st.markdown("**Time Series Models**")
        enable_prophet = st.checkbox(
            "Prophet",
            value=st.session_state.config['ml_models']['enable_prophet']
        )
        st.session_state.config['ml_models']['enable_prophet'] = enable_prophet
        
        enable_arima = st.checkbox(
            "ARIMA",
            value=st.session_state.config['ml_models']['enable_arima']
        )
        st.session_state.config['ml_models']['enable_arima'] = enable_arima
    
    # Model Parameters
    if enable_rf:
        st.subheader("🌳 Random Forest Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            rf_n_estimators = st.slider(
                "Number of Trees",
                min_value=50,
                max_value=500,
                value=int(st.session_state.config['ml_models']['rf_n_estimators']),
                step=50
            )
            st.session_state.config['ml_models']['rf_n_estimators'] = rf_n_estimators
            
        with col2:
            rf_max_depth = st.slider(
                "Max Depth",
                min_value=5,
                max_value=20,
                value=int(st.session_state.config['ml_models']['rf_max_depth']),
                step=1
            )
            st.session_state.config['ml_models']['rf_max_depth'] = rf_max_depth

# Data Settings Configuration
with tab3:
    st.header("Data Management Settings")
    
    st.subheader("📊 Data Refresh")
    col1, col2 = st.columns(2)
    
    with col1:
        # Ensure data config exists
        if 'data' not in st.session_state.config:
            st.session_state.config['data'] = {}
        
        auto_refresh = st.checkbox(
            "Auto Refresh Data",
            value=st.session_state.config['data'].get('auto_refresh', True)
        )
        st.session_state.config['data']['auto_refresh'] = auto_refresh
        
        cache_duration = st.slider(
            "Cache Duration (minutes)",
            min_value=5,
            max_value=60,
            value=int(st.session_state.config['data'].get('cache_duration', 15)),
            step=5
        )
        st.session_state.config['data']['cache_duration'] = cache_duration
    
    with col2:
        interval_options = {
            "1 minute": "1m",
            "5 minutes": "5m",
            "15 minutes": "15m",
            "1 hour": "1h",
            "1 day": "1d"
        }
        
        interval_display = st.selectbox(
            "Data Interval",
            options=list(interval_options.keys()),
            index=4  # Default to 1 day
        )
        st.session_state.config['data']['interval'] = interval_options[interval_display]

# Save/Load Configuration
st.markdown("---")
st.subheader("💾 Configuration Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        # Reset configuration to defaults
        st.session_state.config = {
            'indicators': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2.0,
                'sma_period': 20,
                'ema_period': 20,
                'adx_period': 14,
                'cci_period': 20,
                'williams_period': 14,
                'stoch_k': 14,
                'stoch_d': 3,
                'atr_period': 14,
                'mfi_period': 14
            },
            'ml_models': {
                'enable_random_forest': True,
                'enable_xgboost': True,
                'enable_prophet': True,
                'enable_arima': True,
                'enable_lstm': False,
                'rf_n_estimators': 100,
                'rf_max_depth': 10,
                'xgb_n_estimators': 100,
                'xgb_max_depth': 6,
                'prophet_periods': 30,
                'arima_order': [1, 1, 1]
            },
            'data': {
                'period': '1y',
                'interval': '1d',
                'auto_refresh': True,
                'refresh_minutes': 15,
                'cache_duration': 15
            }
        }
        st.success("Configuration reset to defaults!")
        st.rerun()

with col2:
    # Export configuration
    config_json = json.dumps(st.session_state.config, indent=2)
    st.download_button(
        "📥 Export Config",
        data=config_json,
        file_name=f"stock_analysis_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

with col3:
    st.info("Configuration is automatically saved!")

# Display current configuration summary
st.markdown("---")
st.subheader("📋 Current Configuration Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Technical Indicators:**")
    st.write(f"• RSI Period: {st.session_state.config['indicators']['rsi_period']}")
    st.write(f"• MACD: {st.session_state.config['indicators']['macd_fast']}, {st.session_state.config['indicators']['macd_slow']}, {st.session_state.config['indicators']['macd_signal']}")
    st.write(f"• Bollinger Bands: {st.session_state.config['indicators']['bb_period']}, {st.session_state.config['indicators']['bb_std']}")
    st.write(f"• Moving Averages: SMA({st.session_state.config['indicators']['sma_period']}), EMA({st.session_state.config['indicators']['ema_period']})")

with col2:
    st.markdown("**ML Models:**")
    enabled_models = []
    if st.session_state.config['ml_models']['enable_random_forest']:
        enabled_models.append("Random Forest")
    if st.session_state.config['ml_models']['enable_xgboost']:
        enabled_models.append("XGBoost")
    if st.session_state.config['ml_models']['enable_prophet']:
        enabled_models.append("Prophet")
    if st.session_state.config['ml_models']['enable_arima']:
        enabled_models.append("ARIMA")
    
    st.write(f"• Enabled: {', '.join(enabled_models) if enabled_models else 'None'}")
    st.write(f"• Cache Duration: {st.session_state.config['data']['cache_duration']} min")
    st.write(f"• Auto Refresh: {'Yes' if st.session_state.config['data']['auto_refresh'] else 'No'}")