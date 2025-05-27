import streamlit as st

def initialize_session_state():
    """
    Centralized session state initialization for consistent configuration across all pages
    """
    if 'data_manager' not in st.session_state:
        from utils.data_manager import DataManager
        st.session_state.data_manager = DataManager()
    
    if 'config' not in st.session_state:
        st.session_state.config = {}
    
    # Initialize indicators configuration
    if 'indicators' not in st.session_state.config:
        st.session_state.config['indicators'] = {
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
        }
    
    # Initialize ML models configuration
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
    
    # Initialize data configuration
    if 'data' not in st.session_state.config:
        st.session_state.config['data'] = {
            'period': '1y',
            'interval': '1d',
            'auto_refresh': True,
            'refresh_minutes': 15,
            'cache_duration': 15
        }

def get_config():
    """Get configuration with proper initialization"""
    initialize_session_state()
    return st.session_state.config

def update_config(section, key, value):
    """Safely update configuration"""
    initialize_session_state()
    if section not in st.session_state.config:
        st.session_state.config[section] = {}
    st.session_state.config[section][key] = value