import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

class DataManager:
    """
    Centralized data management with caching and single API call strategy
    """
    
    def __init__(self):
        self.cache_duration = 900  # 15 minutes default
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for data caching"""
        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}
        if 'cache_timestamps' not in st.session_state:
            st.session_state.cache_timestamps = {}
        if 'data_info' not in st.session_state:
            st.session_state.data_info = {}
    
    def get_cache_key(self, symbol, period, interval='1d'):
        """Generate cache key for data storage"""
        return f"{symbol}_{period}_{interval}"
    
    def is_data_cached(self, symbol, period='1y', interval='1d'):
        """Check if data is available in cache"""
        cache_key = self.get_cache_key(symbol, period, interval)
        return cache_key in st.session_state.data_cache
    
    def is_cache_valid(self, symbol, period='1y', interval='1d'):
        """Check if cached data is still valid"""
        cache_key = self.get_cache_key(symbol, period, interval)
        
        if cache_key not in st.session_state.cache_timestamps:
            return False
        
        cache_time = st.session_state.cache_timestamps[cache_key]
        current_time = time.time()
        
        # Check if cache has expired
        if current_time - cache_time > self.cache_duration:
            return False
        
        return True
    
    def get_stock_data(self, symbol, period='1y', interval='1d', force_refresh=False):
        """
        Get stock data with intelligent caching
        Single API call strategy with automatic cache management
        """
        cache_key = self.get_cache_key(symbol, period, interval)
        
        # Check if we should use cached data
        if not force_refresh and self.is_data_cached(symbol, period, interval) and self.is_cache_valid(symbol, period, interval):
            st.success(f"📦 Using cached data for {symbol}")
            return st.session_state.data_cache[cache_key]
        
        try:
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            
            # Determine the optimal period for fetching
            # Always fetch a bit more data than requested for technical indicators
            extended_period = self._get_extended_period(period)
            
            data = ticker.history(
                period=extended_period,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                st.error(f"❌ No data available for {symbol}")
                return None
            
            # Clean and validate data
            data = self._clean_data(data)
            
            # Cache the full dataset
            st.session_state.data_cache[cache_key] = data
            st.session_state.cache_timestamps[cache_key] = time.time()
            
            # Store metadata
            st.session_state.data_info[cache_key] = {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'rows': len(data),
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'fetch_time': datetime.now()
            }
            
            st.success(f"✅ Fresh data loaded for {symbol} ({len(data)} data points)")
            
            # Return data trimmed to requested period if needed
            return self._trim_to_period(data, period)
            
        except Exception as e:
            st.error(f"❌ Error fetching data for {symbol}: {str(e)}")
            logging.error(f"Data fetch error for {symbol}: {str(e)}")
            return None
    
    def _get_extended_period(self, period):
        """
        Get extended period for better technical indicator calculation
        Always fetch more data than needed for accurate indicators
        """
        period_map = {
            '1mo': '3mo',
            '3mo': '6mo', 
            '6mo': '1y',
            '1y': '2y',
            '2y': '5y',
            '5y': '10y',
            '10y': 'max'
        }
        return period_map.get(period, period)
    
    def _trim_to_period(self, data, period):
        """Trim data to requested period while keeping extra for indicators"""
        if period == 'max':
            return data
        
        # Calculate the actual cutoff date
        end_date = data.index[-1]
        
        if period.endswith('mo'):
            months = int(period[:-2])
            start_date = end_date - pd.DateOffset(months=months)
        elif period.endswith('y'):
            years = int(period[:-1])
            start_date = end_date - pd.DateOffset(years=years)
        elif period.endswith('d'):
            days = int(period[:-1])
            start_date = end_date - pd.DateOffset(days=days)
        else:
            return data
        
        # Return data from start_date onwards, but keep some buffer for indicators
        buffer_days = 100  # Keep 100 days buffer for technical indicators
        actual_start = start_date - pd.DateOffset(days=buffer_days)
        
        return data[data.index >= actual_start]
    
    def _clean_data(self, data):
        """Clean and validate stock data"""
        # Remove any rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill any missing values (common for volume on weekends)
        data = data.fillna(method='ffill')
        
        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                st.warning(f"Missing column: {col}")
                return pd.DataFrame()
        
        # Basic data validation
        if len(data) < 2:
            st.warning("Insufficient data points")
            return pd.DataFrame()
        
        # Check for data integrity
        invalid_rows = (
            (data['High'] < data['Low']) |
            (data['High'] < data['Open']) |
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) |
            (data['Low'] > data['Close']) |
            (data['Volume'] < 0)
        )
        
        if invalid_rows.any():
            st.warning(f"Found {invalid_rows.sum()} rows with invalid data. Cleaning...")
            data = data[~invalid_rows]
        
        return data
    
    def get_multiple_stocks(self, symbols, period='1y', interval='1d'):
        """
        Efficiently fetch multiple stocks with batch processing
        """
        results = {}
        
        # Separate cached and non-cached symbols
        cached_symbols = []
        fetch_symbols = []
        
        for symbol in symbols:
            if self.is_data_cached(symbol, period, interval) and self.is_cache_valid(symbol, period, interval):
                cached_symbols.append(symbol)
                cache_key = self.get_cache_key(symbol, period, interval)
                results[symbol] = st.session_state.data_cache[cache_key]
            else:
                fetch_symbols.append(symbol)
        
        if cached_symbols:
            st.info(f"📦 Using cached data for: {', '.join(cached_symbols)}")
        
        # Batch fetch non-cached symbols
        if fetch_symbols:
            try:
                # Use yfinance's multi-ticker download for efficiency
                data_dict = yf.download(
                    fetch_symbols,
                    period=self._get_extended_period(period),
                    interval=interval,
                    group_by='ticker',
                    auto_adjust=True,
                    prepost=True
                )
                
                for symbol in fetch_symbols:
                    if len(fetch_symbols) == 1:
                        symbol_data = data_dict
                    else:
                        symbol_data = data_dict[symbol]
                    
                    if not symbol_data.empty:
                        symbol_data = self._clean_data(symbol_data)
                        cache_key = self.get_cache_key(symbol, period, interval)
                        st.session_state.data_cache[cache_key] = symbol_data
                        st.session_state.cache_timestamps[cache_key] = time.time()
                        results[symbol] = self._trim_to_period(symbol_data, period)
                
                st.success(f"✅ Fetched data for: {', '.join(fetch_symbols)}")
                
            except Exception as e:
                st.error(f"❌ Error fetching batch data: {str(e)}")
        
        return results
    
    def clear_cache(self, symbol=None):
        """Clear cache for specific symbol or all data"""
        if symbol:
            # Clear specific symbol
            keys_to_remove = [key for key in st.session_state.data_cache.keys() if key.startswith(symbol)]
            for key in keys_to_remove:
                del st.session_state.data_cache[key]
                if key in st.session_state.cache_timestamps:
                    del st.session_state.cache_timestamps[key]
                if key in st.session_state.data_info:
                    del st.session_state.data_info[key]
            st.success(f"🗑️ Cleared cache for {symbol}")
        else:
            # Clear all cache
            st.session_state.data_cache.clear()
            st.session_state.cache_timestamps.clear()
            st.session_state.data_info.clear()
            st.success("🗑️ Cleared all cached data")
    
    def get_cache_info(self):
        """Get information about cached data"""
        cache_info = []
        for key, timestamp in st.session_state.cache_timestamps.items():
            age_minutes = (time.time() - timestamp) / 60
            info = st.session_state.data_info.get(key, {})
            cache_info.append({
                'key': key,
                'symbol': info.get('symbol', 'Unknown'),
                'period': info.get('period', 'Unknown'),
                'rows': info.get('rows', 0),
                'age_minutes': age_minutes,
                'valid': age_minutes < (self.cache_duration / 60)
            })
        return cache_info
    
    def set_cache_duration(self, minutes):
        """Set cache duration in minutes"""
        self.cache_duration = minutes * 60
        st.success(f"🕒 Cache duration set to {minutes} minutes")
    
    def get_data_summary(self, symbol, period='1y', interval='1d'):
        """Get summary statistics for cached data"""
        cache_key = self.get_cache_key(symbol, period, interval)
        
        if cache_key not in st.session_state.data_cache:
            return None
        
        data = st.session_state.data_cache[cache_key]
        
        summary = {
            'symbol': symbol,
            'total_rows': len(data),
            'date_range': {
                'start': data.index[0],
                'end': data.index[-1]
            },
            'price_range': {
                'min': data['Low'].min(),
                'max': data['High'].max(),
                'current': data['Close'].iloc[-1]
            },
            'volume': {
                'avg': data['Volume'].mean(),
                'max': data['Volume'].max(),
                'current': data['Volume'].iloc[-1]
            },
            'data_quality': {
                'missing_values': data.isnull().sum().sum(),
                'complete_days': data.notna().all(axis=1).sum()
            }
        }
        
        return summary
