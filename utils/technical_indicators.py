import pandas as pd
import numpy as np
import streamlit as st
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class TechnicalIndicators:
    """
    Comprehensive technical indicators with price prediction capabilities
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.talib_available = TALIB_AVAILABLE
        
        if not self.talib_available:
            st.warning("📊 TA-Lib not available. Using custom implementations.")
    
    def _default_config(self):
        """Default configuration for technical indicators"""
        return {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'sma_period': 20,
            'ema_period': 20,
            'adx_period': 14,
            'cci_period': 14,
            'williams_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'atr_period': 14
        }
    
    def calculate_all_indicators(self, data):
        """Calculate all technical indicators with error handling"""
        if data is None or data.empty:
            st.error("❌ No data provided for indicator calculation")
            return pd.DataFrame()
        
        try:
            df = data.copy()
            
            # Momentum Indicators
            df = self._add_momentum_indicators(df)
            
            # Trend Indicators  
            df = self._add_trend_indicators(df)
            
            # Volatility Indicators
            df = self._add_volatility_indicators(df)
            
            # Volume Indicators
            df = self._add_volume_indicators(df)
            
            # Support/Resistance Levels
            df = self._add_support_resistance(df)
            
            # Advanced Indicators
            df = self._add_advanced_indicators(df)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error calculating indicators: {str(e)}")
            return data
    
    def _add_momentum_indicators(self, df):
        """Add momentum-based indicators"""
        try:
            # RSI
            df['RSI'] = self._calculate_rsi(df['Close'], self.config['rsi_period'])
            
            # MACD
            macd_data = self._calculate_macd(
                df['Close'], 
                self.config['macd_fast'], 
                self.config['macd_slow'], 
                self.config['macd_signal']
            )
            df['MACD'] = macd_data['MACD']
            df['MACD_Signal'] = macd_data['Signal']
            df['MACD_Histogram'] = macd_data['Histogram']
            
            # Stochastic
            stoch_data = self._calculate_stochastic(df)
            df['Stoch_K'] = stoch_data['%K']
            df['Stoch_D'] = stoch_data['%D']
            
            # Williams %R
            df['Williams_R'] = self._calculate_williams_r(df)
            
            # CCI
            df['CCI'] = self._calculate_cci(df)
            
        except Exception as e:
            st.warning(f"⚠️ Error in momentum indicators: {str(e)}")
        
        return df
    
    def _add_trend_indicators(self, df):
        """Add trend-following indicators"""
        try:
            # Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=self.config['sma_period']).mean()
            df['EMA_20'] = df['Close'].ewm(span=self.config['ema_period']).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # ADX
            df['ADX'] = self._calculate_adx(df)
            
            # Parabolic SAR
            df['PSAR'] = self._calculate_parabolic_sar(df)
            
            # Supertrend
            df['Supertrend'] = self._calculate_supertrend(df)
            
            # Ichimoku Cloud
            ichimoku_data = self._calculate_ichimoku(df)
            for key, value in ichimoku_data.items():
                df[key] = value
                
        except Exception as e:
            st.warning(f"⚠️ Error in trend indicators: {str(e)}")
        
        return df
    
    def _add_volatility_indicators(self, df):
        """Add volatility-based indicators"""
        try:
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(df['Close'])
            df['BB_Upper'] = bb_data['Upper']
            df['BB_Middle'] = bb_data['Middle']
            df['BB_Lower'] = bb_data['Lower']
            df['BB_Width'] = bb_data['Width']
            df['BB_Position'] = bb_data['Position']
            
            # ATR
            df['ATR'] = self._calculate_atr(df)
            
            # Keltner Channels
            kc_data = self._calculate_keltner_channels(df)
            df['KC_Upper'] = kc_data['Upper']
            df['KC_Middle'] = kc_data['Middle']  
            df['KC_Lower'] = kc_data['Lower']
            
        except Exception as e:
            st.warning(f"⚠️ Error in volatility indicators: {str(e)}")
        
        return df
    
    def _add_volume_indicators(self, df):
        """Add volume-based indicators"""
        try:
            # VWAP
            df['VWAP'] = self._calculate_vwap(df)
            
            # Volume Moving Average
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            
            # On-Balance Volume
            df['OBV'] = self._calculate_obv(df)
            
            # Money Flow Index
            df['MFI'] = self._calculate_mfi(df)
            
            # Volume Rate of Change
            df['Volume_ROC'] = df['Volume'].pct_change(periods=10) * 100
            
        except Exception as e:
            st.warning(f"⚠️ Error in volume indicators: {str(e)}")
        
        return df
    
    def _add_support_resistance(self, df):
        """Add support and resistance levels"""
        try:
            # Pivot Points
            pivot_data = self._calculate_pivot_points(df)
            for key, value in pivot_data.items():
                df[key] = value
            
            # Fibonacci Retracements
            fib_data = self._calculate_fibonacci_levels(df)
            for key, value in fib_data.items():
                df[key] = value
                
        except Exception as e:
            st.warning(f"⚠️ Error in support/resistance: {str(e)}")
        
        return df
    
    def _add_advanced_indicators(self, df):
        """Add advanced indicators"""
        try:
            # Elder Ray Index
            elder_data = self._calculate_elder_ray(df)
            df['Bull_Power'] = elder_data['Bull']
            df['Bear_Power'] = elder_data['Bear']
            
            # Awesome Oscillator
            df['AO'] = self._calculate_awesome_oscillator(df)
            
            # Chande Momentum Oscillator
            df['CMO'] = self._calculate_cmo(df['Close'])
            
        except Exception as e:
            st.warning(f"⚠️ Error in advanced indicators: {str(e)}")
        
        return df
    
    # Individual indicator calculations
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if self.talib_available and len(prices) >= period:
            return talib.RSI(prices, timeperiod=period)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if self.talib_available and len(prices) >= slow:
            macd, signal_line, histogram = talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return {'MACD': macd, 'Signal': signal_line, 'Histogram': histogram}
        
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {'MACD': macd, 'Signal': signal_line, 'Histogram': histogram}
    
    def _calculate_bollinger_bands(self, prices):
        """Calculate Bollinger Bands"""
        if self.talib_available and len(prices) >= self.config['bb_period']:
            upper, middle, lower = talib.BBANDS(prices, timeperiod=self.config['bb_period'], nbdevup=self.config['bb_std'], nbdevdn=self.config['bb_std'])
        else:
            middle = prices.rolling(window=self.config['bb_period']).mean()
            std = prices.rolling(window=self.config['bb_period']).std()
            upper = middle + (std * self.config['bb_std'])
            lower = middle - (std * self.config['bb_std'])
        
        width = ((upper - lower) / middle) * 100
        position = ((prices - lower) / (upper - lower)) * 100
        
        return {
            'Upper': upper,
            'Middle': middle,
            'Lower': lower,
            'Width': width,
            'Position': position
        }
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        if self.talib_available and len(df) >= period:
            return talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_adx(self, df, period=14):
        """Calculate ADX"""
        if self.talib_available and len(df) >= period * 2:
            return talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=period)
        
        # Custom ADX calculation
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = self._calculate_atr(df, 1)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        if self.talib_available and len(df) >= k_period:
            k, d = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return {'%K': k, '%D': d}
        
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {'%K': k_percent, '%D': d_percent}
    
    def _calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        if self.talib_available and len(df) >= period:
            return talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=period)
        
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
        
        return williams_r
    
    def _calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index"""
        if self.talib_available and len(df) >= period:
            return talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=period)
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci
    
    def _calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    def _calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['Volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_mfi(self, df, period=14):
        """Calculate Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_flow_ratio = positive_mf / negative_mf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi.fillna(50)
    
    def _calculate_parabolic_sar(self, df, af_start=0.02, af_increment=0.02, af_max=0.2):
        """Calculate Parabolic SAR"""
        length = len(df)
        psar = pd.Series(index=df.index, dtype=float)
        
        if length < 2:
            return psar
        
        # Initialize
        psar.iloc[0] = df['Low'].iloc[0]
        af = af_start
        ep = df['High'].iloc[0]
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, length):
            prev_psar = psar.iloc[i-1]
            
            if trend == 1:  # Uptrend
                psar.iloc[i] = prev_psar + af * (ep - prev_psar)
                
                # Check for trend reversal
                if df['Low'].iloc[i] < psar.iloc[i]:
                    trend = -1
                    psar.iloc[i] = ep
                    af = af_start
                    ep = df['Low'].iloc[i]
                else:
                    # Update EP and AF
                    if df['High'].iloc[i] > ep:
                        ep = df['High'].iloc[i]
                        af = min(af + af_increment, af_max)
                    
                    # Make sure PSAR doesn't exceed previous two lows
                    psar.iloc[i] = min(psar.iloc[i], df['Low'].iloc[i-1])
                    if i > 1:
                        psar.iloc[i] = min(psar.iloc[i], df['Low'].iloc[i-2])
            
            else:  # Downtrend
                psar.iloc[i] = prev_psar + af * (ep - prev_psar)
                
                # Check for trend reversal
                if df['High'].iloc[i] > psar.iloc[i]:
                    trend = 1
                    psar.iloc[i] = ep
                    af = af_start
                    ep = df['High'].iloc[i]
                else:
                    # Update EP and AF
                    if df['Low'].iloc[i] < ep:
                        ep = df['Low'].iloc[i]
                        af = min(af + af_increment, af_max)
                    
                    # Make sure PSAR doesn't exceed previous two highs
                    psar.iloc[i] = max(psar.iloc[i], df['High'].iloc[i-1])
                    if i > 1:
                        psar.iloc[i] = max(psar.iloc[i], df['High'].iloc[i-2])
        
        return psar
    
    def _calculate_supertrend(self, df, period=10, multiplier=3):
        """Calculate Supertrend"""
        hl2 = (df['High'] + df['Low']) / 2
        atr = self._calculate_atr(df, period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        # Initialize
        supertrend.iloc[0] = lower_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            # Calculate basic upper and lower bands
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]
            prev_close = df['Close'].iloc[i-1]
            current_close = df['Close'].iloc[i]
            prev_supertrend = supertrend.iloc[i-1]
            prev_direction = direction.iloc[i-1]
            
            # Update bands
            if current_upper < upper_band.iloc[i-1] or prev_close > upper_band.iloc[i-1]:
                final_upper = current_upper
            else:
                final_upper = upper_band.iloc[i-1]
            
            if current_lower > lower_band.iloc[i-1] or prev_close < lower_band.iloc[i-1]:
                final_lower = current_lower
            else:
                final_lower = lower_band.iloc[i-1]
            
            # Determine direction and supertrend value
            if prev_direction == 1 and current_close <= final_lower:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper
            elif prev_direction == -1 and current_close >= final_upper:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower
            else:
                direction.iloc[i] = prev_direction
                if direction.iloc[i] == 1:
                    supertrend.iloc[i] = final_lower
                else:
                    supertrend.iloc[i] = final_upper
        
        return supertrend
    
    def _calculate_ichimoku(self, df):
        """Calculate Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = df['High'].rolling(window=9).max()
        period9_low = df['Low'].rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 displaced 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 displaced 26 periods ahead
        period52_high = df['High'].rolling(window=52).max()
        period52_low = df['Low'].rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price displaced 26 periods behind
        chikou_span = df['Close'].shift(-26)
        
        return {
            'Tenkan_Sen': tenkan_sen,
            'Kijun_Sen': kijun_sen,
            'Senkou_Span_A': senkou_span_a,
            'Senkou_Span_B': senkou_span_b,
            'Chikou_Span': chikou_span
        }
    
    def _calculate_keltner_channels(self, df, period=20, multiplier=2):
        """Calculate Keltner Channels"""
        middle = df['Close'].ewm(span=period).mean()
        atr = self._calculate_atr(df, period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return {
            'Upper': upper,
            'Middle': middle,
            'Lower': lower
        }
    
    def _calculate_pivot_points(self, df):
        """Calculate Pivot Points"""
        # Use previous day's high, low, close for pivot calculation
        high = df['High'].shift(1)
        low = df['Low'].shift(1)
        close = df['Close'].shift(1)
        
        pivot = (high + low + close) / 3
        
        # Support and Resistance levels
        s1 = (2 * pivot) - high
        r1 = (2 * pivot) - low
        s2 = pivot - (high - low)
        r2 = pivot + (high - low)
        s3 = low - 2 * (high - pivot)
        r3 = high + 2 * (pivot - low)
        
        return {
            'Pivot': pivot,
            'S1': s1, 'S2': s2, 'S3': s3,
            'R1': r1, 'R2': r2, 'R3': r3
        }
    
    def _calculate_fibonacci_levels(self, df, lookback=50):
        """Calculate Fibonacci Retracement Levels"""
        # Find the high and low over the lookback period
        high = df['High'].rolling(window=lookback).max()
        low = df['Low'].rolling(window=lookback).min()
        
        diff = high - low
        
        # Fibonacci levels
        level_236 = high - (0.236 * diff)
        level_382 = high - (0.382 * diff)
        level_500 = high - (0.500 * diff)
        level_618 = high - (0.618 * diff)
        level_786 = high - (0.786 * diff)
        
        return {
            'Fib_High': high,
            'Fib_Low': low,
            'Fib_23.6': level_236,
            'Fib_38.2': level_382,
            'Fib_50.0': level_500,
            'Fib_61.8': level_618,
            'Fib_78.6': level_786
        }
    
    def _calculate_elder_ray(self, df, period=13):
        """Calculate Elder Ray Index"""
        ema = df['Close'].ewm(span=period).mean()
        
        bull_power = df['High'] - ema
        bear_power = df['Low'] - ema
        
        return {
            'Bull': bull_power,
            'Bear': bear_power
        }
    
    def _calculate_awesome_oscillator(self, df):
        """Calculate Awesome Oscillator"""
        median_price = (df['High'] + df['Low']) / 2
        
        ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
        
        return ao
    
    def _calculate_cmo(self, prices, period=14):
        """Calculate Chande Momentum Oscillator"""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0).rolling(window=period).sum()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).sum()
        
        cmo = 100 * ((gain - loss) / (gain + loss))
        
        return cmo
    
    def get_indicator_signals(self, df):
        """Generate trading signals from indicators"""
        signals = {}
        
        try:
            # RSI Signals
            if 'RSI' in df.columns:
                rsi_current = df['RSI'].iloc[-1]
                if rsi_current > 70:
                    signals['RSI'] = {'signal': 'SELL', 'strength': 'Strong', 'value': rsi_current}
                elif rsi_current < 30:
                    signals['RSI'] = {'signal': 'BUY', 'strength': 'Strong', 'value': rsi_current}
                else:
                    signals['RSI'] = {'signal': 'NEUTRAL', 'strength': 'Weak', 'value': rsi_current}
            
            # MACD Signals
            if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
                macd_current = df['MACD'].iloc[-1]
                signal_current = df['MACD_Signal'].iloc[-1]
                
                if macd_current > signal_current:
                    signals['MACD'] = {'signal': 'BUY', 'strength': 'Medium', 'value': macd_current - signal_current}
                else:
                    signals['MACD'] = {'signal': 'SELL', 'strength': 'Medium', 'value': macd_current - signal_current}
            
            # Moving Average Signals
            if all(col in df.columns for col in ['SMA_20', 'SMA_50']):
                price = df['Close'].iloc[-1]
                sma20 = df['SMA_20'].iloc[-1]
                sma50 = df['SMA_50'].iloc[-1]
                
                if price > sma20 > sma50:
                    signals['MA'] = {'signal': 'BUY', 'strength': 'Strong', 'value': 'Bullish alignment'}
                elif price < sma20 < sma50:
                    signals['MA'] = {'signal': 'SELL', 'strength': 'Strong', 'value': 'Bearish alignment'}
                else:
                    signals['MA'] = {'signal': 'NEUTRAL', 'strength': 'Weak', 'value': 'Mixed signals'}
            
            # Bollinger Bands Signals
            if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Position']):
                bb_position = df['BB_Position'].iloc[-1]
                
                if bb_position > 95:
                    signals['BB'] = {'signal': 'SELL', 'strength': 'Medium', 'value': 'Price at upper band'}
                elif bb_position < 5:
                    signals['BB'] = {'signal': 'BUY', 'strength': 'Medium', 'value': 'Price at lower band'}
                else:
                    signals['BB'] = {'signal': 'NEUTRAL', 'strength': 'Weak', 'value': f'Position: {bb_position:.1f}%'}
            
        except Exception as e:
            st.warning(f"⚠️ Error generating signals: {str(e)}")
        
        return signals
    
    def get_price_predictions(self, df):
        """Generate price predictions from technical indicators"""
        predictions = {}
        current_price = df['Close'].iloc[-1]
        
        try:
            # RSI-based prediction
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if rsi < 30:
                    # Oversold, expect bounce
                    target = current_price * 1.05  # 5% upside
                    predictions['RSI'] = {
                        'target_price': target,
                        'confidence': 'Medium',
                        'timeframe': '5-10 days',
                        'reasoning': f'RSI oversold at {rsi:.1f}, expecting bounce'
                    }
                elif rsi > 70:
                    # Overbought, expect pullback
                    target = current_price * 0.95  # 5% downside
                    predictions['RSI'] = {
                        'target_price': target,
                        'confidence': 'Medium',
                        'timeframe': '5-10 days',
                        'reasoning': f'RSI overbought at {rsi:.1f}, expecting pullback'
                    }
            
            # Bollinger Bands prediction
            if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Position']):
                bb_position = df['BB_Position'].iloc[-1]
                bb_upper = df['BB_Upper'].iloc[-1]
                bb_lower = df['BB_Lower'].iloc[-1]
                
                if bb_position > 95:
                    predictions['BB'] = {
                        'target_price': bb_lower,
                        'confidence': 'High',
                        'timeframe': '3-7 days',
                        'reasoning': f'Price at upper band ({bb_position:.1f}%), expecting reversion to lower band'
                    }
                elif bb_position < 5:
                    predictions['BB'] = {
                        'target_price': bb_upper,
                        'confidence': 'High',
                        'timeframe': '3-7 days',
                        'reasoning': f'Price at lower band ({bb_position:.1f}%), expecting reversion to upper band'
                    }
            
            # Support/Resistance prediction
            if 'R1' in df.columns and 'S1' in df.columns:
                r1 = df['R1'].iloc[-1]
                s1 = df['S1'].iloc[-1]
                
                if current_price > df['Pivot'].iloc[-1]:
                    predictions['Support_Resistance'] = {
                        'target_price': r1,
                        'confidence': 'Medium',
                        'timeframe': '1-3 days',
                        'reasoning': f'Price above pivot, targeting R1 at ${r1:.2f}'
                    }
                else:
                    predictions['Support_Resistance'] = {
                        'target_price': s1,
                        'confidence': 'Medium',
                        'timeframe': '1-3 days',
                        'reasoning': f'Price below pivot, targeting S1 at ${s1:.2f}'
                    }
            
            # Moving Average prediction
            if all(col in df.columns for col in ['SMA_20', 'SMA_50']):
                sma20 = df['SMA_20'].iloc[-1]
                sma50 = df['SMA_50'].iloc[-1]
                
                if current_price < sma20:
                    predictions['Moving_Average'] = {
                        'target_price': sma20,
                        'confidence': 'Medium',
                        'timeframe': '5-15 days',
                        'reasoning': f'Price below SMA20, expecting reversion to ${sma20:.2f}'
                    }
                elif sma20 > sma50 and current_price > sma20:
                    # Bullish trend continuation
                    target = current_price + (sma20 - sma50)
                    predictions['Moving_Average'] = {
                        'target_price': target,
                        'confidence': 'High',
                        'timeframe': '10-20 days',
                        'reasoning': f'Bullish MA alignment, trend continuation expected'
                    }
        
        except Exception as e:
            st.warning(f"⚠️ Error generating price predictions: {str(e)}")
        
        return predictions
