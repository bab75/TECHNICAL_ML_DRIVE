import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from utils.technical_indicators import TechnicalIndicators
from utils.ml_models import MLModels

class PredictionEngine:
    """
    Centralized prediction engine combining technical analysis and ML models
    """
    
    def __init__(self, config):
        self.config = config
        self.technical_indicators = TechnicalIndicators(config.get('indicators', {}))
        self.ml_models = MLModels(config)
        self.predictions = {}
        self.confidence_scores = {}
        
    def generate_all_predictions(self, data, symbol):
        """Generate comprehensive price predictions"""
        if data is None or data.empty:
            st.error("❌ No data available for predictions")
            return {}
        
        try:
            with st.spinner("🔮 Generating predictions..."):
                # Add technical indicators
                data_with_indicators = self.technical_indicators.calculate_all_indicators(data)
                
                # Generate technical analysis predictions
                technical_predictions = self._generate_technical_predictions(data_with_indicators)
                
                # Generate ML predictions
                ml_predictions = self._generate_ml_predictions(data_with_indicators)
                
                # Combine all predictions
                all_predictions = {
                    'technical': technical_predictions,
                    'ml': ml_predictions,
                    'summary': self._create_prediction_summary(data, technical_predictions, ml_predictions)
                }
                
                self.predictions[symbol] = all_predictions
                return all_predictions
                
        except Exception as e:
            st.error(f"❌ Error generating predictions: {str(e)}")
            return {}
    
    def _generate_technical_predictions(self, data):
        """Generate predictions from technical indicators"""
        predictions = {}
        current_price = data['Close'].iloc[-1]
        
        try:
            # RSI-based predictions
            if 'RSI' in data.columns:
                predictions['RSI'] = self._predict_from_rsi(data, current_price)
            
            # MACD-based predictions
            if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
                predictions['MACD'] = self._predict_from_macd(data, current_price)
            
            # Bollinger Bands predictions
            if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Position']):
                predictions['Bollinger_Bands'] = self._predict_from_bollinger(data, current_price)
            
            # Moving Average predictions
            if all(col in data.columns for col in ['SMA_20', 'SMA_50']):
                predictions['Moving_Averages'] = self._predict_from_ma(data, current_price)
            
            # Support/Resistance predictions
            if 'R1' in data.columns and 'S1' in data.columns:
                predictions['Support_Resistance'] = self._predict_from_sr(data, current_price)
            
            # Momentum predictions
            if 'Stoch_K' in data.columns:
                predictions['Stochastic'] = self._predict_from_stochastic(data, current_price)
            
            # Trend predictions
            if 'ADX' in data.columns:
                predictions['ADX'] = self._predict_from_adx(data, current_price)
            
            # Volume analysis
            predictions['Volume'] = self._predict_from_volume(data, current_price)
            
        except Exception as e:
            st.warning(f"⚠️ Error in technical predictions: {str(e)}")
        
        return predictions
    
    def _predict_from_rsi(self, data, current_price):
        """RSI-based price prediction"""
        rsi = data['RSI'].iloc[-1]
        rsi_14d_avg = data['RSI'].rolling(14).mean().iloc[-1]
        
        if rsi < 30:
            # Oversold - expect bounce
            target = current_price * 1.05
            confidence = "High" if rsi < 25 else "Medium"
            timeframe = "3-7 days"
            reasoning = f"RSI severely oversold at {rsi:.1f}, strong bounce expected"
        elif rsi > 70:
            # Overbought - expect pullback
            target = current_price * 0.95
            confidence = "High" if rsi > 75 else "Medium"
            timeframe = "3-7 days"
            reasoning = f"RSI overbought at {rsi:.1f}, pullback likely"
        elif rsi > rsi_14d_avg + 10:
            # Momentum building
            target = current_price * 1.02
            confidence = "Medium"
            timeframe = "5-10 days"
            reasoning = f"RSI momentum building, short-term upside potential"
        elif rsi < rsi_14d_avg - 10:
            # Losing momentum
            target = current_price * 0.98
            confidence = "Medium"
            timeframe = "5-10 days"
            reasoning = f"RSI losing momentum, short-term weakness expected"
        else:
            # Neutral
            target = current_price
            confidence = "Low"
            timeframe = "Neutral"
            reasoning = f"RSI neutral at {rsi:.1f}, no clear direction"
        
        return {
            'target_price': target,
            'price_change': ((target - current_price) / current_price) * 100,
            'confidence': confidence,
            'timeframe': timeframe,
            'reasoning': reasoning,
            'indicator_value': rsi
        }
    
    def _predict_from_macd(self, data, current_price):
        """MACD-based price prediction"""
        macd = data['MACD'].iloc[-1]
        signal = data['MACD_Signal'].iloc[-1]
        histogram = data['MACD_Histogram'].iloc[-1]
        prev_histogram = data['MACD_Histogram'].iloc[-2]
        
        if macd > signal and histogram > prev_histogram:
            # Bullish momentum accelerating
            target = current_price * 1.04
            confidence = "High"
            timeframe = "5-15 days"
            reasoning = "MACD bullish crossover with accelerating momentum"
        elif macd > signal and histogram < prev_histogram:
            # Bullish but slowing
            target = current_price * 1.02
            confidence = "Medium"
            timeframe = "3-10 days"
            reasoning = "MACD bullish but momentum slowing"
        elif macd < signal and histogram < prev_histogram:
            # Bearish momentum accelerating
            target = current_price * 0.96
            confidence = "High"
            timeframe = "5-15 days"
            reasoning = "MACD bearish crossover with accelerating downside"
        elif macd < signal and histogram > prev_histogram:
            # Bearish but improving
            target = current_price * 0.98
            confidence = "Medium"
            timeframe = "3-10 days"
            reasoning = "MACD bearish but showing signs of improvement"
        else:
            target = current_price
            confidence = "Low"
            timeframe = "Neutral"
            reasoning = "MACD showing mixed signals"
        
        return {
            'target_price': target,
            'price_change': ((target - current_price) / current_price) * 100,
            'confidence': confidence,
            'timeframe': timeframe,
            'reasoning': reasoning,
            'indicator_value': macd - signal
        }
    
    def _predict_from_bollinger(self, data, current_price):
        """Bollinger Bands prediction"""
        bb_position = data['BB_Position'].iloc[-1]
        bb_width = data['BB_Width'].iloc[-1]
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        bb_middle = data['BB_Middle'].iloc[-1]
        
        if bb_position > 95:
            # At upper band
            target = bb_middle
            confidence = "High"
            timeframe = "3-7 days"
            reasoning = f"Price at upper BB ({bb_position:.1f}%), mean reversion expected"
        elif bb_position < 5:
            # At lower band
            target = bb_middle
            confidence = "High"
            timeframe = "3-7 days"
            reasoning = f"Price at lower BB ({bb_position:.1f}%), bounce expected"
        elif bb_position > 80 and bb_width < 5:
            # High in narrow bands (breakout potential)
            target = bb_upper
            confidence = "Medium"
            timeframe = "5-10 days"
            reasoning = "Price high in tight range, upside breakout possible"
        elif bb_position < 20 and bb_width < 5:
            # Low in narrow bands
            target = bb_lower
            confidence = "Medium"
            timeframe = "5-10 days"
            reasoning = "Price low in tight range, downside breakout possible"
        else:
            # Normal range
            if bb_position > 50:
                target = current_price * 1.01
            else:
                target = current_price * 0.99
            confidence = "Low"
            timeframe = "Variable"
            reasoning = f"Price in normal BB range ({bb_position:.1f}%)"
        
        return {
            'target_price': target,
            'price_change': ((target - current_price) / current_price) * 100,
            'confidence': confidence,
            'timeframe': timeframe,
            'reasoning': reasoning,
            'indicator_value': bb_position
        }
    
    def _predict_from_ma(self, data, current_price):
        """Moving Average prediction"""
        sma20 = data['SMA_20'].iloc[-1]
        sma50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else sma20
        
        # Calculate distances
        price_vs_sma20 = ((current_price - sma20) / sma20) * 100
        sma20_vs_sma50 = ((sma20 - sma50) / sma50) * 100
        
        if current_price > sma20 > sma50 and sma20_vs_sma50 > 2:
            # Strong bullish alignment
            target = current_price * 1.03
            confidence = "High"
            timeframe = "10-20 days"
            reasoning = "Strong bullish MA alignment, trend continuation expected"
        elif current_price < sma20 < sma50 and sma20_vs_sma50 < -2:
            # Strong bearish alignment
            target = current_price * 0.97
            confidence = "High"
            timeframe = "10-20 days"
            reasoning = "Strong bearish MA alignment, downtrend continuation expected"
        elif abs(price_vs_sma20) > 5:
            # Price extended from MA
            target = sma20
            confidence = "Medium"
            timeframe = "5-15 days"
            reasoning = f"Price {abs(price_vs_sma20):.1f}% from SMA20, reversion expected"
        else:
            # Near moving averages
            if sma20 > sma50:
                target = current_price * 1.01
            else:
                target = current_price * 0.99
            confidence = "Low"
            timeframe = "Variable"
            reasoning = "Price near moving averages, limited directional bias"
        
        return {
            'target_price': target,
            'price_change': ((target - current_price) / current_price) * 100,
            'confidence': confidence,
            'timeframe': timeframe,
            'reasoning': reasoning,
            'indicator_value': price_vs_sma20
        }
    
    def _predict_from_sr(self, data, current_price):
        """Support/Resistance prediction"""
        pivot = data['Pivot'].iloc[-1]
        r1 = data['R1'].iloc[-1]
        s1 = data['S1'].iloc[-1]
        r2 = data.get('R2', pd.Series([r1])).iloc[-1]
        s2 = data.get('S2', pd.Series([s1])).iloc[-1]
        
        # Determine current position
        if current_price > r1:
            target = r2
            confidence = "Medium"
            timeframe = "1-5 days"
            reasoning = f"Above R1 (${r1:.2f}), targeting R2"
        elif current_price > pivot:
            target = r1
            confidence = "Medium"
            timeframe = "1-3 days"
            reasoning = f"Above pivot (${pivot:.2f}), targeting R1"
        elif current_price < s1:
            target = s2
            confidence = "Medium"
            timeframe = "1-5 days"
            reasoning = f"Below S1 (${s1:.2f}), targeting S2"
        elif current_price < pivot:
            target = s1
            confidence = "Medium"
            timeframe = "1-3 days"
            reasoning = f"Below pivot (${pivot:.2f}), targeting S1"
        else:
            # Near pivot
            target = pivot
            confidence = "Low"
            timeframe = "1-2 days"
            reasoning = f"Near pivot point (${pivot:.2f})"
        
        return {
            'target_price': target,
            'price_change': ((target - current_price) / current_price) * 100,
            'confidence': confidence,
            'timeframe': timeframe,
            'reasoning': reasoning,
            'indicator_value': (current_price - pivot) / pivot * 100
        }
    
    def _predict_from_stochastic(self, data, current_price):
        """Stochastic oscillator prediction"""
        stoch_k = data['Stoch_K'].iloc[-1]
        stoch_d = data['Stoch_D'].iloc[-1]
        
        if stoch_k < 20 and stoch_d < 20:
            # Oversold
            target = current_price * 1.03
            confidence = "Medium"
            timeframe = "3-7 days"
            reasoning = f"Stochastic oversold (%K:{stoch_k:.1f}, %D:{stoch_d:.1f})"
        elif stoch_k > 80 and stoch_d > 80:
            # Overbought
            target = current_price * 0.97
            confidence = "Medium"
            timeframe = "3-7 days"
            reasoning = f"Stochastic overbought (%K:{stoch_k:.1f}, %D:{stoch_d:.1f})"
        elif stoch_k > stoch_d and stoch_k < 80:
            # Bullish crossover
            target = current_price * 1.02
            confidence = "Medium"
            timeframe = "5-10 days"
            reasoning = "Stochastic bullish crossover signal"
        elif stoch_k < stoch_d and stoch_k > 20:
            # Bearish crossover
            target = current_price * 0.98
            confidence = "Medium"
            timeframe = "5-10 days"
            reasoning = "Stochastic bearish crossover signal"
        else:
            target = current_price
            confidence = "Low"
            timeframe = "Neutral"
            reasoning = f"Stochastic neutral (%K:{stoch_k:.1f})"
        
        return {
            'target_price': target,
            'price_change': ((target - current_price) / current_price) * 100,
            'confidence': confidence,
            'timeframe': timeframe,
            'reasoning': reasoning,
            'indicator_value': stoch_k
        }
    
    def _predict_from_adx(self, data, current_price):
        """ADX trend strength prediction"""
        adx = data['ADX'].iloc[-1]
        adx_prev = data['ADX'].iloc[-2]
        
        # Get trend direction from price vs MA
        sma20 = data.get('SMA_20', data['Close']).iloc[-1]
        trend_direction = "up" if current_price > sma20 else "down"
        
        if adx > 25 and adx > adx_prev:
            # Strong and strengthening trend
            if trend_direction == "up":
                target = current_price * 1.04
                reasoning = f"Strong uptrend (ADX:{adx:.1f}) gaining strength"
            else:
                target = current_price * 0.96
                reasoning = f"Strong downtrend (ADX:{adx:.1f}) gaining strength"
            confidence = "High"
            timeframe = "10-20 days"
        elif adx > 25 and adx < adx_prev:
            # Strong but weakening trend
            target = current_price * 1.01 if trend_direction == "up" else current_price * 0.99
            confidence = "Medium"
            timeframe = "5-15 days"
            reasoning = f"Strong trend (ADX:{adx:.1f}) but weakening"
        elif adx < 20:
            # Weak trend, range-bound
            target = current_price
            confidence = "Low"
            timeframe = "Variable"
            reasoning = f"Weak trend strength (ADX:{adx:.1f}), range-bound market"
        else:
            # Moderate trend
            target = current_price * 1.01 if trend_direction == "up" else current_price * 0.99
            confidence = "Medium"
            timeframe = "5-10 days"
            reasoning = f"Moderate trend strength (ADX:{adx:.1f})"
        
        return {
            'target_price': target,
            'price_change': ((target - current_price) / current_price) * 100,
            'confidence': confidence,
            'timeframe': timeframe,
            'reasoning': reasoning,
            'indicator_value': adx
        }
    
    def _predict_from_volume(self, data, current_price):
        """Volume analysis prediction"""
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Price change for volume confirmation
        price_change = data['Close'].pct_change().iloc[-1] * 100
        
        if volume_ratio > 2 and price_change > 2:
            # High volume breakout up
            target = current_price * 1.05
            confidence = "High"
            timeframe = "3-10 days"
            reasoning = f"High volume ({volume_ratio:.1f}x avg) confirms upside breakout"
        elif volume_ratio > 2 and price_change < -2:
            # High volume breakdown
            target = current_price * 0.95
            confidence = "High"
            timeframe = "3-10 days"
            reasoning = f"High volume ({volume_ratio:.1f}x avg) confirms downside breakdown"
        elif volume_ratio > 1.5:
            # Above average volume
            target = current_price * (1 + (price_change * 0.5 / 100))
            confidence = "Medium"
            timeframe = "2-7 days"
            reasoning = f"Above average volume ({volume_ratio:.1f}x) supports move"
        else:
            # Low volume
            target = current_price
            confidence = "Low"
            timeframe = "Uncertain"
            reasoning = f"Low volume ({volume_ratio:.1f}x avg), move lacks conviction"
        
        return {
            'target_price': target,
            'price_change': ((target - current_price) / current_price) * 100,
            'confidence': confidence,
            'timeframe': timeframe,
            'reasoning': reasoning,
            'indicator_value': volume_ratio
        }
    
    def _generate_ml_predictions(self, data):
        """Generate ML model predictions"""
        try:
            # Train ML models
            ml_results = self.ml_models.train_all_models(data)
            
            if not ml_results:
                return {}
            
            # Generate future predictions
            future_predictions = self.ml_models.predict_future_prices(data, days_ahead=5)
            
            # Get model explanations
            explanations = self.ml_models.get_model_explanations(data)
            
            # Combine results
            ml_predictions = {}
            current_price = data['Close'].iloc[-1]
            
            for model_name in ml_results.keys():
                if model_name in future_predictions:
                    future_price = future_predictions[model_name]
                    if isinstance(future_price, (list, np.ndarray)):
                        target_price = future_price[-1] if len(future_price) > 0 else current_price
                    else:
                        target_price = future_price
                    
                    explanation = explanations.get(model_name, {})
                    
                    ml_predictions[model_name] = {
                        'target_price': target_price,
                        'price_change': ((target_price - current_price) / current_price) * 100,
                        'confidence': explanation.get('confidence', 'Unknown'),
                        'timeframe': '5 days',
                        'reasoning': explanation.get('method', 'ML model prediction'),
                        'accuracy': explanation.get('accuracy', 'Unknown'),
                        'key_factors': explanation.get('key_factors', [])
                    }
            
            return ml_predictions
            
        except Exception as e:
            st.warning(f"⚠️ Error in ML predictions: {str(e)}")
            return {}
    
    def _create_prediction_summary(self, data, technical_predictions, ml_predictions):
        """Create comprehensive prediction summary"""
        current_price = data['Close'].iloc[-1]
        
        # Collect all predictions
        all_targets = []
        high_confidence_targets = []
        
        # Technical predictions
        for pred_name, pred_data in technical_predictions.items():
            if isinstance(pred_data, dict) and 'target_price' in pred_data:
                all_targets.append(pred_data['target_price'])
                if pred_data.get('confidence') == 'High':
                    high_confidence_targets.append(pred_data['target_price'])
        
        # ML predictions
        for pred_name, pred_data in ml_predictions.items():
            if isinstance(pred_data, dict) and 'target_price' in pred_data:
                all_targets.append(pred_data['target_price'])
                if pred_data.get('confidence') in ['High', 'Medium']:
                    high_confidence_targets.append(pred_data['target_price'])
        
        if not all_targets:
            return self._create_neutral_summary(current_price)
        
        # Calculate consensus
        consensus_price = np.median(all_targets)
        mean_price = np.mean(all_targets)
        high_conf_consensus = np.median(high_confidence_targets) if high_confidence_targets else consensus_price
        
        # Calculate prediction range
        price_range = {
            'min': min(all_targets),
            'max': max(all_targets),
            'std': np.std(all_targets)
        }
        
        # Determine overall sentiment
        bullish_count = sum(1 for price in all_targets if price > current_price * 1.01)
        bearish_count = sum(1 for price in all_targets if price < current_price * 0.99)
        neutral_count = len(all_targets) - bullish_count - bearish_count
        
        if bullish_count > bearish_count and bullish_count > neutral_count:
            sentiment = "Bullish"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        
        # Overall confidence
        total_predictions = len(all_targets)
        confidence_score = len(high_confidence_targets) / total_predictions if total_predictions > 0 else 0
        
        if confidence_score > 0.6:
            overall_confidence = "High"
        elif confidence_score > 0.3:
            overall_confidence = "Medium"
        else:
            overall_confidence = "Low"
        
        return {
            'consensus_target': high_conf_consensus,
            'price_change': ((high_conf_consensus - current_price) / current_price) * 100,
            'sentiment': sentiment,
            'confidence': overall_confidence,
            'timeframe': '3-10 days',
            'reasoning': f"{sentiment} consensus from {total_predictions} indicators/models",
            'prediction_range': price_range,
            'agreement_level': f"{max(bullish_count, bearish_count, neutral_count)}/{total_predictions}",
            'technical_count': len(technical_predictions),
            'ml_count': len(ml_predictions)
        }
    
    def _create_neutral_summary(self, current_price):
        """Create neutral summary when no predictions available"""
        return {
            'consensus_target': current_price,
            'price_change': 0,
            'sentiment': 'Neutral',
            'confidence': 'Low',
            'timeframe': 'Unknown',
            'reasoning': 'Insufficient data for reliable predictions',
            'prediction_range': {'min': current_price, 'max': current_price, 'std': 0},
            'agreement_level': '0/0',
            'technical_count': 0,
            'ml_count': 0
        }
    
    def get_prediction_comparison(self, symbol):
        """Compare predictions across different methods"""
        if symbol not in self.predictions:
            return None
        
        predictions = self.predictions[symbol]
        technical = predictions.get('technical', {})
        ml = predictions.get('ml', {})
        
        comparison = {
            'methods': [],
            'targets': [],
            'changes': [],
            'confidences': [],
            'timeframes': [],
            'reasonings': []
        }
        
        # Add technical predictions
        for method, pred in technical.items():
            if isinstance(pred, dict) and 'target_price' in pred:
                comparison['methods'].append(f"TA: {method}")
                comparison['targets'].append(pred['target_price'])
                comparison['changes'].append(pred['price_change'])
                comparison['confidences'].append(pred['confidence'])
                comparison['timeframes'].append(pred['timeframe'])
                comparison['reasonings'].append(pred['reasoning'])
        
        # Add ML predictions
        for method, pred in ml.items():
            if isinstance(pred, dict) and 'target_price' in pred:
                comparison['methods'].append(f"ML: {method}")
                comparison['targets'].append(pred['target_price'])
                comparison['changes'].append(pred['price_change'])
                comparison['confidences'].append(pred['confidence'])
                comparison['timeframes'].append(pred['timeframe'])
                comparison['reasonings'].append(pred['reasoning'])
        
        return comparison
    
    def export_predictions(self, symbol):
        """Export predictions to DataFrame for analysis"""
        if symbol not in self.predictions:
            return pd.DataFrame()
        
        predictions = self.predictions[symbol]
        rows = []
        
        # Technical predictions
        for method, pred in predictions.get('technical', {}).items():
            if isinstance(pred, dict):
                rows.append({
                    'Type': 'Technical',
                    'Method': method,
                    'Target_Price': pred.get('target_price', 0),
                    'Price_Change_%': pred.get('price_change', 0),
                    'Confidence': pred.get('confidence', 'Unknown'),
                    'Timeframe': pred.get('timeframe', 'Unknown'),
                    'Reasoning': pred.get('reasoning', '')
                })
        
        # ML predictions
        for method, pred in predictions.get('ml', {}).items():
            if isinstance(pred, dict):
                rows.append({
                    'Type': 'ML',
                    'Method': method,
                    'Target_Price': pred.get('target_price', 0),
                    'Price_Change_%': pred.get('price_change', 0),
                    'Confidence': pred.get('confidence', 'Unknown'),
                    'Timeframe': pred.get('timeframe', 'Unknown'),
                    'Reasoning': pred.get('reasoning', '')
                })
        
        # Summary
        summary = predictions.get('summary', {})
        if summary:
            rows.append({
                'Type': 'Consensus',
                'Method': 'Combined',
                'Target_Price': summary.get('consensus_target', 0),
                'Price_Change_%': summary.get('price_change', 0),
                'Confidence': summary.get('confidence', 'Unknown'),
                'Timeframe': summary.get('timeframe', 'Unknown'),
                'Reasoning': summary.get('reasoning', '')
            })
        
        return pd.DataFrame(rows)
