import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except (ImportError, TypeError, OSError):
    TENSORFLOW_AVAILABLE = False
    tf = None

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class MLModels:
    """
    Comprehensive ML models for stock price prediction with explanations
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Check available libraries
        self.available_models = self._check_available_models()
        
    def _default_config(self):
        """Default configuration for ML models"""
        return {
            'ml_models': {
                'lstm_lookback': 60,
                'lstm_epochs': 50,
                'lstm_batch_size': 32,
                'prophet_seasonality': True,
                'xgb_n_estimators': 100,
                'xgb_max_depth': 6,
                'rf_n_estimators': 100,
                'rf_max_depth': 10,
                'test_size': 0.2,
                'random_state': 42
            }
        }
    
    def _check_available_models(self):
        """Check which ML libraries are available"""
        available = {
            'Random Forest': True,  # sklearn always available
            'Gradient Boosting': True,  # sklearn always available
            'XGBoost': XGBOOST_AVAILABLE,
            'LightGBM': LIGHTGBM_AVAILABLE,
            'Prophet': PROPHET_AVAILABLE,
            'LSTM': TENSORFLOW_AVAILABLE,
            'GRU': TENSORFLOW_AVAILABLE,
            'ARIMA': STATSMODELS_AVAILABLE,
            'SARIMAX': STATSMODELS_AVAILABLE
        }
        
        unavailable = [model for model, avail in available.items() if not avail]
        if unavailable:
            st.info(f"📦 Some models unavailable: {', '.join(unavailable)}")
        
        return available
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        try:
            features_df = df.copy()
            
            # Price-based features
            features_df['Returns'] = features_df['Close'].pct_change()
            features_df['Log_Returns'] = np.log(features_df['Close'] / features_df['Close'].shift(1))
            features_df['Price_Change'] = features_df['Close'] - features_df['Open']
            features_df['High_Low_Ratio'] = features_df['High'] / features_df['Low']
            features_df['Volume_Price_Ratio'] = features_df['Volume'] / features_df['Close']
            
            # Volatility features
            features_df['Volatility_5'] = features_df['Returns'].rolling(5).std()
            features_df['Volatility_20'] = features_df['Returns'].rolling(20).std()
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'Close_Lag_{lag}'] = features_df['Close'].shift(lag)
                features_df[f'Volume_Lag_{lag}'] = features_df['Volume'].shift(lag)
                features_df[f'Returns_Lag_{lag}'] = features_df['Returns'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features_df[f'Close_MA_{window}'] = features_df['Close'].rolling(window).mean()
                features_df[f'Close_Std_{window}'] = features_df['Close'].rolling(window).std()
                features_df[f'Volume_MA_{window}'] = features_df['Volume'].rolling(window).mean()
            
            # Technical indicator ratios (if available)
            if 'RSI' in features_df.columns:
                features_df['RSI_Normalized'] = (features_df['RSI'] - 50) / 50
            
            if 'MACD' in features_df.columns and 'MACD_Signal' in features_df.columns:
                features_df['MACD_Ratio'] = features_df['MACD'] / features_df['MACD_Signal']
            
            # Time-based features
            features_df['Day_of_Week'] = features_df.index.dayofweek
            features_df['Month'] = features_df.index.month
            features_df['Quarter'] = features_df.index.quarter
            
            # Remove infinite and NaN values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            return features_df
            
        except Exception as e:
            st.error(f"❌ Error preparing features: {str(e)}")
            return df
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        try:
            model = RandomForestRegressor(
                n_estimators=self.config['ml_models']['rf_n_estimators'],
                max_depth=self.config['ml_models']['rf_max_depth'],
                random_state=self.config['ml_models']['random_state'],
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Feature importance
            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
            importance = dict(zip(feature_names, model.feature_importances_))
            
            self.models['Random Forest'] = model
            self.feature_importance['Random Forest'] = importance
            self.model_performance['Random Forest'] = {
                'MSE': mse, 'MAE': mae, 'R2': r2,
                'RMSE': np.sqrt(mse)
            }
            
            return predictions, model
            
        except Exception as e:
            st.error(f"❌ Error training Random Forest: {str(e)}")
            return None, None
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            return None, None
        
        try:
            model = xgb.XGBRegressor(
                n_estimators=self.config['ml_models']['xgb_n_estimators'],
                max_depth=self.config['ml_models']['xgb_max_depth'],
                random_state=self.config['ml_models']['random_state'],
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Feature importance
            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
            importance = dict(zip(feature_names, model.feature_importances_))
            
            self.models['XGBoost'] = model
            self.feature_importance['XGBoost'] = importance
            self.model_performance['XGBoost'] = {
                'MSE': mse, 'MAE': mae, 'R2': r2,
                'RMSE': np.sqrt(mse)
            }
            
            return predictions, model
            
        except Exception as e:
            st.error(f"❌ Error training XGBoost: {str(e)}")
            return None, None
    
    def train_lstm(self, data, target_column='Close'):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return None, None
        
        try:
            # Prepare data for LSTM
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[[target_column]])
            
            lookback = self.config['ml_models']['lstm_lookback']
            
            # Create sequences
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Split data
            split_idx = int(len(X) * (1 - self.config['ml_models']['test_size']))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=self.config['ml_models']['lstm_epochs'],
                batch_size=self.config['ml_models']['lstm_batch_size'],
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Inverse transform
            predictions = scaler.inverse_transform(predictions)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            mse = mean_squared_error(y_test_actual, predictions)
            mae = mean_absolute_error(y_test_actual, predictions)
            r2 = r2_score(y_test_actual, predictions)
            
            self.models['LSTM'] = model
            self.scalers['LSTM'] = scaler
            self.model_performance['LSTM'] = {
                'MSE': mse, 'MAE': mae, 'R2': r2,
                'RMSE': np.sqrt(mse)
            }
            
            return predictions.flatten(), model
            
        except Exception as e:
            st.error(f"❌ Error training LSTM: {str(e)}")
            return None, None
    
    def train_prophet(self, data, target_column='Close'):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            return None, None
        
        try:
            # Prepare data for Prophet
            prophet_data = data.reset_index()
            prophet_data = prophet_data[['Date', target_column]].rename(columns={'Date': 'ds', target_column: 'y'})
            
            # Split data
            split_idx = int(len(prophet_data) * (1 - self.config['ml_models']['test_size']))
            train_data = prophet_data[:split_idx]
            test_data = prophet_data[split_idx:]
            
            # Initialize and train Prophet
            model = Prophet(
                yearly_seasonality=self.config['ml_models']['prophet_seasonality'],
                weekly_seasonality=True,
                daily_seasonality=False
            )
            
            model.fit(train_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            
            # Extract test predictions
            predictions = forecast['yhat'][split_idx:].values
            y_test = test_data['y'].values
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            self.models['Prophet'] = model
            self.model_performance['Prophet'] = {
                'MSE': mse, 'MAE': mae, 'R2': r2,
                'RMSE': np.sqrt(mse)
            }
            
            return predictions, model
            
        except Exception as e:
            st.error(f"❌ Error training Prophet: {str(e)}")
            return None, None
    
    def train_arima(self, data, target_column='Close'):
        """Train ARIMA model"""
        if not STATSMODELS_AVAILABLE:
            return None, None
        
        try:
            # Prepare data
            ts_data = data[target_column].dropna()
            
            # Split data
            split_idx = int(len(ts_data) * (1 - self.config['ml_models']['test_size']))
            train_data = ts_data[:split_idx]
            test_data = ts_data[split_idx:]
            
            # Auto ARIMA order selection (simplified)
            # In practice, you'd use auto_arima or grid search
            model = ARIMA(train_data, order=(5,1,0))
            fitted_model = model.fit()
            
            # Make predictions
            predictions = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mse = mean_squared_error(test_data, predictions)
            mae = mean_absolute_error(test_data, predictions)
            r2 = r2_score(test_data, predictions)
            
            self.models['ARIMA'] = fitted_model
            self.model_performance['ARIMA'] = {
                'MSE': mse, 'MAE': mae, 'R2': r2,
                'RMSE': np.sqrt(mse)
            }
            
            return predictions, fitted_model
            
        except Exception as e:
            st.error(f"❌ Error training ARIMA: {str(e)}")
            return None, None
    
    def train_all_models(self, data):
        """Train all available models"""
        results = {}
        
        with st.spinner("🤖 Training ML models..."):
            # Prepare features
            features_df = self.prepare_features(data)
            
            # Define feature columns (exclude target and non-numeric)
            feature_columns = [col for col in features_df.columns 
                             if col not in ['Close', 'Open', 'High', 'Low', 'Volume'] 
                             and features_df[col].dtype in ['int64', 'float64']]
            
            if len(feature_columns) < 5:
                st.warning("⚠️ Insufficient features for ML training")
                return results
            
            # Prepare data for tree-based models
            X = features_df[feature_columns].dropna()
            y = features_df.loc[X.index, 'Close']
            
            if len(X) < 100:
                st.warning("⚠️ Insufficient data for reliable ML training")
                return results
            
            # Split data maintaining time order
            split_idx = int(len(X) * (1 - self.config['ml_models']['test_size']))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train Random Forest
            if self.available_models['Random Forest']:
                st.write("🌲 Training Random Forest...")
                rf_pred, rf_model = self.train_random_forest(X_train, y_train, X_test, y_test)
                if rf_pred is not None:
                    results['Random Forest'] = rf_pred
            
            # Train XGBoost
            if self.available_models['XGBoost']:
                st.write("🚀 Training XGBoost...")
                xgb_pred, xgb_model = self.train_xgboost(X_train, y_train, X_test, y_test)
                if xgb_pred is not None:
                    results['XGBoost'] = xgb_pred
            
            # Train LSTM
            if self.available_models['LSTM']:
                st.write("🧠 Training LSTM...")
                lstm_pred, lstm_model = self.train_lstm(data)
                if lstm_pred is not None:
                    results['LSTM'] = lstm_pred
            
            # Train Prophet
            if self.available_models['Prophet']:
                st.write("📈 Training Prophet...")
                prophet_pred, prophet_model = self.train_prophet(data)
                if prophet_pred is not None:
                    results['Prophet'] = prophet_pred
            
            # Train ARIMA
            if self.available_models['ARIMA']:
                st.write("📊 Training ARIMA...")
                arima_pred, arima_model = self.train_arima(data)
                if arima_pred is not None:
                    results['ARIMA'] = arima_pred
        
        st.success(f"✅ Trained {len(results)} models successfully!")
        return results
    
    def predict_future_prices(self, data, days_ahead=5):
        """Generate future price predictions"""
        predictions = {}
        current_price = data['Close'].iloc[-1]
        
        try:
            for model_name, model in self.models.items():
                if model_name == 'LSTM' and TENSORFLOW_AVAILABLE:
                    # LSTM future prediction
                    scaler = self.scalers.get('LSTM')
                    if scaler:
                        lookback = self.config['ml_models']['lstm_lookback']
                        last_sequence = data['Close'].iloc[-lookback:].values.reshape(-1, 1)
                        scaled_sequence = scaler.transform(last_sequence)
                        
                        future_predictions = []
                        current_sequence = scaled_sequence.copy()
                        
                        for _ in range(days_ahead):
                            next_pred = model.predict(current_sequence.reshape(1, lookback, 1))
                            future_predictions.append(next_pred[0, 0])
                            # Update sequence for next prediction
                            current_sequence = np.roll(current_sequence, -1)
                            current_sequence[-1] = next_pred[0, 0]
                        
                        # Inverse transform
                        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                        predictions[model_name] = future_predictions.flatten()
                
                elif model_name == 'Prophet' and PROPHET_AVAILABLE:
                    # Prophet future prediction
                    future = model.make_future_dataframe(periods=days_ahead)
                    forecast = model.predict(future)
                    predictions[model_name] = forecast['yhat'].iloc[-days_ahead:].values
                
                elif model_name == 'ARIMA' and STATSMODELS_AVAILABLE:
                    # ARIMA future prediction
                    forecast = model.forecast(steps=days_ahead)
                    predictions[model_name] = forecast
                
                elif model_name in ['Random Forest', 'XGBoost']:
                    # Tree-based models need feature engineering for future prediction
                    # This is a simplified approach - in practice, you'd need to project features
                    last_features = self.prepare_features(data).iloc[-1:]
                    feature_columns = [col for col in last_features.columns 
                                     if col not in ['Close', 'Open', 'High', 'Low', 'Volume'] 
                                     and last_features[col].dtype in ['int64', 'float64']]
                    
                    if len(feature_columns) > 0:
                        X_future = last_features[feature_columns].fillna(method='ffill')
                        if not X_future.empty:
                            # Simple projection (repeat last prediction)
                            base_prediction = model.predict(X_future)[0]
                            # Add some trend continuation
                            trend = (current_price - data['Close'].iloc[-5]) / 5  # 5-day trend
                            future_preds = []
                            for i in range(days_ahead):
                                future_preds.append(base_prediction + (trend * (i + 1)))
                            predictions[model_name] = future_preds
        
        except Exception as e:
            st.warning(f"⚠️ Error generating future predictions: {str(e)}")
        
        return predictions
    
    def get_model_explanations(self, data):
        """Get explanations for model predictions"""
        explanations = {}
        
        try:
            current_price = data['Close'].iloc[-1]
            
            for model_name, performance in self.model_performance.items():
                explanation = {
                    'accuracy': f"R² Score: {performance['R2']:.3f}",
                    'error': f"RMSE: ${performance['RMSE']:.2f}",
                    'confidence': self._get_confidence_level(performance['R2']),
                }
                
                # Add model-specific explanations
                if model_name in self.feature_importance:
                    # Get top 3 most important features
                    importance = self.feature_importance[model_name]
                    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    explanation['key_factors'] = [f"{feat}: {imp:.3f}" for feat, imp in top_features]
                
                if model_name == 'LSTM':
                    explanation['method'] = "Deep learning model analyzing 60-day price patterns"
                elif model_name == 'Prophet':
                    explanation['method'] = "Time series decomposition with trend and seasonality"
                elif model_name == 'ARIMA':
                    explanation['method'] = "Statistical time series analysis with autoregression"
                elif model_name == 'Random Forest':
                    explanation['method'] = "Ensemble of decision trees with technical indicators"
                elif model_name == 'XGBoost':
                    explanation['method'] = "Gradient boosting with advanced feature engineering"
                
                explanations[model_name] = explanation
        
        except Exception as e:
            st.warning(f"⚠️ Error generating explanations: {str(e)}")
        
        return explanations
    
    def _get_confidence_level(self, r2_score):
        """Convert R² score to confidence level"""
        if r2_score > 0.8:
            return "High"
        elif r2_score > 0.6:
            return "Medium"
        elif r2_score > 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def get_ensemble_prediction(self, predictions):
        """Create ensemble prediction from multiple models"""
        if not predictions:
            return None
        
        # Weight models by their R² scores
        weights = {}
        total_weight = 0
        
        for model_name in predictions.keys():
            if model_name in self.model_performance:
                r2 = max(0, self.model_performance[model_name]['R2'])  # Ensure non-negative
                weights[model_name] = r2
                total_weight += r2
        
        if total_weight == 0:
            # Equal weights if no performance data
            weights = {model: 1/len(predictions) for model in predictions.keys()}
            total_weight = 1
        else:
            # Normalize weights
            weights = {model: weight/total_weight for model, weight in weights.items()}
        
        # Calculate weighted average
        ensemble_pred = 0
        for model_name, pred in predictions.items():
            if isinstance(pred, (list, np.ndarray)):
                pred_value = pred[-1] if len(pred) > 0 else 0
            else:
                pred_value = pred
            ensemble_pred += pred_value * weights.get(model_name, 0)
        
        return ensemble_pred
    
    def get_prediction_ranges(self, predictions):
        """Calculate prediction ranges and uncertainty"""
        if not predictions:
            return None
        
        pred_values = []
        for pred in predictions.values():
            if isinstance(pred, (list, np.ndarray)):
                pred_values.append(pred[-1] if len(pred) > 0 else 0)
            else:
                pred_values.append(pred)
        
        if not pred_values:
            return None
        
        return {
            'min': min(pred_values),
            'max': max(pred_values),
            'mean': np.mean(pred_values),
            'std': np.std(pred_values),
            'range_pct': ((max(pred_values) - min(pred_values)) / np.mean(pred_values)) * 100
        }
