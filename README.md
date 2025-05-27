# 📈 Advanced Technical Analysis Platform

A comprehensive Streamlit-based stock market analysis platform featuring advanced technical indicators, machine learning predictions, and real-time data visualization.

## 🚀 Features

### 📊 Technical Analysis
- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, ADX, Stochastic, and more
- **Real-time Price Data**: Live stock data from Yahoo Finance API
- **Multiple Time Periods**: 5 days, 15 days, 1 month, 3 months, 6 months, 1 year
- **Interactive Charts**: Plotly-powered visualizations with zoom and pan

### 🤖 Advanced ML Predictions
- **ARIMA Time Series**: Trend-based forecasting
- **Prophet Seasonal**: Facebook's Prophet model for seasonality
- **Random Forest**: Ensemble machine learning predictions
- **LSTM Neural Networks**: Deep learning sequence modeling
- **XGBoost**: Gradient boosting predictions
- **Ensemble Forecasting**: Combines multiple models with confidence scoring

### 📈 Visual Features
- **Color-coded Predictions**: Clear visual distinction between bullish/bearish signals
- **Arrow Indicators**: Visual signals for RSI and MACD
- **Price Target Analysis**: Consensus targets with confidence levels
- **Interactive Configuration**: Customize all indicators and ML models
- **Data Export**: Download raw data as CSV

### 🎯 Prediction System
- **Multiple Signal Types**: Technical analysis + ML predictions
- **Confidence Scoring**: High/Medium/Low confidence ratings
- **Timeframe Analysis**: Short-term (3-7 days) to long-term (30+ days)
- **Risk Assessment**: Comprehensive analysis of prediction reliability

## 🛠️ Installation

### For Local Development
```bash
git clone https://github.com/yourusername/technical-analysis-platform.git
cd technical-analysis-platform
pip install -r packages.txt
streamlit run app_simple.py
```

### For Streamlit Cloud Deployment
1. Fork this repository to your GitHub account
2. Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io)
3. Deploy the app by selecting your repository
4. Set the main file path to `app_simple.py`

## 📁 Project Structure

```
technical-analysis-platform/
├── app_simple.py              # Main application (enhanced version)
├── app.py                     # Alternative main file
├── packages.txt               # Python dependencies
├── README.md                  # This file
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── pages/
│   ├── 1_📊_Analysis.py      # Detailed analysis page
│   ├── 2_⚙️_Configuration.py  # Settings and configuration
│   └── 3_🔮_Predictions.py   # Advanced ML predictions
└── utils/
    ├── config_manager.py     # Configuration management
    ├── data_manager.py       # Data fetching and caching
    ├── technical_indicators.py # Technical analysis calculations
    ├── ml_models.py          # Machine learning models
    ├── prediction_engine.py  # Prediction logic
    └── explanation_engine.py # Prediction explanations
```

## 🎮 Usage

### Quick Start
1. **Launch the app**: Open the main page
2. **Enter stock symbol**: Type any valid ticker (e.g., AAPL, GOOGL, TSLA)
3. **Select time period**: Choose from 5 days to 1 year
4. **Click ANALYZE**: Get comprehensive analysis with predictions

### Advanced Features
- **Configuration Page**: Customize indicator parameters and ML model settings
- **ML Predictions Page**: Access advanced forecasting models
- **Data Export**: Download analysis data for further research

### Supported Stocks
- All stocks available on Yahoo Finance
- Major indices (SPY, QQQ, DIA)
- International markets
- Cryptocurrencies (BTC-USD, ETH-USD)

## 🔧 Configuration

### Technical Indicators
- **RSI Period**: Default 14 days
- **MACD Parameters**: Fast (12), Slow (26), Signal (9)
- **Bollinger Bands**: 20-period, 2 standard deviations
- **Moving Averages**: SMA and EMA with customizable periods

### ML Models
- **ARIMA**: Configurable order parameters
- **Prophet**: Seasonal analysis settings
- **Random Forest**: Tree count and depth settings
- **LSTM**: Sequence length and layers
- **XGBoost**: Boosting parameters

### Data Management
- **Auto Refresh**: Automatic data updates
- **Cache Duration**: Configurable cache time (5-60 minutes)
- **API Rate Limiting**: Intelligent request management

## 📊 Prediction Models

### Technical Analysis Models
1. **RSI Mean Reversion**: Identifies overbought/oversold conditions
2. **MACD Momentum**: Trend change predictions
3. **Bollinger Bands**: Mean reversion analysis
4. **Moving Average Crossovers**: Trend direction signals
5. **Volume Analysis**: Volume-confirmed price movements
6. **Support/Resistance**: Key price level analysis

### Machine Learning Models
1. **ARIMA Time Series**: Statistical forecasting
2. **Prophet Forecast**: Seasonal and trend analysis
3. **Random Forest**: Ensemble tree-based predictions
4. **LSTM Neural Net**: Deep learning sequences
5. **XGBoost**: Gradient boosting ensemble

## 🎨 Visual Features

### Color Coding
- **Current Price**: Bold black text for readability
- **Target Prices**: Bright red for clear visibility
- **Bullish Signals**: Green backgrounds and borders
- **Bearish Signals**: Red backgrounds and borders
- **Neutral Signals**: Yellow/amber backgrounds

### Interactive Elements
- **Zoom and Pan**: Full chart interactivity
- **Hover Details**: Detailed information on hover
- **Responsive Design**: Works on all screen sizes
- **Export Options**: PNG, SVG, HTML downloads

## 🚀 Deployment

### Streamlit Cloud
1. Connect GitHub repository
2. Set main file: `app_simple.py`
3. Configure secrets (if needed)
4. Deploy automatically

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### Docker
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r packages.txt
EXPOSE 8501
CMD ["streamlit", "run", "app_simple.py"]
```

## 🔐 Security

- **No API Keys Required**: Uses free Yahoo Finance data
- **Data Privacy**: No personal data stored
- **Secure Connections**: HTTPS-only data fetching
- **Rate Limiting**: Prevents API abuse

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: GitHub Issues page
- **Documentation**: This README
- **Community**: Streamlit Community Forum

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing free stock data
- **Streamlit**: For the amazing web framework
- **Plotly**: For interactive visualizations
- **scikit-learn**: For machine learning capabilities
- **Prophet**: For time series forecasting

---

**⭐ Star this repository if you find it useful!**

Built with ❤️ using Streamlit, Python, and modern data science libraries.