import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_comprehensive_dashboard(data, symbol, predictions=None, current_price=None):
    """
    Create comprehensive trading dashboard with all key metrics
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'{symbol} - Price Chart', 'Volume Analysis',
            'RSI & Stochastic', 'MACD Analysis', 
            'Bollinger Bands', 'Prediction Summary'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1. Price Chart (Top Left)
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA20',
                line=dict(color='#ff9800', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. Volume Analysis (Top Right)
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            opacity=0.7,
            name='Volume',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. RSI & Stochastic (Middle Left)
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#9c27b0', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
    
    # 4. MACD Analysis (Middle Right)
    if 'MACD' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='#2196f3', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        if 'MACD_Signal' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='#ff5722', width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # MACD zero line
        fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=2)
    
    # 5. Bollinger Bands (Bottom Left)
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2),
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='#e0e0e0', width=1),
                showlegend=False
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='#e0e0e0', width=1),
                fill='tonexty',
                fillcolor='rgba(224, 224, 224, 0.1)',
                showlegend=False
            ),
            row=3, col=1
        )
    
    # 6. Prediction Summary (Bottom Right)
    if predictions and current_price:
        targets = [pred.get('target', current_price) for pred in predictions]
        changes = [((target - current_price) / current_price) * 100 for target in targets]
        
        # Create histogram of predictions
        fig.add_trace(
            go.Histogram(
                x=changes,
                nbinsx=8,
                marker_color='#17a2b8',
                opacity=0.7,
                name='Predictions',
                showlegend=False
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Trading Dashboard',
        height=1000,
        template='plotly_white',
        showlegend=False
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=2)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=2, col=2)
    fig.update_yaxes(title_text="Price ($)", row=3, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=2)
    
    return fig

def create_performance_summary_chart(data, symbol):
    """
    Create performance summary with key metrics
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Performance', 'Volume Trend', 'Volatility', 'Returns Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Price Performance
    normalized_price = (data['Close'] / data['Close'].iloc[0]) * 100
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=normalized_price,
            mode='lines',
            name='Performance',
            line=dict(color='#1f77b4', width=2),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # 2. Volume Trend
    volume_ma = data['Volume'].rolling(10).mean()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=volume_ma,
            mode='lines',
            name='Volume Trend',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=2
    )
    
    # 3. Volatility (Rolling Standard Deviation)
    volatility = data['Close'].pct_change().rolling(10).std() * 100
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=volatility,
            mode='lines',
            name='Volatility',
            line=dict(color='#d62728', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # 4. Returns Distribution
    daily_returns = data['Close'].pct_change().dropna() * 100
    fig.add_trace(
        go.Histogram(
            x=daily_returns,
            nbinsx=20,
            marker_color='#2ca02c',
            opacity=0.7,
            name='Returns'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f'{symbol} - Performance Summary',
        height=600,
        template='plotly_white',
        showlegend=False
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Performance (%)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=2)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    fig.update_xaxes(title_text="Returns (%)", row=2, col=2)
    
    return fig

def create_risk_analysis_chart(data, symbol):
    """
    Create risk analysis dashboard
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price vs Moving Average', 'Support & Resistance', 'Risk Metrics', 'Trend Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Price vs Moving Average
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#ff7f0e', width=2)
            ),
            row=1, col=1
        )
    
    # 2. Support & Resistance Levels
    recent_high = data['High'].rolling(20).max()
    recent_low = data['Low'].rolling(20).min()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=recent_high,
            mode='lines',
            name='Resistance',
            line=dict(color='#d62728', width=1, dash='dash')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=recent_low,
            mode='lines',
            name='Support',
            line=dict(color='#2ca02c', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(44, 160, 44, 0.1)'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=2
    )
    
    # 3. Risk Metrics
    returns = data['Close'].pct_change().dropna()
    rolling_sharpe = returns.rolling(30).mean() / returns.rolling(30).std()
    
    fig.add_trace(
        go.Scatter(
            x=data.index[30:],
            y=rolling_sharpe.dropna(),
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color='#9467bd', width=2)
        ),
        row=2, col=1
    )
    
    # 4. Trend Analysis
    if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
        trend_signal = data['SMA_20'] - data['SMA_50']
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=trend_signal,
                mode='lines',
                name='Trend Signal',
                line=dict(color='#8c564b', width=2),
                fill='tozeroy'
            ),
            row=2, col=2
        )
        
        # Zero line
        fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=2)
    
    fig.update_layout(
        title=f'{symbol} - Risk Analysis',
        height=600,
        template='plotly_white',
        showlegend=False
    )
    
    return fig