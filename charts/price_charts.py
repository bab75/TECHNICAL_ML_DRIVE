import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_price_chart(data, symbol):
    """
    Create comprehensive price chart with technical indicators
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} - Price & Technical Indicators', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Main price chart (candlestick)
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
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
                name='SMA 20',
                line=dict(color='#ff9800', width=2)
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#9c27b0', width=2)
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands if available
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='#e0e0e0', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='#e0e0e0', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(224, 224, 224, 0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Middle'],
                mode='lines',
                name='BB Middle',
                line=dict(color='#2196f3', width=1)
            ),
            row=1, col=1
        )
    
    # Volume chart
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Complete Price Analysis',
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_simple_price_chart(data, symbol):
    """
    Create simple price line chart
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=3)
        )
    )
    
    fig.update_layout(
        title=f'{symbol} - Price Trend',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        template='plotly_white',
        hovermode='x'
    )
    
    return fig

def create_price_with_predictions(data, symbol, predictions=None, current_price=None):
    """
    Create price chart with prediction overlays
    """
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Current price line
    if current_price:
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="black",
            line_width=3,
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="bottom right"
        )
    
    # Add predictions if provided
    if predictions:
        for pred in predictions:
            target = pred.get('target', 0)
            model_name = pred.get('indicator', pred.get('model', 'Unknown'))
            
            color = '#28a745' if target > current_price else '#dc3545'
            
            fig.add_hline(
                y=target,
                line_dash="dot",
                line_color=color,
                line_width=2,
                annotation_text=f"{model_name}: ${target:.2f}",
                annotation_position="top right"
            )
    
    fig.update_layout(
        title=f'{symbol} - Price with Predictions',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        template='plotly_white',
        hovermode='x'
    )
    
    return fig