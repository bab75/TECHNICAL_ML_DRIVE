import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_rsi_chart(data):
    """
    Create RSI chart with overbought/oversold zones
    """
    if 'RSI' not in data.columns:
        return None
    
    fig = go.Figure()
    
    # RSI line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # Overbought zone (70-100)
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor='rgba(220, 53, 69, 0.2)',
        layer="below",
        line_width=0
    )
    
    # Oversold zone (0-30)
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor='rgba(40, 167, 69, 0.2)',
        layer="below",
        line_width=0
    )
    
    # Reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, annotation_text="Overbought (70)")
    fig.add_hline(y=50, line_dash="solid", line_color="gray", line_width=1, annotation_text="Midline (50)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, annotation_text="Oversold (30)")
    
    fig.update_layout(
        title='RSI (Relative Strength Index)',
        xaxis_title='Date',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100]),
        height=300,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_macd_chart(data):
    """
    Create MACD chart with signal line and histogram
    """
    if not all(col in data.columns for col in ['MACD', 'MACD_Signal']):
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('MACD Line & Signal', 'MACD Histogram'),
        row_heights=[0.7, 0.3]
    )
    
    # MACD line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # Signal line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD_Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=1
    )
    
    # MACD Histogram
    if 'MACD_Histogram' in data.columns:
        colors = ['#26a69a' if val >= 0 else '#ef5350' for val in data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Zero line
    fig.add_hline(y=0, line_color="gray", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)
    
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)',
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=1, col=1)
    fig.update_yaxes(title_text="Histogram", row=2, col=1)
    
    return fig

def create_stochastic_chart(data):
    """
    Create Stochastic Oscillator chart
    """
    if not all(col in data.columns for col in ['Stoch_K', 'Stoch_D']):
        return None
    
    fig = go.Figure()
    
    # %K line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Stoch_K'],
            mode='lines',
            name='%K',
            line=dict(color='#1f77b4', width=2)
        )
    )
    
    # %D line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Stoch_D'],
            mode='lines',
            name='%D',
            line=dict(color='#ff7f0e', width=2)
        )
    )
    
    # Overbought/Oversold zones
    fig.add_hrect(y0=80, y1=100, fillcolor='rgba(220, 53, 69, 0.2)', layer="below", line_width=0)
    fig.add_hrect(y0=0, y1=20, fillcolor='rgba(40, 167, 69, 0.2)', layer="below", line_width=0)
    
    # Reference lines
    fig.add_hline(y=80, line_dash="dash", line_color="red", line_width=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", line_width=1)
    
    fig.update_layout(
        title='Stochastic Oscillator',
        xaxis_title='Date',
        yaxis_title='Stochastic (%)',
        yaxis=dict(range=[0, 100]),
        height=300,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_volume_analysis_chart(data):
    """
    Create volume analysis chart with moving average
    """
    fig = go.Figure()
    
    # Volume bars
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        )
    )
    
    # Volume moving average if available
    if 'Volume_SMA' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Volume_SMA'],
                mode='lines',
                name='Volume MA',
                line=dict(color='#ff9800', width=2)
            )
        )
    else:
        # Calculate simple volume MA
        volume_ma = data['Volume'].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=volume_ma,
                mode='lines',
                name='Volume MA (20)',
                line=dict(color='#ff9800', width=2)
            )
        )
    
    fig.update_layout(
        title='Volume Analysis',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=300,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_multi_indicator_dashboard(data):
    """
    Create comprehensive technical indicators dashboard
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Volume'),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price with moving averages
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close',
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
                line=dict(color='#ff9800', width=1)
            ),
            row=1, col=1
        )
    
    # RSI
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
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
    
    # MACD
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
            row=3, col=1
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
                row=3, col=1
            )
    
    # Volume
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(data['Close'], data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7,
            showlegend=False
        ),
        row=4, col=1
    )
    
    fig.update_layout(
        title='Technical Analysis Dashboard',
        height=800,
        template='plotly_white',
        showlegend=True
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    return fig