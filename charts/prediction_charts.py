import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_prediction_comparison_chart(data, symbol, predictions, current_price):
    """
    Create chart comparing all prediction models
    """
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=data.index[-30:],  # Last 30 days
            y=data['Close'].iloc[-30:],
            mode='lines',
            name='Historical Price',
            line=dict(color='#1f77b4', width=3)
        )
    )
    
    # Current price line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="black",
        line_width=3,
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="bottom right"
    )
    
    # Add prediction targets
    colors = ['#28a745', '#17a2b8', '#ffc107', '#dc3545', '#6f42c1', '#20c997']
    
    for i, pred in enumerate(predictions):
        target = pred.get('target', current_price)
        model_name = pred.get('indicator', pred.get('model', f'Model {i+1}'))
        confidence = pred.get('confidence', 'Medium')
        
        color = colors[i % len(colors)]
        
        # Prediction line
        fig.add_hline(
            y=target,
            line_dash="dot",
            line_color=color,
            line_width=2,
            annotation_text=f"{model_name}: ${target:.2f} ({confidence})",
            annotation_position="top left" if i % 2 == 0 else "top right"
        )
    
    fig.update_layout(
        title=f'{symbol} - Prediction Comparison',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_prediction_confidence_chart(predictions, current_price):
    """
    Create chart showing prediction confidence levels
    """
    if not predictions:
        return None
    
    # Prepare data
    models = []
    targets = []
    changes = []
    confidences = []
    colors = []
    
    for pred in predictions:
        model_name = pred.get('indicator', pred.get('model', 'Unknown'))
        target = pred.get('target', current_price)
        confidence = pred.get('confidence', 'Medium')
        
        change_pct = ((target - current_price) / current_price) * 100
        
        models.append(model_name)
        targets.append(target)
        changes.append(change_pct)
        confidences.append(confidence)
        
        # Color based on direction and confidence
        if confidence == 'High':
            color = '#28a745' if change_pct > 0 else '#dc3545'
        elif confidence == 'Medium':
            color = '#ffc107' if change_pct > 0 else '#fd7e14'
        else:
            color = '#6c757d'
        colors.append(color)
    
    fig = go.Figure()
    
    # Bar chart of percentage changes
    fig.add_trace(
        go.Bar(
            x=models,
            y=changes,
            marker_color=colors,
            text=[f"{change:.1f}%" for change in changes],
            textposition='outside',
            name='Price Change %'
        )
    )
    
    fig.update_layout(
        title='Prediction Comparison - Price Change %',
        xaxis_title='Models',
        yaxis_title='Expected Price Change (%)',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    # Add zero line
    fig.add_hline(y=0, line_color="gray", line_width=1)
    
    return fig

def create_prediction_timeline_chart(data, symbol, predictions, current_price, days_ahead=30):
    """
    Create timeline chart showing prediction ranges
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
    
    # Create future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days_ahead+1, freq='D')[1:]
    
    # Calculate prediction ranges
    if predictions:
        targets = [pred.get('target', current_price) for pred in predictions]
        min_target = min(targets)
        max_target = max(targets)
        avg_target = sum(targets) / len(targets)
        
        # Future price range (cone)
        upper_range = [current_price + (max_target - current_price) * (i/days_ahead) for i in range(days_ahead)]
        lower_range = [current_price + (min_target - current_price) * (i/days_ahead) for i in range(days_ahead)]
        avg_range = [current_price + (avg_target - current_price) * (i/days_ahead) for i in range(days_ahead)]
        
        # Upper range
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=upper_range,
                mode='lines',
                name='Upper Range',
                line=dict(color='rgba(40, 167, 69, 0.3)', width=1),
                showlegend=False
            )
        )
        
        # Lower range with fill
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=lower_range,
                mode='lines',
                name='Prediction Range',
                line=dict(color='rgba(40, 167, 69, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(40, 167, 69, 0.1)'
            )
        )
        
        # Average prediction
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=avg_range,
                mode='lines',
                name='Average Prediction',
                line=dict(color='#28a745', width=3, dash='dot')
            )
        )
    
    fig.update_layout(
        title=f'{symbol} - Price Prediction Timeline',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_prediction_accuracy_chart(predictions):
    """
    Create chart showing model accuracy metrics
    """
    if not predictions:
        return None
    
    # Prepare data
    models = []
    confidence_scores = []
    colors = []
    
    confidence_map = {'High': 3, 'Medium': 2, 'Low': 1}
    color_map = {'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'}
    
    for pred in predictions:
        model_name = pred.get('indicator', pred.get('model', 'Unknown'))
        confidence = pred.get('confidence', 'Medium')
        
        models.append(model_name)
        confidence_scores.append(confidence_map[confidence])
        colors.append(color_map[confidence])
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=models,
            y=confidence_scores,
            marker_color=colors,
            text=[f"{conf}" for conf in [pred.get('confidence', 'Medium') for pred in predictions]],
            textposition='inside',
            name='Confidence Level'
        )
    )
    
    fig.update_layout(
        title='Model Confidence Levels',
        xaxis_title='Models',
        yaxis_title='Confidence Score',
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High']
        ),
        height=350,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_ensemble_prediction_chart(data, symbol, predictions, current_price):
    """
    Create comprehensive ensemble prediction visualization
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Price Timeline with Predictions',
            'Model Confidence Levels',
            'Prediction Distribution',
            'Expected Returns'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Price timeline (top left)
    fig.add_trace(
        go.Scatter(
            x=data.index[-20:],
            y=data['Close'].iloc[-20:],
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # Current price
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="black",
        line_width=2,
        row=1, col=1
    )
    
    # 2. Confidence levels (top right)
    if predictions:
        confidences = [pred.get('confidence', 'Medium') for pred in predictions]
        conf_counts = {conf: confidences.count(conf) for conf in ['High', 'Medium', 'Low']}
        
        fig.add_trace(
            go.Bar(
                x=list(conf_counts.keys()),
                y=list(conf_counts.values()),
                marker_color=['#28a745', '#ffc107', '#dc3545'],
                name='Confidence'
            ),
            row=1, col=2
        )
        
        # 3. Prediction distribution (bottom left)
        targets = [pred.get('target', current_price) for pred in predictions]
        fig.add_trace(
            go.Histogram(
                x=targets,
                nbinsx=10,
                marker_color='#17a2b8',
                name='Target Distribution'
            ),
            row=2, col=1
        )
        
        # 4. Expected returns (bottom right)
        returns = [((target - current_price) / current_price) * 100 for target in targets]
        models = [pred.get('indicator', pred.get('model', f'Model {i}')) for i, pred in enumerate(predictions)]
        
        colors = ['#28a745' if ret > 0 else '#dc3545' for ret in returns]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=returns,
                marker_color=colors,
                name='Expected Returns'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title=f'{symbol} - Comprehensive Prediction Analysis',
        height=700,
        template='plotly_white',
        showlegend=False
    )
    
    return fig