import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ExplanationEngine:
    """
    Advanced explanation engine for technical analysis and ML predictions
    Provides detailed reasoning for why different models predict different outcomes
    """
    
    def __init__(self):
        self.explanations = {}
        self.comparison_data = {}
        
    def explain_prediction_differences(self, predictions, current_price):
        """
        Explain why different models/indicators predict different prices
        """
        if not predictions:
            return {}
        
        explanations = {}
        
        try:
            # Collect all predictions
            all_predictions = {}
            
            # Technical predictions
            if 'technical' in predictions:
                for method, pred in predictions['technical'].items():
                    if isinstance(pred, dict) and 'target_price' in pred:
                        all_predictions[f"TA_{method}"] = pred
            
            # ML predictions  
            if 'ml' in predictions:
                for method, pred in predictions['ml'].items():
                    if isinstance(pred, dict) and 'target_price' in pred:
                        all_predictions[f"ML_{method}"] = pred
            
            if len(all_predictions) < 2:
                return {"message": "Need at least 2 predictions to compare differences"}
            
            # Analyze prediction spread
            targets = [pred['target_price'] for pred in all_predictions.values()]
            price_changes = [pred['price_change'] for pred in all_predictions.values()]
            
            spread_analysis = self._analyze_prediction_spread(targets, current_price)
            methodology_analysis = self._analyze_methodology_differences(all_predictions)
            consensus_analysis = self._analyze_consensus_strength(all_predictions, current_price)
            
            explanations = {
                'spread_analysis': spread_analysis,
                'methodology_differences': methodology_analysis,
                'consensus_strength': consensus_analysis,
                'detailed_comparisons': self._create_detailed_comparisons(all_predictions, current_price)
            }
            
        except Exception as e:
            st.error(f"❌ Error explaining prediction differences: {str(e)}")
            explanations = {"error": str(e)}
        
        return explanations
    
    def _analyze_prediction_spread(self, targets, current_price):
        """Analyze the spread and distribution of predictions"""
        if not targets:
            return {}
        
        min_target = min(targets)
        max_target = max(targets)
        median_target = np.median(targets)
        std_target = np.std(targets)
        
        spread_pct = ((max_target - min_target) / current_price) * 100
        
        analysis = {
            'spread_percentage': spread_pct,
            'range': {'min': min_target, 'max': max_target},
            'central_tendency': median_target,
            'volatility': std_target,
            'interpretation': self._interpret_spread(spread_pct)
        }
        
        return analysis
    
    def _interpret_spread(self, spread_pct):
        """Interpret what the prediction spread means"""
        if spread_pct < 2:
            return {
                'level': 'Low',
                'meaning': 'Strong consensus among models',
                'implication': 'High confidence in direction',
                'risk': 'Low prediction uncertainty'
            }
        elif spread_pct < 5:
            return {
                'level': 'Moderate',
                'meaning': 'Reasonable agreement among models',
                'implication': 'Moderate confidence in direction',
                'risk': 'Some prediction uncertainty'
            }
        elif spread_pct < 10:
            return {
                'level': 'High',
                'meaning': 'Significant disagreement among models',
                'implication': 'Lower confidence in specific targets',
                'risk': 'High prediction uncertainty'
            }
        else:
            return {
                'level': 'Very High',
                'meaning': 'Major disagreement among models',
                'implication': 'Very low confidence in predictions',
                'risk': 'Extreme prediction uncertainty'
            }
    
    def _analyze_methodology_differences(self, predictions):
        """Analyze differences based on methodology types"""
        technical_preds = {k: v for k, v in predictions.items() if k.startswith('TA_')}
        ml_preds = {k: v for k, v in predictions.items() if k.startswith('ML_')}
        
        analysis = {}
        
        if technical_preds and ml_preds:
            # Compare technical vs ML approaches
            ta_targets = [pred['target_price'] for pred in technical_preds.values()]
            ml_targets = [pred['target_price'] for pred in ml_preds.values()]
            
            ta_median = np.median(ta_targets)
            ml_median = np.median(ml_targets)
            
            analysis['ta_vs_ml'] = {
                'ta_median': ta_median,
                'ml_median': ml_median,
                'difference': abs(ta_median - ml_median),
                'explanation': self._explain_ta_vs_ml_difference(ta_median, ml_median)
            }
        
        # Analyze within technical indicators
        if len(technical_preds) > 1:
            analysis['technical_consensus'] = self._analyze_technical_consensus(technical_preds)
        
        # Analyze within ML models
        if len(ml_preds) > 1:
            analysis['ml_consensus'] = self._analyze_ml_consensus(ml_preds)
        
        return analysis
    
    def _explain_ta_vs_ml_difference(self, ta_median, ml_median):
        """Explain differences between technical analysis and ML predictions"""
        diff_pct = abs(ta_median - ml_median) / min(ta_median, ml_median) * 100
        
        if diff_pct < 1:
            return {
                'agreement': 'Strong',
                'reason': 'Both technical and ML models see similar patterns',
                'implication': 'High confidence in prediction direction'
            }
        elif diff_pct < 3:
            return {
                'agreement': 'Moderate',
                'reason': 'Technical and ML models have slight differences in interpretation',
                'implication': 'Good confidence but some uncertainty in magnitude'
            }
        else:
            if ta_median > ml_median:
                return {
                    'agreement': 'Disagreement',
                    'reason': 'Technical indicators more bullish than ML models',
                    'implication': 'Technical patterns suggest stronger move than historical data patterns'
                }
            else:
                return {
                    'agreement': 'Disagreement', 
                    'reason': 'ML models more bullish than technical indicators',
                    'implication': 'Historical patterns suggest stronger move than current technical setup'
                }
    
    def _analyze_technical_consensus(self, technical_preds):
        """Analyze consensus among technical indicators"""
        momentum_indicators = ['RSI', 'MACD', 'Stochastic']
        trend_indicators = ['Moving_Averages', 'ADX']
        volatility_indicators = ['Bollinger_Bands']
        
        consensus = {}
        
        # Group by indicator type
        for indicator_type, indicators in [
            ('momentum', momentum_indicators),
            ('trend', trend_indicators), 
            ('volatility', volatility_indicators)
        ]:
            type_preds = {k: v for k, v in technical_preds.items() 
                         if any(ind in k for ind in indicators)}
            
            if type_preds:
                targets = [pred['target_price'] for pred in type_preds.values()]
                consensus[indicator_type] = {
                    'median_target': np.median(targets),
                    'agreement_level': self._calculate_agreement_level(targets),
                    'count': len(type_preds)
                }
        
        return consensus
    
    def _analyze_ml_consensus(self, ml_preds):
        """Analyze consensus among ML models"""
        model_types = {
            'tree_based': ['Random Forest', 'XGBoost'],
            'neural_network': ['LSTM', 'GRU'],
            'statistical': ['ARIMA', 'Prophet']
        }
        
        consensus = {}
        
        for model_type, models in model_types.items():
            type_preds = {k: v for k, v in ml_preds.items() 
                         if any(model in k for model in models)}
            
            if type_preds:
                targets = [pred['target_price'] for pred in type_preds.values()]
                consensus[model_type] = {
                    'median_target': np.median(targets),
                    'agreement_level': self._calculate_agreement_level(targets),
                    'count': len(type_preds)
                }
        
        return consensus
    
    def _calculate_agreement_level(self, targets):
        """Calculate how much predictions agree"""
        if len(targets) < 2:
            return 'N/A'
        
        std_pct = (np.std(targets) / np.mean(targets)) * 100
        
        if std_pct < 1:
            return 'Very High'
        elif std_pct < 2:
            return 'High'
        elif std_pct < 5:
            return 'Moderate'
        else:
            return 'Low'
    
    def _analyze_consensus_strength(self, predictions, current_price):
        """Analyze overall consensus strength"""
        targets = [pred['target_price'] for pred in predictions.values()]
        confidences = [pred.get('confidence', 'Unknown') for pred in predictions.values()]
        
        # Direction consensus
        bullish = sum(1 for target in targets if target > current_price * 1.005)
        bearish = sum(1 for target in targets if target < current_price * 0.995)
        neutral = len(targets) - bullish - bearish
        
        total = len(targets)
        direction_consensus = max(bullish, bearish, neutral) / total
        
        # Confidence consensus
        high_conf = confidences.count('High')
        medium_conf = confidences.count('Medium')
        low_conf = confidences.count('Low')
        
        consensus_strength = {
            'direction': {
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral,
                'consensus_pct': direction_consensus * 100,
                'dominant_view': max([('Bullish', bullish), ('Bearish', bearish), ('Neutral', neutral)], key=lambda x: x[1])[0]
            },
            'confidence': {
                'high': high_conf,
                'medium': medium_conf,
                'low': low_conf,
                'avg_confidence': self._calculate_avg_confidence(confidences)
            },
            'overall_strength': self._calculate_overall_strength(direction_consensus, confidences)
        }
        
        return consensus_strength
    
    def _calculate_avg_confidence(self, confidences):
        """Calculate average confidence level"""
        conf_scores = {'High': 3, 'Medium': 2, 'Low': 1, 'Unknown': 0}
        scores = [conf_scores.get(conf, 0) for conf in confidences]
        avg_score = np.mean(scores) if scores else 0
        
        if avg_score >= 2.5:
            return 'High'
        elif avg_score >= 1.5:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_overall_strength(self, direction_consensus, confidences):
        """Calculate overall prediction strength"""
        conf_score = self._calculate_avg_confidence(confidences)
        conf_weights = {'High': 1.0, 'Medium': 0.7, 'Low': 0.3}
        
        strength_score = direction_consensus * conf_weights.get(conf_score, 0.1)
        
        if strength_score >= 0.8:
            return 'Very Strong'
        elif strength_score >= 0.6:
            return 'Strong'
        elif strength_score >= 0.4:
            return 'Moderate'
        else:
            return 'Weak'
    
    def _create_detailed_comparisons(self, predictions, current_price):
        """Create detailed pairwise comparisons"""
        comparisons = []
        
        pred_list = list(predictions.items())
        
        for i in range(len(pred_list)):
            for j in range(i + 1, len(pred_list)):
                name1, pred1 = pred_list[i]
                name2, pred2 = pred_list[j]
                
                comparison = self._compare_two_predictions(name1, pred1, name2, pred2, current_price)
                comparisons.append(comparison)
        
        return comparisons
    
    def _compare_two_predictions(self, name1, pred1, name2, pred2, current_price):
        """Compare two specific predictions"""
        target1 = pred1['target_price']
        target2 = pred2['target_price']
        
        diff_abs = abs(target1 - target2)
        diff_pct = (diff_abs / current_price) * 100
        
        comparison = {
            'methods': f"{name1} vs {name2}",
            'targets': [target1, target2],
            'difference_abs': diff_abs,
            'difference_pct': diff_pct,
            'timeframes': [pred1.get('timeframe', 'Unknown'), pred2.get('timeframe', 'Unknown')],
            'confidences': [pred1.get('confidence', 'Unknown'), pred2.get('confidence', 'Unknown')],
            'explanation': self._explain_specific_difference(name1, pred1, name2, pred2)
        }
        
        return comparison
    
    def _explain_specific_difference(self, name1, pred1, name2, pred2):
        """Explain difference between two specific predictions"""
        reasoning1 = pred1.get('reasoning', '')
        reasoning2 = pred2.get('reasoning', '')
        
        # Determine which is more bullish
        if pred1['target_price'] > pred2['target_price']:
            more_bullish = name1
            less_bullish = name2
            bull_reasoning = reasoning1
            bear_reasoning = reasoning2
        else:
            more_bullish = name2
            less_bullish = name1
            bull_reasoning = reasoning2
            bear_reasoning = reasoning1
        
        explanation = f"{more_bullish} is more bullish than {less_bullish}. "
        
        # Add methodology-specific explanations
        if 'TA_' in more_bullish and 'ML_' in less_bullish:
            explanation += "Technical analysis suggests stronger patterns than historical ML models indicate. "
        elif 'ML_' in more_bullish and 'TA_' in less_bullish:
            explanation += "ML models detect stronger historical patterns than current technical indicators show. "
        
        # Add timeframe considerations
        tf1 = pred1.get('timeframe', '')
        tf2 = pred2.get('timeframe', '')
        if tf1 != tf2 and 'day' in tf1 and 'day' in tf2:
            explanation += f"Different timeframes may explain variance ({tf1} vs {tf2}). "
        
        return explanation
    
    def generate_prediction_narrative(self, predictions, symbol, current_price):
        """Generate a comprehensive narrative explaining all predictions"""
        if not predictions:
            return "No predictions available to analyze."
        
        try:
            narrative = []
            
            # Summary overview
            summary = predictions.get('summary', {})
            if summary:
                consensus_target = summary.get('consensus_target', current_price)
                sentiment = summary.get('sentiment', 'Neutral')
                confidence = summary.get('confidence', 'Unknown')
                
                narrative.append(f"📊 **Overall Outlook for {symbol}:**")
                narrative.append(f"The consensus view is {sentiment.lower()} with {confidence.lower()} confidence, targeting ${consensus_target:.2f}.")
                narrative.append("")
            
            # Technical analysis narrative
            technical = predictions.get('technical', {})
            if technical:
                narrative.append("🔧 **Technical Analysis Perspective:**")
                
                strong_signals = []
                weak_signals = []
                
                for method, pred in technical.items():
                    if isinstance(pred, dict):
                        conf = pred.get('confidence', 'Low')
                        if conf == 'High':
                            strong_signals.append((method, pred))
                        else:
                            weak_signals.append((method, pred))
                
                if strong_signals:
                    narrative.append("Strong technical signals:")
                    for method, pred in strong_signals:
                        change = pred.get('price_change', 0)
                        reasoning = pred.get('reasoning', '')
                        narrative.append(f"• {method}: {change:+.1f}% - {reasoning}")
                
                if weak_signals:
                    narrative.append("Supporting technical indicators:")
                    for method, pred in weak_signals[:3]:  # Limit to top 3
                        change = pred.get('price_change', 0)
                        narrative.append(f"• {method}: {change:+.1f}%")
                
                narrative.append("")
            
            # ML analysis narrative
            ml = predictions.get('ml', {})
            if ml:
                narrative.append("🤖 **Machine Learning Analysis:**")
                
                best_ml = max(ml.items(), key=lambda x: self._get_confidence_score(x[1].get('confidence', 'Low')))
                if best_ml:
                    method, pred = best_ml
                    change = pred.get('price_change', 0)
                    accuracy = pred.get('accuracy', 'Unknown')
                    narrative.append(f"Best performing model: {method} ({accuracy})")
                    narrative.append(f"Prediction: {change:+.1f}% - {pred.get('reasoning', '')}")
                
                # Add model agreement analysis
                ml_targets = [pred['target_price'] for pred in ml.values() if isinstance(pred, dict) and 'target_price' in pred]
                if len(ml_targets) > 1:
                    agreement = self._calculate_agreement_level(ml_targets)
                    narrative.append(f"ML model agreement level: {agreement}")
                
                narrative.append("")
            
            # Risk and uncertainty
            explanations = self.explain_prediction_differences(predictions, current_price)
            if 'spread_analysis' in explanations:
                spread = explanations['spread_analysis']
                interpretation = spread.get('interpretation', {})
                
                narrative.append("⚠️ **Risk Assessment:**")
                narrative.append(f"Prediction uncertainty: {interpretation.get('level', 'Unknown')}")
                narrative.append(f"Risk level: {interpretation.get('risk', 'Unknown')}")
                narrative.append(f"Implication: {interpretation.get('implication', 'Unknown')}")
                narrative.append("")
            
            # Actionable insights
            narrative.append("💡 **Key Takeaways:**")
            
            if summary:
                agreement = summary.get('agreement_level', '0/0')
                narrative.append(f"• Consensus strength: {agreement} models in agreement")
                
                tech_count = summary.get('technical_count', 0)
                ml_count = summary.get('ml_count', 0)
                narrative.append(f"• Analysis depth: {tech_count} technical indicators + {ml_count} ML models")
            
            # Add specific recommendations based on confidence and consensus
            if summary.get('confidence') == 'High':
                narrative.append("• High confidence prediction - consider position sizing accordingly")
            elif summary.get('confidence') == 'Low':
                narrative.append("• Low confidence - wait for clearer signals or use smaller position sizes")
            
            return "\n".join(narrative)
            
        except Exception as e:
            return f"Error generating narrative: {str(e)}"
    
    def _get_confidence_score(self, confidence):
        """Convert confidence to numeric score"""
        scores = {'High': 3, 'Medium': 2, 'Low': 1, 'Unknown': 0}
        return scores.get(confidence, 0)
    
    def create_prediction_visualization(self, predictions, current_price, symbol):
        """Create comprehensive prediction visualization"""
        try:
            # Collect all predictions
            all_preds = []
            
            if 'technical' in predictions:
                for method, pred in predictions['technical'].items():
                    if isinstance(pred, dict) and 'target_price' in pred:
                        all_preds.append({
                            'Method': f"TA: {method}",
                            'Target': pred['target_price'],
                            'Change_%': pred.get('price_change', 0),
                            'Confidence': pred.get('confidence', 'Unknown'),
                            'Type': 'Technical'
                        })
            
            if 'ml' in predictions:
                for method, pred in predictions['ml'].items():
                    if isinstance(pred, dict) and 'target_price' in pred:
                        all_preds.append({
                            'Method': f"ML: {method}",
                            'Target': pred['target_price'],
                            'Change_%': pred.get('price_change', 0),
                            'Confidence': pred.get('confidence', 'Unknown'),
                            'Type': 'ML'
                        })
            
            if not all_preds:
                return None
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Price Target Distribution',
                    'Prediction Confidence Levels',
                    'Price Change Distribution', 
                    'Method Performance Comparison'
                ),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "bar"}]]
            )
            
            # Plot 1: Price targets scatter
            colors = {'Technical': 'blue', 'ML': 'red'}
            for pred_type in ['Technical', 'ML']:
                type_preds = [p for p in all_preds if p['Type'] == pred_type]
                if type_preds:
                    fig.add_trace(
                        go.Scatter(
                            x=[p['Method'].split(': ')[1] for p in type_preds],
                            y=[p['Target'] for p in type_preds],
                            mode='markers',
                            name=pred_type,
                            marker=dict(
                                color=colors[pred_type],
                                size=[12 if p['Confidence'] == 'High' else 8 if p['Confidence'] == 'Medium' else 6 for p in type_preds]
                            )
                        ),
                        row=1, col=1
                    )
            
            # Add current price line
            fig.add_hline(y=current_price, line_dash="dash", line_color="black", 
                         annotation_text=f"Current: ${current_price:.2f}", row=1, col=1)
            
            # Plot 2: Confidence distribution
            conf_counts = {}
            for pred in all_preds:
                conf = pred['Confidence']
                conf_counts[conf] = conf_counts.get(conf, 0) + 1
            
            fig.add_trace(
                go.Bar(
                    x=list(conf_counts.keys()),
                    y=list(conf_counts.values()),
                    name='Confidence Distribution',
                    marker_color=['green' if c == 'High' else 'orange' if c == 'Medium' else 'red' for c in conf_counts.keys()]
                ),
                row=1, col=2
            )
            
            # Plot 3: Price change histogram
            changes = [p['Change_%'] for p in all_preds]
            fig.add_trace(
                go.Histogram(
                    x=changes,
                    name='Price Change Distribution',
                    nbinsx=10,
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            # Plot 4: Method comparison
            methods = [p['Method'] for p in all_preds]
            targets = [p['Target'] for p in all_preds]
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=targets,
                    name='Target Prices by Method',
                    marker_color=['blue' if 'TA:' in m else 'red' for m in methods]
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f"Comprehensive Prediction Analysis - {symbol}",
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating visualization: {str(e)}")
            return None
    
    def export_explanation_report(self, predictions, symbol, current_price):
        """Export comprehensive explanation report"""
        try:
            report = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'current_price': current_price,
                'narrative': self.generate_prediction_narrative(predictions, symbol, current_price),
                'differences_analysis': self.explain_prediction_differences(predictions, current_price),
                'prediction_summary': predictions.get('summary', {}),
                'technical_count': len(predictions.get('technical', {})),
                'ml_count': len(predictions.get('ml', {}))
            }
            
            return report
            
        except Exception as e:
            st.error(f"❌ Error exporting report: {str(e)}")
            return {}
