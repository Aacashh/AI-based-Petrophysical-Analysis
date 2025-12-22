"""
Visualization Components for AI/ML Demo

Provides architecture diagrams, animated charts, and comparison plots
for the AI/ML Laboratory demo page.

Uses Plotly for interactive visualizations and custom HTML/CSS for diagrams.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# =============================================================================
# COLOR SCHEME (matching app theme)
# =============================================================================

COLORS = {
    'primary': '#00D4AA',
    'secondary': '#2d3748',
    'background': '#1a1d24',
    'text': '#ffffff',
    'text_muted': '#8892a0',
    'success': '#00D4AA',
    'warning': '#FFB800',
    'error': '#FF4444',
    'shallow': '#00AA00',
    'deep': '#CC0000',
    'corrected': '#0066CC',
    'anomaly': '#FF6600',
    'grid': '#3d4852'
}


# =============================================================================
# ARCHITECTURE DIAGRAMS
# =============================================================================

def render_lstm_architecture() -> str:
    """
    Generate HTML/SVG for LSTM architecture visualization.
    
    Returns:
        HTML string with animated LSTM diagram
    """
    return '''
    <div style="background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%); 
                padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h4 style="color: #00D4AA; margin-bottom: 15px; text-align: center;">
            LSTM Multi-Curve Alignment Architecture
        </h4>
        <svg viewBox="0 0 800 400" style="width: 100%; max-width: 800px; margin: 0 auto; display: block;">
            <!-- Background -->
            <rect width="800" height="400" fill="transparent"/>
            
            <!-- Input Logs -->
            <g transform="translate(50, 50)">
                <rect x="0" y="0" width="80" height="120" rx="5" fill="#2d3748" stroke="#00D4AA" stroke-width="2"/>
                <text x="40" y="-10" fill="#8892a0" text-anchor="middle" font-size="12">Shallow Log</text>
                <text x="40" y="60" fill="#00AA00" text-anchor="middle" font-size="11">GR, RHOB</text>
                <text x="40" y="80" fill="#00AA00" text-anchor="middle" font-size="11">NPHI, RT</text>
            </g>
            
            <g transform="translate(50, 230)">
                <rect x="0" y="0" width="80" height="120" rx="5" fill="#2d3748" stroke="#00D4AA" stroke-width="2"/>
                <text x="40" y="-10" fill="#8892a0" text-anchor="middle" font-size="12">Deep Log</text>
                <text x="40" y="60" fill="#CC0000" text-anchor="middle" font-size="11">GR, RHOB</text>
                <text x="40" y="80" fill="#CC0000" text-anchor="middle" font-size="11">NPHI, RT</text>
            </g>
            
            <!-- Arrows to LSTM -->
            <path d="M 135 110 L 200 110 L 200 150 L 250 150" stroke="#00D4AA" stroke-width="2" fill="none" marker-end="url(#arrowhead)">
                <animate attributeName="stroke-dashoffset" from="100" to="0" dur="2s" repeatCount="indefinite"/>
            </path>
            <path d="M 135 290 L 200 290 L 200 250 L 250 250" stroke="#00D4AA" stroke-width="2" fill="none">
                <animate attributeName="stroke-dashoffset" from="100" to="0" dur="2s" repeatCount="indefinite"/>
            </path>
            
            <!-- LSTM Encoders -->
            <g transform="translate(250, 120)">
                <rect x="0" y="0" width="120" height="60" rx="8" fill="#1a5a4a" stroke="#00D4AA" stroke-width="2"/>
                <text x="60" y="25" fill="#ffffff" text-anchor="middle" font-size="11" font-weight="bold">Bi-LSTM</text>
                <text x="60" y="42" fill="#8892a0" text-anchor="middle" font-size="10">Encoder</text>
            </g>
            
            <g transform="translate(250, 220)">
                <rect x="0" y="0" width="120" height="60" rx="8" fill="#1a5a4a" stroke="#00D4AA" stroke-width="2"/>
                <text x="60" y="25" fill="#ffffff" text-anchor="middle" font-size="11" font-weight="bold">Bi-LSTM</text>
                <text x="60" y="42" fill="#8892a0" text-anchor="middle" font-size="10">Encoder</text>
            </g>
            
            <!-- Shared Weights Indicator -->
            <text x="310" y="195" fill="#FFB800" text-anchor="middle" font-size="10">Shared Weights</text>
            <path d="M 280 180 L 280 195 L 340 195 L 340 210" stroke="#FFB800" stroke-width="1" stroke-dasharray="4,2" fill="none"/>
            
            <!-- Arrows to Attention -->
            <path d="M 375 150 L 430 150 L 430 180 L 460 180" stroke="#00D4AA" stroke-width="2" fill="none"/>
            <path d="M 375 250 L 430 250 L 430 220 L 460 220" stroke="#00D4AA" stroke-width="2" fill="none"/>
            
            <!-- Attention Layer -->
            <g transform="translate(460, 160)">
                <rect x="0" y="0" width="100" height="80" rx="8" fill="#4a3a1a" stroke="#FFB800" stroke-width="2"/>
                <text x="50" y="30" fill="#ffffff" text-anchor="middle" font-size="11" font-weight="bold">Self</text>
                <text x="50" y="48" fill="#ffffff" text-anchor="middle" font-size="11" font-weight="bold">Attention</text>
                <text x="50" y="68" fill="#8892a0" text-anchor="middle" font-size="9">Weight curves</text>
            </g>
            
            <!-- Arrow to Fusion -->
            <path d="M 565 200 L 620 200" stroke="#00D4AA" stroke-width="2" fill="none"/>
            
            <!-- Fusion Layer -->
            <g transform="translate(620, 165)">
                <rect x="0" y="0" width="80" height="70" rx="8" fill="#2d3748" stroke="#00D4AA" stroke-width="2"/>
                <text x="40" y="28" fill="#ffffff" text-anchor="middle" font-size="11" font-weight="bold">Fusion</text>
                <text x="40" y="48" fill="#8892a0" text-anchor="middle" font-size="10">Concat+Dense</text>
            </g>
            
            <!-- Arrow to Output -->
            <path d="M 705 200 L 750 200" stroke="#00D4AA" stroke-width="2" fill="none"/>
            
            <!-- Output -->
            <g transform="translate(750, 165)">
                <circle cx="25" cy="35" r="30" fill="#00D4AA" opacity="0.2" stroke="#00D4AA" stroke-width="2"/>
                <text x="25" y="30" fill="#ffffff" text-anchor="middle" font-size="10" font-weight="bold">Shift</text>
                <text x="25" y="45" fill="#8892a0" text-anchor="middle" font-size="9">± conf</text>
            </g>
            
            <!-- Arrow marker definition -->
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#00D4AA"/>
                </marker>
            </defs>
        </svg>
        
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 15px; flex-wrap: wrap;">
            <div style="text-align: center; padding: 8px 15px; background: #1a5a4a; border-radius: 5px;">
                <span style="color: #00D4AA; font-size: 14px; font-weight: bold;">32</span>
                <span style="color: #8892a0; font-size: 11px; display: block;">Hidden Units</span>
            </div>
            <div style="text-align: center; padding: 8px 15px; background: #4a3a1a; border-radius: 5px;">
                <span style="color: #FFB800; font-size: 14px; font-weight: bold;">8</span>
                <span style="color: #8892a0; font-size: 11px; display: block;">Attention Heads</span>
            </div>
            <div style="text-align: center; padding: 8px 15px; background: #2d3748; border-radius: 5px;">
                <span style="color: #ffffff; font-size: 14px; font-weight: bold;">~5K</span>
                <span style="color: #8892a0; font-size: 11px; display: block;">Parameters</span>
            </div>
        </div>
    </div>
    '''


def render_cnn_architecture() -> str:
    """
    Generate HTML/SVG for Siamese CNN architecture visualization.
    
    Returns:
        HTML string with CNN diagram
    """
    return '''
    <div style="background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%); 
                padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h4 style="color: #00D4AA; margin-bottom: 15px; text-align: center;">
            Siamese 1D CNN Pattern Matcher
        </h4>
        <svg viewBox="0 0 800 350" style="width: 100%; max-width: 800px; margin: 0 auto; display: block;">
            <!-- Shallow Branch -->
            <g transform="translate(50, 30)">
                <text x="40" y="0" fill="#8892a0" text-anchor="middle" font-size="12">Shallow Log</text>
                <rect x="0" y="10" width="80" height="100" rx="5" fill="#2d3748" stroke="#00AA00" stroke-width="2"/>
                
                <!-- Conv blocks -->
                <rect x="100" y="20" width="60" height="30" rx="3" fill="#1a5a4a" stroke="#00D4AA"/>
                <text x="130" y="40" fill="#fff" font-size="9" text-anchor="middle">Conv 1D</text>
                
                <rect x="100" y="55" width="60" height="30" rx="3" fill="#1a5a4a" stroke="#00D4AA"/>
                <text x="130" y="75" fill="#fff" font-size="9" text-anchor="middle">Conv 1D</text>
                
                <rect x="100" y="90" width="60" height="30" rx="3" fill="#1a5a4a" stroke="#00D4AA"/>
                <text x="130" y="110" fill="#fff" font-size="9" text-anchor="middle">Conv 1D</text>
                
                <!-- Pooling -->
                <rect x="180" y="40" width="50" height="60" rx="3" fill="#4a3a1a" stroke="#FFB800"/>
                <text x="205" y="65" fill="#fff" font-size="8" text-anchor="middle">Global</text>
                <text x="205" y="78" fill="#fff" font-size="8" text-anchor="middle">Pool</text>
                
                <!-- Embedding -->
                <circle cx="280" cy="70" r="25" fill="#00D4AA" opacity="0.3" stroke="#00D4AA" stroke-width="2"/>
                <text x="280" y="74" fill="#fff" font-size="9" text-anchor="middle">Emb</text>
            </g>
            
            <!-- Shared Weights Label -->
            <g transform="translate(310, 140)">
                <rect x="0" y="0" width="120" height="30" rx="15" fill="#2d3748" stroke="#FFB800" stroke-dasharray="4,2"/>
                <text x="60" y="20" fill="#FFB800" font-size="10" text-anchor="middle">SHARED WEIGHTS</text>
            </g>
            
            <!-- Deep Branch -->
            <g transform="translate(50, 200)">
                <text x="40" y="0" fill="#8892a0" text-anchor="middle" font-size="12">Deep Log</text>
                <rect x="0" y="10" width="80" height="100" rx="5" fill="#2d3748" stroke="#CC0000" stroke-width="2"/>
                
                <!-- Conv blocks -->
                <rect x="100" y="20" width="60" height="30" rx="3" fill="#1a5a4a" stroke="#00D4AA"/>
                <text x="130" y="40" fill="#fff" font-size="9" text-anchor="middle">Conv 1D</text>
                
                <rect x="100" y="55" width="60" height="30" rx="3" fill="#1a5a4a" stroke="#00D4AA"/>
                <text x="130" y="75" fill="#fff" font-size="9" text-anchor="middle">Conv 1D</text>
                
                <rect x="100" y="90" width="60" height="30" rx="3" fill="#1a5a4a" stroke="#00D4AA"/>
                <text x="130" y="110" fill="#fff" font-size="9" text-anchor="middle">Conv 1D</text>
                
                <!-- Pooling -->
                <rect x="180" y="40" width="50" height="60" rx="3" fill="#4a3a1a" stroke="#FFB800"/>
                <text x="205" y="65" fill="#fff" font-size="8" text-anchor="middle">Global</text>
                <text x="205" y="78" fill="#fff" font-size="8" text-anchor="middle">Pool</text>
                
                <!-- Embedding -->
                <circle cx="280" cy="70" r="25" fill="#00D4AA" opacity="0.3" stroke="#00D4AA" stroke-width="2"/>
                <text x="280" y="74" fill="#fff" font-size="9" text-anchor="middle">Emb</text>
            </g>
            
            <!-- Similarity Computation -->
            <g transform="translate(450, 120)">
                <path d="M -100 -20 L -50 30 M -100 130 L -50 80" stroke="#00D4AA" stroke-width="2"/>
                
                <rect x="0" y="0" width="100" height="110" rx="8" fill="#2d3748" stroke="#00D4AA" stroke-width="2"/>
                <text x="50" y="35" fill="#fff" font-size="11" text-anchor="middle" font-weight="bold">Cosine</text>
                <text x="50" y="55" fill="#fff" font-size="11" text-anchor="middle" font-weight="bold">Similarity</text>
                <text x="50" y="80" fill="#8892a0" font-size="9" text-anchor="middle">emb₁ · emb₂</text>
                <text x="50" y="95" fill="#8892a0" font-size="9" text-anchor="middle">‖emb₁‖ ‖emb₂‖</text>
            </g>
            
            <!-- Output -->
            <g transform="translate(600, 140)">
                <path d="M -45 35 L -10 35" stroke="#00D4AA" stroke-width="2"/>
                
                <rect x="0" y="0" width="140" height="70" rx="8" fill="#00D4AA" opacity="0.2" stroke="#00D4AA" stroke-width="2"/>
                <text x="70" y="25" fill="#fff" font-size="12" text-anchor="middle" font-weight="bold">Similarity Score</text>
                <text x="70" y="45" fill="#00D4AA" font-size="14" text-anchor="middle" font-weight="bold">0.0 - 1.0</text>
                <text x="70" y="60" fill="#8892a0" font-size="9" text-anchor="middle">Best match = highest</text>
            </g>
        </svg>
        
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 15px; flex-wrap: wrap;">
            <div style="padding: 6px 12px; background: #1a5a4a; border-radius: 4px;">
                <span style="color: #00D4AA; font-size: 12px;">3 Conv Layers</span>
            </div>
            <div style="padding: 6px 12px; background: #4a3a1a; border-radius: 4px;">
                <span style="color: #FFB800; font-size: 12px;">64-dim Embedding</span>
            </div>
            <div style="padding: 6px 12px; background: #2d3748; border-radius: 4px;">
                <span style="color: #fff; font-size: 12px;">~10K Parameters</span>
            </div>
        </div>
    </div>
    '''


def render_outlier_detection_diagram() -> str:
    """
    Generate HTML/SVG for outlier detection visualization.
    
    Returns:
        HTML string with Isolation Forest diagram
    """
    return '''
    <div style="background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%); 
                padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h4 style="color: #00D4AA; margin-bottom: 15px; text-align: center;">
            Isolation Forest Outlier Detection
        </h4>
        <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px;">
            <div style="text-align: center;">
                <div style="width: 120px; height: 120px; border: 2px solid #00D4AA; border-radius: 10px; 
                            display: flex; align-items: center; justify-content: center; background: #2d3748;">
                    <span style="color: #00D4AA; font-size: 36px;">📊</span>
                </div>
                <p style="color: #8892a0; margin-top: 8px; font-size: 12px;">Multi-Curve<br/>Input</p>
            </div>
            
            <div style="color: #00D4AA; font-size: 24px;">→</div>
            
            <div style="text-align: center;">
                <div style="width: 120px; height: 120px; border: 2px solid #FFB800; border-radius: 10px; 
                            display: flex; align-items: center; justify-content: center; background: #4a3a1a;">
                    <span style="font-size: 36px;">🌲</span>
                </div>
                <p style="color: #8892a0; margin-top: 8px; font-size: 12px;">100 Random<br/>Trees</p>
            </div>
            
            <div style="color: #00D4AA; font-size: 24px;">→</div>
            
            <div style="text-align: center;">
                <div style="width: 120px; height: 120px; border: 2px solid #FF6600; border-radius: 10px; 
                            display: flex; align-items: center; justify-content: center; background: #3a2a1a;">
                    <span style="color: #FF6600; font-size: 36px;">⚠️</span>
                </div>
                <p style="color: #8892a0; margin-top: 8px; font-size: 12px;">Anomaly<br/>Scores</p>
            </div>
        </div>
        
        <p style="color: #8892a0; text-align: center; margin-top: 15px; font-size: 11px;">
            Points isolated quickly (short path) = Anomalies | Points isolated slowly = Normal
        </p>
    </div>
    '''


# =============================================================================
# PLOTLY VISUALIZATIONS
# =============================================================================

def create_outlier_plot(
    depths: np.ndarray,
    signal: np.ndarray,
    anomaly_mask: np.ndarray,
    anomaly_scores: np.ndarray,
    curve_name: str = "Signal"
) -> go.Figure:
    """
    Create interactive outlier detection visualization.
    
    Args:
        depths: Depth array
        signal: Signal values
        anomaly_mask: Boolean mask of anomalies
        anomaly_scores: Continuous anomaly scores
        curve_name: Name of the curve
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=[f'{curve_name} with Outliers', 'Anomaly Score']
    )
    
    # Normal points
    normal_mask = ~anomaly_mask
    fig.add_trace(
        go.Scatter(
            x=signal[normal_mask],
            y=depths[normal_mask],
            mode='markers',
            marker=dict(color=COLORS['shallow'], size=4, opacity=0.6),
            name='Normal',
            hovertemplate='Depth: %{y:.2f}m<br>Value: %{x:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Anomaly points
    fig.add_trace(
        go.Scatter(
            x=signal[anomaly_mask],
            y=depths[anomaly_mask],
            mode='markers',
            marker=dict(color=COLORS['anomaly'], size=8, symbol='x'),
            name='Anomaly',
            hovertemplate='Depth: %{y:.2f}m<br>Value: %{x:.2f}<br>ANOMALY<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Anomaly score heatmap
    fig.add_trace(
        go.Scatter(
            x=anomaly_scores,
            y=depths,
            mode='lines',
            line=dict(color=COLORS['primary'], width=2),
            name='Anomaly Score',
            fill='tozerox',
            fillcolor='rgba(0, 212, 170, 0.2)'
        ),
        row=1, col=2
    )
    
    fig.update_yaxes(autorange='reversed', title_text='Depth (m)', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=2)
    fig.update_xaxes(title_text=curve_name, row=1, col=1)
    fig.update_xaxes(title_text='Score', row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['secondary'],
        font=dict(color=COLORS['text']),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        height=500,
        margin=dict(l=60, r=20, t=50, b=40)
    )
    
    return fig


def create_optimization_plot(
    convergence_history: List[float],
    param_history: List[Dict[str, float]],
    cost_history: List[float]
) -> go.Figure:
    """
    Create Bayesian optimization visualization.
    
    Args:
        convergence_history: Best cost at each iteration
        param_history: Parameter values at each iteration
        cost_history: Cost at each iteration
        
    Returns:
        Plotly figure with convergence and parameter plots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Convergence', 'Parameter Exploration',
            'Search Window vs Cost', 'DTW Window vs Cost'
        ],
        specs=[[{}, {}], [{}, {}]]
    )
    
    iterations = list(range(1, len(convergence_history) + 1))
    
    # Convergence plot
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=convergence_history,
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=6),
            name='Best Cost'
        ),
        row=1, col=1
    )
    
    # Cost at each evaluation
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=cost_history,
            mode='markers',
            marker=dict(color=COLORS['text_muted'], size=4, opacity=0.5),
            name='Evaluated Cost'
        ),
        row=1, col=1
    )
    
    # Parameter space - 3D scatter
    if param_history:
        search_vals = [p['search_window'] for p in param_history]
        dtw_vals = [p['dtw_window'] for p in param_history]
        
        # Color by cost (normalized)
        costs = np.array(cost_history)
        colors = (costs - costs.min()) / (costs.max() - costs.min() + 1e-6)
        
        fig.add_trace(
            go.Scatter(
                x=search_vals,
                y=dtw_vals,
                mode='markers',
                marker=dict(
                    color=colors,
                    colorscale='Viridis_r',
                    size=8,
                    showscale=True,
                    colorbar=dict(title='Cost', x=1.02, len=0.4, y=0.8)
                ),
                name='Evaluations'
            ),
            row=1, col=2
        )
        
        # Best point
        best_idx = np.argmin(cost_history)
        fig.add_trace(
            go.Scatter(
                x=[search_vals[best_idx]],
                y=[dtw_vals[best_idx]],
                mode='markers',
                marker=dict(color=COLORS['success'], size=15, symbol='star'),
                name='Best'
            ),
            row=1, col=2
        )
        
        # Search window vs cost
        fig.add_trace(
            go.Scatter(
                x=search_vals,
                y=cost_history,
                mode='markers',
                marker=dict(color=COLORS['primary'], size=6, opacity=0.6),
                name='Search Window'
            ),
            row=2, col=1
        )
        
        # DTW window vs cost
        fig.add_trace(
            go.Scatter(
                x=dtw_vals,
                y=cost_history,
                mode='markers',
                marker=dict(color=COLORS['warning'], size=6, opacity=0.6),
                name='DTW Window'
            ),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text='Iteration', row=1, col=1)
    fig.update_yaxes(title_text='Cost', row=1, col=1)
    fig.update_xaxes(title_text='Search Window (m)', row=1, col=2)
    fig.update_yaxes(title_text='DTW Window (m)', row=1, col=2)
    fig.update_xaxes(title_text='Search Window (m)', row=2, col=1)
    fig.update_yaxes(title_text='Cost', row=2, col=1)
    fig.update_xaxes(title_text='DTW Window (m)', row=2, col=2)
    fig.update_yaxes(title_text='Cost', row=2, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['secondary'],
        font=dict(color=COLORS['text']),
        showlegend=False,
        height=600,
        margin=dict(l=60, r=80, t=50, b=40)
    )
    
    return fig


def create_training_progress_plot(
    loss_history: List[float],
    model_name: str = "Model"
) -> go.Figure:
    """
    Create real-time training progress visualization.
    
    Args:
        loss_history: List of loss values per epoch
        model_name: Name of the model
        
    Returns:
        Plotly figure
    """
    epochs = list(range(1, len(loss_history) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=loss_history,
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)',
            name='Training Loss'
        )
    )
    
    # Add trend line
    if len(loss_history) > 5:
        z = np.polyfit(epochs, loss_history, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=p(epochs),
                mode='lines',
                line=dict(color=COLORS['warning'], width=1, dash='dash'),
                name='Trend'
            )
        )
    
    fig.update_layout(
        title=dict(text=f'{model_name} Training Progress', font=dict(size=14)),
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['secondary'],
        font=dict(color=COLORS['text']),
        showlegend=True,
        legend=dict(x=0.7, y=0.98),
        height=300,
        margin=dict(l=60, r=20, t=50, b=40)
    )
    
    return fig


def create_comparison_plot(
    depths: np.ndarray,
    before_signal: np.ndarray,
    after_signal: np.ndarray,
    labels: Tuple[str, str] = ("Before", "After")
) -> go.Figure:
    """
    Create before/after comparison visualization.
    
    Args:
        depths: Depth array
        before_signal: Signal before processing
        after_signal: Signal after processing
        labels: Labels for before/after
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.4, 0.4, 0.2],
        subplot_titles=[labels[0], labels[1], 'Difference']
    )
    
    # Before
    fig.add_trace(
        go.Scatter(
            x=before_signal,
            y=depths,
            mode='lines',
            line=dict(color=COLORS['deep'], width=1.5),
            name=labels[0]
        ),
        row=1, col=1
    )
    
    # After
    fig.add_trace(
        go.Scatter(
            x=after_signal,
            y=depths,
            mode='lines',
            line=dict(color=COLORS['corrected'], width=1.5),
            name=labels[1]
        ),
        row=1, col=2
    )
    
    # Difference
    diff = after_signal - before_signal
    fig.add_trace(
        go.Scatter(
            x=diff,
            y=depths,
            mode='lines',
            line=dict(color=COLORS['primary'], width=1.5),
            fill='tozerox',
            fillcolor='rgba(0, 212, 170, 0.2)',
            name='Difference'
        ),
        row=1, col=3
    )
    
    for col in [1, 2, 3]:
        fig.update_yaxes(autorange='reversed', row=1, col=col)
    
    fig.update_yaxes(title_text='Depth (m)', row=1, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['secondary'],
        font=dict(color=COLORS['text']),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        height=500,
        margin=dict(l=60, r=20, t=50, b=40)
    )
    
    return fig


def create_uncertainty_plot(
    depths: np.ndarray,
    mean_prediction: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    reference: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Create uncertainty band visualization.
    
    Args:
        depths: Depth array
        mean_prediction: Mean predicted values
        ci_lower: Lower confidence bound
        ci_upper: Upper confidence bound
        reference: Optional reference signal
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([ci_upper, ci_lower[::-1]]),
            y=np.concatenate([depths, depths[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 212, 170, 0.2)',
            line=dict(color='rgba(0, 212, 170, 0)'),
            name='95% CI',
            hoverinfo='skip'
        )
    )
    
    # Mean prediction
    fig.add_trace(
        go.Scatter(
            x=mean_prediction,
            y=depths,
            mode='lines',
            line=dict(color=COLORS['primary'], width=2),
            name='Mean Prediction'
        )
    )
    
    # Reference if provided
    if reference is not None:
        fig.add_trace(
            go.Scatter(
                x=reference,
                y=depths,
                mode='lines',
                line=dict(color=COLORS['text_muted'], width=1, dash='dot'),
                name='Reference'
            )
        )
    
    fig.update_yaxes(autorange='reversed', title_text='Depth (m)')
    fig.update_xaxes(title_text='Value')
    
    fig.update_layout(
        title='Prediction with Uncertainty Bands',
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['secondary'],
        font=dict(color=COLORS['text']),
        showlegend=True,
        legend=dict(x=0.7, y=0.02),
        height=500,
        margin=dict(l=60, r=20, t=50, b=40)
    )
    
    return fig


def create_similarity_heatmap(
    similarity_map: np.ndarray,
    positions: np.ndarray,
    best_position: int
) -> go.Figure:
    """
    Create similarity map visualization for CNN pattern matching.
    
    Args:
        similarity_map: Similarity scores at each position
        positions: Position indices
        best_position: Best match position
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Similarity curve
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=similarity_map,
            mode='lines',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.2)',
            name='Similarity'
        )
    )
    
    # Best match marker
    best_idx = np.argmin(np.abs(positions - best_position))
    fig.add_trace(
        go.Scatter(
            x=[positions[best_idx]],
            y=[similarity_map[best_idx]],
            mode='markers',
            marker=dict(color=COLORS['success'], size=15, symbol='star'),
            name='Best Match'
        )
    )
    
    # Threshold line
    fig.add_hline(
        y=0.5,
        line_dash='dash',
        line_color=COLORS['warning'],
        annotation_text='Match Threshold'
    )
    
    fig.update_layout(
        title='Pattern Similarity Map',
        xaxis_title='Position (samples)',
        yaxis_title='Similarity Score',
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['secondary'],
        font=dict(color=COLORS['text']),
        showlegend=True,
        height=350,
        margin=dict(l=60, r=20, t=50, b=40)
    )
    
    return fig


def create_metrics_dashboard(metrics: Dict[str, Any]) -> str:
    """
    Create HTML metrics dashboard.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        HTML string
    """
    cards_html = ""
    
    for name, value in metrics.items():
        if isinstance(value, float):
            display_value = f"{value:.4f}"
        elif isinstance(value, int):
            display_value = f"{value:,}"
        else:
            display_value = str(value)
        
        cards_html += f'''
        <div style="flex: 1; min-width: 150px; background: #2d3748; padding: 15px; 
                    border-radius: 8px; text-align: center; border: 1px solid #3d4852;">
            <div style="color: #00D4AA; font-size: 24px; font-weight: bold; 
                        font-family: 'JetBrains Mono', monospace;">{display_value}</div>
            <div style="color: #8892a0; font-size: 12px; margin-top: 5px;">{name}</div>
        </div>
        '''
    
    return f'''
    <div style="display: flex; gap: 15px; flex-wrap: wrap; margin: 15px 0;">
        {cards_html}
    </div>
    '''


def create_feature_importance_chart(
    feature_importance: Dict[str, float]
) -> go.Figure:
    """
    Create feature importance bar chart.
    
    Args:
        feature_importance: Dictionary of feature names and importance scores
        
    Returns:
        Plotly figure
    """
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    features = [features[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale=[[0, COLORS['secondary']], [1, COLORS['primary']]],
            ),
            text=[f'{v:.2%}' for v in importances],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title='Feature Contribution to Anomalies',
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_dark',
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['secondary'],
        font=dict(color=COLORS['text']),
        height=max(200, len(features) * 40),
        margin=dict(l=100, r=20, t=50, b=40)
    )
    
    return fig
