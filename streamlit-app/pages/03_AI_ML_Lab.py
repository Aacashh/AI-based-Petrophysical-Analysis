"""
WellLog Analyzer Pro - AI/ML Laboratory
Advanced Machine Learning features for well log analysis.

Features:
- Tier 1: Outlier Detection, Bayesian Optimization
- Tier 2: LSTM Alignment, CNN Pattern Matching, Uncertainty Quantification

This is a demo page showcasing genuine AI/ML capabilities with
educational visualizations showing what happens "under the hood".
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.las_parser import load_las, extract_header_info, detect_depth_units, get_available_curves
from shared.curve_mapping import get_curve_mapping
from shared.data_processing import process_data

# Page configuration
st.set_page_config(
    page_title="AI/ML Lab | WellLog Analyzer Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #3d4852;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #00D4AA;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 0.9rem;
        color: #8892a0;
        margin-top: 0.3rem;
    }
    
    .tier-header {
        background: linear-gradient(90deg, #00D4AA 0%, #00A080 100%);
        color: #1a1d24;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 25px 0 15px 0;
    }
    
    .feature-card {
        background: linear-gradient(180deg, #2d3748 0%, #1a1d24 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3d4852;
        margin-bottom: 15px;
    }
    
    .feature-title {
        color: #00D4AA;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        color: #8892a0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #3d4852;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00D4AA;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8892a0;
        margin-top: 5px;
    }
    
    .algorithm-box {
        background: #1a1d24;
        border-left: 4px solid #00D4AA;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
        font-family: 'Inter', sans-serif;
    }
    
    .algorithm-box code {
        font-family: 'JetBrains Mono', monospace;
        background: #2d3748;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.85rem;
        color: #00D4AA;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-ready { background: #00D4AA; color: #1a1d24; }
    .badge-processing { background: #FFB800; color: #1a1d24; }
    .badge-error { background: #FF4444; color: white; }
    
    .info-panel {
        background: #2d3748;
        border: 1px solid #3d4852;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🧠 AI/ML Laboratory</div>
    <div class="main-subtitle">Advanced Machine Learning for Well Log Analysis • Tier 1 & Tier 2 Features</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("### 📁 Data Upload")
    
    shallow_file = st.file_uploader(
        "Reference Log (Shallow)",
        type=['las', 'LAS'],
        key='ml_shallow',
        help="Upload the shallow/reference log run"
    )
    
    deep_file = st.file_uploader(
        "Target Log (Deep)",
        type=['las', 'LAS'],
        key='ml_deep',
        help="Upload the deep/target log run"
    )
    
    st.markdown("---")
    
    st.markdown("### 🎯 Feature Selection")
    
    tier1_enabled = st.checkbox("Enable Tier 1 Features", value=True)
    tier2_enabled = st.checkbox("Enable Tier 2 Features", value=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📚 About AI/ML Lab
    
    **Tier 1: Quick Wins**
    - Outlier Detection
    - Bayesian Optimization
    
    **Tier 2: Core ML**
    - LSTM Alignment
    - CNN Pattern Matching
    - Uncertainty Quantification
    
    ---
    
    *Models train on your data in real-time*
    """)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Check if files are uploaded
if shallow_file and deep_file:
    try:
        # Load LAS files
        shallow_las = load_las(shallow_file)
        deep_las = load_las(deep_file)
        
        shallow_unit = detect_depth_units(shallow_las)
        deep_unit = detect_depth_units(deep_las)
        
        shallow_mapping = get_curve_mapping(shallow_las)
        deep_mapping = get_curve_mapping(deep_las)
        
        shallow_df = process_data(shallow_las, shallow_mapping)
        deep_df = process_data(deep_las, deep_mapping)
        
        shallow_header = extract_header_info(shallow_las)
        deep_header = extract_header_info(deep_las)
        
        # Display file info
        st.markdown("### 📥 Loaded Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-panel">
                <strong style="color: #00D4AA;">Shallow Run:</strong> {shallow_header['WELL']}<br>
                <span style="color: #8892a0;">Depth: {shallow_df['DEPTH'].min():.1f} - {shallow_df['DEPTH'].max():.1f} {shallow_unit}</span><br>
                <span style="color: #8892a0;">Samples: {len(shallow_df):,}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-panel">
                <strong style="color: #00D4AA;">Deep Run:</strong> {deep_header['WELL']}<br>
                <span style="color: #8892a0;">Depth: {deep_df['DEPTH'].min():.1f} - {deep_df['DEPTH'].max():.1f} {deep_unit}</span><br>
                <span style="color: #8892a0;">Samples: {len(deep_df):,}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Find common curves
        shallow_curves = [c for c in shallow_df.columns if c != 'DEPTH']
        deep_curves = [c for c in deep_df.columns if c != 'DEPTH']
        common_curves = list(set(shallow_curves) & set(deep_curves))
        
        if not common_curves:
            st.error("No common curves found between files!")
            st.stop()
        
        # Curve selection
        selected_curves = st.multiselect(
            "Select curves for analysis",
            options=common_curves,
            default=common_curves[:min(4, len(common_curves))],
            help="Choose curves to use for ML analysis"
        )
        
        if not selected_curves:
            st.warning("Please select at least one curve")
            st.stop()
        
        # Primary curve for correlation
        primary_curve = st.selectbox(
            "Primary curve for alignment",
            options=selected_curves,
            index=0 if 'GR' not in selected_curves else selected_curves.index('GR') if 'GR' in selected_curves else 0
        )
        
        # =================================================================
        # TIER 1: QUICK WINS
        # =================================================================
        
        if tier1_enabled:
            st.markdown('<div class="tier-header">🎯 TIER 1: Quick Wins</div>', unsafe_allow_html=True)
            
            tier1_tab1, tier1_tab2 = st.tabs(["🔍 Outlier Detection", "⚡ Bayesian Optimization"])
            
            # ---------------------------------------------------------
            # OUTLIER DETECTION
            # ---------------------------------------------------------
            with tier1_tab1:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-title">🔍 ML-Based Outlier Detection</div>
                    <div class="feature-desc">
                        Detect anomalous data points using Isolation Forest and Local Outlier Factor (LOF).
                        These algorithms identify points that don't follow the normal data distribution.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Import ML components
                from ml_components.outlier_detection import (
                    detect_outliers_isolation_forest,
                    detect_outliers_lof,
                    clean_outliers
                )
                from ml_components.visualizations import (
                    render_outlier_detection_diagram,
                    create_outlier_plot,
                    create_feature_importance_chart,
                    create_metrics_dashboard
                )
                
                # Architecture diagram
                with st.expander("📐 How It Works", expanded=False):
                    st.markdown(render_outlier_detection_diagram(), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="algorithm-box">
                    <strong>Isolation Forest Algorithm:</strong><br>
                    1. Build random trees by selecting random features and split values<br>
                    2. Anomalies are isolated in fewer splits (shorter path length)<br>
                    3. Points with short average path length are flagged as outliers<br><br>
                    <strong>Why it works:</strong> Anomalies are "few and different" - they're easier to isolate randomly.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Controls
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    outlier_method = st.selectbox(
                        "Detection Method",
                        options=['Isolation Forest', 'Local Outlier Factor', 'Ensemble (Both)'],
                        index=0
                    )
                
                with col2:
                    contamination = st.slider(
                        "Contamination %",
                        min_value=1,
                        max_value=20,
                        value=5,
                        help="Expected percentage of outliers"
                    ) / 100
                
                with col3:
                    target_log = st.selectbox(
                        "Analyze Log",
                        options=['Shallow', 'Deep'],
                        index=0
                    )
                
                # Run detection
                if st.button("🔍 Run Outlier Detection", type="primary", use_container_width=True):
                    
                    target_df = shallow_df if target_log == 'Shallow' else deep_df
                    
                    with st.status("🔄 Running ML Outlier Detection...", expanded=True) as status:
                        
                        st.write("📊 Preparing data...")
                        
                        st.write(f"🌲 Running {outlier_method}...")
                        
                        if outlier_method == 'Isolation Forest':
                            result = detect_outliers_isolation_forest(
                                target_df, selected_curves, contamination
                            )
                        elif outlier_method == 'Local Outlier Factor':
                            result = detect_outliers_lof(
                                target_df, selected_curves, contamination=contamination
                            )
                        else:
                            from ml_components.outlier_detection import detect_outliers_ensemble
                            result = detect_outliers_ensemble(
                                target_df, selected_curves, contamination
                            )
                        
                        st.write(f"✅ Found {result.num_anomalies} anomalies ({result.contamination_actual*100:.1f}%)")
                        
                        status.update(label="✅ Detection Complete!", state="complete")
                    
                    # Results
                    st.markdown("### 📊 Results")
                    
                    # Metrics
                    metrics = {
                        'Anomalies Found': result.num_anomalies,
                        'Contamination': f"{result.contamination_actual*100:.2f}%",
                        'Confidence': f"{result.confidence*100:.1f}%",
                        'Method': result.method
                    }
                    st.markdown(create_metrics_dashboard(metrics), unsafe_allow_html=True)
                    
                    # Visualization
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = create_outlier_plot(
                            target_df['DEPTH'].values,
                            target_df[primary_curve].values,
                            result.anomaly_mask,
                            result.anomaly_scores,
                            primary_curve
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Feature Importance")
                        fig_imp = create_feature_importance_chart(result.feature_importance)
                        st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Store result
                    st.session_state['outlier_result'] = result
            
            # ---------------------------------------------------------
            # BAYESIAN OPTIMIZATION
            # ---------------------------------------------------------
            with tier1_tab2:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-title">⚡ Bayesian Hyperparameter Optimization</div>
                    <div class="feature-desc">
                        Automatically find optimal splicing parameters (search window, DTW window, grid step)
                        using Gaussian Process optimization. Learns from each evaluation to explore efficiently.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                from ml_components.bayesian_optimizer import (
                    optimize_splicing_params,
                    quick_grid_search,
                    OptimizationConfig
                )
                from ml_components.visualizations import create_optimization_plot
                
                with st.expander("📐 How It Works", expanded=False):
                    st.markdown("""
                    <div class="algorithm-box">
                    <strong>Gaussian Process Optimization:</strong><br>
                    1. Start with random parameter samples<br>
                    2. Build probabilistic model (GP) of the objective function<br>
                    3. Use acquisition function (Expected Improvement) to select next point<br>
                    4. Evaluate objective and update model<br>
                    5. Repeat until convergence<br><br>
                    <strong>Why it's better than grid search:</strong> Learns from evaluations to focus on promising regions.
                    </div>
                    """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    opt_method = st.selectbox(
                        "Optimization Method",
                        options=['Bayesian (GP)', 'Quick Grid Search'],
                        index=0
                    )
                
                with col2:
                    n_iterations = st.slider(
                        "Max Iterations",
                        min_value=10,
                        max_value=50,
                        value=25
                    )
                
                if st.button("⚡ Run Optimization", type="primary", use_container_width=True):
                    
                    # Get primary curve data
                    shallow_depth = shallow_df['DEPTH'].values
                    shallow_signal = shallow_df[primary_curve].values
                    deep_depth = deep_df['DEPTH'].values
                    deep_signal = deep_df[primary_curve].values
                    
                    progress_placeholder = st.empty()
                    chart_placeholder = st.empty()
                    
                    convergence_data = []
                    
                    def opt_callback(iteration, cost, params):
                        convergence_data.append(cost)
                        progress_placeholder.progress(
                            iteration / n_iterations,
                            text=f"Iteration {iteration}/{n_iterations} | Cost: {cost:.4f}"
                        )
                    
                    with st.status("🔄 Running Optimization...", expanded=True) as status:
                        
                        if opt_method == 'Bayesian (GP)':
                            st.write("🧠 Initializing Gaussian Process model...")
                            
                            config = OptimizationConfig(n_calls=n_iterations)
                            
                            result = optimize_splicing_params(
                                shallow_depth, shallow_signal,
                                deep_depth, deep_signal,
                                config=config,
                                progress_callback=opt_callback
                            )
                        else:
                            st.write("📊 Running grid search...")
                            result = quick_grid_search(
                                shallow_depth, shallow_signal,
                                deep_depth, deep_signal,
                                progress_callback=opt_callback
                            )
                        
                        status.update(label="✅ Optimization Complete!", state="complete")
                    
                    progress_placeholder.empty()
                    
                    # Results
                    st.markdown("### 📊 Optimization Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{result.best_search_window:.1f}m</div>
                            <div class="metric-label">Search Window</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{result.best_dtw_window:.1f}m</div>
                            <div class="metric-label">DTW Window</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{result.best_grid_step:.3f}m</div>
                            <div class="metric-label">Grid Step</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        improvement = result.improvement_percent or 0
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{improvement:+.1f}%</div>
                            <div class="metric-label">Improvement</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualization
                    fig = create_optimization_plot(
                        result.convergence_history,
                        result.param_history,
                        result.cost_history
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.session_state['opt_result'] = result
        
        # =================================================================
        # TIER 2: CORE ML
        # =================================================================
        
        if tier2_enabled:
            st.markdown('<div class="tier-header">🧬 TIER 2: Core ML</div>', unsafe_allow_html=True)
            
            tier2_tab1, tier2_tab2, tier2_tab3 = st.tabs([
                "🔮 LSTM Alignment",
                "🎯 CNN Pattern Matcher",
                "📊 Uncertainty Quantification"
            ])
            
            # ---------------------------------------------------------
            # LSTM ALIGNMENT
            # ---------------------------------------------------------
            with tier2_tab1:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-title">🔮 LSTM Multi-Curve Alignment</div>
                    <div class="feature-desc">
                        Train a lightweight LSTM network to learn optimal depth alignment from multiple curves.
                        Uses self-attention to weight important log sections.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                from ml_components.lstm_alignment import (
                    train_lstm_aligner,
                    get_model_summary
                )
                from ml_components.visualizations import (
                    render_lstm_architecture,
                    create_training_progress_plot
                )
                
                with st.expander("📐 Network Architecture", expanded=True):
                    st.markdown(render_lstm_architecture(), unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    lstm_epochs = st.slider("Training Epochs", 20, 100, 50, key='lstm_epochs')
                
                with col2:
                    lstm_lr = st.select_slider(
                        "Learning Rate",
                        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                        value=0.001,
                        key='lstm_lr'
                    )
                
                if st.button("🚀 Train LSTM Model", type="primary", use_container_width=True, key='train_lstm'):
                    
                    shallow_depth = shallow_df['DEPTH'].values
                    shallow_signal = shallow_df[primary_curve].values
                    deep_depth = deep_df['DEPTH'].values
                    deep_signal = deep_df[primary_curve].values
                    
                    progress_bar = st.progress(0)
                    loss_placeholder = st.empty()
                    
                    loss_history = []
                    
                    def lstm_callback(epoch, loss):
                        loss_history.append(loss)
                        progress_bar.progress(epoch / lstm_epochs)
                        
                        fig = create_training_progress_plot(loss_history, "LSTM")
                        loss_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    with st.status("🧠 Training LSTM Model...", expanded=True) as status:
                        
                        st.write("📐 Building model architecture...")
                        st.write(f"🏋️ Training for {lstm_epochs} epochs...")
                        
                        result = train_lstm_aligner(
                            shallow_depth, shallow_signal,
                            deep_depth, deep_signal,
                            epochs=lstm_epochs,
                            learning_rate=lstm_lr,
                            progress_callback=lstm_callback
                        )
                        
                        status.update(label="✅ Training Complete!", state="complete")
                    
                    progress_bar.empty()
                    
                    # Results
                    st.markdown("### 📊 Training Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{result.predicted_shift:.3f}m</div>
                            <div class="metric-label">Predicted Shift</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{result.confidence*100:.1f}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{result.final_loss:.4f}</div>
                            <div class="metric-label">Final Loss</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Model summary
                    with st.expander("📋 Model Summary"):
                        summary = get_model_summary(result.model)
                        st.json(summary)
                    
                    st.session_state['lstm_result'] = result
            
            # ---------------------------------------------------------
            # CNN PATTERN MATCHER
            # ---------------------------------------------------------
            with tier2_tab2:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-title">🎯 Siamese CNN Pattern Matcher</div>
                    <div class="feature-desc">
                        Learn to compare log segments using a Siamese 1D CNN. The network learns
                        an embedding space where similar patterns are close together.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                from ml_components.cnn_pattern_matcher import (
                    train_pattern_matcher,
                    compute_similarity_map,
                    get_model_summary as get_cnn_summary
                )
                from ml_components.visualizations import (
                    render_cnn_architecture,
                    create_training_progress_plot,
                    create_similarity_heatmap
                )
                
                with st.expander("📐 Network Architecture", expanded=True):
                    st.markdown(render_cnn_architecture(), unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    cnn_epochs = st.slider("Training Epochs", 10, 50, 30, key='cnn_epochs')
                
                with col2:
                    n_pairs = st.slider("Training Pairs", 50, 200, 100, key='n_pairs')
                
                if st.button("🚀 Train CNN Model", type="primary", use_container_width=True, key='train_cnn'):
                    
                    shallow_depth = shallow_df['DEPTH'].values
                    shallow_signal = shallow_df[primary_curve].values
                    deep_depth = deep_df['DEPTH'].values
                    deep_signal = deep_df[primary_curve].values
                    
                    progress_bar = st.progress(0)
                    loss_placeholder = st.empty()
                    
                    loss_history = []
                    
                    def cnn_callback(epoch, loss):
                        loss_history.append(loss)
                        progress_bar.progress(epoch / cnn_epochs)
                        
                        fig = create_training_progress_plot(loss_history, "Siamese CNN")
                        loss_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    with st.status("🧠 Training CNN Model...", expanded=True) as status:
                        
                        st.write("📐 Building Siamese architecture...")
                        st.write(f"🏋️ Training on {n_pairs} contrastive pairs...")
                        
                        result = train_pattern_matcher(
                            shallow_depth, shallow_signal,
                            deep_depth, deep_signal,
                            epochs=cnn_epochs,
                            n_pairs=n_pairs,
                            progress_callback=cnn_callback
                        )
                        
                        st.write("🔍 Computing similarity map...")
                        
                        similarity_result = compute_similarity_map(
                            result.model,
                            shallow_depth, shallow_signal,
                            deep_depth, deep_signal
                        )
                        
                        status.update(label="✅ Training Complete!", state="complete")
                    
                    progress_bar.empty()
                    
                    # Results
                    st.markdown("### 📊 Pattern Matching Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{similarity_result.best_match_similarity:.3f}</div>
                            <div class="metric-label">Best Similarity</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{similarity_result.depth_offset:.3f}m</div>
                            <div class="metric-label">Depth Offset</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-highlight">
                            <div class="metric-value">{similarity_result.confidence*100:.1f}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Similarity map
                    fig = create_similarity_heatmap(
                        similarity_result.similarity_map,
                        np.arange(len(similarity_result.similarity_map)),
                        similarity_result.best_match_position
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.session_state['cnn_result'] = result
                    st.session_state['similarity_result'] = similarity_result
            
            # ---------------------------------------------------------
            # UNCERTAINTY QUANTIFICATION
            # ---------------------------------------------------------
            with tier2_tab3:
                st.markdown("""
                <div class="feature-card">
                    <div class="feature-title">📊 Uncertainty Quantification</div>
                    <div class="feature-desc">
                        Estimate prediction confidence using Monte Carlo Dropout. Run multiple
                        forward passes with dropout enabled to sample from the posterior.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                from ml_components.uncertainty import (
                    predict_with_uncertainty,
                    get_prediction_quality_metrics
                )
                from ml_components.visualizations import create_uncertainty_plot
                
                with st.expander("📐 How It Works", expanded=False):
                    st.markdown("""
                    <div class="algorithm-box">
                    <strong>Monte Carlo Dropout:</strong><br>
                    1. Train model with Dropout layers<br>
                    2. At inference, keep Dropout enabled<br>
                    3. Run N forward passes (default: 50)<br>
                    4. Each pass samples different neurons<br>
                    5. Variation across passes = uncertainty<br><br>
                    <strong>Result:</strong> Point estimate ± 95% confidence interval
                    </div>
                    """, unsafe_allow_html=True)
                
                # Check if LSTM model exists
                if 'lstm_result' not in st.session_state:
                    st.info("⚠️ Train an LSTM model first to enable uncertainty quantification")
                else:
                    n_mc_samples = st.slider("MC Samples", 20, 100, 50, key='n_mc')
                    
                    if st.button("📊 Run Uncertainty Analysis", type="primary", use_container_width=True):
                        
                        model = st.session_state['lstm_result'].model
                        
                        shallow_depth = shallow_df['DEPTH'].values
                        shallow_signal = shallow_df[primary_curve].values
                        deep_depth = deep_df['DEPTH'].values
                        deep_signal = deep_df[primary_curve].values
                        
                        from ml_components.lstm_alignment import prepare_alignment_data
                        import torch
                        
                        with st.status("🔄 Running Monte Carlo Inference...", expanded=True) as status:
                            
                            st.write(f"🎲 Running {n_mc_samples} forward passes...")
                            
                            # Find overlap
                            overlap_start = max(np.nanmin(shallow_depth), np.nanmin(deep_depth))
                            overlap_end = min(np.nanmax(shallow_depth), np.nanmax(deep_depth))
                            
                            shallow_windows, deep_windows, _ = prepare_alignment_data(
                                shallow_depth, shallow_signal,
                                deep_depth, deep_signal,
                                overlap_start, overlap_end,
                                n_samples=1
                            )
                            
                            def forward_fn(m):
                                shift, conf, _, _ = m(shallow_windows, deep_windows)
                                return shift
                            
                            uq_result = predict_with_uncertainty(
                                model, forward_fn, n_samples=n_mc_samples
                            )
                            
                            status.update(label="✅ Analysis Complete!", state="complete")
                        
                        # Results
                        st.markdown("### 📊 Uncertainty Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-highlight">
                                <div class="metric-value">{uq_result.mean_prediction:.3f}m</div>
                                <div class="metric-label">Mean Prediction</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-highlight">
                                <div class="metric-value">±{uq_result.std_prediction:.3f}m</div>
                                <div class="metric-label">Std Deviation</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-highlight">
                                <div class="metric-value">[{uq_result.ci_lower:.3f}, {uq_result.ci_upper:.3f}]</div>
                                <div class="metric-label">95% CI</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            grade = 'A' if uq_result.confidence_score > 0.9 else 'B' if uq_result.confidence_score > 0.7 else 'C'
                            st.markdown(f"""
                            <div class="metric-highlight">
                                <div class="metric-value">{grade}</div>
                                <div class="metric-label">Quality Grade</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Distribution histogram
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=uq_result.prediction_samples,
                            nbinsx=30,
                            marker_color='rgba(0, 212, 170, 0.6)',
                            name='MC Samples'
                        ))
                        
                        # Mean line
                        fig.add_vline(
                            x=uq_result.mean_prediction,
                            line_dash="solid",
                            line_color="#00D4AA",
                            annotation_text="Mean"
                        )
                        
                        # CI lines
                        fig.add_vline(x=uq_result.ci_lower, line_dash="dash", line_color="#FFB800")
                        fig.add_vline(x=uq_result.ci_upper, line_dash="dash", line_color="#FFB800")
                        
                        fig.update_layout(
                            title='Monte Carlo Prediction Distribution',
                            xaxis_title='Predicted Shift (m)',
                            yaxis_title='Count',
                            template='plotly_dark',
                            paper_bgcolor='#1a1d24',
                            plot_bgcolor='#2d3748',
                            height=350
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Quality metrics
                        metrics = get_prediction_quality_metrics(uq_result)
                        
                        with st.expander("📋 Quality Metrics"):
                            st.json(metrics)
                        
                        st.session_state['uq_result'] = uq_result
        
        # =================================================================
        # COMPARISON DASHBOARD
        # =================================================================
        
        st.markdown("---")
        st.markdown("### 📊 Results Comparison Dashboard")
        
        # Check what results we have
        has_outlier = 'outlier_result' in st.session_state
        has_opt = 'opt_result' in st.session_state
        has_lstm = 'lstm_result' in st.session_state
        has_cnn = 'cnn_result' in st.session_state
        has_uq = 'uq_result' in st.session_state
        
        if not any([has_outlier, has_opt, has_lstm, has_cnn, has_uq]):
            st.info("Run some analyses above to see comparison results here.")
        else:
            comparison_data = []
            
            if has_lstm:
                comparison_data.append({
                    'Method': 'LSTM',
                    'Predicted Shift': f"{st.session_state['lstm_result'].predicted_shift:.3f}m",
                    'Confidence': f"{st.session_state['lstm_result'].confidence*100:.1f}%",
                    'Final Loss': f"{st.session_state['lstm_result'].final_loss:.4f}"
                })
            
            if has_cnn and 'similarity_result' in st.session_state:
                comparison_data.append({
                    'Method': 'Siamese CNN',
                    'Predicted Shift': f"{st.session_state['similarity_result'].depth_offset:.3f}m",
                    'Confidence': f"{st.session_state['similarity_result'].confidence*100:.1f}%",
                    'Final Loss': f"{st.session_state['cnn_result'].final_loss:.4f}"
                })
            
            if has_uq:
                comparison_data.append({
                    'Method': 'MC Dropout',
                    'Predicted Shift': f"{st.session_state['uq_result'].mean_prediction:.3f}m ± {st.session_state['uq_result'].std_prediction:.3f}m",
                    'Confidence': f"{st.session_state['uq_result'].confidence_score*100:.1f}%",
                    'Final Loss': 'N/A'
                })
            
            if comparison_data:
                st.table(pd.DataFrame(comparison_data))
            
            # Export button
            if st.button("💾 Export All Results", use_container_width=True):
                st.success("Results exported! (Demo mode - implement actual export)")
    
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        st.exception(e)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; background: linear-gradient(180deg, #2d3748 0%, #1a1d24 100%); 
                border-radius: 15px; margin-top: 20px; border: 1px solid #3d4852;">
        <div style="font-size: 5rem; margin-bottom: 20px;">🧠</div>
        <h2 style="color: #00D4AA; margin-bottom: 15px;">AI/ML Laboratory</h2>
        <p style="color: #8892a0; font-size: 1.1rem; max-width: 700px; margin: 0 auto; line-height: 1.6;">
            Upload two LAS files to explore advanced machine learning capabilities
            for well log analysis. All models train on your data in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature overview
    st.markdown("### 🎯 Available Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 Tier 1: Quick Wins</div>
            <div class="feature-desc">
                <strong>Outlier Detection</strong><br>
                Isolation Forest & LOF for anomaly detection<br><br>
                <strong>Bayesian Optimization</strong><br>
                Automatic hyperparameter tuning with GP
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🧬 Tier 2: Core ML</div>
            <div class="feature-desc">
                <strong>LSTM Alignment</strong><br>
                Bi-directional LSTM with attention<br><br>
                <strong>CNN Pattern Matcher</strong><br>
                Siamese network for similarity learning<br><br>
                <strong>Uncertainty Quantification</strong><br>
                Monte Carlo Dropout for confidence bounds
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📚 Getting Started
    
    1. **Upload** a shallow (reference) and deep (target) LAS file using the sidebar
    2. **Select** curves to analyze
    3. **Run** the ML features - each trains a lightweight model on your data
    4. **Compare** results across different methods
    """)
