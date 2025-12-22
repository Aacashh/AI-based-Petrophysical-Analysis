"""
ML Components Module for WellLog Analyzer Pro

This module provides AI/ML capabilities for well log analysis:

Tier 1 (Quick Wins):
- Outlier Detection: Isolation Forest, Local Outlier Factor
- Bayesian Optimization: Hyperparameter tuning for splicing

Tier 2 (Core ML):
- LSTM Alignment: Multi-curve sequence alignment
- CNN Pattern Matcher: Siamese network for similarity matching
- Uncertainty Quantification: Monte Carlo Dropout
"""

from .outlier_detection import (
    detect_outliers_isolation_forest,
    detect_outliers_lof,
    OutlierDetectionResult
)

from .bayesian_optimizer import (
    optimize_splicing_params,
    OptimizationResult
)

from .lstm_alignment import (
    LightweightLSTMAligner,
    train_lstm_aligner,
    predict_alignment
)

from .cnn_pattern_matcher import (
    SiameseCNN,
    train_pattern_matcher,
    compute_similarity_map
)

from .uncertainty import (
    predict_with_uncertainty,
    UncertaintyResult
)

from .visualizations import (
    render_lstm_architecture,
    render_cnn_architecture,
    create_metrics_dashboard,
    create_comparison_plot,
    create_outlier_plot,
    create_optimization_plot
)

__all__ = [
    # Tier 1
    'detect_outliers_isolation_forest',
    'detect_outliers_lof',
    'OutlierDetectionResult',
    'optimize_splicing_params',
    'OptimizationResult',
    # Tier 2
    'LightweightLSTMAligner',
    'train_lstm_aligner',
    'predict_alignment',
    'SiameseCNN',
    'train_pattern_matcher',
    'compute_similarity_map',
    'predict_with_uncertainty',
    'UncertaintyResult',
    # Visualizations
    'render_lstm_architecture',
    'render_cnn_architecture',
    'create_metrics_dashboard',
    'create_comparison_plot',
    'create_outlier_plot',
    'create_optimization_plot',
]
