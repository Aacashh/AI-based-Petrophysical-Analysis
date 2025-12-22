"""
ML-Based Outlier Detection for Well Log Data

Implements Isolation Forest, Local Outlier Factor (LOF), and Angular-Based
Outlier Detection (ABOD) algorithms for detecting anomalous data points
in well log curves.

Reference: INTERNAL_AI_ML_ENHANCEMENT_RESEARCH.md - Tier 1.1
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Try to import PyOD for ABOD - graceful fallback if not available
try:
    from pyod.models.abod import ABOD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False


@dataclass
class OutlierDetectionResult:
    """Container for outlier detection results."""
    anomaly_mask: np.ndarray  # Boolean mask: True = anomaly
    anomaly_scores: np.ndarray  # Continuous anomaly scores (-1 to 1 range normalized)
    num_anomalies: int
    contamination_actual: float  # Actual percentage of anomalies
    method: str  # 'isolation_forest' or 'lof'
    feature_columns: List[str]
    confidence: float  # Model confidence based on data quality
    
    # Per-feature statistics
    feature_importance: Dict[str, float]  # Which features contributed most to anomalies


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    feature_columns: List[str],
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42
) -> OutlierDetectionResult:
    """
    Detect outliers using Isolation Forest algorithm.
    
    Isolation Forest isolates anomalies by randomly selecting features and
    split values. Anomalies are easier to isolate (shorter path lengths).
    
    Args:
        df: DataFrame with well log data
        feature_columns: List of curve names to use for detection
        contamination: Expected proportion of outliers (0.0 to 0.5)
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        
    Returns:
        OutlierDetectionResult with detection results and metrics
    """
    # Validate inputs
    available_cols = [c for c in feature_columns if c in df.columns]
    if not available_cols:
        raise ValueError(f"None of the specified columns found in DataFrame: {feature_columns}")
    
    # Extract and prepare data
    X = df[available_cols].copy()
    
    # Handle missing values - fill with column median
    X = X.fillna(X.median())
    
    # Replace any remaining NaN/inf with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Standardize features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and fit model
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Predict: -1 for outliers, 1 for inliers
    predictions = model.fit_predict(X_scaled)
    
    # Get anomaly scores (negative = more anomalous)
    raw_scores = model.decision_function(X_scaled)
    
    # Normalize scores to -1 to 1 range (more negative = more anomalous)
    scores_min, scores_max = raw_scores.min(), raw_scores.max()
    if scores_max - scores_min > 0:
        normalized_scores = 2 * (raw_scores - scores_min) / (scores_max - scores_min) - 1
    else:
        normalized_scores = np.zeros_like(raw_scores)
    
    # Create anomaly mask
    anomaly_mask = predictions == -1
    num_anomalies = np.sum(anomaly_mask)
    
    # Calculate feature importance based on anomalous points
    feature_importance = {}
    if num_anomalies > 0:
        normal_data = X_scaled[~anomaly_mask]
        anomaly_data = X_scaled[anomaly_mask]
        
        for i, col in enumerate(available_cols):
            # Calculate how different anomaly values are from normal distribution
            normal_mean = normal_data[:, i].mean() if len(normal_data) > 0 else 0
            normal_std = normal_data[:, i].std() if len(normal_data) > 0 else 1
            
            if normal_std > 0:
                z_scores = np.abs((anomaly_data[:, i] - normal_mean) / normal_std)
                feature_importance[col] = float(np.mean(z_scores))
            else:
                feature_importance[col] = 0.0
    else:
        feature_importance = {col: 0.0 for col in available_cols}
    
    # Normalize feature importance to sum to 1
    total_importance = sum(feature_importance.values())
    if total_importance > 0:
        feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
    
    # Calculate confidence based on data quality
    confidence = calculate_detection_confidence(df, available_cols, num_anomalies)
    
    return OutlierDetectionResult(
        anomaly_mask=anomaly_mask,
        anomaly_scores=normalized_scores,
        num_anomalies=num_anomalies,
        contamination_actual=num_anomalies / len(df) if len(df) > 0 else 0.0,
        method='isolation_forest',
        feature_columns=available_cols,
        confidence=confidence,
        feature_importance=feature_importance
    )


def detect_outliers_lof(
    df: pd.DataFrame,
    feature_columns: List[str],
    n_neighbors: int = 20,
    contamination: float = 0.05
) -> OutlierDetectionResult:
    """
    Detect outliers using Local Outlier Factor (LOF) algorithm.
    
    LOF measures local density deviation of a data point with respect to
    its neighbors. Points with substantially lower density are outliers.
    
    Args:
        df: DataFrame with well log data
        feature_columns: List of curve names to use for detection
        n_neighbors: Number of neighbors to use for LOF calculation
        contamination: Expected proportion of outliers (0.0 to 0.5)
        
    Returns:
        OutlierDetectionResult with detection results and metrics
    """
    # Validate inputs
    available_cols = [c for c in feature_columns if c in df.columns]
    if not available_cols:
        raise ValueError(f"None of the specified columns found in DataFrame: {feature_columns}")
    
    # Extract and prepare data
    X = df[available_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Adjust n_neighbors if necessary
    n_neighbors = min(n_neighbors, len(X) - 1)
    n_neighbors = max(n_neighbors, 2)
    
    # Create and fit model
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
        n_jobs=-1
    )
    
    # Fit and predict
    predictions = model.fit_predict(X_scaled)
    
    # Get LOF scores (negative_outlier_factor_)
    raw_scores = model.negative_outlier_factor_
    
    # Normalize scores to -1 to 1 range
    scores_min, scores_max = raw_scores.min(), raw_scores.max()
    if scores_max - scores_min > 0:
        normalized_scores = 2 * (raw_scores - scores_min) / (scores_max - scores_min) - 1
    else:
        normalized_scores = np.zeros_like(raw_scores)
    
    # Create anomaly mask
    anomaly_mask = predictions == -1
    num_anomalies = np.sum(anomaly_mask)
    
    # Calculate feature importance
    feature_importance = {}
    if num_anomalies > 0:
        normal_data = X_scaled[~anomaly_mask]
        anomaly_data = X_scaled[anomaly_mask]
        
        for i, col in enumerate(available_cols):
            normal_mean = normal_data[:, i].mean() if len(normal_data) > 0 else 0
            normal_std = normal_data[:, i].std() if len(normal_data) > 0 else 1
            
            if normal_std > 0:
                z_scores = np.abs((anomaly_data[:, i] - normal_mean) / normal_std)
                feature_importance[col] = float(np.mean(z_scores))
            else:
                feature_importance[col] = 0.0
    else:
        feature_importance = {col: 0.0 for col in available_cols}
    
    # Normalize feature importance
    total_importance = sum(feature_importance.values())
    if total_importance > 0:
        feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
    
    # Calculate confidence
    confidence = calculate_detection_confidence(df, available_cols, num_anomalies)
    
    return OutlierDetectionResult(
        anomaly_mask=anomaly_mask,
        anomaly_scores=normalized_scores,
        num_anomalies=num_anomalies,
        contamination_actual=num_anomalies / len(df) if len(df) > 0 else 0.0,
        method='lof',
        feature_columns=available_cols,
        confidence=confidence,
        feature_importance=feature_importance
    )


def detect_outliers_abod(
    df: pd.DataFrame,
    feature_columns: List[str],
    contamination: float = 0.05,
    n_neighbors: int = 10
) -> OutlierDetectionResult:
    """
    Detect outliers using Angular-Based Outlier Detection (ABOD) algorithm.
    
    ABOD uses the variance of angles between a point and all pairs of other
    points. Outliers have smaller angle variance because they lie in a
    different region of the feature space.
    
    This is particularly effective for high-dimensional data and detects
    outliers that are not easily found by distance-based methods.
    
    Args:
        df: DataFrame with well log data
        feature_columns: List of curve names to use for detection
        contamination: Expected proportion of outliers (0.0 to 0.5)
        n_neighbors: Number of neighbors for fast ABOD approximation
        
    Returns:
        OutlierDetectionResult with detection results and metrics
        
    Raises:
        ImportError: If PyOD is not installed
    """
    if not PYOD_AVAILABLE:
        raise ImportError(
            "PyOD library is required for ABOD. Install with: pip install pyod"
        )
    
    # Validate inputs
    available_cols = [c for c in feature_columns if c in df.columns]
    if not available_cols:
        raise ValueError(f"None of the specified columns found in DataFrame: {feature_columns}")
    
    # Extract and prepare data
    X = df[available_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Adjust n_neighbors if necessary
    n_neighbors = min(n_neighbors, len(X) - 1)
    n_neighbors = max(n_neighbors, 3)
    
    # Create and fit ABOD model
    # Use FastABOD (n_neighbors > 0) for efficiency on larger datasets
    model = ABOD(
        contamination=contamination,
        n_neighbors=n_neighbors,
        method='fast'  # Use fast approximation for efficiency
    )
    
    # Fit and predict
    model.fit(X_scaled)
    predictions = model.labels_  # 0 for inliers, 1 for outliers
    
    # Get decision scores (lower = more anomalous for ABOD)
    raw_scores = model.decision_scores_
    
    # Normalize scores to -1 to 1 range (more negative = more anomalous)
    # Note: For ABOD, higher decision_score = more anomalous
    scores_min, scores_max = raw_scores.min(), raw_scores.max()
    if scores_max - scores_min > 0:
        # Invert so that -1 = most anomalous, 1 = most normal
        normalized_scores = 1 - 2 * (raw_scores - scores_min) / (scores_max - scores_min)
    else:
        normalized_scores = np.zeros_like(raw_scores)
    
    # Create anomaly mask (1 = outlier in PyOD)
    anomaly_mask = predictions == 1
    num_anomalies = np.sum(anomaly_mask)
    
    # Calculate feature importance
    feature_importance = {}
    if num_anomalies > 0:
        normal_data = X_scaled[~anomaly_mask]
        anomaly_data = X_scaled[anomaly_mask]
        
        for i, col in enumerate(available_cols):
            normal_mean = normal_data[:, i].mean() if len(normal_data) > 0 else 0
            normal_std = normal_data[:, i].std() if len(normal_data) > 0 else 1
            
            if normal_std > 0:
                z_scores = np.abs((anomaly_data[:, i] - normal_mean) / normal_std)
                feature_importance[col] = float(np.mean(z_scores))
            else:
                feature_importance[col] = 0.0
    else:
        feature_importance = {col: 0.0 for col in available_cols}
    
    # Normalize feature importance
    total_importance = sum(feature_importance.values())
    if total_importance > 0:
        feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
    
    # Calculate confidence
    confidence = calculate_detection_confidence(df, available_cols, num_anomalies)
    
    return OutlierDetectionResult(
        anomaly_mask=anomaly_mask,
        anomaly_scores=normalized_scores,
        num_anomalies=num_anomalies,
        contamination_actual=num_anomalies / len(df) if len(df) > 0 else 0.0,
        method='abod',
        feature_columns=available_cols,
        confidence=confidence,
        feature_importance=feature_importance
    )


def calculate_detection_confidence(
    df: pd.DataFrame,
    feature_columns: List[str],
    num_anomalies: int
) -> float:
    """
    Calculate confidence score for outlier detection based on data quality.
    
    Factors considered:
    - Sample size (more samples = higher confidence)
    - Missing value ratio
    - Feature variance (low variance = less reliable)
    """
    confidence = 1.0
    
    # Sample size factor
    n_samples = len(df)
    if n_samples < 50:
        confidence *= 0.6
    elif n_samples < 100:
        confidence *= 0.8
    elif n_samples < 500:
        confidence *= 0.9
    
    # Missing value penalty
    X = df[feature_columns]
    missing_ratio = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
    confidence *= (1 - missing_ratio * 0.5)
    
    # Variance check - very low variance features are unreliable
    for col in feature_columns:
        if col in df.columns:
            col_std = df[col].std()
            if col_std is not None and col_std < 0.001:
                confidence *= 0.9
    
    # Anomaly ratio sanity check
    if n_samples > 0:
        anomaly_ratio = num_anomalies / n_samples
        if anomaly_ratio > 0.3:  # Suspiciously high
            confidence *= 0.7
    
    return min(max(confidence, 0.0), 1.0)


def detect_outliers_ensemble(
    df: pd.DataFrame,
    feature_columns: List[str],
    contamination: float = 0.05,
    include_abod: bool = False
) -> OutlierDetectionResult:
    """
    Ensemble method combining multiple outlier detection algorithms.
    
    A point is considered an outlier if multiple methods agree.
    This reduces false positives at the cost of some sensitivity.
    
    Args:
        df: DataFrame with well log data
        feature_columns: List of curve names to use
        contamination: Expected proportion of outliers
        include_abod: Whether to include ABOD in the ensemble (requires PyOD)
        
    Returns:
        OutlierDetectionResult with combined results
    """
    # Run core methods
    result_if = detect_outliers_isolation_forest(df, feature_columns, contamination)
    result_lof = detect_outliers_lof(df, feature_columns, contamination=contamination)
    
    results = [result_if, result_lof]
    method_name = 'ensemble_if_lof'
    
    # Optionally include ABOD
    if include_abod and PYOD_AVAILABLE:
        try:
            result_abod = detect_outliers_abod(df, feature_columns, contamination=contamination)
            results.append(result_abod)
            method_name = 'ensemble_if_lof_abod'
        except Exception:
            pass  # Fall back to IF + LOF only
    
    # Ensemble: require majority agreement for outliers
    n_methods = len(results)
    vote_threshold = n_methods // 2 + 1  # Majority vote
    
    # Count votes for each point
    votes = np.zeros(len(df), dtype=int)
    score_sum = np.zeros(len(df), dtype=float)
    
    for result in results:
        votes += result.anomaly_mask.astype(int)
        score_sum += result.anomaly_scores
    
    # Points are outliers if they get majority votes
    combined_mask = votes >= vote_threshold
    combined_scores = score_sum / n_methods
    
    num_anomalies = np.sum(combined_mask)
    
    # Combine feature importance from all methods
    feature_importance = {}
    all_cols = set()
    for result in results:
        all_cols.update(result.feature_importance.keys())
    
    for col in all_cols:
        importances = [r.feature_importance.get(col, 0) for r in results]
        feature_importance[col] = sum(importances) / len(importances)
    
    # Take max confidence from all methods
    max_confidence = max(r.confidence for r in results)
    
    return OutlierDetectionResult(
        anomaly_mask=combined_mask,
        anomaly_scores=combined_scores,
        num_anomalies=num_anomalies,
        contamination_actual=num_anomalies / len(df) if len(df) > 0 else 0.0,
        method=method_name,
        feature_columns=feature_columns,
        confidence=max_confidence,
        feature_importance=feature_importance
    )


def clean_outliers(
    df: pd.DataFrame,
    result: OutlierDetectionResult,
    method: str = 'interpolate'
) -> pd.DataFrame:
    """
    Clean detected outliers from the data.
    
    Args:
        df: Original DataFrame
        result: OutlierDetectionResult from detection
        method: 'interpolate', 'median', or 'remove'
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if method == 'remove':
        # Simply remove outlier rows
        df_clean = df_clean[~result.anomaly_mask].reset_index(drop=True)
        
    elif method == 'median':
        # Replace outliers with column median
        for col in result.feature_columns:
            if col in df_clean.columns:
                median_val = df_clean.loc[~result.anomaly_mask, col].median()
                df_clean.loc[result.anomaly_mask, col] = median_val
                
    elif method == 'interpolate':
        # Interpolate outliers from surrounding values
        for col in result.feature_columns:
            if col in df_clean.columns:
                # Set outliers to NaN
                df_clean.loc[result.anomaly_mask, col] = np.nan
                # Interpolate
                df_clean[col] = df_clean[col].interpolate(method='linear')
                # Fill any remaining NaN at edges with median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean
