"""
Tool Startup Noise Detection and Removal for Well Log Data

Implements algorithms to detect and remove tool startup noise, which
typically appears as constant/flat values or straight lines at the
beginning of logging runs.

Method: Rolling variance + slope analysis to identify low-variation
regions that indicate tool warmup or calibration periods.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class NoiseDetectionResult:
    """Container for noise detection results."""
    noise_mask: np.ndarray  # Boolean mask: True = noise
    rolling_variance: np.ndarray  # Rolling variance values
    rolling_slope: np.ndarray  # Rolling slope/gradient values
    noise_depth_start: float  # Start depth of noise region
    noise_depth_end: float  # End depth of noise region
    noise_samples: int  # Number of noisy samples
    noise_percentage: float  # Percentage of data flagged as noise
    feature_columns: List[str]  # Columns analyzed


@dataclass
class NoiseRemovalResult:
    """Container for noise removal operation results."""
    cleaned_df: pd.DataFrame  # DataFrame with noise removed/cleaned
    original_samples: int  # Original number of samples
    removed_samples: int  # Number of samples removed/modified
    method: str  # 'remove', 'interpolate', or 'trim'
    noise_result: NoiseDetectionResult  # Detection result reference


def detect_tool_startup_noise(
    df: pd.DataFrame,
    feature_columns: List[str],
    depth_column: str = 'DEPTH',
    window: int = 10,
    variance_threshold: float = 0.01,
    slope_threshold: float = 0.001,
    min_noise_samples: int = 5,
    check_start_only: bool = True
) -> NoiseDetectionResult:
    """
    Detect tool startup noise characterized by constant/flat values.
    
    Tool startup noise typically appears at the beginning of a log as:
    - Constant values (zero variance)
    - Straight lines (constant slope near zero)
    - Values that haven't stabilized to formation response
    
    Args:
        df: DataFrame with well log data
        feature_columns: Curves to analyze for noise
        depth_column: Name of depth column
        window: Window size for rolling calculations
        variance_threshold: Max variance to be considered "flat" (normalized)
        slope_threshold: Max slope to be considered "constant" (normalized)
        min_noise_samples: Minimum samples to flag as noise region
        check_start_only: Only check from start of log (True) or both ends (False)
        
    Returns:
        NoiseDetectionResult with detection details
    """
    # Validate inputs
    available_cols = [c for c in feature_columns if c in df.columns]
    if not available_cols:
        raise ValueError(f"None of the specified columns found: {feature_columns}")
    
    if depth_column not in df.columns:
        raise ValueError(f"Depth column '{depth_column}' not found")
    
    n_samples = len(df)
    
    # Initialize combined noise mask
    combined_noise_mask = np.zeros(n_samples, dtype=bool)
    all_variances = []
    all_slopes = []
    
    for col in available_cols:
        signal = df[col].values.copy()
        
        # Handle NaN values
        signal_series = pd.Series(signal)
        signal_filled = signal_series.fillna(method='ffill').fillna(method='bfill').fillna(0)
        signal_clean = signal_filled.values
        
        # Normalize signal for consistent thresholds
        signal_std = np.nanstd(signal_clean)
        if signal_std > 0:
            signal_normalized = (signal_clean - np.nanmean(signal_clean)) / signal_std
        else:
            signal_normalized = signal_clean
        
        # Calculate rolling variance
        rolling_var = pd.Series(signal_normalized).rolling(
            window=window, center=False, min_periods=1
        ).var().fillna(0).values
        
        # Calculate rolling slope (gradient)
        rolling_slope = np.zeros_like(signal_normalized)
        for i in range(window, len(signal_normalized)):
            window_data = signal_normalized[i-window:i]
            if len(window_data) >= 2:
                # Linear regression slope
                x = np.arange(len(window_data))
                slope = np.polyfit(x, window_data, 1)[0] if len(window_data) > 1 else 0
                rolling_slope[i] = abs(slope)
        
        # First few samples get the first calculated values
        rolling_slope[:window] = rolling_slope[window] if window < len(rolling_slope) else 0
        
        all_variances.append(rolling_var)
        all_slopes.append(rolling_slope)
        
        # Detect flat/constant regions
        low_variance = rolling_var < variance_threshold
        low_slope = rolling_slope < slope_threshold
        noise_mask = low_variance & low_slope
        
        # Combine with OR logic (noise in any column)
        combined_noise_mask |= noise_mask
    
    # Average variance and slope across all columns for visualization
    avg_variance = np.mean(all_variances, axis=0) if all_variances else np.zeros(n_samples)
    avg_slope = np.mean(all_slopes, axis=0) if all_slopes else np.zeros(n_samples)
    
    if check_start_only:
        # Only keep noise mask from the start until first valid data
        # Find first position where we have valid (non-noise) data
        if combined_noise_mask.any():
            # Find contiguous noise region from start
            first_valid = _find_first_valid_index(combined_noise_mask)
            startup_mask = np.zeros(n_samples, dtype=bool)
            if first_valid > min_noise_samples:
                startup_mask[:first_valid] = True
            combined_noise_mask = startup_mask
    
    # Apply minimum noise samples threshold
    if np.sum(combined_noise_mask) < min_noise_samples:
        combined_noise_mask = np.zeros(n_samples, dtype=bool)
    
    # Calculate noise region boundaries
    depth_values = df[depth_column].values
    noise_indices = np.where(combined_noise_mask)[0]
    
    if len(noise_indices) > 0:
        noise_depth_start = depth_values[noise_indices[0]]
        noise_depth_end = depth_values[noise_indices[-1]]
        noise_samples = len(noise_indices)
    else:
        noise_depth_start = depth_values[0]
        noise_depth_end = depth_values[0]
        noise_samples = 0
    
    noise_percentage = (noise_samples / n_samples * 100) if n_samples > 0 else 0.0
    
    return NoiseDetectionResult(
        noise_mask=combined_noise_mask,
        rolling_variance=avg_variance,
        rolling_slope=avg_slope,
        noise_depth_start=noise_depth_start,
        noise_depth_end=noise_depth_end,
        noise_samples=noise_samples,
        noise_percentage=noise_percentage,
        feature_columns=available_cols
    )


def _find_first_valid_index(noise_mask: np.ndarray) -> int:
    """Find the first index where data transitions from noise to valid."""
    # Look for first False (valid) value
    for i, is_noise in enumerate(noise_mask):
        if not is_noise:
            return i
    return len(noise_mask)


def detect_tool_shutdown_noise(
    df: pd.DataFrame,
    feature_columns: List[str],
    depth_column: str = 'DEPTH',
    window: int = 10,
    variance_threshold: float = 0.01,
    slope_threshold: float = 0.001,
    min_noise_samples: int = 5
) -> NoiseDetectionResult:
    """
    Detect tool shutdown noise at the end of a log.
    
    Similar to startup noise but at the bottom of the log.
    
    Args:
        df: DataFrame with well log data
        feature_columns: Curves to analyze
        depth_column: Name of depth column
        window: Window size for rolling calculations
        variance_threshold: Max variance threshold
        slope_threshold: Max slope threshold
        min_noise_samples: Minimum samples to flag
        
    Returns:
        NoiseDetectionResult for end-of-log noise
    """
    # Detect noise without start-only restriction
    result = detect_tool_startup_noise(
        df, feature_columns, depth_column,
        window, variance_threshold, slope_threshold,
        min_noise_samples, check_start_only=False
    )
    
    # Only keep noise from the end
    n_samples = len(df)
    noise_mask = result.noise_mask.copy()
    
    if noise_mask.any():
        # Find last contiguous noise region
        last_valid = _find_last_valid_index(noise_mask)
        shutdown_mask = np.zeros(n_samples, dtype=bool)
        if (n_samples - last_valid - 1) > min_noise_samples:
            shutdown_mask[last_valid+1:] = True
        noise_mask = shutdown_mask
    
    # Recalculate boundaries
    depth_values = df[depth_column].values
    noise_indices = np.where(noise_mask)[0]
    
    if len(noise_indices) > 0:
        noise_depth_start = depth_values[noise_indices[0]]
        noise_depth_end = depth_values[noise_indices[-1]]
        noise_samples = len(noise_indices)
    else:
        noise_depth_start = depth_values[-1]
        noise_depth_end = depth_values[-1]
        noise_samples = 0
    
    noise_percentage = (noise_samples / n_samples * 100) if n_samples > 0 else 0.0
    
    return NoiseDetectionResult(
        noise_mask=noise_mask,
        rolling_variance=result.rolling_variance,
        rolling_slope=result.rolling_slope,
        noise_depth_start=noise_depth_start,
        noise_depth_end=noise_depth_end,
        noise_samples=noise_samples,
        noise_percentage=noise_percentage,
        feature_columns=result.feature_columns
    )


def _find_last_valid_index(noise_mask: np.ndarray) -> int:
    """Find the last index where data is valid (not noise)."""
    for i in range(len(noise_mask) - 1, -1, -1):
        if not noise_mask[i]:
            return i
    return -1


def remove_noise(
    df: pd.DataFrame,
    noise_result: NoiseDetectionResult,
    method: str = 'trim',
    depth_column: str = 'DEPTH'
) -> NoiseRemovalResult:
    """
    Remove detected noise from the DataFrame.
    
    Args:
        df: Original DataFrame
        noise_result: NoiseDetectionResult from detection
        method: Removal method:
            - 'trim': Remove noisy rows entirely
            - 'interpolate': Replace noisy values with interpolated values
            - 'nan': Replace noisy values with NaN
        depth_column: Name of depth column
        
    Returns:
        NoiseRemovalResult with cleaned data
    """
    original_samples = len(df)
    noise_mask = noise_result.noise_mask
    
    if method == 'trim':
        # Remove noisy rows entirely
        cleaned_df = df[~noise_mask].reset_index(drop=True)
        removed_samples = np.sum(noise_mask)
        
    elif method == 'interpolate':
        # Replace noisy values with interpolated values
        cleaned_df = df.copy()
        for col in noise_result.feature_columns:
            if col in cleaned_df.columns:
                # Set noisy values to NaN
                cleaned_df.loc[noise_mask, col] = np.nan
                # Interpolate
                cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
                # Fill edges with first/last valid value
                cleaned_df[col] = cleaned_df[col].fillna(method='bfill').fillna(method='ffill')
        removed_samples = 0  # No rows removed, just modified
        
    elif method == 'nan':
        # Replace noisy values with NaN
        cleaned_df = df.copy()
        for col in noise_result.feature_columns:
            if col in cleaned_df.columns:
                cleaned_df.loc[noise_mask, col] = np.nan
        removed_samples = 0  # No rows removed
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trim', 'interpolate', or 'nan'")
    
    return NoiseRemovalResult(
        cleaned_df=cleaned_df,
        original_samples=original_samples,
        removed_samples=removed_samples,
        method=method,
        noise_result=noise_result
    )


def detect_spike_noise(
    df: pd.DataFrame,
    feature_columns: List[str],
    window: int = 5,
    std_threshold: float = 3.0
) -> np.ndarray:
    """
    Detect spike noise (sudden jumps in values).
    
    Spikes are detected using a moving median filter and flagging
    points that deviate significantly from the local median.
    
    Args:
        df: DataFrame with well log data
        feature_columns: Curves to analyze
        window: Window size for median filter
        std_threshold: Number of standard deviations for spike detection
        
    Returns:
        Boolean mask where True indicates a spike
    """
    n_samples = len(df)
    spike_mask = np.zeros(n_samples, dtype=bool)
    
    available_cols = [c for c in feature_columns if c in df.columns]
    
    for col in available_cols:
        signal = df[col].values.copy()
        signal_series = pd.Series(signal)
        
        # Calculate rolling median and std
        rolling_median = signal_series.rolling(window=window, center=True, min_periods=1).median()
        rolling_std = signal_series.rolling(window=window, center=True, min_periods=1).std()
        
        # Detect spikes: points far from local median
        deviation = np.abs(signal_series - rolling_median)
        is_spike = deviation > (std_threshold * rolling_std)
        
        # Handle NaN
        is_spike = is_spike.fillna(False).values
        
        spike_mask |= is_spike
    
    return spike_mask


def despike_signal(
    df: pd.DataFrame,
    feature_columns: List[str],
    spike_mask: np.ndarray,
    method: str = 'median'
) -> pd.DataFrame:
    """
    Remove spike noise from signals.
    
    Args:
        df: DataFrame with well log data
        feature_columns: Curves to despike
        spike_mask: Boolean mask of spikes
        method: 'median' (replace with rolling median) or 'interpolate'
        
    Returns:
        DataFrame with spikes removed
    """
    df_clean = df.copy()
    available_cols = [c for c in feature_columns if c in df.columns]
    
    for col in available_cols:
        if method == 'median':
            # Replace spikes with rolling median
            rolling_median = df[col].rolling(window=5, center=True, min_periods=1).median()
            df_clean.loc[spike_mask, col] = rolling_median[spike_mask]
        elif method == 'interpolate':
            # Set spikes to NaN and interpolate
            df_clean.loc[spike_mask, col] = np.nan
            df_clean[col] = df_clean[col].interpolate(method='linear')
            df_clean[col] = df_clean[col].fillna(method='bfill').fillna(method='ffill')
    
    return df_clean


def get_noise_quality_report(
    noise_result: NoiseDetectionResult,
    df: pd.DataFrame,
    depth_column: str = 'DEPTH'
) -> Dict[str, any]:
    """
    Generate a quality report for noise detection results.
    
    Args:
        noise_result: NoiseDetectionResult from detection
        df: Original DataFrame
        depth_column: Name of depth column
        
    Returns:
        Dictionary with quality metrics
    """
    depth_values = df[depth_column].values
    total_depth = depth_values[-1] - depth_values[0]
    noise_depth_range = noise_result.noise_depth_end - noise_result.noise_depth_start
    
    return {
        'total_samples': len(df),
        'noise_samples': noise_result.noise_samples,
        'noise_percentage': noise_result.noise_percentage,
        'total_depth_range': total_depth,
        'noise_depth_range': noise_depth_range,
        'noise_depth_percentage': (noise_depth_range / total_depth * 100) if total_depth > 0 else 0,
        'noise_start_depth': noise_result.noise_depth_start,
        'noise_end_depth': noise_result.noise_depth_end,
        'analyzed_curves': noise_result.feature_columns,
        'avg_variance_in_noise': np.mean(noise_result.rolling_variance[noise_result.noise_mask]) if noise_result.noise_samples > 0 else 0,
        'avg_slope_in_noise': np.mean(noise_result.rolling_slope[noise_result.noise_mask]) if noise_result.noise_samples > 0 else 0
    }

