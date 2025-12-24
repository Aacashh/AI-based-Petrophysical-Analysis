"""
ML-Based Log Splicing Module

Implements the Simple ML-Based Log Splicing Workflow:
1. Overlap detection - Find common depth (Rule-based logic)
2. Resampling - Align samples (Linear interpolation)
3. QC - Remove spikes (Hampel filter)
4. Stability metric - Detect bad data (Rolling variance)
5. Splice detection - Pick depth (CPD – PELT)
6. Run selection - Decide trust (Rule-based or RF/GBDT)
7. Transition - Smooth join (Weighted blending)

This module provides an industry-standard ML-enhanced approach to log splicing
that handles outliers, detects optimal splice points using change point detection,
and creates smooth transitions between log segments.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.ndimage import median_filter
from typing import Tuple, Dict, Optional, List, NamedTuple
from dataclasses import dataclass, field
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MLSplicingResult:
    """Container for ML-based splicing operation results."""
    # Final merged data
    merged_depth: np.ndarray
    merged_signal: np.ndarray
    
    # Quality metrics
    splice_point: float
    splice_quality_score: float  # 0-1, higher is better
    
    # Overlap info
    overlap_start: float
    overlap_end: float
    
    # QC metrics
    shallow_stability: float  # Rolling variance metric
    deep_stability: float
    
    # Detected change points
    change_points: List[float]  # Depths of detected change points
    
    # Run selection info
    trust_shallow: float  # 0-1, confidence in shallow run
    trust_deep: float  # 0-1, confidence in deep run
    
    # Blending zone
    blend_start: float
    blend_end: float


@dataclass
class QCResult:
    """Container for QC (despiking) results."""
    cleaned_signal: np.ndarray
    spike_mask: np.ndarray  # True where spikes were detected
    num_spikes_removed: int


@dataclass
class StabilityResult:
    """Container for stability metric results."""
    rolling_variance: np.ndarray
    mean_stability: float  # Average 1/(1 + variance)
    bad_zones: np.ndarray  # Boolean mask for unstable zones


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_GRID_STEP = 0.1524  # meters (0.5 feet)
DEFAULT_HAMPEL_WINDOW = 7  # samples for Hampel filter
DEFAULT_HAMPEL_THRESHOLD = 3.0  # MAD multiplier
DEFAULT_VARIANCE_WINDOW = 21  # samples for rolling variance
DEFAULT_BLEND_ZONE = 3.0  # meters for blending transition
DEFAULT_STABILITY_THRESHOLD = 0.5  # Threshold for "bad" data


# =============================================================================
# STEP 1: OVERLAP DETECTION (Rule-based logic)
# =============================================================================

def find_overlap_region(
    shallow_depth: np.ndarray, 
    deep_depth: np.ndarray
) -> Tuple[float, float, float]:
    """
    Find the overlapping depth region between shallow and deep logs.
    
    Uses rule-based logic:
    - Overlap start = max(min_shallow, min_deep)
    - Overlap end = min(max_shallow, max_deep)
    
    Args:
        shallow_depth: Depth array from shallow log
        deep_depth: Depth array from deep log
        
    Returns:
        Tuple of (overlap_start, overlap_end, overlap_length)
        
    Raises:
        ValueError: If no overlap exists
    """
    # Get valid depth ranges
    shallow_min = np.nanmin(shallow_depth)
    shallow_max = np.nanmax(shallow_depth)
    deep_min = np.nanmin(deep_depth)
    deep_max = np.nanmax(deep_depth)
    
    # Calculate overlap
    overlap_start = max(shallow_min, deep_min)
    overlap_end = min(shallow_max, deep_max)
    overlap_length = overlap_end - overlap_start
    
    if overlap_length <= 0:
        raise ValueError(
            f"No overlap between logs. "
            f"Shallow: {shallow_min:.1f}-{shallow_max:.1f}m, "
            f"Deep: {deep_min:.1f}-{deep_max:.1f}m"
        )
    
    return overlap_start, overlap_end, overlap_length


# =============================================================================
# STEP 2: RESAMPLING (Linear interpolation)
# =============================================================================

def resample_to_common_grid(
    depth: np.ndarray, 
    signal: np.ndarray,
    target_grid: np.ndarray
) -> np.ndarray:
    """
    Resample signal to a common depth grid using linear interpolation.
    
    Args:
        depth: Original depth array
        signal: Original signal array
        target_grid: Target depth grid
        
    Returns:
        Resampled signal on target grid
    """
    # Handle NaN values
    valid_mask = ~np.isnan(signal) & ~np.isnan(depth)
    
    if np.sum(valid_mask) < 2:
        return np.full_like(target_grid, np.nan)
    
    valid_depth = depth[valid_mask]
    valid_signal = signal[valid_mask]
    
    # Sort by depth (required for np.interp)
    sort_idx = np.argsort(valid_depth)
    valid_depth = valid_depth[sort_idx]
    valid_signal = valid_signal[sort_idx]
    
    # Linear interpolation - NaN outside data range
    resampled = np.interp(
        target_grid, 
        valid_depth, 
        valid_signal,
        left=np.nan, 
        right=np.nan
    )
    
    return resampled


def create_common_grid(
    depth1: np.ndarray, 
    depth2: np.ndarray,
    step: float = DEFAULT_GRID_STEP
) -> np.ndarray:
    """
    Create a common depth grid covering both logs.
    
    Args:
        depth1: First depth array
        depth2: Second depth array
        step: Grid step in meters
        
    Returns:
        Common depth grid
    """
    min_depth = min(np.nanmin(depth1), np.nanmin(depth2))
    max_depth = max(np.nanmax(depth1), np.nanmax(depth2))
    
    # Round to nice values
    min_depth = np.floor(min_depth / step) * step
    max_depth = np.ceil(max_depth / step) * step
    
    return np.arange(min_depth, max_depth + step / 2, step)


# =============================================================================
# STEP 3: QC - HAMPEL FILTER (Remove spikes)
# =============================================================================

def hampel_filter(
    signal: np.ndarray,
    window_size: int = DEFAULT_HAMPEL_WINDOW,
    threshold: float = DEFAULT_HAMPEL_THRESHOLD
) -> QCResult:
    """
    Apply Hampel filter to remove spikes from the signal.
    
    The Hampel filter is a robust outlier detection method that uses
    the Median Absolute Deviation (MAD) to identify spikes.
    
    Formula:
        MAD = median(|x_i - median(x)|)
        Spike if |x_i - median(window)| > threshold * MAD * 1.4826
        
    The factor 1.4826 makes MAD consistent with standard deviation
    for normally distributed data.
    
    Args:
        signal: Input signal array
        window_size: Size of the sliding window (must be odd)
        threshold: Number of MAD units for spike detection
        
    Returns:
        QCResult with cleaned signal and spike information
    """
    n = len(signal)
    cleaned = signal.copy()
    spike_mask = np.zeros(n, dtype=bool)
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    
    # MAD scaling factor for normal distribution
    k = 1.4826
    
    for i in range(n):
        # Define window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        window = signal[start:end]
        
        # Skip if window contains only NaN
        valid_window = window[~np.isnan(window)]
        if len(valid_window) < 3:
            continue
        
        # Calculate median and MAD
        median_val = np.median(valid_window)
        mad = np.median(np.abs(valid_window - median_val))
        
        # Avoid division by zero
        if mad < 1e-10:
            mad = np.std(valid_window) / k
            if mad < 1e-10:
                continue
        
        # Check if current point is a spike
        if not np.isnan(signal[i]):
            if np.abs(signal[i] - median_val) > threshold * k * mad:
                spike_mask[i] = True
                cleaned[i] = median_val  # Replace with median
    
    return QCResult(
        cleaned_signal=cleaned,
        spike_mask=spike_mask,
        num_spikes_removed=int(np.sum(spike_mask))
    )


# =============================================================================
# STEP 4: STABILITY METRIC (Rolling variance)
# =============================================================================

def calculate_rolling_variance(
    signal: np.ndarray,
    window_size: int = DEFAULT_VARIANCE_WINDOW
) -> StabilityResult:
    """
    Calculate rolling variance to detect unstable/bad data zones.
    
    High variance indicates noise, tool problems, or wash-out zones.
    Low variance indicates stable, trustworthy readings.
    
    Stability metric = 1 / (1 + variance)
    Higher values = more stable data
    
    Args:
        signal: Input signal array (should be cleaned first)
        window_size: Size of rolling window
        
    Returns:
        StabilityResult with variance and stability metrics
    """
    n = len(signal)
    rolling_var = np.full(n, np.nan)
    
    half_window = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        window = signal[start:end]
        valid_window = window[~np.isnan(window)]
        
        if len(valid_window) >= 3:
            rolling_var[i] = np.var(valid_window)
    
    # Calculate stability metric (inverse of variance)
    # Normalize variance first
    var_max = np.nanmax(rolling_var)
    if var_max > 0:
        norm_var = rolling_var / var_max
    else:
        norm_var = np.zeros_like(rolling_var)
    
    stability = 1.0 / (1.0 + norm_var * 10)  # Scale factor for sensitivity
    
    # Mean stability (ignoring NaN)
    mean_stability = np.nanmean(stability)
    
    # Identify bad zones (high variance)
    var_threshold = np.nanpercentile(rolling_var[~np.isnan(rolling_var)], 90)
    bad_zones = rolling_var > var_threshold
    
    return StabilityResult(
        rolling_variance=rolling_var,
        mean_stability=mean_stability,
        bad_zones=bad_zones
    )


# =============================================================================
# STEP 5: SPLICE DETECTION - CPD-PELT (Change Point Detection)
# =============================================================================

def detect_change_points_pelt(
    signal: np.ndarray,
    depth: np.ndarray,
    min_segment_length: int = 10,
    penalty: float = None
) -> List[float]:
    """
    Detect change points using PELT (Pruned Exact Linear Time) algorithm.
    
    PELT is an efficient exact segmentation algorithm that identifies
    points where the statistical properties of the signal change.
    This is used to find natural splice points where the transition
    between logs would be least noticeable.
    
    Implementation uses a simplified version based on cumulative sum
    and variance changes when ruptures library is not available.
    
    Args:
        signal: Input signal array
        depth: Corresponding depth array
        min_segment_length: Minimum samples between change points
        penalty: Penalty for adding change points (auto-computed if None)
        
    Returns:
        List of depths where change points were detected
    """
    # Remove NaN for analysis
    valid_mask = ~np.isnan(signal)
    valid_signal = signal[valid_mask]
    valid_depth = depth[valid_mask]
    
    if len(valid_signal) < min_segment_length * 2:
        return []
    
    # Try to use ruptures library for proper PELT
    try:
        import ruptures as rpt
        
        # Auto-compute penalty based on signal statistics
        if penalty is None:
            penalty = np.log(len(valid_signal)) * np.var(valid_signal)
        
        # PELT with RBF kernel
        algo = rpt.Pelt(model="rbf", min_size=min_segment_length).fit(valid_signal)
        change_indices = algo.predict(pen=penalty)
        
        # Remove the last index (always equal to len(signal))
        change_indices = [idx for idx in change_indices if idx < len(valid_signal)]
        
        # Convert to depths
        change_depths = [valid_depth[idx] if idx < len(valid_depth) else valid_depth[-1] 
                        for idx in change_indices]
        
        return change_depths
        
    except ImportError:
        # Fallback: Simple change point detection using CUSUM
        return _cusum_change_points(valid_signal, valid_depth, min_segment_length)


def _cusum_change_points(
    signal: np.ndarray,
    depth: np.ndarray,
    min_segment_length: int
) -> List[float]:
    """
    Fallback change point detection using CUSUM (Cumulative Sum).
    
    Detects points where the cumulative deviation from the mean
    changes direction significantly.
    """
    n = len(signal)
    if n < min_segment_length * 2:
        return []
    
    # Normalize signal
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-10:
        return []
    
    normalized = (signal - mean) / std
    
    # Calculate CUSUM
    cusum = np.cumsum(normalized)
    
    # Find peaks in absolute CUSUM (potential change points)
    abs_cusum = np.abs(cusum - np.mean(cusum))
    
    # Find local maxima
    change_indices = []
    for i in range(min_segment_length, n - min_segment_length):
        window = abs_cusum[i - min_segment_length:i + min_segment_length]
        if abs_cusum[i] == np.max(window):
            change_indices.append(i)
    
    # Filter to significant changes (top 3 or fewer)
    if len(change_indices) > 3:
        scores = [abs_cusum[i] for i in change_indices]
        sorted_indices = np.argsort(scores)[::-1][:3]
        change_indices = [change_indices[i] for i in sorted(sorted_indices)]
    
    # Convert to depths
    change_depths = [depth[idx] for idx in change_indices]
    
    return change_depths


def find_optimal_splice_point(
    shallow_signal: np.ndarray,
    deep_signal: np.ndarray,
    depth: np.ndarray,
    overlap_start: float,
    overlap_end: float,
    shallow_stability: np.ndarray,
    deep_stability: np.ndarray,
    change_points: List[float]
) -> Tuple[float, float]:
    """
    Find the optimal splice point within the overlap region.
    
    Selection criteria (in order of priority):
    1. Closest to a detected change point (natural transition)
    2. Where both signals have similar values (smooth transition)
    3. Where both signals are stable (low variance)
    4. Default to midpoint if no good candidate
    
    Args:
        shallow_signal: Signal from shallow log (on common grid)
        deep_signal: Signal from deep log (on common grid)
        depth: Common depth grid
        overlap_start: Start of overlap region
        overlap_end: End of overlap region
        shallow_stability: Stability metric for shallow
        deep_stability: Stability metric for deep
        change_points: Detected change points from PELT
        
    Returns:
        Tuple of (optimal_splice_depth, quality_score)
    """
    # Get overlap region indices
    overlap_mask = (depth >= overlap_start) & (depth <= overlap_end)
    overlap_indices = np.where(overlap_mask)[0]
    
    if len(overlap_indices) < 2:
        # Return midpoint with low quality
        return (overlap_start + overlap_end) / 2, 0.3
    
    overlap_depths = depth[overlap_mask]
    shallow_overlap = shallow_signal[overlap_mask]
    deep_overlap = deep_signal[overlap_mask]
    
    # Calculate signal difference (want minimum)
    signal_diff = np.abs(shallow_overlap - deep_overlap)
    signal_diff_normalized = signal_diff / (np.nanmax(signal_diff) + 1e-10)
    
    # Combined stability (want maximum)
    combined_stability = (
        shallow_stability[overlap_mask] * deep_stability[overlap_mask]
    )
    combined_stability = np.nan_to_num(combined_stability, nan=0.0)
    
    # Initialize score (higher = better splice point)
    scores = np.zeros(len(overlap_depths))
    
    # Factor 1: Signal similarity (40% weight)
    similarity_score = 1.0 - signal_diff_normalized
    scores += 0.4 * np.nan_to_num(similarity_score, nan=0.0)
    
    # Factor 2: Stability (30% weight)
    scores += 0.3 * combined_stability
    
    # Factor 3: Proximity to change points (30% weight)
    if change_points:
        cp_score = np.zeros(len(overlap_depths))
        for d in overlap_depths:
            min_dist = min(abs(d - cp) for cp in change_points)
            # Gaussian weight - closer to change point is better
            cp_score[overlap_depths == d] = np.exp(-min_dist**2 / 10)
        scores += 0.3 * cp_score
    else:
        # No change points - prefer middle of overlap
        middle = (overlap_start + overlap_end) / 2
        dist_from_middle = np.abs(overlap_depths - middle)
        middle_score = 1.0 - dist_from_middle / (dist_from_middle.max() + 1e-10)
        scores += 0.3 * middle_score
    
    # Find best splice point
    best_idx = np.argmax(scores)
    best_depth = overlap_depths[best_idx]
    best_score = scores[best_idx]
    
    return float(best_depth), float(best_score)


# =============================================================================
# STEP 6: RUN SELECTION (Rule-based trust scoring)
# =============================================================================

def calculate_trust_scores(
    shallow_stability: StabilityResult,
    deep_stability: StabilityResult,
    shallow_signal: np.ndarray,
    deep_signal: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate trust scores for each log run.
    
    Trust is based on:
    - Data stability (lower variance = higher trust)
    - Coverage (fewer NaN values = higher trust)
    - Signal consistency (fewer spikes = higher trust)
    
    Args:
        shallow_stability: Stability result for shallow run
        deep_stability: Stability result for deep run
        shallow_signal: Shallow signal array
        deep_signal: Deep signal array
        
    Returns:
        Tuple of (shallow_trust, deep_trust) scores 0-1
    """
    # Stability component (40% weight)
    shallow_stab = shallow_stability.mean_stability
    deep_stab = deep_stability.mean_stability
    
    # Coverage component (30% weight)
    shallow_coverage = 1.0 - np.mean(np.isnan(shallow_signal))
    deep_coverage = 1.0 - np.mean(np.isnan(deep_signal))
    
    # Consistency component (30% weight) - inverse of bad zones fraction
    shallow_bad_frac = np.mean(shallow_stability.bad_zones)
    deep_bad_frac = np.mean(deep_stability.bad_zones)
    shallow_consistency = 1.0 - shallow_bad_frac
    deep_consistency = 1.0 - deep_bad_frac
    
    # Combined trust scores
    shallow_trust = (
        0.4 * shallow_stab +
        0.3 * shallow_coverage +
        0.3 * shallow_consistency
    )
    
    deep_trust = (
        0.4 * deep_stab +
        0.3 * deep_coverage +
        0.3 * deep_consistency
    )
    
    # Normalize to 0-1
    shallow_trust = np.clip(shallow_trust, 0, 1)
    deep_trust = np.clip(deep_trust, 0, 1)
    
    return float(shallow_trust), float(deep_trust)


# =============================================================================
# STEP 7: TRANSITION - WEIGHTED BLENDING
# =============================================================================

def weighted_blend_transition(
    shallow_signal: np.ndarray,
    deep_signal: np.ndarray,
    depth: np.ndarray,
    splice_point: float,
    blend_zone: float = DEFAULT_BLEND_ZONE,
    shallow_trust: float = 0.5,
    deep_trust: float = 0.5
) -> np.ndarray:
    """
    Create smooth transition using weighted blending.
    
    The blending zone extends equally above and below the splice point.
    Weight transitions smoothly from shallow to deep using a sigmoid function.
    Trust scores influence the blending behavior.
    
    Formula:
        blended = w_shallow * shallow + w_deep * deep
        
    Where weights are computed using sigmoid blending and adjusted by trust.
    
    Args:
        shallow_signal: Signal from shallow log (on common grid)
        deep_signal: Signal from deep log (on common grid)
        depth: Common depth grid
        splice_point: Depth of the splice point
        blend_zone: Width of blending zone in meters
        shallow_trust: Trust score for shallow log
        deep_trust: Trust score for deep log
        
    Returns:
        Blended signal array
    """
    # Initialize with NaN
    blended = np.full_like(depth, np.nan, dtype=float)
    
    # Define blend zone
    blend_start = splice_point - blend_zone / 2
    blend_end = splice_point + blend_zone / 2
    
    # Process each depth
    for i, d in enumerate(depth):
        shallow_val = shallow_signal[i]
        deep_val = deep_signal[i]
        
        if d < blend_start:
            # Pure shallow region
            blended[i] = shallow_val
            
        elif d > blend_end:
            # Pure deep region
            blended[i] = deep_val
            
        else:
            # Blend zone - use sigmoid transition
            # Position in blend zone (0 to 1)
            t = (d - blend_start) / blend_zone
            
            # Sigmoid weight (smooth S-curve)
            # Steeper curve = sharper transition
            steepness = 5.0
            w_deep = 1.0 / (1.0 + np.exp(-steepness * (t - 0.5)))
            w_shallow = 1.0 - w_deep
            
            # Adjust weights by trust scores
            total_trust = shallow_trust + deep_trust
            if total_trust > 0:
                trust_shallow = shallow_trust / total_trust
                trust_deep = deep_trust / total_trust
                
                # Blend trust adjustment (subtle influence)
                w_shallow = w_shallow * (0.7 + 0.3 * trust_shallow)
                w_deep = w_deep * (0.7 + 0.3 * trust_deep)
                
                # Renormalize
                w_total = w_shallow + w_deep
                w_shallow /= w_total
                w_deep /= w_total
            
            # Handle NaN values
            if np.isnan(shallow_val) and np.isnan(deep_val):
                blended[i] = np.nan
            elif np.isnan(shallow_val):
                blended[i] = deep_val
            elif np.isnan(deep_val):
                blended[i] = shallow_val
            else:
                blended[i] = w_shallow * shallow_val + w_deep * deep_val
    
    return blended


# =============================================================================
# MASTER ORCHESTRATION FUNCTION
# =============================================================================

def ml_splice_logs(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    grid_step: float = DEFAULT_GRID_STEP,
    hampel_window: int = DEFAULT_HAMPEL_WINDOW,
    hampel_threshold: float = DEFAULT_HAMPEL_THRESHOLD,
    variance_window: int = DEFAULT_VARIANCE_WINDOW,
    blend_zone: float = DEFAULT_BLEND_ZONE,
    use_pelt: bool = True,
    progress_callback: Optional[callable] = None
) -> MLSplicingResult:
    """
    Master function for ML-based log splicing.
    
    Implements the complete workflow:
    1. Overlap detection (Rule-based logic)
    2. Resampling (Linear interpolation)
    3. QC - Remove spikes (Hampel filter)
    4. Stability metric (Rolling variance)
    5. Splice detection (CPD – PELT)
    6. Run selection (Rule-based trust scoring)
    7. Transition (Weighted blending)
    
    Args:
        shallow_depth: Depth array from shallow log
        shallow_signal: Signal array from shallow log
        deep_depth: Depth array from deep log
        deep_signal: Signal array from deep log
        grid_step: Common grid step in meters
        hampel_window: Window size for Hampel filter
        hampel_threshold: Threshold for spike detection
        variance_window: Window size for rolling variance
        blend_zone: Width of blending zone in meters
        use_pelt: Whether to use PELT for change point detection
        progress_callback: Optional callback(step, message) for progress
        
    Returns:
        MLSplicingResult with all outputs and metrics
    """
    def report(step, msg):
        if progress_callback:
            progress_callback(step, msg)
    
    # =========================================================================
    # STEP 1: OVERLAP DETECTION
    # =========================================================================
    report("overlap", "Detecting overlap region...")
    
    overlap_start, overlap_end, overlap_length = find_overlap_region(
        shallow_depth, deep_depth
    )
    
    report("overlap", 
           f"Overlap found: {overlap_start:.1f}m to {overlap_end:.1f}m "
           f"({overlap_length:.1f}m)")
    
    # =========================================================================
    # STEP 2: RESAMPLING
    # =========================================================================
    report("resampling", f"Creating common grid (step={grid_step}m)...")
    
    common_grid = create_common_grid(shallow_depth, deep_depth, grid_step)
    
    shallow_resampled = resample_to_common_grid(
        shallow_depth, shallow_signal, common_grid
    )
    deep_resampled = resample_to_common_grid(
        deep_depth, deep_signal, common_grid
    )
    
    report("resampling", f"Resampled to {len(common_grid)} samples")
    
    # =========================================================================
    # STEP 3: QC - HAMPEL FILTER
    # =========================================================================
    report("qc", "Applying Hampel filter to remove spikes...")
    
    shallow_qc = hampel_filter(
        shallow_resampled, 
        window_size=hampel_window,
        threshold=hampel_threshold
    )
    deep_qc = hampel_filter(
        deep_resampled,
        window_size=hampel_window,
        threshold=hampel_threshold
    )
    
    report("qc", 
           f"QC complete. Spikes removed: "
           f"shallow={shallow_qc.num_spikes_removed}, "
           f"deep={deep_qc.num_spikes_removed}")
    
    # =========================================================================
    # STEP 4: STABILITY METRIC
    # =========================================================================
    report("stability", "Calculating rolling variance...")
    
    shallow_stability = calculate_rolling_variance(
        shallow_qc.cleaned_signal,
        window_size=variance_window
    )
    deep_stability = calculate_rolling_variance(
        deep_qc.cleaned_signal,
        window_size=variance_window
    )
    
    report("stability", 
           f"Stability scores: "
           f"shallow={shallow_stability.mean_stability:.3f}, "
           f"deep={deep_stability.mean_stability:.3f}")
    
    # =========================================================================
    # STEP 5: SPLICE DETECTION (CPD-PELT)
    # =========================================================================
    change_points = []
    
    if use_pelt:
        report("splice_detection", "Running PELT change point detection...")
        
        # Combine signals in overlap for change point detection
        overlap_mask = (common_grid >= overlap_start) & (common_grid <= overlap_end)
        overlap_depths = common_grid[overlap_mask]
        
        # Use difference signal for change point detection
        diff_signal = np.abs(
            shallow_qc.cleaned_signal[overlap_mask] - 
            deep_qc.cleaned_signal[overlap_mask]
        )
        
        change_points = detect_change_points_pelt(
            diff_signal, 
            overlap_depths,
            min_segment_length=max(10, int(2.0 / grid_step))  # ~2m minimum
        )
        
        report("splice_detection", 
               f"Detected {len(change_points)} change point(s)")
    
    # =========================================================================
    # STEP 6: RUN SELECTION (Trust scores)
    # =========================================================================
    report("run_selection", "Calculating trust scores...")
    
    shallow_trust, deep_trust = calculate_trust_scores(
        shallow_stability,
        deep_stability,
        shallow_qc.cleaned_signal,
        deep_qc.cleaned_signal
    )
    
    report("run_selection",
           f"Trust scores: shallow={shallow_trust:.3f}, deep={deep_trust:.3f}")
    
    # Find optimal splice point
    report("splice_detection", "Finding optimal splice point...")
    
    # Create stability arrays on common grid
    shallow_stab_array = 1.0 / (1.0 + shallow_stability.rolling_variance + 1e-10)
    deep_stab_array = 1.0 / (1.0 + deep_stability.rolling_variance + 1e-10)
    
    splice_point, splice_quality = find_optimal_splice_point(
        shallow_qc.cleaned_signal,
        deep_qc.cleaned_signal,
        common_grid,
        overlap_start,
        overlap_end,
        shallow_stab_array,
        deep_stab_array,
        change_points
    )
    
    report("splice_detection",
           f"Optimal splice point: {splice_point:.1f}m (quality={splice_quality:.3f})")
    
    # =========================================================================
    # STEP 7: TRANSITION - WEIGHTED BLENDING
    # =========================================================================
    report("transition", f"Applying weighted blending (zone={blend_zone}m)...")
    
    blended_signal = weighted_blend_transition(
        shallow_qc.cleaned_signal,
        deep_qc.cleaned_signal,
        common_grid,
        splice_point,
        blend_zone=blend_zone,
        shallow_trust=shallow_trust,
        deep_trust=deep_trust
    )
    
    # Create final merged signal
    # Above overlap: pure shallow
    # Below overlap: pure deep
    # In overlap: blended
    
    merged_signal = np.full_like(common_grid, np.nan, dtype=float)
    
    for i, d in enumerate(common_grid):
        if d < overlap_start:
            # Pure shallow
            merged_signal[i] = shallow_qc.cleaned_signal[i]
        elif d > overlap_end:
            # Pure deep
            merged_signal[i] = deep_qc.cleaned_signal[i]
        else:
            # Use blended value
            merged_signal[i] = blended_signal[i]
    
    report("complete", "ML-based splicing complete!")
    
    blend_start = splice_point - blend_zone / 2
    blend_end = splice_point + blend_zone / 2
    
    return MLSplicingResult(
        merged_depth=common_grid,
        merged_signal=merged_signal,
        splice_point=splice_point,
        splice_quality_score=splice_quality,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        shallow_stability=shallow_stability.mean_stability,
        deep_stability=deep_stability.mean_stability,
        change_points=change_points,
        trust_shallow=shallow_trust,
        trust_deep=deep_trust,
        blend_start=blend_start,
        blend_end=blend_end
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_common_curves(curves1: List[str], curves2: List[str]) -> List[str]:
    """
    Find curves present in both lists.
    
    Args:
        curves1: List of curve names from first file
        curves2: List of curve names from second file
        
    Returns:
        List of common curve names
    """
    set1 = set(c.upper() for c in curves1)
    set2 = set(c.upper() for c in curves2)
    
    common = set1 & set2
    
    # Return in original case from first list
    return [c for c in curves1 if c.upper() in common]


def get_recommended_correlation_curve(common_curves: List[str]) -> Optional[str]:
    """
    Get recommended curve for correlation.
    
    Priority: GR > RHOB > NPHI > first available
    
    Args:
        common_curves: List of available curve names
        
    Returns:
        Recommended curve name or None
    """
    priority = ['GR', 'GRC', 'SGR', 'CGR', 'RHOB', 'RHOZ', 'NPHI', 'TNPH']
    
    upper_curves = {c.upper(): c for c in common_curves}
    
    for p in priority:
        if p in upper_curves:
            return upper_curves[p]
    
    # Return first non-depth curve
    depth_names = {'DEPT', 'DEPTH', 'MD', 'TVD'}
    for c in common_curves:
        if c.upper() not in depth_names:
            return c
    
    return None

