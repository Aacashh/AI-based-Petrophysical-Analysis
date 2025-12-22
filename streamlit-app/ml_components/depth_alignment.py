"""
Depth Alignment Module for Well Log Data

Implements ML-based depth alignment algorithms:
1. Correlation-based alignment (cross-correlation)
2. Siamese Neural Network for feature matching

Used to align measurements from different tools that may have
depth discrepancies due to cable stretch, tool positioning, etc.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
from scipy import signal as scipy_signal
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class AlignmentResult:
    """Container for depth alignment results."""
    optimal_shift: float  # Optimal depth shift in same units as input
    optimal_shift_samples: int  # Shift in number of samples
    correlation_coefficient: float  # Max correlation value
    confidence: float  # Confidence score (0-1)
    method: str  # 'correlation' or 'siamese'
    correlation_curve: np.ndarray  # Full correlation function
    shift_range: np.ndarray  # Range of shifts tested


@dataclass
class SiameseAlignmentResult:
    """Container for Siamese network alignment results."""
    optimal_shift: float  # Optimal depth shift
    similarity_score: float  # Best similarity score
    confidence: float  # Model confidence
    similarity_map: np.ndarray  # Similarity at each position
    model: nn.Module  # Trained model reference
    training_loss: List[float]  # Training loss history


# =============================================================================
# CORRELATION-BASED ALIGNMENT
# =============================================================================

def align_by_correlation(
    reference_depth: np.ndarray,
    reference_signal: np.ndarray,
    target_depth: np.ndarray,
    target_signal: np.ndarray,
    max_shift_meters: float = 20.0,
    grid_step: float = 0.1524
) -> AlignmentResult:
    """
    Align two log curves using cross-correlation.
    
    This method finds the optimal bulk shift that maximizes correlation
    between the reference and target signals.
    
    Args:
        reference_depth: Depth array of reference log
        reference_signal: Signal values of reference log
        target_depth: Depth array of target log
        target_signal: Signal values of target log
        max_shift_meters: Maximum shift to search (+/-)
        grid_step: Depth step for resampling
        
    Returns:
        AlignmentResult with optimal shift and correlation details
    """
    # Resample both signals to common grid
    common_grid = _create_common_grid(reference_depth, target_depth, grid_step)
    
    ref_resampled = _resample_signal(reference_depth, reference_signal, common_grid)
    tgt_resampled = _resample_signal(target_depth, target_signal, common_grid)
    
    # Normalize signals
    ref_normalized = _zscore_normalize(ref_resampled)
    tgt_normalized = _zscore_normalize(tgt_resampled)
    
    # Fill NaN with 0 for correlation
    ref_clean = np.nan_to_num(ref_normalized, nan=0.0)
    tgt_clean = np.nan_to_num(tgt_normalized, nan=0.0)
    
    # Compute cross-correlation
    correlation = scipy_signal.correlate(ref_clean, tgt_clean, mode='full')
    
    # Normalize correlation
    correlation = correlation / (len(ref_clean) * max(ref_clean.std() * tgt_clean.std(), 1e-6))
    
    # Create lag array (in samples)
    n = len(ref_clean)
    lags = np.arange(-(n-1), n)
    lag_meters = lags * grid_step
    
    # Limit search to specified window
    max_samples = int(max_shift_meters / grid_step)
    center = n - 1
    search_start = max(0, center - max_samples)
    search_end = min(len(correlation), center + max_samples + 1)
    
    # Find peak within window
    search_region = correlation[search_start:search_end]
    peak_idx_local = np.argmax(search_region)
    peak_idx_global = search_start + peak_idx_local
    
    # Get optimal shift
    optimal_shift_samples = lags[peak_idx_global]
    optimal_shift_meters = optimal_shift_samples * grid_step
    max_correlation = correlation[peak_idx_global]
    
    # Calculate confidence based on correlation peak characteristics
    confidence = _calculate_correlation_confidence(
        correlation, peak_idx_global, max_correlation
    )
    
    return AlignmentResult(
        optimal_shift=optimal_shift_meters,
        optimal_shift_samples=optimal_shift_samples,
        correlation_coefficient=float(max_correlation),
        confidence=confidence,
        method='correlation',
        correlation_curve=correlation,
        shift_range=lag_meters
    )


def _create_common_grid(depth1: np.ndarray, depth2: np.ndarray, step: float) -> np.ndarray:
    """Create common depth grid covering both logs."""
    min_depth = min(np.nanmin(depth1), np.nanmin(depth2))
    max_depth = max(np.nanmax(depth1), np.nanmax(depth2))
    min_depth = np.floor(min_depth / step) * step
    max_depth = np.ceil(max_depth / step) * step
    return np.arange(min_depth, max_depth + step/2, step)


def _resample_signal(depth: np.ndarray, signal: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Resample signal to target grid."""
    valid_mask = ~np.isnan(signal) & ~np.isnan(depth)
    if np.sum(valid_mask) < 2:
        return np.full_like(grid, np.nan)
    
    valid_depth = depth[valid_mask]
    valid_signal = signal[valid_mask]
    
    sort_idx = np.argsort(valid_depth)
    valid_depth = valid_depth[sort_idx]
    valid_signal = valid_signal[sort_idx]
    
    return np.interp(grid, valid_depth, valid_signal, left=np.nan, right=np.nan)


def _zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalize a signal."""
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (signal - mean) / std


def _calculate_correlation_confidence(
    correlation: np.ndarray,
    peak_idx: int,
    peak_value: float
) -> float:
    """Calculate confidence based on correlation peak quality."""
    confidence = 1.0
    
    # Peak value factor
    if peak_value < 0.5:
        confidence *= peak_value * 2  # Linear scaling below 0.5
    
    # Peak sharpness factor (ratio of peak to neighboring values)
    if peak_idx > 0 and peak_idx < len(correlation) - 1:
        neighbors_mean = (correlation[peak_idx-1] + correlation[peak_idx+1]) / 2
        if peak_value > 0:
            sharpness = 1 - (neighbors_mean / peak_value)
            confidence *= min(1.0, max(0.5, sharpness + 0.5))
    
    # Secondary peak check
    sorted_corr = np.sort(correlation)[::-1]
    if len(sorted_corr) > 1 and sorted_corr[0] > 0:
        second_peak_ratio = sorted_corr[1] / sorted_corr[0]
        if second_peak_ratio > 0.9:  # Secondary peak almost as high
            confidence *= 0.7
    
    return min(max(confidence, 0.0), 1.0)


# =============================================================================
# SIAMESE NEURAL NETWORK
# =============================================================================

class SiameseEncoder(nn.Module):
    """1D CNN encoder for Siamese network."""
    
    def __init__(self, input_channels: int = 1, embedding_dim: int = 64):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        features = self.conv_layers(x)
        features = features.squeeze(-1)  # (batch, 128)
        embedding = self.fc(features)
        return embedding


class SiameseDepthAligner(nn.Module):
    """Siamese network for depth alignment via similarity learning."""
    
    def __init__(self, input_channels: int = 1, embedding_dim: int = 64):
        super().__init__()
        self.encoder = SiameseEncoder(input_channels, embedding_dim)
    
    def forward_one(self, x):
        """Encode a single input."""
        return self.encoder(x)
    
    def forward(self, x1, x2):
        """
        Forward pass for pair comparison.
        
        Returns cosine similarity between embeddings.
        """
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity, emb1, emb2


class AlignmentPairDataset(Dataset):
    """Dataset for training Siamese alignment network."""
    
    def __init__(
        self,
        ref_signal: np.ndarray,
        target_signal: np.ndarray,
        window_size: int = 50,
        n_pairs: int = 100
    ):
        self.ref_signal = ref_signal
        self.target_signal = target_signal
        self.window_size = window_size
        self.n_pairs = n_pairs
        
        # Pre-generate pairs
        self.pairs = self._generate_pairs()
    
    def _generate_pairs(self):
        pairs = []
        n_ref = len(self.ref_signal) - self.window_size
        n_tgt = len(self.target_signal) - self.window_size
        
        for _ in range(self.n_pairs):
            # Random reference position
            ref_start = np.random.randint(0, n_ref)
            
            # Matching pair (positive) or non-matching (negative)
            is_positive = np.random.random() > 0.5
            
            if is_positive:
                # Same position (or very close) = positive pair
                shift = np.random.randint(-5, 6)
                tgt_start = max(0, min(n_tgt - 1, ref_start + shift))
                label = 1.0
            else:
                # Random different position = negative pair
                tgt_start = np.random.randint(0, n_tgt)
                while abs(tgt_start - ref_start) < self.window_size:
                    tgt_start = np.random.randint(0, n_tgt)
                label = 0.0
            
            pairs.append((ref_start, tgt_start, label))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        ref_start, tgt_start, label = self.pairs[idx]
        
        ref_window = self.ref_signal[ref_start:ref_start + self.window_size]
        tgt_window = self.target_signal[tgt_start:tgt_start + self.window_size]
        
        # Normalize windows
        ref_window = (ref_window - np.nanmean(ref_window)) / (np.nanstd(ref_window) + 1e-6)
        tgt_window = (tgt_window - np.nanmean(tgt_window)) / (np.nanstd(tgt_window) + 1e-6)
        
        # Handle NaN
        ref_window = np.nan_to_num(ref_window, nan=0.0)
        tgt_window = np.nan_to_num(tgt_window, nan=0.0)
        
        # Add channel dimension
        ref_tensor = torch.FloatTensor(ref_window).unsqueeze(0)
        tgt_tensor = torch.FloatTensor(tgt_window).unsqueeze(0)
        
        return ref_tensor, tgt_tensor, torch.FloatTensor([label])


def train_siamese_aligner(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    window_size: int = 50,
    n_pairs: int = 200,
    epochs: int = 30,
    learning_rate: float = 0.001,
    progress_callback: Optional[Callable[[int, float], None]] = None
) -> Tuple[SiameseDepthAligner, List[float]]:
    """
    Train a Siamese network for depth alignment.
    
    Args:
        reference_signal: Reference log signal
        target_signal: Target log signal
        window_size: Size of comparison windows
        n_pairs: Number of training pairs to generate
        epochs: Number of training epochs
        learning_rate: Learning rate
        progress_callback: Optional callback(epoch, loss)
        
    Returns:
        Tuple of (trained model, loss history)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    dataset = AlignmentPairDataset(reference_signal, target_signal, window_size, n_pairs)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = SiameseDepthAligner(input_channels=1, embedding_dim=64).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for ref_batch, tgt_batch, labels in dataloader:
            ref_batch = ref_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            
            similarity, _, _ = model(ref_batch, tgt_batch)
            loss = criterion(similarity, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        if progress_callback:
            progress_callback(epoch + 1, avg_loss)
    
    return model, loss_history


def find_alignment_siamese(
    model: SiameseDepthAligner,
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    window_size: int = 50,
    step: int = 5,
    depth_step: float = 0.1524
) -> SiameseAlignmentResult:
    """
    Find optimal alignment using trained Siamese network.
    
    Slides a window over the target and computes similarity at each position.
    
    Args:
        model: Trained SiameseDepthAligner
        reference_signal: Reference log signal
        target_signal: Target log signal
        window_size: Size of comparison windows
        step: Step size for sliding (samples)
        depth_step: Depth per sample for conversion
        
    Returns:
        SiameseAlignmentResult with optimal shift
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Normalize signals
    ref_norm = _zscore_normalize(reference_signal)
    tgt_norm = _zscore_normalize(target_signal)
    
    ref_norm = np.nan_to_num(ref_norm, nan=0.0)
    tgt_norm = np.nan_to_num(tgt_norm, nan=0.0)
    
    # Use middle section of reference as anchor
    ref_start = len(ref_norm) // 2 - window_size // 2
    ref_window = ref_norm[ref_start:ref_start + window_size]
    
    if len(ref_window) < window_size:
        # Pad if necessary
        ref_window = np.pad(ref_window, (0, window_size - len(ref_window)))
    
    ref_tensor = torch.FloatTensor(ref_window).unsqueeze(0).unsqueeze(0).to(device)
    
    # Slide over target
    similarities = []
    positions = []
    
    with torch.no_grad():
        for pos in range(0, len(tgt_norm) - window_size, step):
            tgt_window = tgt_norm[pos:pos + window_size]
            tgt_tensor = torch.FloatTensor(tgt_window).unsqueeze(0).unsqueeze(0).to(device)
            
            similarity, _, _ = model(ref_tensor, tgt_tensor)
            similarities.append(similarity.item())
            positions.append(pos)
    
    similarities = np.array(similarities)
    positions = np.array(positions)
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_position = positions[best_idx]
    best_similarity = similarities[best_idx]
    
    # Calculate shift relative to reference position
    optimal_shift_samples = best_position - ref_start
    optimal_shift_meters = optimal_shift_samples * depth_step
    
    # Confidence based on similarity and peak characteristics
    confidence = best_similarity  # Similarity is already 0-1
    if np.std(similarities) > 0:
        # Boost confidence if peak is distinct
        peak_prominence = (best_similarity - np.mean(similarities)) / np.std(similarities)
        confidence = min(1.0, confidence * (0.5 + 0.5 * min(peak_prominence / 3, 1.0)))
    
    return SiameseAlignmentResult(
        optimal_shift=optimal_shift_meters,
        similarity_score=best_similarity,
        confidence=confidence,
        similarity_map=similarities,
        model=model,
        training_loss=[]
    )


def align_by_siamese(
    reference_depth: np.ndarray,
    reference_signal: np.ndarray,
    target_depth: np.ndarray,
    target_signal: np.ndarray,
    window_size: int = 50,
    n_pairs: int = 200,
    epochs: int = 30,
    grid_step: float = 0.1524,
    progress_callback: Optional[Callable[[int, float], None]] = None
) -> SiameseAlignmentResult:
    """
    Full pipeline: train Siamese network and find optimal alignment.
    
    Args:
        reference_depth: Depth array of reference log
        reference_signal: Signal values of reference log
        target_depth: Depth array of target log
        target_signal: Signal values of target log
        window_size: Size of comparison windows
        n_pairs: Number of training pairs
        epochs: Training epochs
        grid_step: Depth step for resampling
        progress_callback: Optional callback(epoch, loss)
        
    Returns:
        SiameseAlignmentResult with shift and trained model
    """
    # Resample to common grid
    common_grid = _create_common_grid(reference_depth, target_depth, grid_step)
    ref_resampled = _resample_signal(reference_depth, reference_signal, common_grid)
    tgt_resampled = _resample_signal(target_depth, target_signal, common_grid)
    
    # Train model
    model, loss_history = train_siamese_aligner(
        ref_resampled, tgt_resampled,
        window_size=window_size,
        n_pairs=n_pairs,
        epochs=epochs,
        progress_callback=progress_callback
    )
    
    # Find alignment
    result = find_alignment_siamese(
        model, ref_resampled, tgt_resampled,
        window_size=window_size,
        depth_step=grid_step
    )
    
    # Add training loss to result
    result.training_loss = loss_history
    
    return result


def apply_depth_shift(
    df: pd.DataFrame,
    shift: float,
    depth_column: str = 'DEPTH'
) -> pd.DataFrame:
    """
    Apply a depth shift to a DataFrame.
    
    Args:
        df: DataFrame with log data
        shift: Shift to apply (positive = shift deeper)
        depth_column: Name of depth column
        
    Returns:
        DataFrame with shifted depths
    """
    df_shifted = df.copy()
    df_shifted[depth_column] = df_shifted[depth_column] + shift
    return df_shifted


def get_model_summary(model: SiameseDepthAligner) -> Dict:
    """Get summary of Siamese model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'encoder_type': 'Conv1D',
        'embedding_dim': 64,
        'architecture': [
            'Conv1d(1, 32, k=5) + BN + ReLU + MaxPool',
            'Conv1d(32, 64, k=5) + BN + ReLU + MaxPool',
            'Conv1d(64, 128, k=3) + BN + ReLU',
            'AdaptiveAvgPool1d(1)',
            'Linear(128, 64) + ReLU + Dropout(0.3)'
        ]
    }

