"""
Siamese 1D CNN for Log Pattern Matching

Implements a Siamese network with shared 1D CNN backbone for learning
similarity between well log segments. Replaces cross-correlation with
learned feature matching.

Reference: INTERNAL_AI_ML_ENHANCEMENT_RESEARCH.md - Tier 2.2
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class PatternMatchResult:
    """Container for pattern matching results."""
    model: 'SiameseCNN'
    loss_history: List[float]
    final_loss: float
    epochs_trained: int
    embeddings: Optional[np.ndarray] = None


@dataclass
class SimilarityResult:
    """Container for similarity computation results."""
    similarity_map: np.ndarray  # [n_positions] similarity scores
    best_match_position: int
    best_match_similarity: float
    depth_offset: float  # Best match depth offset in meters
    confidence: float
    shallow_embedding: np.ndarray
    deep_embeddings: np.ndarray


class CNN1DBlock(nn.Module):
    """
    Single 1D CNN block with conv, batch norm, activation, and pooling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class SiameseCNN(nn.Module):
    """
    Siamese 1D CNN for learning log similarity.
    
    Architecture:
    - Shared CNN backbone with 3 conv blocks
    - Global average pooling
    - Dense projection to embedding space
    - Cosine similarity for comparison
    
    Designed for fast training and interpretable embeddings.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        embedding_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Shared CNN backbone
        self.backbone = nn.Sequential(
            # Block 1: 1 -> 16 channels
            CNN1DBlock(input_channels, 16, kernel_size=7, pool_size=2),
            nn.Dropout(dropout),
            
            # Block 2: 16 -> 32 channels
            CNN1DBlock(16, 32, kernel_size=5, pool_size=2),
            nn.Dropout(dropout),
            
            # Block 3: 32 -> 64 channels
            CNN1DBlock(32, 64, kernel_size=3, pool_size=2),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Embedding projection
        self.projection = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embedding_dim)
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a log segment to embedding space.
        
        Args:
            x: [batch, channels, seq_len]
            
        Returns:
            embedding: [batch, embedding_dim] L2-normalized
        """
        # CNN backbone
        features = self.backbone(x)  # [batch, 64, seq_len//8]
        
        # Global pooling
        pooled = self.global_pool(features)  # [batch, 64, 1]
        pooled = pooled.squeeze(-1)  # [batch, 64]
        
        # Project to embedding space
        embedding = self.projection(pooled)  # [batch, embedding_dim]
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def forward(
        self,
        shallow_log: torch.Tensor,
        deep_log: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute similarity between shallow and deep log segments.
        
        Args:
            shallow_log: [batch, channels, seq_len]
            deep_log: [batch, channels, seq_len]
            
        Returns:
            similarity: [batch] cosine similarity (-1 to 1)
            shallow_emb: [batch, embedding_dim]
            deep_emb: [batch, embedding_dim]
        """
        shallow_emb = self.encode(shallow_log)
        deep_emb = self.encode(deep_log)
        
        # Cosine similarity (embeddings are already normalized)
        similarity = torch.sum(shallow_emb * deep_emb, dim=-1)
        
        return similarity, shallow_emb, deep_emb


def prepare_contrastive_pairs(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    window_size: int = 64,
    n_positive: int = 50,
    n_negative: int = 50
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare contrastive training pairs.
    
    Positive pairs: Same depth region from both logs
    Negative pairs: Different depth regions
    
    Args:
        shallow_depth, shallow_signal: Shallow log data
        deep_depth, deep_signal: Deep log data
        window_size: Window size for segments
        n_positive: Number of positive pairs
        n_negative: Number of negative pairs
        
    Returns:
        shallow_segments: [n_pairs, 1, window_size]
        deep_segments: [n_pairs, 1, window_size]
        labels: [n_pairs] 1 for positive, 0 for negative
    """
    # Find overlap region
    overlap_start = max(np.nanmin(shallow_depth), np.nanmin(deep_depth))
    overlap_end = min(np.nanmax(shallow_depth), np.nanmax(deep_depth))
    
    # Extract overlap
    shallow_mask = (shallow_depth >= overlap_start) & (shallow_depth <= overlap_end)
    deep_mask = (deep_depth >= overlap_start) & (deep_depth <= overlap_end)
    
    shallow_overlap = shallow_signal[shallow_mask]
    deep_overlap = deep_signal[deep_mask]
    
    # Normalize
    shallow_norm = (shallow_overlap - np.nanmean(shallow_overlap)) / max(np.nanstd(shallow_overlap), 0.001)
    deep_norm = (deep_overlap - np.nanmean(deep_overlap)) / max(np.nanstd(deep_overlap), 0.001)
    
    shallow_norm = np.nan_to_num(shallow_norm, 0)
    deep_norm = np.nan_to_num(deep_norm, 0)
    
    shallow_segments = []
    deep_segments = []
    labels = []
    
    max_start = min(len(shallow_norm), len(deep_norm)) - window_size
    
    if max_start <= 0:
        # Not enough data, create minimal pairs
        seg = np.zeros(window_size)
        seg[:min(window_size, len(shallow_norm))] = shallow_norm[:min(window_size, len(shallow_norm))]
        shallow_segments = [seg] * (n_positive + n_negative)
        deep_segments = [seg] * (n_positive + n_negative)
        labels = [1] * n_positive + [0] * n_negative
    else:
        # Positive pairs: same region
        for _ in range(n_positive):
            start = np.random.randint(0, max_start)
            shallow_seg = shallow_norm[start:start + window_size]
            deep_seg = deep_norm[start:start + window_size]
            
            shallow_segments.append(shallow_seg)
            deep_segments.append(deep_seg)
            labels.append(1)
        
        # Negative pairs: different regions
        for _ in range(n_negative):
            start_s = np.random.randint(0, max_start)
            start_d = np.random.randint(0, max_start)
            
            # Ensure they're different
            while abs(start_s - start_d) < window_size // 2:
                start_d = np.random.randint(0, max_start)
            
            shallow_seg = shallow_norm[start_s:start_s + window_size]
            deep_seg = deep_norm[start_d:start_d + window_size]
            
            shallow_segments.append(shallow_seg)
            deep_segments.append(deep_seg)
            labels.append(0)
    
    # Convert to tensors
    shallow_tensor = torch.FloatTensor(np.array(shallow_segments)).unsqueeze(1)
    deep_tensor = torch.FloatTensor(np.array(deep_segments)).unsqueeze(1)
    label_tensor = torch.FloatTensor(labels)
    
    return shallow_tensor, deep_tensor, label_tensor


def contrastive_loss(
    similarity: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.5
) -> torch.Tensor:
    """
    Contrastive loss for Siamese training.
    
    Positive pairs should have similarity close to 1.
    Negative pairs should have similarity below margin.
    
    Args:
        similarity: [batch] predicted similarities
        labels: [batch] 1 for positive, 0 for negative
        margin: Margin for negative pairs
        
    Returns:
        loss: Scalar loss value
    """
    # Positive loss: (1 - similarity)^2
    pos_loss = labels * (1 - similarity) ** 2
    
    # Negative loss: max(0, similarity - margin)^2
    neg_loss = (1 - labels) * F.relu(similarity - margin) ** 2
    
    return torch.mean(pos_loss + neg_loss)


def train_pattern_matcher(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    epochs: int = 30,
    learning_rate: float = 0.001,
    window_size: int = 64,
    n_pairs: int = 100,
    progress_callback: Optional[Callable[[int, float], None]] = None
) -> PatternMatchResult:
    """
    Train Siamese CNN on well log data.
    
    Args:
        shallow_depth, shallow_signal: Shallow log data
        deep_depth, deep_signal: Deep log data
        epochs: Number of training epochs
        learning_rate: Learning rate
        window_size: Window size for segments
        n_pairs: Total number of training pairs
        progress_callback: Optional callback(epoch, loss)
        
    Returns:
        PatternMatchResult with trained model
    """
    # Prepare training data
    n_positive = n_pairs // 2
    n_negative = n_pairs - n_positive
    
    shallow_segs, deep_segs, labels = prepare_contrastive_pairs(
        shallow_depth, shallow_signal,
        deep_depth, deep_signal,
        window_size=window_size,
        n_positive=n_positive,
        n_negative=n_negative
    )
    
    # Create model
    model = SiameseCNN(
        input_channels=1,
        embedding_dim=64,
        dropout=0.2
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    loss_history = []
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        similarity, _, _ = model(shallow_segs, deep_segs)
        
        # Compute loss
        loss = contrastive_loss(similarity, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if progress_callback:
            progress_callback(epoch + 1, loss_val)
    
    return PatternMatchResult(
        model=model,
        loss_history=loss_history,
        final_loss=loss_history[-1] if loss_history else 0,
        epochs_trained=epochs
    )


def compute_similarity_map(
    model: SiameseCNN,
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    window_size: int = 64,
    stride: int = 8
) -> SimilarityResult:
    """
    Compute similarity map between logs using sliding window.
    
    Args:
        model: Trained SiameseCNN
        shallow_depth, shallow_signal: Shallow log data
        deep_depth, deep_signal: Deep log data
        window_size: Window size for comparison
        stride: Stride for sliding window
        
    Returns:
        SimilarityResult with similarity map and best match
    """
    # Find overlap region
    overlap_start = max(np.nanmin(shallow_depth), np.nanmin(deep_depth))
    overlap_end = min(np.nanmax(shallow_depth), np.nanmax(deep_depth))
    
    # Extract and normalize overlap
    shallow_mask = (shallow_depth >= overlap_start) & (shallow_depth <= overlap_end)
    deep_mask = (deep_depth >= overlap_start) & (deep_depth <= overlap_end)
    
    shallow_overlap = shallow_signal[shallow_mask]
    deep_overlap = deep_signal[deep_mask]
    shallow_overlap_depth = shallow_depth[shallow_mask]
    
    shallow_norm = (shallow_overlap - np.nanmean(shallow_overlap)) / max(np.nanstd(shallow_overlap), 0.001)
    deep_norm = (deep_overlap - np.nanmean(deep_overlap)) / max(np.nanstd(deep_overlap), 0.001)
    
    shallow_norm = np.nan_to_num(shallow_norm, 0)
    deep_norm = np.nan_to_num(deep_norm, 0)
    
    # Get reference window from middle of shallow log
    ref_start = len(shallow_norm) // 2 - window_size // 2
    ref_start = max(0, min(ref_start, len(shallow_norm) - window_size))
    ref_window = shallow_norm[ref_start:ref_start + window_size]
    
    if len(ref_window) < window_size:
        ref_window = np.pad(ref_window, (0, window_size - len(ref_window)))
    
    ref_tensor = torch.FloatTensor(ref_window).unsqueeze(0).unsqueeze(0)  # [1, 1, window_size]
    
    # Slide over deep log
    similarities = []
    positions = []
    deep_embeddings = []
    
    model.eval()
    with torch.no_grad():
        shallow_emb = model.encode(ref_tensor)
        
        for i in range(0, len(deep_norm) - window_size + 1, stride):
            deep_window = deep_norm[i:i + window_size]
            deep_tensor = torch.FloatTensor(deep_window).unsqueeze(0).unsqueeze(0)
            
            similarity, _, deep_emb = model(ref_tensor, deep_tensor)
            
            similarities.append(similarity.item())
            positions.append(i)
            deep_embeddings.append(deep_emb.numpy()[0])
    
    similarities = np.array(similarities)
    positions = np.array(positions)
    deep_embeddings = np.array(deep_embeddings)
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_position = positions[best_idx]
    best_similarity = similarities[best_idx]
    
    # Calculate depth offset
    ref_depth = shallow_overlap_depth[ref_start] if ref_start < len(shallow_overlap_depth) else overlap_start
    match_depth = overlap_start + (best_position / len(deep_norm)) * (overlap_end - overlap_start)
    depth_offset = match_depth - ref_depth
    
    # Confidence based on similarity distribution
    similarity_std = np.std(similarities)
    confidence = min(1.0, best_similarity * (1 + similarity_std))
    
    return SimilarityResult(
        similarity_map=similarities,
        best_match_position=best_position,
        best_match_similarity=best_similarity,
        depth_offset=depth_offset,
        confidence=confidence,
        shallow_embedding=shallow_emb.numpy()[0],
        deep_embeddings=deep_embeddings
    )


def visualize_feature_embeddings(
    model: SiameseCNN,
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    window_size: int = 64,
    n_samples: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature embeddings for visualization.
    
    Args:
        model: Trained SiameseCNN
        shallow_depth, shallow_signal: Shallow log data
        deep_depth, deep_signal: Deep log data
        window_size: Window size
        n_samples: Number of samples per log
        
    Returns:
        shallow_embeddings: [n_samples, embedding_dim]
        deep_embeddings: [n_samples, embedding_dim]
        depths: [n_samples] corresponding depths
    """
    # Find overlap
    overlap_start = max(np.nanmin(shallow_depth), np.nanmin(deep_depth))
    overlap_end = min(np.nanmax(shallow_depth), np.nanmax(deep_depth))
    
    shallow_mask = (shallow_depth >= overlap_start) & (shallow_depth <= overlap_end)
    deep_mask = (deep_depth >= overlap_start) & (deep_depth <= overlap_end)
    
    shallow_overlap = shallow_signal[shallow_mask]
    deep_overlap = deep_signal[deep_mask]
    shallow_depths = shallow_depth[shallow_mask]
    
    # Normalize
    shallow_norm = (shallow_overlap - np.nanmean(shallow_overlap)) / max(np.nanstd(shallow_overlap), 0.001)
    deep_norm = (deep_overlap - np.nanmean(deep_overlap)) / max(np.nanstd(deep_overlap), 0.001)
    
    shallow_norm = np.nan_to_num(shallow_norm, 0)
    deep_norm = np.nan_to_num(deep_norm, 0)
    
    # Sample positions
    max_start = min(len(shallow_norm), len(deep_norm)) - window_size
    if max_start <= 0:
        # Return empty if not enough data
        return np.zeros((1, model.embedding_dim)), np.zeros((1, model.embedding_dim)), np.array([overlap_start])
    
    positions = np.linspace(0, max_start, n_samples, dtype=int)
    
    shallow_embeddings = []
    deep_embeddings = []
    depths = []
    
    model.eval()
    with torch.no_grad():
        for pos in positions:
            shallow_window = shallow_norm[pos:pos + window_size]
            deep_window = deep_norm[pos:pos + window_size]
            
            if len(shallow_window) < window_size:
                shallow_window = np.pad(shallow_window, (0, window_size - len(shallow_window)))
            if len(deep_window) < window_size:
                deep_window = np.pad(deep_window, (0, window_size - len(deep_window)))
            
            shallow_tensor = torch.FloatTensor(shallow_window).unsqueeze(0).unsqueeze(0)
            deep_tensor = torch.FloatTensor(deep_window).unsqueeze(0).unsqueeze(0)
            
            s_emb = model.encode(shallow_tensor)
            d_emb = model.encode(deep_tensor)
            
            shallow_embeddings.append(s_emb.numpy()[0])
            deep_embeddings.append(d_emb.numpy()[0])
            
            if pos < len(shallow_depths):
                depths.append(shallow_depths[pos])
            else:
                depths.append(overlap_start + pos * (overlap_end - overlap_start) / len(shallow_norm))
    
    return np.array(shallow_embeddings), np.array(deep_embeddings), np.array(depths)


def get_model_summary(model: SiameseCNN) -> Dict:
    """
    Get summary of model architecture.
    
    Returns:
        Dictionary with model details for visualization
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'name': 'Siamese 1D CNN',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'embedding_dim': model.embedding_dim,
        'architecture': [
            {'name': 'Conv Block 1', 'type': '1D Conv + BN + ReLU + Pool', 'channels': '1 -> 16'},
            {'name': 'Conv Block 2', 'type': '1D Conv + BN + ReLU + Pool', 'channels': '16 -> 32'},
            {'name': 'Conv Block 3', 'type': '1D Conv + BN + ReLU + Pool', 'channels': '32 -> 64'},
            {'name': 'Global Pool', 'type': 'Adaptive Avg Pool', 'output': '64'},
            {'name': 'Projection', 'type': 'Dense + L2 Norm', 'output': str(model.embedding_dim)}
        ]
    }
