"""
Lightweight LSTM for Multi-Curve Well Log Alignment

Implements a simplified LSTM architecture that can be trained on the fly
using overlap region data. Designed for demo purposes with fast training.

Reference: INTERNAL_AI_ML_ENHANCEMENT_RESEARCH.md - Tier 2.1
Paper: arXiv:2307.10253 - Efficient Selective Attention LSTM
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LSTMTrainingResult:
    """Container for LSTM training results."""
    model: 'LightweightLSTMAligner'
    loss_history: List[float]
    final_loss: float
    epochs_trained: int
    predicted_shift: float
    confidence: float
    attention_weights: Optional[np.ndarray] = None


@dataclass
class AlignmentPrediction:
    """Container for alignment prediction."""
    depth_shift: float
    confidence: float
    attention_weights: np.ndarray
    prediction_samples: Optional[List[float]] = None  # For uncertainty


class LightweightLSTMAligner(nn.Module):
    """
    Lightweight LSTM for predicting depth shift between two log runs.
    
    Architecture:
    - Bidirectional LSTM encoder for each input log
    - Self-attention layer for importance weighting
    - Fusion layer combining both encodings
    - Dense head predicting shift amount
    
    Designed for fast training on CPU (seconds, not hours).
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Shared LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Fusion layer
        fusion_input_size = hidden_size * self.num_directions * 2  # Both logs
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)  # [shift, log_confidence]
        )
        
    def apply_attention(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention to LSTM output.
        
        Args:
            lstm_output: [batch, seq_len, hidden*directions]
            
        Returns:
            context: [batch, hidden*directions] - weighted sum
            weights: [batch, seq_len] - attention weights
        """
        # Compute attention scores
        scores = self.attention(lstm_output)  # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch, seq_len, 1]
        
        # Weighted sum
        context = torch.sum(lstm_output * weights, dim=1)  # [batch, hidden*directions]
        
        return context, weights.squeeze(-1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single log sequence.
        
        Args:
            x: [batch, seq_len, 1]
            
        Returns:
            context: [batch, hidden*directions] - encoded representation
            attention_weights: [batch, seq_len]
        """
        lstm_out, _ = self.encoder(x)  # [batch, seq_len, hidden*directions]
        context, attention_weights = self.apply_attention(lstm_out)
        return context, attention_weights
    
    def forward(
        self,
        shallow_log: torch.Tensor,
        deep_log: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict depth shift between shallow and deep logs.
        
        Args:
            shallow_log: [batch, seq_len, 1] - reference log
            deep_log: [batch, seq_len, 1] - target log
            
        Returns:
            shift: [batch, 1] - predicted depth shift in meters
            confidence: [batch, 1] - confidence score (0-1)
            shallow_attention: [batch, seq_len] - attention weights
            deep_attention: [batch, seq_len] - attention weights
        """
        # Encode both logs
        shallow_context, shallow_attn = self.encode(shallow_log)
        deep_context, deep_attn = self.encode(deep_log)
        
        # Fuse representations
        combined = torch.cat([shallow_context, deep_context], dim=-1)
        fused = self.fusion(combined)
        
        # Predict shift and confidence
        output = self.output_head(fused)
        shift = output[:, 0:1]  # Raw shift prediction
        confidence = torch.sigmoid(output[:, 1:2])  # 0-1 confidence
        
        return shift, confidence, shallow_attn, deep_attn


def prepare_alignment_data(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    overlap_start: float,
    overlap_end: float,
    window_size: int = 64,
    n_samples: int = 50,
    true_shift: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare training data from overlap region.
    
    Creates pairs of windowed sequences with known shifts for training.
    
    Args:
        shallow_depth: Shallow log depth array
        shallow_signal: Shallow log signal array
        deep_depth: Deep log depth array
        deep_signal: Deep log signal array
        overlap_start: Start of overlap region
        overlap_end: End of overlap region
        window_size: Size of each training window
        n_samples: Number of training samples to generate
        true_shift: Known true shift (if available)
        
    Returns:
        shallow_windows: [n_samples, window_size, 1]
        deep_windows: [n_samples, window_size, 1]
        shifts: [n_samples, 1] - target shifts
    """
    # Extract overlap region
    shallow_mask = (shallow_depth >= overlap_start) & (shallow_depth <= overlap_end)
    deep_mask = (deep_depth >= overlap_start) & (deep_depth <= overlap_end)
    
    shallow_overlap = shallow_signal[shallow_mask]
    deep_overlap = deep_signal[deep_mask]
    
    # Normalize signals
    shallow_mean, shallow_std = np.nanmean(shallow_overlap), np.nanstd(shallow_overlap)
    deep_mean, deep_std = np.nanmean(deep_overlap), np.nanstd(deep_overlap)
    
    shallow_std = max(shallow_std, 0.001)
    deep_std = max(deep_std, 0.001)
    
    shallow_norm = (shallow_overlap - shallow_mean) / shallow_std
    deep_norm = (deep_overlap - deep_mean) / deep_std
    
    # Fill NaN with 0
    shallow_norm = np.nan_to_num(shallow_norm, 0)
    deep_norm = np.nan_to_num(deep_norm, 0)
    
    # Generate training windows with synthetic shifts
    shallow_windows = []
    deep_windows = []
    shifts = []
    
    max_start = len(shallow_norm) - window_size - 10
    
    for _ in range(n_samples):
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start)
        
        # Extract shallow window
        shallow_win = shallow_norm[start_idx:start_idx + window_size]
        
        # Add synthetic shift to deep
        shift_samples = np.random.randint(-5, 6)  # ±5 sample shift
        deep_start = max(0, start_idx + shift_samples)
        deep_end = min(len(deep_norm), deep_start + window_size)
        
        deep_win = deep_norm[deep_start:deep_end]
        
        # Pad if necessary
        if len(shallow_win) < window_size:
            shallow_win = np.pad(shallow_win, (0, window_size - len(shallow_win)))
        if len(deep_win) < window_size:
            deep_win = np.pad(deep_win, (0, window_size - len(deep_win)))
        
        shallow_windows.append(shallow_win[:window_size])
        deep_windows.append(deep_win[:window_size])
        
        # Calculate shift in depth units (assuming uniform sampling)
        depth_step = (overlap_end - overlap_start) / max(len(shallow_overlap) - 1, 1)
        shift_meters = shift_samples * depth_step
        shifts.append(shift_meters)
    
    # Convert to tensors
    shallow_tensor = torch.FloatTensor(np.array(shallow_windows)).unsqueeze(-1)
    deep_tensor = torch.FloatTensor(np.array(deep_windows)).unsqueeze(-1)
    shift_tensor = torch.FloatTensor(np.array(shifts)).unsqueeze(-1)
    
    return shallow_tensor, deep_tensor, shift_tensor


def train_lstm_aligner(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    epochs: int = 50,
    learning_rate: float = 0.001,
    window_size: int = 64,
    n_train_samples: int = 100,
    progress_callback: Optional[Callable[[int, float], None]] = None
) -> LSTMTrainingResult:
    """
    Train LSTM aligner on uploaded data.
    
    This is designed for fast training on CPU using the overlap region data.
    
    Args:
        shallow_depth: Shallow log depth array
        shallow_signal: Shallow log signal array
        deep_depth: Deep log depth array
        deep_signal: Deep log signal array
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        window_size: Size of training windows
        n_train_samples: Number of training samples
        progress_callback: Optional callback(epoch, loss)
        
    Returns:
        LSTMTrainingResult with trained model and metrics
    """
    # Find overlap region
    overlap_start = max(np.nanmin(shallow_depth), np.nanmin(deep_depth))
    overlap_end = min(np.nanmax(shallow_depth), np.nanmax(deep_depth))
    
    # Prepare training data
    shallow_windows, deep_windows, target_shifts = prepare_alignment_data(
        shallow_depth, shallow_signal,
        deep_depth, deep_signal,
        overlap_start, overlap_end,
        window_size=window_size,
        n_samples=n_train_samples
    )
    
    # Create model
    model = LightweightLSTMAligner(
        input_size=1,
        hidden_size=32,
        num_layers=1,
        dropout=0.2
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    loss_history = []
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predicted_shift, confidence, shallow_attn, deep_attn = model(
            shallow_windows, deep_windows
        )
        
        # Compute loss (weighted by confidence)
        shift_loss = criterion(predicted_shift, target_shifts)
        
        # Add confidence regularization (encourage high confidence for good predictions)
        conf_loss = 0.1 * torch.mean((1 - confidence) ** 2)
        
        loss = shift_loss + conf_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if progress_callback:
            progress_callback(epoch + 1, loss_val)
    
    # Final inference on full overlap
    model.eval()
    with torch.no_grad():
        # Prepare full overlap data for inference
        full_shallow, full_deep, _ = prepare_alignment_data(
            shallow_depth, shallow_signal,
            deep_depth, deep_signal,
            overlap_start, overlap_end,
            window_size=window_size,
            n_samples=1
        )
        
        shift_pred, conf_pred, shallow_attn, deep_attn = model(full_shallow, full_deep)
        
        predicted_shift = shift_pred.item()
        confidence = conf_pred.item()
        
        # Get attention weights
        attention_weights = (shallow_attn.numpy() + deep_attn.numpy()) / 2
    
    return LSTMTrainingResult(
        model=model,
        loss_history=loss_history,
        final_loss=loss_history[-1] if loss_history else 0,
        epochs_trained=epochs,
        predicted_shift=predicted_shift,
        confidence=confidence,
        attention_weights=attention_weights[0] if len(attention_weights.shape) > 1 else attention_weights
    )


def predict_alignment(
    model: LightweightLSTMAligner,
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    window_size: int = 64
) -> AlignmentPrediction:
    """
    Predict alignment using trained model.
    
    Args:
        model: Trained LightweightLSTMAligner
        shallow_depth: Shallow log depth array
        shallow_signal: Shallow log signal array
        deep_depth: Deep log depth array
        deep_signal: Deep log signal array
        window_size: Window size for inference
        
    Returns:
        AlignmentPrediction with results
    """
    # Find overlap region
    overlap_start = max(np.nanmin(shallow_depth), np.nanmin(deep_depth))
    overlap_end = min(np.nanmax(shallow_depth), np.nanmax(deep_depth))
    
    # Prepare data
    shallow_windows, deep_windows, _ = prepare_alignment_data(
        shallow_depth, shallow_signal,
        deep_depth, deep_signal,
        overlap_start, overlap_end,
        window_size=window_size,
        n_samples=1
    )
    
    model.eval()
    with torch.no_grad():
        shift_pred, conf_pred, shallow_attn, deep_attn = model(shallow_windows, deep_windows)
        
        attention_weights = (shallow_attn.numpy() + deep_attn.numpy()) / 2
        
        return AlignmentPrediction(
            depth_shift=shift_pred.item(),
            confidence=conf_pred.item(),
            attention_weights=attention_weights[0] if len(attention_weights.shape) > 1 else attention_weights
        )


def get_model_summary(model: LightweightLSTMAligner) -> Dict:
    """
    Get summary of model architecture.
    
    Returns:
        Dictionary with model details for visualization
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'name': 'Lightweight LSTM Aligner',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'bidirectional': model.bidirectional,
        'architecture': [
            {'name': 'LSTM Encoder', 'type': 'Bidirectional LSTM', 'params': sum(p.numel() for p in model.encoder.parameters())},
            {'name': 'Attention', 'type': 'Self-Attention', 'params': sum(p.numel() for p in model.attention.parameters())},
            {'name': 'Fusion', 'type': 'Dense', 'params': sum(p.numel() for p in model.fusion.parameters())},
            {'name': 'Output', 'type': 'Dense + Sigmoid', 'params': sum(p.numel() for p in model.output_head.parameters())}
        ]
    }
