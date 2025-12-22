"""
Uncertainty Quantification with Monte Carlo Dropout

Provides confidence intervals on predictions using MC Dropout,
a Bayesian approximation technique that uses dropout at inference time.

Reference: INTERNAL_AI_ML_ENHANCEMENT_RESEARCH.md - Tier 2.4
Paper: Gal & Ghahramani (2016) - Dropout as Bayesian Approximation
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import torch
import torch.nn as nn


@dataclass
class UncertaintyResult:
    """Container for uncertainty quantification results."""
    # Point estimate
    mean_prediction: float
    
    # Uncertainty measures
    std_prediction: float
    ci_lower: float  # Lower 95% CI
    ci_upper: float  # Upper 95% CI
    
    # Full distribution
    prediction_samples: np.ndarray
    
    # Quality metrics
    epistemic_uncertainty: float  # Model uncertainty
    total_uncertainty: float
    confidence_score: float  # 0-1, higher = more confident
    
    # QC flags
    high_uncertainty: bool  # True if uncertainty exceeds threshold


def enable_mc_dropout(model: nn.Module):
    """
    Enable dropout layers for Monte Carlo inference.
    
    By default, dropout is disabled during eval(). This function
    enables dropout while keeping other layers (e.g., BatchNorm) in eval mode.
    
    Args:
        model: PyTorch model with Dropout layers
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def disable_mc_dropout(model: nn.Module):
    """
    Disable dropout layers, returning to normal eval mode.
    
    Args:
        model: PyTorch model
    """
    model.eval()


def predict_with_uncertainty(
    model: nn.Module,
    forward_fn: Callable,
    n_samples: int = 50,
    confidence_level: float = 0.95,
    uncertainty_threshold: float = 0.5
) -> UncertaintyResult:
    """
    Make prediction with uncertainty estimation using MC Dropout.
    
    Performs multiple forward passes with dropout enabled to sample
    from the approximate posterior distribution.
    
    Args:
        model: Trained model with Dropout layers
        forward_fn: Function that takes model and returns prediction
        n_samples: Number of MC samples
        confidence_level: Confidence level for CI (default 95%)
        uncertainty_threshold: Threshold for high uncertainty flag
        
    Returns:
        UncertaintyResult with point estimate and uncertainty bounds
    """
    predictions = []
    
    # Enable MC Dropout
    model.train()  # Enables dropout
    enable_mc_dropout(model)
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = forward_fn(model)
            if isinstance(pred, torch.Tensor):
                pred = pred.item()
            predictions.append(pred)
    
    # Disable dropout
    model.eval()
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    
    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(predictions, 100 * alpha / 2)
    ci_upper = np.percentile(predictions, 100 * (1 - alpha / 2))
    
    # Epistemic uncertainty (model uncertainty)
    epistemic = std_pred
    
    # Total uncertainty
    total_uncertainty = std_pred
    
    # Confidence score (inverse of normalized uncertainty)
    # Scale uncertainty relative to prediction magnitude
    if abs(mean_pred) > 0.001:
        relative_uncertainty = std_pred / abs(mean_pred)
    else:
        relative_uncertainty = std_pred
    
    confidence_score = 1.0 / (1.0 + relative_uncertainty)
    
    # High uncertainty flag
    high_uncertainty = relative_uncertainty > uncertainty_threshold
    
    return UncertaintyResult(
        mean_prediction=mean_pred,
        std_prediction=std_pred,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        prediction_samples=predictions,
        epistemic_uncertainty=epistemic,
        total_uncertainty=total_uncertainty,
        confidence_score=confidence_score,
        high_uncertainty=high_uncertainty
    )


def predict_sequence_with_uncertainty(
    model: nn.Module,
    input_data: torch.Tensor,
    n_samples: int = 50,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Make sequence predictions with uncertainty bands.
    
    Useful for predicting depth corrections along the entire log.
    
    Args:
        model: Trained model
        input_data: Input tensor
        n_samples: Number of MC samples
        confidence_level: Confidence level for bands
        
    Returns:
        mean: Mean prediction at each point
        std: Standard deviation at each point
        lower: Lower confidence bound
        upper: Upper confidence bound
    """
    all_predictions = []
    
    model.train()
    enable_mc_dropout(model)
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(input_data)
            if isinstance(output, tuple):
                output = output[0]  # Take first output if multiple
            all_predictions.append(output.numpy())
    
    model.eval()
    
    all_predictions = np.array(all_predictions)  # [n_samples, ...]
    
    # Statistics along sample dimension
    mean = np.mean(all_predictions, axis=0)
    std = np.std(all_predictions, axis=0)
    
    # Confidence bounds
    alpha = 1 - confidence_level
    lower = np.percentile(all_predictions, 100 * alpha / 2, axis=0)
    upper = np.percentile(all_predictions, 100 * (1 - alpha / 2), axis=0)
    
    return mean, std, lower, upper


def compute_calibration_score(
    predictions: np.ndarray,
    true_values: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Compute calibration score for uncertainty estimates.
    
    A well-calibrated model should have approximately (confidence_level)% 
    of true values within the predicted CI.
    
    Args:
        predictions: Predicted values
        true_values: Ground truth values
        ci_lower: Lower bounds
        ci_upper: Upper bounds
        confidence_level: Expected coverage
        
    Returns:
        Calibration score (1.0 = perfectly calibrated)
    """
    within_ci = np.logical_and(
        true_values >= ci_lower,
        true_values <= ci_upper
    )
    actual_coverage = np.mean(within_ci)
    
    # Score: 1 - |expected - actual|
    return 1.0 - abs(confidence_level - actual_coverage)


def decompose_uncertainty(
    model: nn.Module,
    forward_fn: Callable,
    n_samples: int = 50
) -> Tuple[float, float, float]:
    """
    Decompose total uncertainty into epistemic and aleatoric components.
    
    Epistemic: Model uncertainty (can be reduced with more data)
    Aleatoric: Data noise (irreducible)
    
    Note: This is an approximation. True aleatoric uncertainty requires
    a model that predicts variance.
    
    Args:
        model: Trained model
        forward_fn: Forward function
        n_samples: Number of MC samples
        
    Returns:
        total: Total uncertainty
        epistemic: Model uncertainty
        aleatoric: Estimated aleatoric (approximation)
    """
    predictions = []
    
    model.train()
    enable_mc_dropout(model)
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = forward_fn(model)
            if isinstance(pred, torch.Tensor):
                pred = pred.item()
            predictions.append(pred)
    
    model.eval()
    
    predictions = np.array(predictions)
    
    # Total variance
    total_var = np.var(predictions)
    
    # Epistemic = variance of means (but we only have one input, so this is the total)
    epistemic_var = total_var
    
    # Aleatoric approximation (in practice, would need model to predict this)
    aleatoric_var = 0.0  # Cannot estimate without heteroscedastic model
    
    return np.sqrt(total_var), np.sqrt(epistemic_var), np.sqrt(aleatoric_var)


def uncertainty_weighted_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Loss function that weights errors by uncertainty.
    
    Predictions with high uncertainty contribute less to the loss.
    This encourages the model to be confident when it's accurate.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        uncertainties: Predicted uncertainties (log variance)
        epsilon: Small constant for numerical stability
        
    Returns:
        Weighted loss
    """
    # Heteroscedastic loss (Kendall & Gal, 2017)
    precision = torch.exp(-uncertainties)
    diff = (predictions - targets) ** 2
    
    loss = torch.mean(precision * diff + uncertainties)
    
    return loss


class UncertaintyAwareLSTM(nn.Module):
    """
    LSTM that also predicts uncertainty (heteroscedastic model).
    
    Predicts both the shift and log-variance of the prediction.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate heads for mean and variance
        self.mean_head = nn.Linear(hidden_size, 1)
        self.var_head = nn.Linear(hidden_size, 1)  # Outputs log-variance
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.encoder(x)
        scores = self.attention(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        return context, weights.squeeze(-1)
    
    def forward(
        self,
        shallow_log: torch.Tensor,
        deep_log: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: Predicted mean shift
            log_var: Predicted log-variance (aleatoric uncertainty)
            attention: Attention weights
        """
        shallow_ctx, shallow_attn = self.encode(shallow_log)
        deep_ctx, deep_attn = self.encode(deep_log)
        
        combined = torch.cat([shallow_ctx, deep_ctx], dim=-1)
        fused = self.fusion(combined)
        
        mean = self.mean_head(fused)
        log_var = self.var_head(fused)
        
        # Combine attention weights
        attention = (shallow_attn + deep_attn) / 2
        
        return mean, log_var, attention


def train_uncertainty_aware_model(
    model: UncertaintyAwareLSTM,
    shallow_data: torch.Tensor,
    deep_data: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 50,
    learning_rate: float = 0.001,
    progress_callback: Optional[Callable] = None
) -> List[float]:
    """
    Train model with uncertainty-aware loss.
    
    Args:
        model: UncertaintyAwareLSTM
        shallow_data: Shallow log windows
        deep_data: Deep log windows
        targets: Target shifts
        epochs: Training epochs
        learning_rate: Learning rate
        progress_callback: Optional callback
        
    Returns:
        Loss history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        mean, log_var, _ = model(shallow_data, deep_data)
        
        loss = uncertainty_weighted_loss(mean, targets, log_var)
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if progress_callback:
            progress_callback(epoch + 1, loss_val)
    
    return loss_history


def get_prediction_quality_metrics(result: UncertaintyResult) -> dict:
    """
    Generate quality metrics for uncertainty result.
    
    Args:
        result: UncertaintyResult from prediction
        
    Returns:
        Dictionary of quality metrics
    """
    return {
        'point_estimate': result.mean_prediction,
        'uncertainty_range': result.ci_upper - result.ci_lower,
        'relative_uncertainty': result.std_prediction / max(abs(result.mean_prediction), 0.001),
        'confidence_score': result.confidence_score,
        'n_samples': len(result.prediction_samples),
        'quality_grade': _get_quality_grade(result.confidence_score),
        'requires_review': result.high_uncertainty
    }


def _get_quality_grade(confidence: float) -> str:
    """Convert confidence score to quality grade."""
    if confidence >= 0.9:
        return 'A'
    elif confidence >= 0.75:
        return 'B'
    elif confidence >= 0.5:
        return 'C'
    else:
        return 'D'
