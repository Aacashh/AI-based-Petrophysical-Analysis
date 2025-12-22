"""
Bayesian Hyperparameter Optimization for Log Splicing

Uses Gaussian Process optimization to find optimal splicing parameters
(search window, DTW window, grid step) that minimize alignment cost.

Reference: INTERNAL_AI_ML_ENHANCEMENT_RESEARCH.md - Tier 1.2
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable, Dict, Any
import sys
import os

# Add parent paths for shared module access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.splicing import splice_logs, SplicingResult


@dataclass
class OptimizationResult:
    """Container for Bayesian optimization results."""
    # Best parameters found
    best_search_window: float
    best_dtw_window: float
    best_grid_step: float
    
    # Optimization metrics
    best_cost: float
    n_iterations: int
    convergence_history: List[float]  # Best cost at each iteration
    
    # Parameter exploration history
    param_history: List[Dict[str, float]]  # All evaluated parameter sets
    cost_history: List[float]  # Cost for each evaluation
    
    # Uncertainty estimates
    param_uncertainties: Dict[str, Tuple[float, float]]  # {param: (lower_95, upper_95)}
    
    # Comparison with defaults
    default_cost: Optional[float] = None
    improvement_percent: Optional[float] = None


@dataclass 
class OptimizationConfig:
    """Configuration for Bayesian optimization."""
    # Parameter bounds
    search_window_bounds: Tuple[float, float] = (5.0, 50.0)
    dtw_window_bounds: Tuple[float, float] = (1.0, 15.0)
    grid_step_bounds: Tuple[float, float] = (0.05, 0.5)
    
    # Optimization settings
    n_calls: int = 25  # Total function evaluations
    n_initial_points: int = 5  # Random initial samples
    random_state: int = 42
    
    # Convergence criteria
    min_improvement: float = 0.01  # Stop if improvement < this
    patience: int = 5  # Iterations without improvement before early stop


def create_objective_function(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    progress_callback: Optional[Callable[[int, float, Dict], None]] = None
) -> Callable:
    """
    Create objective function for optimization.
    
    The objective minimizes the DTW cost (alignment quality).
    Lower cost = better alignment.
    
    Args:
        shallow_depth: Shallow log depth array
        shallow_signal: Shallow log signal array
        deep_depth: Deep log depth array
        deep_signal: Deep log signal array
        progress_callback: Optional callback(iteration, cost, params)
        
    Returns:
        Objective function for skopt
    """
    iteration_counter = [0]
    
    def objective(params):
        search_window, dtw_window, grid_step = params
        
        try:
            result = splice_logs(
                shallow_depth=shallow_depth,
                shallow_signal=shallow_signal,
                deep_depth=deep_depth,
                deep_signal=deep_signal,
                grid_step=grid_step,
                max_search_meters=search_window,
                max_elastic_meters=dtw_window,
                progress_callback=None
            )
            
            cost = result.dtw_cost
            
            # Add small penalty for extreme parameters to encourage reasonable values
            penalty = 0.0
            if search_window > 40:
                penalty += 0.01 * (search_window - 40)
            if dtw_window > 10:
                penalty += 0.01 * (dtw_window - 10)
            
            total_cost = cost + penalty
            
        except Exception as e:
            # Return high cost for failed evaluations
            total_cost = 1e6
        
        iteration_counter[0] += 1
        
        if progress_callback:
            progress_callback(
                iteration_counter[0],
                total_cost,
                {
                    'search_window': search_window,
                    'dtw_window': dtw_window,
                    'grid_step': grid_step
                }
            )
        
        return total_cost
    
    return objective


def optimize_splicing_params(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    config: Optional[OptimizationConfig] = None,
    progress_callback: Optional[Callable[[int, float, Dict], None]] = None
) -> OptimizationResult:
    """
    Use Bayesian optimization to find optimal splicing parameters.
    
    This uses Gaussian Process surrogate model to efficiently explore
    the parameter space and find parameters that minimize DTW cost.
    
    Args:
        shallow_depth: Shallow log depth array
        shallow_signal: Shallow log signal array
        deep_depth: Deep log depth array
        deep_signal: Deep log signal array
        config: Optimization configuration (optional)
        progress_callback: Optional callback for progress updates
        
    Returns:
        OptimizationResult with best parameters and metrics
    """
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.callbacks import EarlyStopper
    
    if config is None:
        config = OptimizationConfig()
    
    # Track optimization history
    param_history = []
    cost_history = []
    convergence_history = []
    best_so_far = float('inf')
    
    def tracking_callback(iteration, cost, params):
        nonlocal best_so_far
        param_history.append(params.copy())
        cost_history.append(cost)
        best_so_far = min(best_so_far, cost)
        convergence_history.append(best_so_far)
        
        if progress_callback:
            progress_callback(iteration, cost, params)
    
    # Create objective function
    objective = create_objective_function(
        shallow_depth, shallow_signal,
        deep_depth, deep_signal,
        progress_callback=tracking_callback
    )
    
    # Define parameter space
    space = [
        Real(config.search_window_bounds[0], config.search_window_bounds[1], 
             name='search_window', prior='uniform'),
        Real(config.dtw_window_bounds[0], config.dtw_window_bounds[1],
             name='dtw_window', prior='uniform'),
        Real(config.grid_step_bounds[0], config.grid_step_bounds[1],
             name='grid_step', prior='log-uniform')
    ]
    
    # Define early stopping callback
    class ConvergenceChecker:
        def __init__(self, min_improvement, patience):
            self.min_improvement = min_improvement
            self.patience = patience
            self.best = float('inf')
            self.no_improvement_count = 0
            
        def __call__(self, res):
            current_best = res.fun
            if self.best - current_best > self.min_improvement:
                self.best = current_best
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            return self.no_improvement_count >= self.patience
    
    convergence_checker = ConvergenceChecker(
        config.min_improvement, config.patience
    )
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=config.n_calls,
        n_initial_points=config.n_initial_points,
        random_state=config.random_state,
        callback=[convergence_checker],
        acq_func='EI',  # Expected Improvement
        n_jobs=1  # Sequential for progress tracking
    )
    
    # Extract best parameters
    best_search_window, best_dtw_window, best_grid_step = result.x
    best_cost = result.fun
    
    # Estimate parameter uncertainties from posterior
    param_uncertainties = estimate_uncertainties(
        param_history, cost_history,
        best_search_window, best_dtw_window, best_grid_step
    )
    
    # Calculate default cost for comparison
    default_cost = None
    improvement_percent = None
    try:
        default_result = splice_logs(
            shallow_depth=shallow_depth,
            shallow_signal=shallow_signal,
            deep_depth=deep_depth,
            deep_signal=deep_signal,
            grid_step=0.1524,  # Default
            max_search_meters=20.0,  # Default
            max_elastic_meters=5.0,  # Default
            progress_callback=None
        )
        default_cost = default_result.dtw_cost
        
        if default_cost > 0:
            improvement_percent = 100 * (default_cost - best_cost) / default_cost
    except Exception:
        pass
    
    return OptimizationResult(
        best_search_window=best_search_window,
        best_dtw_window=best_dtw_window,
        best_grid_step=best_grid_step,
        best_cost=best_cost,
        n_iterations=len(cost_history),
        convergence_history=convergence_history,
        param_history=param_history,
        cost_history=cost_history,
        param_uncertainties=param_uncertainties,
        default_cost=default_cost,
        improvement_percent=improvement_percent
    )


def estimate_uncertainties(
    param_history: List[Dict[str, float]],
    cost_history: List[float],
    best_search: float,
    best_dtw: float,
    best_grid: float,
    percentile: float = 10
) -> Dict[str, Tuple[float, float]]:
    """
    Estimate parameter uncertainties from optimization history.
    
    Uses the top percentile of evaluations to estimate confidence bounds.
    
    Args:
        param_history: History of evaluated parameters
        cost_history: History of costs
        best_*: Best parameter values
        percentile: Top percentile to consider
        
    Returns:
        Dictionary of {param_name: (lower_bound, upper_bound)}
    """
    if not param_history:
        return {
            'search_window': (best_search * 0.9, best_search * 1.1),
            'dtw_window': (best_dtw * 0.9, best_dtw * 1.1),
            'grid_step': (best_grid * 0.9, best_grid * 1.1)
        }
    
    # Get top performing parameter sets
    costs = np.array(cost_history)
    threshold = np.percentile(costs, percentile)
    
    top_params = [p for p, c in zip(param_history, cost_history) if c <= threshold]
    
    if len(top_params) < 3:
        # Not enough data, return wide bounds
        return {
            'search_window': (best_search * 0.8, best_search * 1.2),
            'dtw_window': (best_dtw * 0.8, best_dtw * 1.2),
            'grid_step': (best_grid * 0.8, best_grid * 1.2)
        }
    
    # Extract arrays for each parameter
    search_vals = [p['search_window'] for p in top_params]
    dtw_vals = [p['dtw_window'] for p in top_params]
    grid_vals = [p['grid_step'] for p in top_params]
    
    return {
        'search_window': (np.percentile(search_vals, 5), np.percentile(search_vals, 95)),
        'dtw_window': (np.percentile(dtw_vals, 5), np.percentile(dtw_vals, 95)),
        'grid_step': (np.percentile(grid_vals, 5), np.percentile(grid_vals, 95))
    }


def run_optimization_with_restarts(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    n_restarts: int = 3,
    config: Optional[OptimizationConfig] = None,
    progress_callback: Optional[Callable] = None
) -> OptimizationResult:
    """
    Run optimization multiple times with different random seeds.
    
    This helps avoid local minima and provides more robust results.
    
    Args:
        shallow_depth, shallow_signal: Shallow log data
        deep_depth, deep_signal: Deep log data
        n_restarts: Number of optimization runs
        config: Optimization config
        progress_callback: Progress callback
        
    Returns:
        Best result across all runs
    """
    if config is None:
        config = OptimizationConfig()
    
    best_result = None
    
    for i in range(n_restarts):
        # Update random seed
        restart_config = OptimizationConfig(
            search_window_bounds=config.search_window_bounds,
            dtw_window_bounds=config.dtw_window_bounds,
            grid_step_bounds=config.grid_step_bounds,
            n_calls=config.n_calls // n_restarts,  # Split budget
            n_initial_points=config.n_initial_points,
            random_state=config.random_state + i,
            min_improvement=config.min_improvement,
            patience=config.patience
        )
        
        result = optimize_splicing_params(
            shallow_depth, shallow_signal,
            deep_depth, deep_signal,
            config=restart_config,
            progress_callback=progress_callback
        )
        
        if best_result is None or result.best_cost < best_result.best_cost:
            best_result = result
    
    return best_result


def quick_grid_search(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    progress_callback: Optional[Callable] = None
) -> OptimizationResult:
    """
    Fast grid search for quick parameter exploration.
    
    Less accurate than Bayesian optimization but much faster.
    Useful for initial exploration or when time is limited.
    
    Args:
        shallow_depth, shallow_signal: Shallow log data
        deep_depth, deep_signal: Deep log data
        progress_callback: Progress callback
        
    Returns:
        OptimizationResult with best parameters from grid
    """
    # Coarse grid
    search_windows = [10.0, 20.0, 30.0, 40.0]
    dtw_windows = [2.0, 5.0, 8.0, 12.0]
    grid_steps = [0.1, 0.15, 0.2, 0.3]
    
    best_cost = float('inf')
    best_params = None
    param_history = []
    cost_history = []
    iteration = 0
    
    for sw in search_windows:
        for dw in dtw_windows:
            for gs in grid_steps:
                iteration += 1
                
                try:
                    result = splice_logs(
                        shallow_depth=shallow_depth,
                        shallow_signal=shallow_signal,
                        deep_depth=deep_depth,
                        deep_signal=deep_signal,
                        grid_step=gs,
                        max_search_meters=sw,
                        max_elastic_meters=dw,
                        progress_callback=None
                    )
                    cost = result.dtw_cost
                except Exception:
                    cost = 1e6
                
                params = {'search_window': sw, 'dtw_window': dw, 'grid_step': gs}
                param_history.append(params)
                cost_history.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_params = params
                
                if progress_callback:
                    progress_callback(iteration, cost, params)
    
    # Build convergence history
    convergence = []
    best_so_far = float('inf')
    for c in cost_history:
        best_so_far = min(best_so_far, c)
        convergence.append(best_so_far)
    
    return OptimizationResult(
        best_search_window=best_params['search_window'],
        best_dtw_window=best_params['dtw_window'],
        best_grid_step=best_params['grid_step'],
        best_cost=best_cost,
        n_iterations=len(cost_history),
        convergence_history=convergence,
        param_history=param_history,
        cost_history=cost_history,
        param_uncertainties={
            'search_window': (best_params['search_window'] - 5, best_params['search_window'] + 5),
            'dtw_window': (best_params['dtw_window'] - 2, best_params['dtw_window'] + 2),
            'grid_step': (best_params['grid_step'] * 0.8, best_params['grid_step'] * 1.2)
        },
        default_cost=None,
        improvement_percent=None
    )
