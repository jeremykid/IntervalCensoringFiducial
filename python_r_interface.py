"""
Python interface for R-based Interval Censoring Fiducial algorithm.

This module provides a clean interface to run the R algorithm from Python,
handling data conversion and subprocess management.
"""

import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


class IntervalCensoringFiducialR:
    """
    Interface to R-based interval censoring fiducial algorithm.
    
    This class handles running the R algorithm via Rscript subprocess,
    managing temporary files, and converting results back to Python.
    """
    
    def __init__(self, r_script_dir: str = None):
        """
        Initialize the R interface.
        
        Parameters
        ----------
        r_script_dir : str, optional
            Directory containing algorithm.R, LinInterpolation.R, and run_algo_wrapper.R
            If None, uses the same directory as this Python file.
        """
        if r_script_dir is None:
            r_script_dir = Path(__file__).parent
        else:
            r_script_dir = Path(r_script_dir)
            
        self.r_script_dir = r_script_dir
        self.wrapper_script = self.r_script_dir / "run_algo_wrapper.R"
        
        # Validate that required files exist
        required_files = ["algorithm.R", "LinInterpolation.R", "run_algo_wrapper.R"]
        for fname in required_files:
            fpath = self.r_script_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Required R file not found: {fpath}")
    
    def fit_predict(
        self, 
        left: np.ndarray, 
        right: np.ndarray,
        seed: int = 123,
        mfid: int = 1000,
        mburn: int = 100,
        alpha: float = 0.05,
        cleanup: bool = True,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit the fiducial model and predict survival function.
        
        Parameters
        ----------
        left : np.ndarray
            Left endpoints of interval-censored observations
        right : np.ndarray
            Right endpoints of interval-censored observations
            Use np.inf for right-censored observations
        seed : int, default=123
            Random seed for reproducibility
        mfid : int, default=1000
            Number of fiducial samples to generate
        mburn : int, default=100
            Number of burn-in samples
        alpha : float, default=0.05
            Significance level for confidence intervals
        cleanup : bool, default=True
            Whether to remove temporary CSV files after completion
        verbose : bool, default=True
            Whether to print R output
            
        Returns
        -------
        survival_df : pd.DataFrame
            DataFrame with columns:
            - time: time points
            - S_hat: survival function estimate (mean of fiducial samples)
            - S_low: lower confidence bound
            - S_high: upper confidence bound
            - S_point_li: point estimate from linear interpolation
            - F_mean: CDF estimate (mean)
            - F_low: CDF lower bound
            - F_high: CDF upper bound
        metadata : dict
            Additional information about the run
        """
        # Validate inputs
        left = np.asarray(left)
        right = np.asarray(right)
        
        if left.shape != right.shape:
            raise ValueError("left and right must have same shape")
        
        if np.any(left < 0):
            raise ValueError("left values must be non-negative")
        
        if np.any(left > right):
            raise ValueError("left must be <= right for all observations")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_in:
            input_csv = f_in.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_out:
            output_csv = f_out.name
        
        try:
            # Write input data
            input_df = pd.DataFrame({'left': left, 'right': right})
            input_df.to_csv(input_csv, index=False)
            
            if verbose:
                print(f"Input data shape: {input_df.shape}")
                print(f"Left range: [{left.min():.2f}, {left.max():.2f}]")
                finite_right = right[np.isfinite(right)]
                if len(finite_right) > 0:
                    print(f"Right range (finite): [{finite_right.min():.2f}, {finite_right.max():.2f}]")
                print(f"Right-censored: {np.sum(np.isinf(right))} / {len(right)}")
            
            # Build command
            cmd = [
                "Rscript",
                str(self.wrapper_script),
                input_csv,
                output_csv,
                str(seed),
                str(mfid),
                str(mburn),
                str(alpha)
            ]
            
            if verbose:
                print(f"\nRunning R algorithm...")
                print(f"Command: {' '.join(cmd)}\n")
            
            # Run R script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.r_script_dir)
            )
            
            # Print R output
            if verbose and result.stdout:
                print("=== R Output ===")
                print(result.stdout)
            
            if result.returncode != 0:
                print("=== R Error ===")
                print(result.stderr)
                raise RuntimeError(f"R script failed with code {result.returncode}")
            
            # Read results
            survival_df = pd.read_csv(output_csv)
            survival_df = survival_df.set_index('time')
            
            # Extract metadata
            metadata = {
                'seed': seed,
                'mfid': mfid,
                'mburn': mburn,
                'alpha': alpha,
                'n_obs': len(left),
                'n_right_censored': np.sum(np.isinf(right)),
                'time_range': (survival_df.index.min(), survival_df.index.max()),
                'r_stdout': result.stdout,
                'r_stderr': result.stderr
            }
            
            if verbose:
                print(f"\n=== Results Summary ===")
                print(f"Time points: {len(survival_df)}")
                print(f"Time range: [{survival_df.index.min():.2f}, {survival_df.index.max():.2f}]")
                print(f"S(t) at first time: {survival_df['S_hat'].iloc[0]:.4f}")
                print(f"S(t) at last time: {survival_df['S_hat'].iloc[-1]:.4f}")
            
            return survival_df, metadata
            
        finally:
            # Cleanup temporary files
            if cleanup:
                for fpath in [input_csv, output_csv]:
                    try:
                        os.unlink(fpath)
                    except:
                        pass


def run_fiducial_algorithm(
    df: pd.DataFrame,
    left_col: str = 'left',
    right_col: str = 'right',
    r_script_dir: str = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to run fiducial algorithm on a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing interval-censored data
    left_col : str, default='left'
        Name of column containing left endpoints
    right_col : str, default='right'
        Name of column containing right endpoints
    r_script_dir : str, optional
        Directory containing R scripts
    **kwargs
        Additional arguments passed to IntervalCensoringFiducialR.fit_predict
        
    Returns
    -------
    survival_df : pd.DataFrame
        Survival function estimates with confidence intervals
    metadata : dict
        Run metadata
    """
    if left_col not in df.columns or right_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{left_col}' and '{right_col}'")
    
    fitter = IntervalCensoringFiducialR(r_script_dir=r_script_dir)
    return fitter.fit_predict(
        left=df[left_col].values,
        right=df[right_col].values,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Create synthetic interval-censored data
    np.random.seed(42)
    n = 100
    
    # Simulate event times
    true_times = np.random.exponential(scale=10, size=n)
    
    # Simulate interval censoring
    # Each observation has a random interval width
    interval_widths = np.random.uniform(1, 5, size=n)
    left = np.maximum(0, true_times - interval_widths/2)
    right = true_times + interval_widths/2
    
    # Add some right-censored observations (set right = inf)
    n_censored = 20
    censored_idx = np.random.choice(n, size=n_censored, replace=False)
    right[censored_idx] = np.inf
    
    # Create DataFrame
    df = pd.DataFrame({'left': left, 'right': right})
    
    print("=== Running Fiducial Algorithm on Synthetic Data ===\n")
    
    # Run algorithm
    survival_df, metadata = run_fiducial_algorithm(
        df,
        seed=123,
        mfid=500,  # Fewer samples for quick testing
        mburn=50,
        alpha=0.05,
        verbose=True
    )
    
    print("\n=== Survival Function Estimates (first 10 time points) ===")
    print(survival_df.head(10))
    
    print("\n=== Survival Function Estimates (last 10 time points) ===")
    print(survival_df.tail(10))
    
    # Optional: plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot survival function
        ax1.plot(survival_df.index, survival_df['S_hat'], 'b-', label='S(t) estimate', linewidth=2)
        ax1.fill_between(survival_df.index, survival_df['S_low'], survival_df['S_high'], 
                         alpha=0.3, label=f'{100*(1-metadata["alpha"])}% CI')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Survival Probability')
        ax1.set_title('Survival Function with Confidence Intervals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot CDF
        ax2.plot(survival_df.index, survival_df['F_mean'], 'r-', label='F(t) estimate', linewidth=2)
        ax2.fill_between(survival_df.index, survival_df['F_low'], survival_df['F_high'], 
                         alpha=0.3, label=f'{100*(1-metadata["alpha"])}% CI')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cumulative Distribution Function')
        ax2.set_title('CDF with Confidence Intervals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fiducial_survival_estimate.png', dpi=150)
        print("\nPlot saved to: fiducial_survival_estimate.png")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plot")
