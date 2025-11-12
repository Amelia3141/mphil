###
# GPU-accelerated ZScoreSustainMissingData implementation
# 
# This module provides a PyTorch-based implementation of ZScoreSustainMissingData
# with significant performance improvements through GPU acceleration.
#
# Authors: GPU Migration Team
###

import torch
import numpy as np
from typing import Optional, Tuple, Union
from .ZScoreSustainMissingData import ZscoreSustainMissingData
from .torch_backend import TorchSustainBackend, create_torch_backend
from .torch_data_classes import TorchZScoreSustainData, create_torch_zscore_data
from .torch_likelihood import TorchZScoreMissingDataLikelihoodCalculator, create_zscore_missing_data_likelihood_calculator


class TorchZScoreSustainMissingData(ZscoreSustainMissingData):
    """
    GPU-accelerated version of ZScoreSustainMissingData.
    
    This class extends the original ZScoreSustainMissingData with PyTorch-based
    GPU acceleration while maintaining full compatibility with the original API.
    """
    
    def __init__(self,
                 data: np.ndarray,
                 Z_vals: np.ndarray,
                 Z_max: np.ndarray,
                 biomarker_labels: list,
                 N_startpoints: int,
                 N_S_max: int,
                 N_iterations_MCMC: int,
                 output_folder: str,
                 dataset_name: str,
                 use_parallel_startpoints: bool,
                 seed: Optional[int] = None,
                 use_gpu: bool = True,
                 device_id: Optional[int] = None):
        """
        Initialize GPU-accelerated ZScoreSustainMissingData.
        
        Args:
            data: Z-score data array (M, B) where M=subjects, B=biomarkers
            Z_vals: Z-score thresholds matrix (B, num_thresholds)
            Z_max: Maximum z-scores for each biomarker (B,)
            biomarker_labels: List of biomarker names
            N_startpoints: Number of startpoints for optimization
            N_S_max: Maximum number of subtypes
            N_iterations_MCMC: Number of MCMC iterations
            output_folder: Output directory for results
            dataset_name: Name for output files
            use_parallel_startpoints: Whether to use parallel startpoints
            seed: Random seed
            use_gpu: Whether to use GPU acceleration
            device_id: Specific GPU device ID
        """
        # Initialize the original class first
        super().__init__(
            data, Z_vals, Z_max, biomarker_labels,
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints, seed
        )
        
        # Initialize PyTorch backend
        self.torch_backend = create_torch_backend(use_gpu=use_gpu, device_id=device_id)
        self.use_gpu = self.torch_backend.use_gpu
        
        # Create PyTorch-enabled data object
        # Access the sustainData attribute directly (it should be available after super().__init__)
        sustain_data = getattr(self, '_ZscoreSustainMissingData__sustainData', None)
        if sustain_data is None:
            # Fallback: try different possible attribute names
            for attr_name in ['__sustainData', '_sustainData', 'sustainData']:
                sustain_data = getattr(self, attr_name, None)
                if sustain_data is not None:
                    break
        
        if sustain_data is None:
            raise AttributeError("Could not find sustainData attribute. Available attributes: " + 
                               str([attr for attr in dir(self) if not attr.startswith('__')]))
        
        self.torch_sustain_data = create_torch_zscore_data(
            data, sustain_data.getNumStages(), self.torch_backend
        )
        
        # Create GPU-accelerated likelihood calculator
        self.torch_likelihood_calculator = create_zscore_missing_data_likelihood_calculator(
            self.torch_backend,
            self.stage_biomarker_index,
            self.stage_zscore,
            np.array(self.min_biomarker_zscore),
            np.array(self.max_biomarker_zscore),
            np.array(self.std_biomarker_zscore)
        )
        
        print(f"TorchZScoreSustainMissingData initialized with {'GPU' if self.use_gpu else 'CPU'} acceleration")
    
    def _calculate_likelihood(self, sustainData, S: np.ndarray, f: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU-accelerated likelihood computation.
        
        This method uses the PyTorch backend for GPU acceleration while maintaining
        the same interface as the original implementation.
        
        Args:
            sustainData: SuStaIn data object
            S: Sequence matrix (N_S, N)
            f: Fraction vector (N_S,)
            
        Returns:
            Tuple of (loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k)
        """
        # Use GPU-accelerated computation if available
        if self.use_gpu:
            try:
                return self.torch_likelihood_calculator.calculate_likelihood(
                    self.torch_sustain_data, S, f
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("GPU out of memory, falling back to CPU computation")
                    self.torch_backend.clear_cache()
                    # Fall back to original CPU implementation
                    return super()._calculate_likelihood(sustainData, S, f)
                else:
                    raise
        else:
            # Use original CPU implementation
            return super()._calculate_likelihood(sustainData, S, f)
    
    def _calculate_likelihood_stage(self, sustainData, S: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated stage likelihood computation.
        
        Args:
            sustainData: SuStaIn data object
            S: Single sequence (N,)
            
        Returns:
            Likelihood array (M, N+1)
        """
        if self.use_gpu:
            try:
                # Convert to PyTorch tensor and compute
                S_torch = self.torch_backend.to_torch(S)
                result_torch = self.torch_likelihood_calculator._calculate_likelihood_stage_torch(
                    self.torch_sustain_data, S_torch
                )
                return self.torch_backend.to_numpy(result_torch)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("GPU out of memory, falling back to CPU computation")
                    self.torch_backend.clear_cache()
                    return super()._calculate_likelihood_stage(sustainData, S)
                else:
                    raise
        else:
            return super()._calculate_likelihood_stage(sustainData, S)
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics from GPU computations."""
        return self.torch_backend.get_performance_stats()
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if self.use_gpu:
            self.torch_backend.clear_cache()
    
    def switch_to_cpu(self):
        """Switch to CPU-only computation."""
        self.use_gpu = False
        print("Switched to CPU-only computation")
    
    def switch_to_gpu(self, device_id: Optional[int] = None):
        """Switch to GPU computation."""
        try:
            self.torch_backend = create_torch_backend(use_gpu=True, device_id=device_id)
            self.use_gpu = self.torch_backend.use_gpu
            
            if self.use_gpu:
                # Recreate PyTorch components
                # Access the sustainData attribute
                sustain_data = getattr(self, '_ZscoreSustainMissingData__sustainData', None)
                if sustain_data is None:
                    # Fallback: try different possible attribute names
                    for attr_name in ['__sustainData', '_sustainData', 'sustainData']:
                        sustain_data = getattr(self, attr_name, None)
                        if sustain_data is not None:
                            break
                
                if sustain_data is None:
                    raise AttributeError("Could not find sustainData attribute")
                
                self.torch_sustain_data = create_torch_zscore_data(
                    sustain_data.data, sustain_data.getNumStages(), self.torch_backend
                )
                
                self.torch_likelihood_calculator = create_zscore_missing_data_likelihood_calculator(
                    self.torch_backend,
                    self.stage_biomarker_index,
                    self.stage_zscore,
                    np.array(self.min_biomarker_zscore),
                    np.array(self.max_biomarker_zscore),
                    np.array(self.std_biomarker_zscore)
                )
                print("Switched to GPU computation")
            else:
                print("GPU not available, staying on CPU")
        except Exception as e:
            print(f"Failed to switch to GPU: {e}")
            self.use_gpu = False


# Factory function for creating GPU-accelerated instances
def create_torch_zscore_sustain_missing_data(
    data: np.ndarray,
    Z_vals: np.ndarray,
    Z_max: np.ndarray,
    biomarker_labels: list,
    N_startpoints: int = 25,
    N_S_max: int = 3,
    N_iterations_MCMC: int = 100000,
    output_folder: str = "./output",
    dataset_name: str = "dataset",
    use_parallel_startpoints: bool = True,
    seed: Optional[int] = None,
    use_gpu: bool = True,
    device_id: Optional[int] = None
) -> TorchZScoreSustainMissingData:
    """
    Factory function to create a GPU-accelerated ZScoreSustainMissingData instance.
    
    Args:
        data: Z-score data array (M, B)
        Z_vals: Z-score thresholds matrix (B, num_thresholds)
        Z_max: Maximum z-scores for each biomarker (B,)
        biomarker_labels: List of biomarker names
        N_startpoints: Number of startpoints for optimization
        N_S_max: Maximum number of subtypes
        N_iterations_MCMC: Number of MCMC iterations
        output_folder: Output directory for results
        dataset_name: Name for output files
        use_parallel_startpoints: Whether to use parallel startpoints
        seed: Random seed
        use_gpu: Whether to use GPU acceleration
        device_id: Specific GPU device ID
        
    Returns:
        GPU-accelerated ZScoreSustainMissingData instance
    """
    return TorchZScoreSustainMissingData(
        data, Z_vals, Z_max, biomarker_labels,
        N_startpoints, N_S_max, N_iterations_MCMC,
        output_folder, dataset_name, use_parallel_startpoints,
        seed, use_gpu, device_id
    )


# Example usage and testing functions
def benchmark_gpu_vs_cpu(data: np.ndarray, Z_vals: np.ndarray, Z_max: np.ndarray,
                        biomarker_labels: list, num_iterations: int = 10) -> dict:
    """
    Benchmark GPU vs CPU performance for ZScoreSustainMissingData.
    
    Args:
        data: Test data
        Z_vals: Z-score thresholds
        Z_max: Maximum z-scores
        biomarker_labels: Biomarker labels
        num_iterations: Number of benchmark iterations
        
    Returns:
        Dictionary with performance comparison
    """
    import time
    
    # Create test sequences and fractions
    N = Z_vals.shape[0] * Z_vals.shape[1]  # Approximate number of stages
    S_test = np.random.permutation(N).reshape(1, N)
    f_test = np.array([1.0])
    
    # Benchmark CPU version
    cpu_times = []
    cpu_sustain = ZscoreSustainMissingData(
        data, Z_vals, Z_max, biomarker_labels,
        1, 1, 100, "./temp", "cpu_test", False, 42
    )
    
    for _ in range(num_iterations):
        start_time = time.time()
        _ = cpu_sustain._calculate_likelihood_stage(cpu_sustain._ZscoreSustainMissingData__sustainData, S_test[0])
        cpu_times.append(time.time() - start_time)
    
    # Benchmark GPU version
    gpu_times = []
    gpu_sustain = TorchZScoreSustainMissingData(
        data, Z_vals, Z_max, biomarker_labels,
        1, 1, 100, "./temp", "gpu_test", False, 42, use_gpu=True
    )
    
    for _ in range(num_iterations):
        start_time = time.time()
        _ = gpu_sustain._calculate_likelihood_stage(gpu_sustain._ZscoreSustainMissingData__sustainData, S_test[0])
        gpu_times.append(time.time() - start_time)
    
    return {
        'cpu_mean_time': np.mean(cpu_times),
        'cpu_std_time': np.std(cpu_times),
        'gpu_mean_time': np.mean(gpu_times),
        'gpu_std_time': np.std(gpu_times),
        'speedup': np.mean(cpu_times) / np.mean(gpu_times),
        'gpu_available': gpu_sustain.use_gpu
    }
