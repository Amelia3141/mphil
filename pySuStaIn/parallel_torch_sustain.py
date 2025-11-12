###
# Parallel Torch SuStaIn Implementation
# 
# This module provides parallel execution capabilities for TorchZScoreSustainMissingData
# with both CPU and GPU acceleration options.
#
# Authors: GPU Migration Team
###

import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Any
from .TorchZScoreSustainMissingData import TorchZScoreSustainMissingData
from .parallel_mcmc import ParallelMCMCManager, integrate_parallel_mcmc_with_sustain


class ParallelTorchZScoreSustainMissingData(TorchZScoreSustainMissingData):
    """
    GPU-accelerated ZScoreSustainMissingData with parallel MCMC capabilities.
    
    This class extends TorchZScoreSustainMissingData with parallel execution
    of multiple MCMC chains for significant performance improvements.
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
                 device_id: Optional[int] = None,
                 # New parallel MCMC parameters
                 use_parallel_mcmc: bool = True,
                 n_mcmc_chains: int = 4,
                 mcmc_backend: str = 'process',
                 parallel_workers: Optional[int] = None):
        """
        Initialize parallel GPU-accelerated ZScoreSustainMissingData.
        
        Args:
            data: Z-score data array (M, B) where M=subjects, B=biomarkers
            Z_vals: Z-score thresholds matrix (B, num_thresholds)
            Z_max: Maximum z-scores for each biomarker (B,)
            biomarker_labels: List of biomarker names
            N_startpoints: Number of startpoints for optimization
            N_S_max: Maximum number of subtypes
            N_iterations_MCMC: Number of MCMC iterations per chain
            output_folder: Output directory for results
            dataset_name: Name for output files
            use_parallel_startpoints: Whether to use parallel startpoints
            seed: Random seed
            use_gpu: Whether to use GPU acceleration
            device_id: Specific GPU device ID
            use_parallel_mcmc: Whether to use parallel MCMC chains
            n_mcmc_chains: Number of MCMC chains to run in parallel
            mcmc_backend: Backend for parallel MCMC ('process', 'thread', 'gpu')
            parallel_workers: Number of parallel workers (default: min(n_mcmc_chains, cpu_count))
        """
        # Initialize parent class
        super().__init__(
            data, Z_vals, Z_max, biomarker_labels,
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints,
            seed, use_gpu, device_id
        )
        
        # Ensure the __sustainData attribute is accessible for AbstractSustain methods
        # The parent class stores it as _ZscoreSustainMissingData__sustainData due to name mangling
        if hasattr(self, '_ZscoreSustainMissingData__sustainData'):
            self._AbstractSustain__sustainData = self._ZscoreSustainMissingData__sustainData
        
        # Store parallel MCMC parameters
        self.use_parallel_mcmc = use_parallel_mcmc
        self.n_mcmc_chains = n_mcmc_chains
        self.mcmc_backend = mcmc_backend
        self.parallel_workers = parallel_workers
        
        # Initialize parallel MCMC manager if requested
        if self.use_parallel_mcmc:
            self._setup_parallel_mcmc()
        
        print(f"ParallelTorchZScoreSustainMissingData initialized with:")
        print(f"  - GPU acceleration: {'Yes' if self.use_gpu else 'No'}")
        print(f"  - Parallel MCMC: {'Yes' if self.use_parallel_mcmc else 'No'}")
        if self.use_parallel_mcmc:
            print(f"  - MCMC chains: {self.n_mcmc_chains}")
            print(f"  - MCMC backend: {self.mcmc_backend}")
    
    def _setup_parallel_mcmc(self):
        """Setup parallel MCMC manager."""
        self.parallel_mcmc_manager = ParallelMCMCManager(
            n_chains=self.n_mcmc_chains,
            n_workers=self.parallel_workers,
            backend=self.mcmc_backend,
            use_gpu=self.use_gpu,
            device_ids=[self.torch_backend.device_manager.device.index] if self.use_gpu else None
        )
        
        # Override the uncertainty estimation method
        self._original_estimate_uncertainty = super()._estimate_uncertainty_sustain_model
        self._estimate_uncertainty_sustain_model = self._parallel_estimate_uncertainty_sustain_model
    
    def _parallel_estimate_uncertainty_sustain_model(self, sustainData, seq_init, f_init):
        """
        Parallel version of uncertainty estimation using multiple MCMC chains.
        
        Args:
            sustainData: SuStaIn data object
            seq_init: Initial sequence matrix
            f_init: Initial fraction vector
            
        Returns:
            Tuple of (ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood)
        """
        if not self.use_parallel_mcmc:
            # Fall back to original method
            return self._original_estimate_uncertainty(sustainData, seq_init, f_init)
        
        print(f"Running parallel MCMC with {self.n_mcmc_chains} chains...")
        start_time = time.time()
        
        # Get MCMC settings
        seq_sigma_opt, f_sigma_opt = self._optimise_mcmc_settings(sustainData, seq_init, f_init)
        
        # Generate random seeds for each chain
        seeds = [np.random.randint(0, 2**32) for _ in range(self.n_mcmc_chains)]
        
        # Run parallel MCMC
        samples_sequences, samples_fs, samples_likelihoods, chain_times = \
            self.parallel_mcmc_manager.run_parallel_mcmc(
                self, sustainData, seq_init, f_init,
                self.N_iterations_MCMC, seq_sigma_opt, f_sigma_opt, seeds
            )
        
        # Combine results from all chains
        from .parallel_mcmc import combine_mcmc_results
        combined_sequences, combined_fs, combined_likelihoods, stats = combine_mcmc_results(
            samples_sequences, samples_fs, samples_likelihoods, chain_times
        )
        
        # Find best result across all chains
        best_idx = np.argmax(combined_likelihoods)
        ml_sequence = combined_sequences[:, :, best_idx]
        ml_f = combined_fs[:, best_idx]
        ml_likelihood = combined_likelihoods[best_idx]
        
        total_time = time.time() - start_time
        
        print(f"Parallel MCMC completed in {total_time:.2f} seconds")
        print(f"  - Total iterations: {stats['total_iterations']}")
        print(f"  - Speedup: {stats['speedup']:.2f}x")
        print(f"  - Efficiency: {stats['efficiency']:.2f}")
        
        return ml_sequence, ml_f, ml_likelihood, combined_sequences, combined_fs, combined_likelihoods
    
    def benchmark_parallel_performance(self, 
                                     test_data: Optional[np.ndarray] = None,
                                     n_iterations: int = 1000) -> Dict[str, Any]:
        """
        Benchmark parallel MCMC performance.
        
        Args:
            test_data: Test data to use (default: use current data)
            n_iterations: Number of iterations for benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        if test_data is None:
            test_data = self._ZscoreSustainMissingData__sustainData.data
        
        # Create test sequences
        N = self._ZscoreSustainMissingData__sustainData.getNumStages()
        seq_init = np.random.permutation(N).reshape(1, N)
        f_init = np.array([1.0])
        
        from .parallel_mcmc import benchmark_parallel_mcmc
        
        results = benchmark_parallel_mcmc(
            self, self._ZscoreSustainMissingData__sustainData,
            seq_init, f_init, n_iterations,
            n_chains_list=[1, 2, 4, 8]
        )
        
        return results
    
    def get_parallel_stats(self) -> Dict[str, Any]:
        """Get statistics about parallel execution."""
        if not hasattr(self, 'parallel_mcmc_manager'):
            return {'parallel_mcmc': False}
        
        return {
            'parallel_mcmc': True,
            'n_chains': self.n_mcmc_chains,
            'backend': self.mcmc_backend,
            'workers': self.parallel_workers,
            'gpu_acceleration': self.use_gpu
        }
    
    def switch_parallel_backend(self, backend: str, n_workers: Optional[int] = None):
        """
        Switch parallel MCMC backend.
        
        Args:
            backend: New backend ('process', 'thread', 'gpu')
            n_workers: Number of workers (default: keep current)
        """
        if not self.use_parallel_mcmc:
            print("Parallel MCMC not enabled")
            return
        
        if n_workers is None:
            n_workers = self.parallel_workers
        
        self.mcmc_backend = backend
        self.parallel_workers = n_workers
        
        # Recreate parallel manager
        self._setup_parallel_mcmc()
        
        print(f"Switched to {backend} backend with {n_workers} workers")
    
    def disable_parallel_mcmc(self):
        """Disable parallel MCMC and revert to original method."""
        self.use_parallel_mcmc = False
        self._estimate_uncertainty_sustain_model = self._original_estimate_uncertainty
        print("Parallel MCMC disabled")


# Factory function for creating parallel instances
def create_parallel_torch_zscore_sustain_missing_data(
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
    device_id: Optional[int] = None,
    use_parallel_mcmc: bool = True,
    n_mcmc_chains: int = 4,
    mcmc_backend: str = 'process',
    parallel_workers: Optional[int] = None
) -> ParallelTorchZScoreSustainMissingData:
    """
    Factory function to create a parallel GPU-accelerated ZScoreSustainMissingData instance.
    
    Args:
        data: Z-score data array (M, B)
        Z_vals: Z-score thresholds matrix (B, num_thresholds)
        Z_max: Maximum z-scores for each biomarker (B,)
        biomarker_labels: List of biomarker names
        N_startpoints: Number of startpoints for optimization
        N_S_max: Maximum number of subtypes
        N_iterations_MCMC: Number of MCMC iterations per chain
        output_folder: Output directory for results
        dataset_name: Name for output files
        use_parallel_startpoints: Whether to use parallel startpoints
        seed: Random seed
        use_gpu: Whether to use GPU acceleration
        device_id: Specific GPU device ID
        use_parallel_mcmc: Whether to use parallel MCMC chains
        n_mcmc_chains: Number of MCMC chains to run in parallel
        mcmc_backend: Backend for parallel MCMC ('process', 'thread', 'gpu')
        parallel_workers: Number of parallel workers
        
    Returns:
        Parallel GPU-accelerated ZScoreSustainMissingData instance
    """
    return ParallelTorchZScoreSustainMissingData(
        data, Z_vals, Z_max, biomarker_labels,
        N_startpoints, N_S_max, N_iterations_MCMC,
        output_folder, dataset_name, use_parallel_startpoints,
        seed, use_gpu, device_id,
        use_parallel_mcmc, n_mcmc_chains, mcmc_backend, parallel_workers
    )


# Example usage and testing functions
def demo_parallel_sustain():
    """Demonstrate parallel SuStaIn usage."""
    import numpy as np
    
    # Create sample data
    n_subjects = 100
    n_biomarkers = 5
    data = np.random.randn(n_subjects, n_biomarkers)
    Z_vals = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    Z_max = np.array([5, 5, 5, 5, 5])
    biomarker_labels = [f'biomarker_{i}' for i in range(n_biomarkers)]
    
    # Create parallel SuStaIn instance
    sustain = create_parallel_torch_zscore_sustain_missing_data(
        data=data,
        Z_vals=Z_vals,
        Z_max=Z_max,
        biomarker_labels=biomarker_labels,
        N_startpoints=10,
        N_S_max=2,
        N_iterations_MCMC=1000,
        use_parallel_mcmc=True,
        n_mcmc_chains=4,
        mcmc_backend='process',
        use_gpu=False  # Use CPU for demo
    )
    
    print("Running parallel SuStaIn algorithm...")
    start_time = time.time()
    
    # Run algorithm
    sustain.run_sustain_algorithm()
    
    total_time = time.time() - start_time
    print(f"Algorithm completed in {total_time:.2f} seconds")
    
    # Get parallel stats
    stats = sustain.get_parallel_stats()
    print(f"Parallel stats: {stats}")
    
    return sustain


if __name__ == "__main__":
    # Run demo
    demo_parallel_sustain()
