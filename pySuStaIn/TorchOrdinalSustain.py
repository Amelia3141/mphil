###
# GPU-accelerated OrdinalSustain implementation
#
# This module provides a PyTorch-based implementation of OrdinalSustain
# with significant performance improvements through GPU acceleration.
#
# Authors: GPU Migration Team
###

import torch
import numpy as np
from typing import Optional, Tuple, Union
from .OrdinalSustain import OrdinalSustain, OrdinalSustainData
from .torch_backend import TorchSustainBackend, create_torch_backend
from .torch_data_classes import TorchOrdinalSustainData, create_torch_ordinal_data
from .torch_likelihood import TorchOrdinalLikelihoodCalculator, create_ordinal_likelihood_calculator


class TorchOrdinalSustain(OrdinalSustain):
    """
    GPU-accelerated version of OrdinalSustain.

    This class extends the original OrdinalSustain with PyTorch-based
    GPU acceleration while maintaining full compatibility with the original API.
    """

    def __init__(self,
                 prob_nl: np.ndarray,
                 prob_score: np.ndarray,
                 score_vals: np.ndarray,
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
        Initialize GPU-accelerated OrdinalSustain.

        Args:
            prob_nl: Probability of normal class (M, B) where M=subjects, B=biomarkers
            prob_score: Probability of each score (M, B, num_scores)
            score_vals: Score values matrix (B, num_scores)
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
            prob_nl, prob_score, score_vals, biomarker_labels,
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints, seed
        )

        # Initialize PyTorch backend
        self.torch_backend = create_torch_backend(use_gpu=use_gpu, device_id=device_id)
        self.use_gpu = self.torch_backend.use_gpu

        # Access the sustainData attribute from parent class
        # Try different possible attribute names for compatibility
        sustain_data = None
        for attr_name in ['_OrdinalSustain__sustainData', '__sustainData', '_sustainData', 'sustainData']:
            sustain_data = getattr(self, attr_name, None)
            if sustain_data is not None:
                break

        if sustain_data is None:
            raise AttributeError("Could not find sustainData attribute. Available attributes: " +
                               str([attr for attr in dir(self) if 'sustain' in attr.lower()]))

        # Create PyTorch-enabled data object
        self.torch_sustain_data = create_torch_ordinal_data(
            prob_nl, prob_score, sustain_data.getNumStages(), self.torch_backend
        )

        # Create GPU-accelerated likelihood calculator
        self.torch_likelihood_calculator = create_ordinal_likelihood_calculator(
            self.torch_backend,
            self.stage_biomarker_index,
            self.stage_score
        )

        print(f"TorchOrdinalSustain initialized with {'GPU' if self.use_gpu else 'CPU'} acceleration")

    def _calculate_likelihood(self, sustainData, S: np.ndarray, f: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        GPU-accelerated likelihood computation.

        This method uses the PyTorch backend for GPU acceleration while maintaining
        the same interface as the original implementation.

        Args:
            sustainData: SuStaIn data object (may be subset via reindex)
            S: Sequence matrix (N_S, N)
            f: Fraction vector (N_S,)

        Returns:
            Tuple of (loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k)
        """
        # Use GPU-accelerated computation if available
        if self.use_gpu:
            try:
                # Check if sustainData is a subset of the full dataset
                M_subset = sustainData.getNumSamples()
                M_full = self.torch_sustain_data.getNumSamples()

                # Determine which torch data object to use
                if M_subset < M_full:
                    # sustainData is a subset - need to create TorchOrdinalSustainData from it
                    from .torch_data_classes import TorchOrdinalSustainData
                    torch_data_subset = TorchOrdinalSustainData(
                        sustainData.prob_nl,
                        sustainData.prob_score,
                        sustainData.getNumStages(),
                        self.torch_backend
                    )
                    data_to_use = torch_data_subset
                else:
                    # Use full dataset
                    data_to_use = self.torch_sustain_data

                return self.torch_likelihood_calculator.calculate_likelihood(
                    data_to_use, S, f
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
            sustainData: SuStaIn data object (may be subset via reindex)
            S: Single sequence (N,)

        Returns:
            Likelihood array (M, N+1) where M is number of subjects in sustainData
        """
        if self.use_gpu:
            try:
                # Check if sustainData is a subset of the full dataset
                M_subset = sustainData.getNumSamples()
                M_full = self.torch_sustain_data.getNumSamples()

                # Determine which torch data object to use
                if M_subset < M_full:
                    # sustainData is a subset - need to create TorchOrdinalSustainData from it
                    from .torch_data_classes import TorchOrdinalSustainData
                    torch_data_subset = TorchOrdinalSustainData(
                        sustainData.prob_nl,
                        sustainData.prob_score,
                        sustainData.getNumStages(),
                        self.torch_backend
                    )
                    data_to_use = torch_data_subset
                else:
                    # Use full dataset
                    data_to_use = self.torch_sustain_data

                # Convert to PyTorch tensor and compute
                S_torch = self.torch_backend.to_torch(S)
                result_torch = self.torch_likelihood_calculator._calculate_likelihood_stage_torch(
                    data_to_use, S_torch
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
                sustain_data = None
                for attr_name in ['_OrdinalSustain__sustainData', '__sustainData', '_sustainData', 'sustainData']:
                    sustain_data = getattr(self, attr_name, None)
                    if sustain_data is not None:
                        break

                if sustain_data is None:
                    raise AttributeError("Could not find sustainData attribute")

                self.torch_sustain_data = create_torch_ordinal_data(
                    sustain_data.prob_nl, sustain_data.prob_score,
                    sustain_data.getNumStages(), self.torch_backend
                )

                self.torch_likelihood_calculator = create_ordinal_likelihood_calculator(
                    self.torch_backend,
                    self.stage_biomarker_index,
                    self.stage_score
                )
                print("Switched to GPU computation")
            else:
                print("GPU not available, staying on CPU")
        except Exception as e:
            print(f"Failed to switch to GPU: {e}")
            self.use_gpu = False


# Factory function for creating GPU-accelerated instances
def create_torch_ordinal_sustain(
    prob_nl: np.ndarray,
    prob_score: np.ndarray,
    score_vals: np.ndarray,
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
) -> TorchOrdinalSustain:
    """
    Factory function to create a GPU-accelerated OrdinalSustain instance.

    Args:
        prob_nl: Probability of normal class (M, B)
        prob_score: Probability of each score (M, B, num_scores)
        score_vals: Score values matrix (B, num_scores)
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
        GPU-accelerated OrdinalSustain instance
    """
    return TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints, N_S_max, N_iterations_MCMC,
        output_folder, dataset_name, use_parallel_startpoints,
        seed, use_gpu, device_id
    )


# Example usage and testing functions
def benchmark_gpu_vs_cpu(prob_nl: np.ndarray, prob_score: np.ndarray, score_vals: np.ndarray,
                        biomarker_labels: list, num_iterations: int = 10) -> dict:
    """
    Benchmark GPU vs CPU performance for OrdinalSustain.

    Args:
        prob_nl: Test probability data for normal class
        prob_score: Test probability data for scores
        score_vals: Score values matrix
        biomarker_labels: Biomarker labels
        num_iterations: Number of benchmark iterations

    Returns:
        Dictionary with performance comparison
    """
    import time

    # Create test sequences and fractions
    N = score_vals.shape[0] * (score_vals.shape[1] - 1)  # Number of stages (exclude 0 scores)
    S_test = np.random.permutation(N).reshape(1, N)
    f_test = np.array([1.0])

    # Benchmark CPU version
    cpu_times = []
    cpu_sustain = OrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 1, 100, "./temp", "cpu_test", False, 42
    )

    # Access the sustainData attribute
    cpu_sustain_data = None
    for attr_name in ['_OrdinalSustain__sustainData', '__sustainData', '_sustainData', 'sustainData']:
        cpu_sustain_data = getattr(cpu_sustain, attr_name, None)
        if cpu_sustain_data is not None:
            break

    for _ in range(num_iterations):
        start_time = time.time()
        _ = cpu_sustain._calculate_likelihood_stage(cpu_sustain_data, S_test[0])
        cpu_times.append(time.time() - start_time)

    # Benchmark GPU version
    gpu_times = []
    gpu_sustain = TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 1, 100, "./temp", "gpu_test", False, 42, use_gpu=True
    )

    # Access the sustainData attribute
    gpu_sustain_data = None
    for attr_name in ['_OrdinalSustain__sustainData', '__sustainData', '_sustainData', 'sustainData']:
        gpu_sustain_data = getattr(gpu_sustain, attr_name, None)
        if gpu_sustain_data is not None:
            break

    for _ in range(num_iterations):
        start_time = time.time()
        _ = gpu_sustain._calculate_likelihood_stage(gpu_sustain_data, S_test[0])
        gpu_times.append(time.time() - start_time)

    return {
        'cpu_mean_time': np.mean(cpu_times),
        'cpu_std_time': np.std(cpu_times),
        'gpu_mean_time': np.mean(gpu_times),
        'gpu_std_time': np.std(gpu_times),
        'speedup': np.mean(cpu_times) / np.mean(gpu_times),
        'gpu_available': gpu_sustain.use_gpu
    }
