###
# GPU-accelerated OrdinalSustain following the fastSuStaIn pattern.
#
# Strategy: Override ONLY _calculate_likelihood_stage() and _calculate_likelihood()
# to dispatch to GPU. Everything else (EM, MCMC, _optimise_parameters) runs
# identically to CPU OrdinalSustain. This guarantees identical results — same
# algorithm, same RNG sequence, just each likelihood call is GPU-accelerated.
#
# Key difference from previous approach:
#   OLD: Batched 128 MCMC proposals on GPU (different Markov chain, confounding)
#   NEW: Each likelihood call routes to GPU individually (same Markov chain, no confounding)
#
# The speedup comes from within each _calculate_likelihood_stage() call:
#   - torch.prod() across all M subjects runs on GPU cores in parallel
#   - Data (prob_nl, prob_score) transferred to GPU once, stays cached
#   - Only the tiny sequence S (~22 elements) crosses CPU-GPU boundary per call
###

import numpy as np
import torch
from typing import Optional

from .OrdinalSustain import OrdinalSustain, OrdinalSustainData
from .torch_backend import create_torch_backend
from .torch_data_classes import TorchOrdinalSustainData, create_torch_ordinal_data
from .torch_likelihood import create_ordinal_likelihood_calculator


class TorchOrdinalSustain(OrdinalSustain):
    """
    GPU-accelerated OrdinalSustain.

    Drop-in replacement for OrdinalSustain. Produces identical results
    (when using float64) because only the likelihood computation is
    dispatched to GPU — all algorithm logic remains unchanged.
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
                 device_id: Optional[int] = None,
                 force_float64: bool = False):
        """
        Initialize GPU-accelerated OrdinalSustain.

        All parameters are identical to OrdinalSustain, plus:
            use_gpu: Whether to use GPU acceleration (default True)
            device_id: Specific CUDA device ID (default None = auto)
            force_float64: Use float64 on GPU for exact CPU equivalence (default False)
        """
        # Initialize the parent CPU OrdinalSustain exactly as normal
        super().__init__(
            prob_nl, prob_score, score_vals, biomarker_labels,
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints, seed
        )

        # Set up GPU backend
        self.torch_backend = create_torch_backend(
            use_gpu=use_gpu, device_id=device_id, force_float64=force_float64
        )
        self.use_gpu = self.torch_backend.use_gpu

        # Create the likelihood calculator with model parameters on GPU
        self.torch_likelihood_calculator = create_ordinal_likelihood_calculator(
            self.torch_backend, self.stage_biomarker_index, self.stage_score
        )

        # Cache for TorchOrdinalSustainData objects to avoid repeated wrapping
        self._torch_data_cache = {}

        device_name = 'GPU' if self.use_gpu else 'CPU'
        dtype_name = 'float64' if force_float64 else ('float32' if self.use_gpu else 'float64')
        print(f"TorchOrdinalSustain: {device_name} ({dtype_name})")

    def _ensure_torch_data(self, sustainData) -> TorchOrdinalSustainData:
        """
        Wrap an OrdinalSustainData as TorchOrdinalSustainData if needed.

        Uses identity-based caching so the same sustainData object doesn't
        get re-wrapped and re-transferred to GPU on every call.

        Args:
            sustainData: OrdinalSustainData or TorchOrdinalSustainData

        Returns:
            TorchOrdinalSustainData with lazy GPU tensor caching
        """
        if isinstance(sustainData, TorchOrdinalSustainData):
            return sustainData

        # Cache by object id — same sustainData object reuses the same wrapper
        cache_key = id(sustainData)
        if cache_key in self._torch_data_cache:
            cached = self._torch_data_cache[cache_key]
            # Verify the cached wrapper still matches (same shape = same object)
            if cached.getNumSamples() == sustainData.getNumSamples():
                return cached

        # Create new wrapper — GPU tensors are lazily created on first access
        torch_data = create_torch_ordinal_data(
            sustainData.prob_nl, sustainData.prob_score,
            sustainData.getNumStages(), self.torch_backend
        )

        # Keep cache small — only store a few recent entries
        if len(self._torch_data_cache) > 10:
            self._torch_data_cache.clear()
        self._torch_data_cache[cache_key] = torch_data

        return torch_data

    def _calculate_likelihood_stage(self, sustainData, S):
        """
        GPU-accelerated likelihood computation for a single sequence.

        Dispatches to TorchOrdinalLikelihoodCalculator on GPU, falling
        back to CPU if GPU fails (e.g., out of memory).

        Args:
            sustainData: Data object (may be full dataset or subset)
            S: Sequence array (N,) — the event ordering for one subtype

        Returns:
            p_perm_k: numpy array (M, N+1) of stage likelihoods
        """
        if self.use_gpu:
            try:
                torch_data = self._ensure_torch_data(sustainData)
                S_torch = self.torch_backend.to_torch(S)
                result = self.torch_likelihood_calculator._calculate_likelihood_stage_torch(
                    torch_data, S_torch
                )
                return self.torch_backend.to_numpy(result)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.torch_backend.clear_cache()
                    self._torch_data_cache.clear()
                    print(f"GPU OOM in _calculate_likelihood_stage, falling back to CPU")
                else:
                    raise
        return super()._calculate_likelihood_stage(sustainData, S)

    def _calculate_likelihood(self, sustainData, S, f):
        """
        GPU-accelerated mixture likelihood computation.

        Dispatches to TorchLikelihoodCalculator.calculate_likelihood() on GPU,
        falling back to CPU if GPU fails.

        Args:
            sustainData: Data object (may be full dataset or subset)
            S: Sequence matrix (N_S, N) for all subtypes
            f: Fraction vector (N_S,) for each subtype

        Returns:
            Tuple of (loglike, total_prob_subj, total_prob_stage,
                      total_prob_cluster, p_perm_k)
        """
        if self.use_gpu:
            try:
                torch_data = self._ensure_torch_data(sustainData)
                return self.torch_likelihood_calculator.calculate_likelihood(
                    torch_data, S, f
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.torch_backend.clear_cache()
                    self._torch_data_cache.clear()
                    print(f"GPU OOM in _calculate_likelihood, falling back to CPU")
                else:
                    raise
        return super()._calculate_likelihood(sustainData, S, f)

    def get_performance_stats(self):
        """Get GPU performance statistics."""
        return self.torch_backend.get_performance_stats()


def create_torch_ordinal_sustain(
    prob_nl, prob_score, score_vals, biomarker_labels,
    N_startpoints=25, N_S_max=3, N_iterations_MCMC=100000,
    output_folder="./output", dataset_name="dataset",
    use_parallel_startpoints=True, seed=None,
    use_gpu=True, device_id=None, force_float64=False
):
    """Factory function for creating TorchOrdinalSustain."""
    return TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints, N_S_max, N_iterations_MCMC,
        output_folder, dataset_name, use_parallel_startpoints,
        seed, use_gpu, device_id, force_float64
    )
