"""
GPU-accelerated Z-score SuStaIn with optional process-parallel MCMC chains.

This is `TorchZScoreSustainMissingData` with `_estimate_uncertainty_sustain_model`
overridden to run `n_mcmc_chains` independent chains in separate processes
via `parallel_mcmc.run_parallel_chains`.

Notes:
    * The MCMC inside each chain runs on whatever device the parent
      instance was configured for. GPU memory is per-process, so running
      `n_chains` chains on a single GPU will allocate `n_chains` copies
      of the data tensors. For the CUDA case use `mcmc_backend='process'`
      with a small `n_mcmc_chains`, or run chains on CPU and reserve GPU
      for the EM startpoints.
    * Each chain re-runs `_optimise_mcmc_settings`. They tune
      independently and may pick slightly different proposal sigmas.
      That is intentional — pooled chains are then summarised together.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from .TorchZScoreSustainMissingData import TorchZScoreSustainMissingData
from .parallel_mcmc import combine_chain_samples, run_parallel_chains


class ParallelTorchZScoreSustainMissingData(TorchZScoreSustainMissingData):
    """`TorchZScoreSustainMissingData` plus multi-chain MCMC."""

    def __init__(
        self,
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
        n_mcmc_chains: int = 1,
        n_mcmc_workers: Optional[int] = None,
    ):
        super().__init__(
            data, Z_vals, Z_max, biomarker_labels,
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints,
            seed, use_gpu, device_id,
        )

        # Stash the kwargs so workers can rebuild a sibling instance
        self._mcmc_init_kwargs: Dict[str, Any] = dict(
            data=data, Z_vals=Z_vals, Z_max=Z_max,
            biomarker_labels=biomarker_labels,
            N_startpoints=N_startpoints, N_S_max=N_S_max,
            N_iterations_MCMC=N_iterations_MCMC,
            output_folder=output_folder, dataset_name=dataset_name,
            use_parallel_startpoints=False,
            seed=seed, use_gpu=use_gpu, device_id=device_id,
        )
        self.n_mcmc_chains = int(n_mcmc_chains)
        self.n_mcmc_workers = n_mcmc_workers
        self._last_parallel_stats: Optional[Dict[str, Any]] = None

    def _estimate_uncertainty_sustain_model(self, sustainData, seq_init, f_init):
        if self.n_mcmc_chains <= 1:
            return super()._estimate_uncertainty_sustain_model(
                sustainData, seq_init, f_init
            )

        seq_hash = hash(np.asarray(seq_init).tobytes()) & 0xFFFFFFFF
        master_seed = int(self.seed ^ seq_hash)
        mcmc_rng = np.random.default_rng(np.random.SeedSequence(master_seed))

        # Tune sigmas once in the parent so all chains share the same
        # proposal scale (small one-off cost, gives reproducible chains)
        seq_sigma_opt, f_sigma_opt = self._optimise_mcmc_settings(
            sustainData, seq_init, f_init, mcmc_rng
        )

        wall_t0 = time.perf_counter()
        result = run_parallel_chains(
            sustain_class=type(self),
            init_kwargs=self._mcmc_init_kwargs,
            seq_init=seq_init,
            f_init=f_init,
            n_iterations=self.N_iterations_MCMC,
            seq_sigma=seq_sigma_opt,
            f_sigma=f_sigma_opt,
            n_chains=self.n_mcmc_chains,
            n_workers=self.n_mcmc_workers,
            master_seed=master_seed,
        )
        wall_time = time.perf_counter() - wall_t0
        self._last_parallel_stats = {
            "wall_time": wall_time,
            "speedup": result["speedup"],
            "efficiency": result["efficiency"],
            "n_workers": result["n_workers"],
            "chain_times": result["chain_times"],
        }

        pooled = combine_chain_samples(result)
        return (
            pooled["ml_sequence"], pooled["ml_f"], pooled["ml_likelihood"],
            pooled["samples_sequence"], pooled["samples_f"],
            pooled["samples_likelihood"],
        )

    def get_parallel_stats(self) -> Optional[Dict[str, Any]]:
        """Stats from the most recent `_estimate_uncertainty_sustain_model` call.

        Returns None if no parallel MCMC has been run yet (i.e. `n_mcmc_chains <= 1`
        or no MCMC has been triggered).
        """
        return self._last_parallel_stats


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
    n_mcmc_chains: int = 4,
    n_mcmc_workers: Optional[int] = None,
) -> ParallelTorchZScoreSustainMissingData:
    """Factory for ParallelTorchZScoreSustainMissingData."""
    return ParallelTorchZScoreSustainMissingData(
        data, Z_vals, Z_max, biomarker_labels,
        N_startpoints, N_S_max, N_iterations_MCMC,
        output_folder, dataset_name, use_parallel_startpoints,
        seed, use_gpu, device_id, n_mcmc_chains, n_mcmc_workers,
    )
