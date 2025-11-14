"""
Process-based parallel MCMC for OrdinalSustain.
Avoids dill serialization by passing only numpy arrays.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import time
from typing import Tuple, List
from functools import partial


def _worker_mcmc_chain(args):
    """
    Worker function for running a single MCMC chain in a separate process.

    This function recreates the OrdinalSustain object from scratch using only
    numpy arrays, avoiding the need for dill serialization.

    Args:
        args: Tuple of (chain_idx, seed, prob_nl, prob_score, score_vals, biomarker_labels,
                       S_init, f_init, n_iterations, seq_sigma, f_sigma)

    Returns:
        Tuple of (samples_sequence, samples_f, samples_likelihood, chain_time, chain_idx)
    """
    (chain_idx, seed, prob_nl, prob_score, score_vals, biomarker_labels,
     S_init, f_init, n_iterations, seq_sigma, f_sigma) = args

    print(f"Process {mp.current_process().name}: Starting chain {chain_idx + 1}")
    start_time = time.time()

    try:
        # Import here to avoid pickling issues
        from pySuStaIn.OrdinalSustain import OrdinalSustain

        # Recreate OrdinalSustain object in this worker process
        # This avoids serialization - we only pass numpy arrays
        sustain = OrdinalSustain(
            prob_nl=prob_nl,
            prob_score=prob_score,
            score_vals=score_vals,
            biomarker_labels=biomarker_labels,
            N_startpoints=1,
            N_S_max=1,
            N_iterations_MCMC=n_iterations,
            output_folder="./temp",
            dataset_name=f"chain_{chain_idx}",
            use_parallel_startpoints=False,
            seed=seed
        )

        # Get the sustain data
        sustain_data = getattr(sustain, '_OrdinalSustain__sustainData')

        # Run MCMC using the real SuStaIn _perform_mcmc method
        ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood = \
            sustain._perform_mcmc(sustain_data, S_init, f_init, n_iterations, seq_sigma, f_sigma)

        chain_time = time.time() - start_time
        print(f"Process {mp.current_process().name}: Chain {chain_idx + 1} completed in {chain_time:.2f}s")

        return (samples_sequence, samples_f, samples_likelihood, chain_time, chain_idx)

    except Exception as e:
        chain_time = time.time() - start_time
        print(f"Process {mp.current_process().name}: Chain {chain_idx + 1} failed: {e}")
        import traceback
        traceback.print_exc()

        # Return empty results
        N = S_init.shape[1]
        N_S = S_init.shape[0]
        return (
            np.zeros((N_S, N, n_iterations)),
            np.zeros((N_S, n_iterations)),
            np.zeros(n_iterations),
            chain_time,
            chain_idx
        )


class ProcessParallelMCMCManager:
    """
    Manages parallel MCMC chains using multiprocessing.
    Avoids dill serialization by passing only numpy arrays.
    """

    def __init__(self, n_chains: int = 4, n_workers: int = None):
        """
        Initialize process-based parallel MCMC manager.

        Args:
            n_chains: Number of MCMC chains to run in parallel
            n_workers: Number of worker processes (default: min(n_chains, cpu_count))
        """
        self.n_chains = n_chains
        self.n_workers = n_workers or min(n_chains, mp.cpu_count())

        print(f"ProcessParallelMCMCManager initialized:")
        print(f"  - Chains: {self.n_chains}")
        print(f"  - Workers: {self.n_workers}")
        print(f"  - Using TRUE multiprocessing (escapes GIL!)")

    def run_parallel_mcmc(self,
                         prob_nl: np.ndarray,
                         prob_score: np.ndarray,
                         score_vals: np.ndarray,
                         biomarker_labels: list,
                         S_init: np.ndarray,
                         f_init: np.ndarray,
                         n_iterations: int,
                         seq_sigma: float,
                         f_sigma: float,
                         seeds: List[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray],
                                                            List[np.ndarray], List[float]]:
        """
        Run MCMC chains in parallel using multiprocessing.

        This avoids dill serialization by passing only numpy arrays
        and recreating the OrdinalSustain object in each worker.

        Args:
            prob_nl: Probability of normal class (M, B)
            prob_score: Probability of each score (M, N_scores)
            score_vals: Score values matrix (B, N_scores)
            biomarker_labels: List of biomarker names
            S_init: Initial sequence matrix
            f_init: Initial fraction vector
            n_iterations: Number of MCMC iterations per chain
            seq_sigma: Sequence perturbation sigma
            f_sigma: Fraction perturbation sigma
            seeds: List of random seeds for each chain

        Returns:
            Tuple of (samples_sequences, samples_fs, samples_likelihoods, chain_times)
        """
        if seeds is None:
            seeds = [np.random.randint(0, 2**31) for _ in range(self.n_chains)]

        print(f"\nRunning {self.n_chains} MCMC chains in parallel using multiprocessing...")
        print(f"  Workers: {self.n_workers}")
        start_time = time.time()

        # Prepare arguments for each chain
        # Only passing numpy arrays - NO complex objects!
        chain_args = [
            (i, seeds[i], prob_nl, prob_score, score_vals, biomarker_labels,
             S_init, f_init, n_iterations, seq_sigma, f_sigma)
            for i in range(self.n_chains)
        ]

        # Run chains in parallel using multiprocessing Pool
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(_worker_mcmc_chain, chain_args)

        total_time = time.time() - start_time
        print(f"Parallel MCMC completed in {total_time:.2f} seconds")

        # Extract results
        samples_sequences = [r[0] for r in results]
        samples_fs = [r[1] for r in results]
        samples_likelihoods = [r[2] for r in results]
        chain_times = [r[3] for r in results]

        return samples_sequences, samples_fs, samples_likelihoods, chain_times


# Test if this can be imported
if __name__ == "__main__":
    print("Process-based parallel MCMC module loaded successfully")
    print(f"CPUs available: {mp.cpu_count()}")
