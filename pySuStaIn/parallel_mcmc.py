"""
Process-based parallel MCMC for SuStaIn.

Runs independent MCMC chains in separate OS processes via the standard
library `multiprocessing` module. Each chain constructs its own SuStaIn
instance inside its worker process, which avoids having to pickle a
parent instance (which may carry unpicklable state such as a
`ThreadPoolExecutor`) and ensures full isolation of NumPy's global RNG.

This module replaces the previous thread-based implementation, which:
    - used `ThreadPoolExecutor` (Python threads share a GIL, so the
      NumPy-bound likelihood evaluation ran serially regardless of
      worker count);
    - contained a `_run_simplified_mcmc` fallback that returned mock
      samples with `rng.random()` as the likelihood, silently making
      benchmarks meaningless;
    - mutated `np.random` global state from multiple threads;
    - reported a "speedup" of `max(chain_times) / mean(chain_times)`,
      which is a load-imbalance ratio rather than a speedup.

Usage:

    from pySuStaIn import OrdinalSustain
    from pySuStaIn.parallel_mcmc import run_parallel_chains

    init_kwargs = dict(
        prob_nl=prob_nl, prob_score=prob_score, score_vals=score_vals,
        biomarker_labels=labels,
        N_startpoints=10, N_S_max=2, N_iterations_MCMC=10000,
        output_folder="./out", dataset_name="test",
        use_parallel_startpoints=False, seed=42,
    )

    result = run_parallel_chains(
        sustain_class=OrdinalSustain,
        init_kwargs=init_kwargs,
        seq_init=seq_init, f_init=f_init,
        n_iterations=10000,
        seq_sigma=1.0, f_sigma=0.01,
        n_chains=4,
    )

    print(result["wall_time"], result["speedup"])
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np


def _chain_worker(spec: Tuple[Any, ...]) -> Dict[str, Any]:
    """Run one MCMC chain in a worker process.

    Receives a fully-picklable spec, constructs a fresh SuStaIn instance,
    runs `_perform_mcmc`, and returns its outputs. The instance is built
    inside the worker so the parent process never has to pickle it.
    """
    (module_name, class_name, init_kwargs, seq_init, f_init,
     n_iterations, seq_sigma, f_sigma, seed, chain_idx) = spec

    module = importlib.import_module(module_name)
    sustain_class = getattr(module, class_name)

    # Each worker gets its own seed and runs serially internally
    kw = dict(init_kwargs)
    kw["seed"] = int(seed)
    kw["use_parallel_startpoints"] = False  # no nested parallelism

    sustain = sustain_class(**kw)

    # AbstractSustain stores the data under a name-mangled attribute.
    sustain_data = getattr(sustain, "_AbstractSustain__sustainData", None)
    if sustain_data is None:
        raise RuntimeError(
            f"Could not locate sustainData on {class_name} instance. "
            "Workers expect AbstractSustain's standard attribute layout."
        )

    rng = np.random.default_rng(int(seed))

    t0 = time.perf_counter()
    ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood = (
        sustain._perform_mcmc(
            sustain_data, seq_init, f_init,
            n_iterations, seq_sigma, f_sigma, rng=rng,
        )
    )
    chain_time = time.perf_counter() - t0

    return {
        "chain_idx": int(chain_idx),
        "seed": int(seed),
        "ml_sequence": np.asarray(ml_sequence),
        "ml_f": np.asarray(ml_f),
        "ml_likelihood": float(np.squeeze(ml_likelihood)),
        "samples_sequence": np.asarray(samples_sequence),
        "samples_f": np.asarray(samples_f),
        "samples_likelihood": np.asarray(samples_likelihood),
        "chain_time": chain_time,
    }


def _default_seeds(n_chains: int, master_seed: int = 12345) -> List[int]:
    """Deterministic, well-spaced seeds for `n_chains` chains."""
    ss = np.random.SeedSequence(master_seed)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(n_chains)]


def run_parallel_chains(
    sustain_class: Type,
    init_kwargs: Dict[str, Any],
    seq_init: np.ndarray,
    f_init: np.ndarray,
    n_iterations: int,
    seq_sigma,
    f_sigma,
    n_chains: int,
    seeds: Optional[List[int]] = None,
    n_workers: Optional[int] = None,
    master_seed: int = 12345,
) -> Dict[str, Any]:
    """Run `n_chains` independent MCMC chains in parallel processes.

    Args:
        sustain_class: SuStaIn class (e.g. `OrdinalSustain`). Must be
            importable by module path — not a class defined in `__main__`
            or inside another function — because workers import it by
            `(module_name, class_name)`.
        init_kwargs: kwargs for `sustain_class(**init_kwargs)`. Must be
            picklable (numpy arrays and primitives are; lambdas and open
            file handles are not). The chain's `seed` is set
            automatically; `use_parallel_startpoints` is forced to False
            to avoid nested parallelism inside each worker.
        seq_init: initial sequence matrix, shape (N_S, N).
        f_init: initial fraction vector, shape (N_S,).
        n_iterations: MCMC iterations per chain.
        seq_sigma, f_sigma: proposal scales (the values
            `_optimise_mcmc_settings` would return).
        n_chains: number of chains to run.
        seeds: optional explicit seeds, length `n_chains`. If None,
            seeds are spawned deterministically from `master_seed`.
        n_workers: number of worker processes. Defaults to
            `min(n_chains, mp.cpu_count())`.
        master_seed: seed source when `seeds` is None.

    Returns:
        A dict with:
            "chains": list of per-chain dicts (see `_chain_worker`)
                ordered by chain_idx.
            "wall_time": total wall-clock seconds for the parallel run.
            "chain_times": per-chain execution times (seconds).
            "speedup": sum(chain_times) / wall_time. A chain count of 1
                gives ~1.0; a fully parallel run on enough cores
                approaches `n_chains`.
            "efficiency": speedup / n_workers.
            "n_workers": worker count actually used.
    """
    if sustain_class.__module__ == "__main__":
        raise ValueError(
            "sustain_class must be defined in an importable module, not "
            "__main__ — workers cannot resolve __main__ classes by name."
        )
    if seeds is None:
        seeds = _default_seeds(n_chains, master_seed=master_seed)
    elif len(seeds) != n_chains:
        raise ValueError(
            f"len(seeds) ({len(seeds)}) != n_chains ({n_chains})"
        )
    if n_workers is None:
        n_workers = min(n_chains, mp.cpu_count())

    module_name = sustain_class.__module__
    class_name = sustain_class.__name__

    specs = [
        (
            module_name, class_name, init_kwargs,
            np.asarray(seq_init), np.asarray(f_init),
            int(n_iterations), seq_sigma, f_sigma,
            int(seed), idx,
        )
        for idx, seed in enumerate(seeds)
    ]

    # Always use spawn — works on macOS/Windows, doesn't inherit
    # parent state (notably matplotlib backends, CUDA contexts, etc.)
    ctx = mp.get_context("spawn")

    chain_results: List[Optional[Dict[str, Any]]] = [None] * n_chains

    wall_t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        future_to_idx = {executor.submit(_chain_worker, s): s[-1] for s in specs}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()  # propagate exceptions
            chain_results[idx] = result
    wall_time = time.perf_counter() - wall_t0

    # All slots must be filled (futures either return or raise)
    for i, r in enumerate(chain_results):
        if r is None:
            raise RuntimeError(f"Chain {i} produced no result.")

    chain_times = [r["chain_time"] for r in chain_results]
    serial_time = float(sum(chain_times))
    speedup = serial_time / wall_time if wall_time > 0 else float("nan")

    return {
        "chains": chain_results,  # ordered by chain_idx
        "wall_time": wall_time,
        "chain_times": chain_times,
        "speedup": speedup,
        "efficiency": speedup / n_workers if n_workers > 0 else float("nan"),
        "n_workers": n_workers,
    }


def combine_chain_samples(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pool samples across chains for downstream summarisation.

    Concatenates along the MCMC-iteration axis. Each chain remains
    a separate Markov chain — this is purely a convenience for code
    that consumes a single (N_S, N, total_iters) sample tensor.

    Returns:
        Dict with keys:
            "samples_sequence": (N_S, N, n_chains * n_iterations)
            "samples_f":        (N_S, n_chains * n_iterations)
            "samples_likelihood": (n_chains * n_iterations, 1)
            "ml_sequence", "ml_f", "ml_likelihood":
                argmax across all chains.
            "per_chain_ml_likelihood": list of length n_chains.
    """
    chains = result["chains"]
    samples_sequence = np.concatenate(
        [c["samples_sequence"] for c in chains], axis=2
    )
    samples_f = np.concatenate([c["samples_f"] for c in chains], axis=1)

    likelihoods = [np.asarray(c["samples_likelihood"]).reshape(-1, 1) for c in chains]
    samples_likelihood = np.concatenate(likelihoods, axis=0)

    # Find ML across the pooled samples
    flat = samples_likelihood.ravel()
    best = int(np.argmax(flat))
    ml_likelihood = float(flat[best])
    ml_sequence = samples_sequence[:, :, best]
    ml_f = samples_f[:, best]

    return {
        "samples_sequence": samples_sequence,
        "samples_f": samples_f,
        "samples_likelihood": samples_likelihood,
        "ml_sequence": ml_sequence,
        "ml_f": ml_f,
        "ml_likelihood": ml_likelihood,
        "per_chain_ml_likelihood": [c["ml_likelihood"] for c in chains],
    }
