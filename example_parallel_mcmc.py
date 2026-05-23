#!/usr/bin/env python3
"""
Demonstrate process-parallel MCMC chains for OrdinalSustain.

Runs the same model in N_S=1 mode with `n_chains` independent MCMC chains,
each in its own worker process. Reports wall time, summed chain time, and
the resulting speedup. Each chain has its own deterministic seed, so
results are reproducible.

Requires `OrdinalSustain` to be importable. The module-path requirement of
`multiprocessing` is satisfied: the class lives in `pySuStaIn.OrdinalSustain`.

Example:
    python example_parallel_mcmc.py
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from pySuStaIn.OrdinalSustain import OrdinalSustain
from pySuStaIn.parallel_mcmc import combine_chain_samples, run_parallel_chains


def make_synthetic_ordinal_data(
    n_subjects: int = 200,
    n_biomarkers: int = 4,
    n_scores: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Construct probability matrices for OrdinalSustain on synthetic data.

    Returns the matrices that OrdinalSustain.__init__ consumes directly
    (no internal estimation needed).
    """
    rng = np.random.default_rng(seed)
    score_vals = np.tile(np.arange(1, n_scores + 1), (n_biomarkers, 1))

    # Soft normal/score probabilities — realistic enough to drive MCMC
    prob_nl = rng.dirichlet(np.ones(2), size=(n_subjects, n_biomarkers))[..., 0]
    prob_nl = np.clip(prob_nl, 1e-3, 1 - 1e-3)

    prob_score = rng.dirichlet(np.ones(n_scores), size=(n_subjects, n_biomarkers))
    prob_score = np.clip(prob_score, 1e-3, 1 - 1e-3)

    biomarker_labels = [f"bm_{i}" for i in range(n_biomarkers)]
    return prob_nl, prob_score, score_vals, biomarker_labels


def build_sustain_kwargs(
    prob_nl,
    prob_score,
    score_vals,
    biomarker_labels,
    *,
    n_iterations_mcmc: int,
    output_folder: str = "./tmp_parallel_demo",
    seed: int = 42,
):
    return dict(
        prob_nl=prob_nl,
        prob_score=prob_score,
        score_vals=score_vals,
        biomarker_labels=biomarker_labels,
        N_startpoints=5,
        N_S_max=1,
        N_iterations_MCMC=n_iterations_mcmc,
        output_folder=output_folder,
        dataset_name="parallel_demo",
        use_parallel_startpoints=False,
        seed=seed,
    )


def run_initial_em_and_get_init(sustain: OrdinalSustain):
    """Run the single-subtype EM to get a sensible `seq_init`, `f_init`."""
    rng = np.random.default_rng(sustain.seed)
    sustain_data = sustain._AbstractSustain__sustainData
    seq_init = sustain._initialise_sequence(sustain_data, rng)
    f_init = np.array([1.0])
    ml_seq, ml_f, _ = sustain._optimise_parameters(sustain_data, seq_init, f_init, rng)
    return ml_seq, ml_f


def demo(n_iterations: int = 2000, chain_counts=(1, 2, 4)):
    prob_nl, prob_score, score_vals, biomarker_labels = make_synthetic_ordinal_data()
    init_kwargs = build_sustain_kwargs(
        prob_nl, prob_score, score_vals, biomarker_labels,
        n_iterations_mcmc=n_iterations,
    )

    # Build one parent instance just to derive seq_init / f_init from EM.
    parent = OrdinalSustain(**init_kwargs)
    seq_init, f_init = run_initial_em_and_get_init(parent)

    # Reasonable MCMC proposal scales (would normally come from
    # `_optimise_mcmc_settings`; using defaults keeps the demo fast).
    seq_sigma = 1.0
    f_sigma = 0.01

    print(f"{'n_chains':>10} {'wall(s)':>10} {'sum(s)':>10} {'speedup':>10}")
    print("-" * 44)
    for n_chains in chain_counts:
        t0 = time.perf_counter()
        result = run_parallel_chains(
            sustain_class=OrdinalSustain,
            init_kwargs=init_kwargs,
            seq_init=seq_init,
            f_init=f_init,
            n_iterations=n_iterations,
            seq_sigma=seq_sigma,
            f_sigma=f_sigma,
            n_chains=n_chains,
        )
        wall = time.perf_counter() - t0
        sum_t = sum(result["chain_times"])
        print(
            f"{n_chains:>10d} {wall:>10.2f} {sum_t:>10.2f} {result['speedup']:>10.2f}"
        )

        pooled = combine_chain_samples(result)
        # Smoke test: shapes line up
        assert pooled["samples_sequence"].shape[2] == n_chains * n_iterations
        assert pooled["samples_f"].shape[1] == n_chains * n_iterations


if __name__ == "__main__":
    demo()
