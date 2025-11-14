#!/usr/bin/env python
"""
Test process-based parallel MCMC for OrdinalSustain.
This uses TRUE multiprocessing (escapes GIL) for real speedup.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pySuStaIn.OrdinalSustain import OrdinalSustain
from pySuStaIn.ParallelOrdinalSustain import ParallelOrdinalSustain


def generate_test_data(n_subjects=8000, n_biomarkers=13, n_scores=3, seed=42):
    """Generate synthetic test data matching user's dataset."""
    np.random.seed(seed)

    # Set the proportion of individuals with correct scores to 0.9
    p_correct = 0.9
    p_nl_dist = np.full((n_scores + 1), (1 - p_correct) / n_scores)
    p_nl_dist[0] = p_correct

    p_score_dist = np.full((n_scores, n_scores + 1), (1 - p_correct) / n_scores)
    for score in range(n_scores):
        p_score_dist[score, score + 1] = p_correct

    # Generate data
    data = np.random.choice(range(n_scores + 1), n_subjects * n_biomarkers,
                          replace=True, p=p_nl_dist)
    data = data.reshape((n_subjects, n_biomarkers))

    # Turn the data into probabilities
    prob_nl = p_nl_dist[data]

    prob_score = np.zeros((n_subjects, n_biomarkers, n_scores))
    for n in range(n_biomarkers):
        for z in range(n_scores):
            for score in range(n_scores + 1):
                prob_score[data[:, n] == score, n, z] = p_score_dist[z, score]

    # Create score_vals matrix
    score_vals = np.tile(np.arange(1, n_scores + 1), (n_biomarkers, 1))

    # Create biomarker labels
    biomarker_labels = [f"SymptomDomain_{i+1}" for i in range(n_biomarkers)]

    return prob_nl, prob_score, score_vals, biomarker_labels


def test_process_parallel():
    print("="*80)
    print("PROCESS-BASED PARALLEL MCMC TEST")
    print("Testing TRUE multiprocessing (escapes GIL!)")
    print("="*80)

    # Generate test data
    print("\n📊 Generating test data...")
    prob_nl, prob_score, score_vals, biomarker_labels = generate_test_data(
        n_subjects=8000,
        n_biomarkers=13,
        n_scores=3
    )
    print(f"✓ Data generated: {prob_nl.shape[0]:,} subjects, {prob_nl.shape[1]} biomarkers")

    # Reduced iterations for faster testing
    n_iterations = 500

    print("\n" + "="*80)
    print("TEST 1: Serial (1 chain) - Baseline")
    print("="*80)

    serial_start = time.time()

    serial_sustain = OrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints=1,
        N_S_max=1,
        N_iterations_MCMC=n_iterations,
        output_folder="./temp",
        dataset_name="serial_test",
        use_parallel_startpoints=False,
        seed=42
    )

    serial_data = getattr(serial_sustain, '_OrdinalSustain__sustainData')
    rng = np.random.default_rng(42)
    S_init = serial_sustain._initialise_sequence(serial_data, rng).reshape(1, -1)
    f_init = np.array([1.0])

    print("\nRunning serial MCMC...")
    ml_seq, ml_f, ml_like, samples_seq, samples_f, samples_like = \
        serial_sustain._estimate_uncertainty_sustain_model(serial_data, S_init, f_init)

    serial_time = time.time() - serial_start

    print(f"\n✓ Serial completed in {serial_time:.2f} seconds")

    # Test process-based parallel with 4 chains
    print("\n" + "="*80)
    print("TEST 2: Process Parallel (4 chains) - TRUE Multiprocessing")
    print("="*80)

    parallel_start = time.time()

    parallel_sustain = ParallelOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints=1,
        N_S_max=1,
        N_iterations_MCMC=n_iterations,
        output_folder="./temp",
        dataset_name="parallel_test",
        use_parallel_startpoints=False,
        seed=42,
        use_parallel_mcmc=True,
        n_mcmc_chains=4,
        mcmc_backend='process'  # ← USE PROCESS BACKEND!
    )

    parallel_data = getattr(parallel_sustain, '_OrdinalSustain__sustainData')
    rng2 = np.random.default_rng(42)
    S_init2 = parallel_sustain._initialise_sequence(parallel_data, rng2).reshape(1, -1)
    f_init2 = np.array([1.0])

    print("\nRunning process-based parallel MCMC...")
    ml_seq2, ml_f2, ml_like2, samples_seq2, samples_f2, samples_like2 = \
        parallel_sustain._estimate_uncertainty_sustain_model(parallel_data, S_init2, f_init2)

    parallel_time = time.time() - parallel_start

    print(f"\n✓ Process parallel completed in {parallel_time:.2f} seconds")

    # Calculate speedup
    speedup = serial_time / parallel_time

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nSerial (1 chain):          {serial_time:>8.2f}s")
    print(f"Process Parallel (4 chains): {parallel_time:>8.2f}s")
    print(f"Speedup:                    {speedup:>8.2f}x")

    if speedup > 1.5:
        print(f"\n✅ SUCCESS! Process parallelism achieved {speedup:.2f}x speedup!")
        print(f"   This escapes Python's GIL for TRUE parallel execution.")
        print(f"\n🎯 For your 30-day run:")
        print(f"   • With this speedup: ~{30/speedup:.1f} days")
        print(f"   • Time saved: ~{30 - 30/speedup:.1f} days")
    else:
        print(f"\n⚠️  Speedup lower than expected ({speedup:.2f}x)")
        print(f"   Expected 2-4x with 4 cores")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_process_parallel()
