#!/usr/bin/env python3
"""
Quick test to verify process-based parallel MCMC speedup
Uses smaller iteration count for faster validation
"""

import numpy as np
import time
from pySuStaIn.OrdinalSustain import OrdinalSustain
from pySuStaIn.ParallelOrdinalSustain import ParallelOrdinalSustain

print("=" * 80)
print("QUICK PROCESS-BASED PARALLEL MCMC SPEEDUP TEST")
print("=" * 80)

# Generate test data (same size as user's dataset)
np.random.seed(42)
n_subjects = 8000
n_biomarkers = 13
n_levels = 3

print(f"\nGenerating test data: {n_subjects} subjects, {n_biomarkers} biomarkers")

# Generate test data using the same method as test_process_parallel.py
p_correct = 0.9
p_nl_dist = np.full((n_levels + 1), (1 - p_correct) / n_levels)
p_nl_dist[0] = p_correct

p_score_dist = np.full((n_levels, n_levels + 1), (1 - p_correct) / n_levels)
for score in range(n_levels):
    p_score_dist[score, score + 1] = p_correct

data = np.random.choice(range(n_levels + 1), n_subjects * n_biomarkers,
                      replace=True, p=p_nl_dist)
data = data.reshape((n_subjects, n_biomarkers))

prob_nl = p_nl_dist[data]

prob_score = np.zeros((n_subjects, n_biomarkers, n_levels))
for n in range(n_biomarkers):
    for z in range(n_levels):
        for score in range(n_levels + 1):
            prob_score[data[:, n] == score, n, z] = p_score_dist[z, score]

score_vals = np.tile(np.arange(1, n_levels + 1), (n_biomarkers, 1))
biomarker_labels = [f"BM{i+1}" for i in range(n_biomarkers)]

# Test with fewer iterations for quick verification
N_ITERATIONS = 500  # Reduced from 10000

print(f"\nTesting with {N_ITERATIONS} MCMC iterations per chain")

# Serial baseline
print("\n" + "=" * 80)
print("TEST 1: Serial (1 chain)")
print("=" * 80)

serial_sustain = OrdinalSustain(
    prob_nl=prob_nl,
    prob_score=prob_score,
    score_vals=score_vals,
    biomarker_labels=biomarker_labels,
    N_startpoints=1,
    N_S_max=1,
    N_iterations_MCMC=N_ITERATIONS,
    output_folder="./temp_serial",
    dataset_name="test_serial",
    use_parallel_startpoints=False
)

serial_start = time.time()
serial_sustain.run_sustain_algorithm()
serial_time = time.time() - serial_start

print(f"✓ Serial completed in {serial_time:.2f} seconds")

# Parallel test
print("\n" + "=" * 80)
print("TEST 2: Process Parallel (4 chains)")
print("=" * 80)

parallel_sustain = ParallelOrdinalSustain(
    prob_nl=prob_nl,
    prob_score=prob_score,
    score_vals=score_vals,
    biomarker_labels=biomarker_labels,
    N_startpoints=1,
    N_S_max=1,
    N_iterations_MCMC=N_ITERATIONS,
    output_folder="./temp_parallel",
    dataset_name="test_parallel",
    use_parallel_startpoints=False,
    use_parallel_mcmc=True,
    n_mcmc_chains=4,
    mcmc_backend='process'
)

parallel_start = time.time()
parallel_sustain.run_sustain_algorithm()
parallel_time = time.time() - parallel_start

print(f"✓ Parallel completed in {parallel_time:.2f} seconds")

# Calculate speedup
speedup = serial_time / parallel_time

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"Serial time (1 chain):      {serial_time:.2f}s")
print(f"Parallel time (4 chains):   {parallel_time:.2f}s")
print(f"Speedup:                    {speedup:.2f}x")
print(f"Efficiency:                 {(speedup/4)*100:.1f}%")

# Project to user's 30-day scenario
if speedup > 1:
    original_days = 30
    projected_days = original_days / speedup
    time_saved = original_days - projected_days

    print(f"\n📊 Projection for your 30-day run:")
    print(f"   - Original: {original_days} days")
    print(f"   - With 4-chain parallel: ~{projected_days:.1f} days")
    print(f"   - Time saved: ~{time_saved:.1f} days")
else:
    print("\n⚠️ Warning: No speedup achieved. Process parallelism may need tuning.")

print("=" * 80)
