#!/usr/bin/env python
"""
Test GPU performance with user's actual dataset size:
- 8000 subjects
- 13 biomarkers (symptom domains)
- 3 severity levels
- 39 total stages (13 × 3)
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pySuStaIn.OrdinalSustain import OrdinalSustain
from pySuStaIn.TorchOrdinalSustain import TorchOrdinalSustain


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


def main():
    print("="*80)
    print("GPU BENCHMARK: User's Actual Dataset Size")
    print("="*80)
    print("\nDataset configuration:")
    print("  • Subjects: 8,000")
    print("  • Symptom domains (biomarkers): 13")
    print("  • Severity levels per domain: 3")
    print("  • Total disease events (stages): 39 (13 × 3)")
    print("\n" + "="*80)

    # Generate test data
    print("\nGenerating test data...")
    prob_nl, prob_score, score_vals, biomarker_labels = generate_test_data(
        n_subjects=8000,
        n_biomarkers=13,
        n_scores=3
    )
    print(f"✓ Data generated: {prob_nl.shape[0]} subjects, {prob_nl.shape[1]} biomarkers")

    # Create CPU instance
    print("\n" + "-"*80)
    print("Creating CPU instance (OrdinalSustain)...")
    print("-"*80)
    cpu_sustain = OrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints=1,
        N_S_max=1,
        N_iterations_MCMC=100,
        output_folder="./temp",
        dataset_name="cpu_test",
        use_parallel_startpoints=False,
        seed=42
    )
    cpu_data = getattr(cpu_sustain, '_OrdinalSustain__sustainData')
    print(f"✓ CPU instance created")

    # Create GPU instance
    print("\n" + "-"*80)
    print("Creating GPU instance (TorchOrdinalSustain)...")
    print("-"*80)
    try:
        gpu_sustain = TorchOrdinalSustain(
            prob_nl, prob_score, score_vals, biomarker_labels,
            N_startpoints=1,
            N_S_max=1,
            N_iterations_MCMC=100,
            output_folder="./temp",
            dataset_name="gpu_test",
            use_parallel_startpoints=False,
            seed=42,
            use_gpu=True,
            device_id=0
        )
        gpu_data = getattr(gpu_sustain, '_OrdinalSustain__sustainData')

        if not gpu_sustain.use_gpu:
            print("⚠️  GPU not available, cannot benchmark")
            return

        print(f"✓ GPU instance created")
        print(f"  Device: {gpu_sustain.torch_backend.device_manager.device}")
    except Exception as e:
        print(f"❌ GPU initialization failed: {e}")
        return

    # Create test sequence
    N = cpu_data.getNumStages()
    print(f"\nNumber of stages: {N}")
    S_test = np.random.permutation(N).astype(float)

    # Benchmark CPU
    print("\n" + "="*80)
    print("BENCHMARKING CPU")
    print("="*80)
    n_iterations = 10
    cpu_times = []

    print(f"Running {n_iterations} iterations...")
    for i in range(n_iterations):
        start = time.time()
        _ = cpu_sustain._calculate_likelihood_stage(cpu_data, S_test)
        cpu_times.append(time.time() - start)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{n_iterations} iterations")

    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    cpu_min = np.min(cpu_times)
    cpu_max = np.max(cpu_times)

    print(f"\nCPU Results:")
    print(f"  Mean: {cpu_mean*1000:.2f}ms ± {cpu_std*1000:.2f}ms")
    print(f"  Min:  {cpu_min*1000:.2f}ms")
    print(f"  Max:  {cpu_max*1000:.2f}ms")

    # Benchmark GPU
    print("\n" + "="*80)
    print("BENCHMARKING GPU")
    print("="*80)

    # Warmup
    print("Warming up GPU (5 iterations)...")
    for _ in range(5):
        _ = gpu_sustain._calculate_likelihood_stage(gpu_data, S_test)
    print("✓ Warmup complete")

    gpu_times = []
    print(f"\nRunning {n_iterations} iterations...")
    for i in range(n_iterations):
        start = time.time()
        _ = gpu_sustain._calculate_likelihood_stage(gpu_data, S_test)
        gpu_times.append(time.time() - start)
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{n_iterations} iterations")

    gpu_mean = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    gpu_min = np.min(gpu_times)
    gpu_max = np.max(gpu_times)

    print(f"\nGPU Results:")
    print(f"  Mean: {gpu_mean*1000:.2f}ms ± {gpu_std*1000:.2f}ms")
    print(f"  Min:  {gpu_min*1000:.2f}ms")
    print(f"  Max:  {gpu_max*1000:.2f}ms")

    # Calculate speedup
    speedup = cpu_mean / gpu_mean

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nDataset: 8,000 subjects × 13 biomarkers × 3 levels = 39 stages")
    print(f"\nCPU:    {cpu_mean*1000:>8.2f}ms")
    print(f"GPU:    {gpu_mean*1000:>8.2f}ms")
    print(f"Speedup: {speedup:>7.2f}x")

    if speedup > 1.0:
        print(f"\n✅ GPU is {speedup:.2f}x FASTER than CPU")
    else:
        print(f"\n❌ GPU is {1/speedup:.2f}x SLOWER than CPU")

    print("\n" + "="*80)

    # Validate correctness
    print("\nValidating correctness...")
    cpu_result = cpu_sustain._calculate_likelihood_stage(cpu_data, S_test)
    gpu_result = gpu_sustain._calculate_likelihood_stage(gpu_data, S_test)

    max_diff = np.max(np.abs(cpu_result - gpu_result))
    mean_diff = np.mean(np.abs(cpu_result - gpu_result))

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✅ GPU results match CPU (within tolerance)")
    else:
        print("  ⚠️  GPU results differ from CPU")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
