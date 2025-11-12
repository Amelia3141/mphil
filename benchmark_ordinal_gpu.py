#!/usr/bin/env python
"""
Benchmark and validate GPU-accelerated OrdinalSustain implementation.

This script compares the performance and correctness of the GPU-accelerated
TorchOrdinalSustain against the original CPU-based OrdinalSustain implementation.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add pySuStaIn to path
sys.path.insert(0, str(Path(__file__).parent))

from pySuStaIn.OrdinalSustain import OrdinalSustain
from pySuStaIn.TorchOrdinalSustain import TorchOrdinalSustain


def generate_test_data(n_subjects=1000, n_biomarkers=10, n_scores=3, seed=42):
    """
    Generate synthetic test data for OrdinalSustain.

    Args:
        n_subjects: Number of subjects
        n_biomarkers: Number of biomarkers
        n_scores: Number of score levels (excluding 0)
        seed: Random seed

    Returns:
        Tuple of (prob_nl, prob_score, score_vals, biomarker_labels)
    """
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
    biomarker_labels = [f"Biomarker_{i}" for i in range(n_biomarkers)]

    return prob_nl, prob_score, score_vals, biomarker_labels


def validate_correctness(prob_nl, prob_score, score_vals, biomarker_labels,
                        n_tests=5, tolerance=1e-5):
    """
    Validate that GPU and CPU implementations produce the same results.

    Args:
        prob_nl: Probability of normal class
        prob_score: Probability of scores
        score_vals: Score values
        biomarker_labels: Biomarker labels
        n_tests: Number of random sequences to test
        tolerance: Numerical tolerance for comparison

    Returns:
        Boolean indicating if validation passed
    """
    print("\n" + "=" * 80)
    print("VALIDATION TEST: Comparing GPU vs CPU Correctness")
    print("=" * 80)

    # Create both implementations
    cpu_sustain = OrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 1, 100, "./temp", "cpu_test", False, 42
    )

    try:
        gpu_sustain = TorchOrdinalSustain(
            prob_nl, prob_score, score_vals, biomarker_labels,
            1, 1, 100, "./temp", "gpu_test", False, 42, use_gpu=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize GPU implementation: {e}")
        return False

    if not gpu_sustain.use_gpu:
        print("⚠️  GPU not available, skipping validation")
        return False

    # Access sustainData
    cpu_sustain_data = getattr(cpu_sustain, '_OrdinalSustain__sustainData', None)
    gpu_sustain_data = getattr(gpu_sustain, '_OrdinalSustain__sustainData', None)

    if cpu_sustain_data is None or gpu_sustain_data is None:
        print("❌ Could not access sustainData")
        return False

    # Test with random sequences
    N = score_vals.shape[0] * (score_vals.shape[1] - 1)
    all_passed = True

    for test_idx in range(n_tests):
        print(f"\n Test {test_idx + 1}/{n_tests}:")

        # Generate random sequence
        S_test = np.random.permutation(N).astype(float)

        # Compute likelihoods
        try:
            cpu_result = cpu_sustain._calculate_likelihood_stage(cpu_sustain_data, S_test)
            gpu_result = gpu_sustain._calculate_likelihood_stage(gpu_sustain_data, S_test)

            # Compare results
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            mean_diff = np.mean(np.abs(cpu_result - gpu_result))
            rel_diff = max_diff / (np.mean(np.abs(cpu_result)) + 1e-10)

            print(f"    Shape: CPU={cpu_result.shape}, GPU={gpu_result.shape}")
            print(f"    Max absolute diff: {max_diff:.2e}")
            print(f"    Mean absolute diff: {mean_diff:.2e}")
            print(f"    Relative diff: {rel_diff:.2e}")

            if max_diff > tolerance:
                print(f"    ❌ FAILED: Difference exceeds tolerance ({tolerance:.2e})")
                all_passed = False
            else:
                print(f"    ✓ PASSED")

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            all_passed = False

    print("\n" + "-" * 80)
    if all_passed:
        print("✓ All validation tests PASSED")
    else:
        print("❌ Some validation tests FAILED")
    print("=" * 80)

    return all_passed


def benchmark_performance(prob_nl, prob_score, score_vals, biomarker_labels,
                         n_iterations=20):
    """
    Benchmark GPU vs CPU performance.

    Args:
        prob_nl: Probability of normal class
        prob_score: Probability of scores
        score_vals: Score values
        biomarker_labels: Biomarker labels
        n_iterations: Number of benchmark iterations

    Returns:
        Dictionary with performance statistics
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK: GPU vs CPU")
    print("=" * 80)

    print(f"\nDataset size:")
    print(f"  - Subjects: {prob_nl.shape[0]}")
    print(f"  - Biomarkers: {prob_nl.shape[1]}")
    print(f"  - Scores: {prob_score.shape[2]}")
    print(f"  - Iterations: {n_iterations}")

    # Create test sequence
    N = score_vals.shape[0] * (score_vals.shape[1] - 1)
    S_test = np.random.permutation(N).astype(float)
    f_test = np.array([1.0])

    # Benchmark CPU
    print("\n" + "-" * 80)
    print("Benchmarking CPU implementation...")
    print("-" * 80)

    cpu_sustain = OrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 1, 100, "./temp", "cpu_test", False, 42
    )
    cpu_sustain_data = getattr(cpu_sustain, '_OrdinalSustain__sustainData', None)

    cpu_times = []
    for i in range(n_iterations):
        start = time.time()
        _ = cpu_sustain._calculate_likelihood_stage(cpu_sustain_data, S_test)
        elapsed = time.time() - start
        cpu_times.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i + 1}/{n_iterations}: {elapsed:.4f}s")

    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    print(f"\nCPU Results:")
    print(f"  Mean time: {cpu_mean:.4f}s ± {cpu_std:.4f}s")
    print(f"  Min time: {np.min(cpu_times):.4f}s")
    print(f"  Max time: {np.max(cpu_times):.4f}s")

    # Benchmark GPU
    print("\n" + "-" * 80)
    print("Benchmarking GPU implementation...")
    print("-" * 80)

    try:
        gpu_sustain = TorchOrdinalSustain(
            prob_nl, prob_score, score_vals, biomarker_labels,
            1, 1, 100, "./temp", "gpu_test", False, 42, use_gpu=True
        )

        if not gpu_sustain.use_gpu:
            print("⚠️  GPU not available")
            return None

        gpu_sustain_data = getattr(gpu_sustain, '_OrdinalSustain__sustainData', None)

        # Warmup
        print("Warming up GPU...")
        for _ in range(3):
            _ = gpu_sustain._calculate_likelihood_stage(gpu_sustain_data, S_test)

        gpu_times = []
        for i in range(n_iterations):
            start = time.time()
            _ = gpu_sustain._calculate_likelihood_stage(gpu_sustain_data, S_test)
            elapsed = time.time() - start
            gpu_times.append(elapsed)
            if (i + 1) % 5 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}: {elapsed:.4f}s")

        gpu_mean = np.mean(gpu_times)
        gpu_std = np.std(gpu_times)
        print(f"\nGPU Results:")
        print(f"  Mean time: {gpu_mean:.4f}s ± {gpu_std:.4f}s")
        print(f"  Min time: {np.min(gpu_times):.4f}s")
        print(f"  Max time: {np.max(gpu_times):.4f}s")

        # Calculate speedup
        speedup = cpu_mean / gpu_mean
        print("\n" + "=" * 80)
        print(f"SPEEDUP: {speedup:.2f}x")
        print("=" * 80)

        # Get performance stats
        perf_stats = gpu_sustain.get_performance_stats()
        if perf_stats['computation_times']:
            print("\nDetailed GPU timing:")
            for op_name, op_time in perf_stats['computation_times'].items():
                print(f"  {op_name}: {op_time:.6f}s")

        return {
            'cpu_mean': cpu_mean,
            'cpu_std': cpu_std,
            'gpu_mean': gpu_mean,
            'gpu_std': gpu_std,
            'speedup': speedup,
            'n_subjects': prob_nl.shape[0],
            'n_biomarkers': prob_nl.shape[1]
        }

    except Exception as e:
        print(f"\n❌ GPU benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main benchmark script."""
    print("\n" + "=" * 80)
    print("GPU-ACCELERATED ORDINALSUSTAIN BENCHMARK")
    print("=" * 80)

    # Test configurations
    configs = [
        {"n_subjects": 100, "n_biomarkers": 5, "n_scores": 3},
        {"n_subjects": 500, "n_biomarkers": 10, "n_scores": 3},
        {"n_subjects": 1000, "n_biomarkers": 10, "n_scores": 3},
        {"n_subjects": 2000, "n_biomarkers": 15, "n_scores": 3},
    ]

    results = []

    for config in configs:
        print(f"\n\n{'#' * 80}")
        print(f"Testing configuration: {config}")
        print('#' * 80)

        # Generate test data
        prob_nl, prob_score, score_vals, biomarker_labels = generate_test_data(**config)

        # Validate correctness (only for first config)
        if len(results) == 0:
            validate_correctness(prob_nl, prob_score, score_vals, biomarker_labels)

        # Benchmark performance
        result = benchmark_performance(prob_nl, prob_score, score_vals, biomarker_labels)
        if result is not None:
            results.append(result)

    # Summary
    if results:
        print("\n\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\n{'Subjects':<12} {'Biomarkers':<12} {'CPU (s)':<15} {'GPU (s)':<15} {'Speedup':<10}")
        print("-" * 80)
        for r in results:
            print(f"{r['n_subjects']:<12} {r['n_biomarkers']:<12} "
                  f"{r['cpu_mean']:.4f}±{r['cpu_std']:.4f}  "
                  f"{r['gpu_mean']:.4f}±{r['gpu_std']:.4f}  "
                  f"{r['speedup']:.2f}x")
        print("=" * 80)


if __name__ == "__main__":
    main()
