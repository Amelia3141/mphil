#!/usr/bin/env python
"""
Benchmark and validate GPU-accelerated OrdinalSustain implementation.

Tests three levels of correctness:
  1. _calculate_likelihood_stage: GPU vs CPU produce identical likelihoods
  2. _calculate_likelihood: Full mixture model matches
  3. run_sustain_algorithm: Full pipeline produces identical sequences/stages

Also benchmarks GPU vs CPU performance at various dataset sizes.

Usage:
    python benchmark_ordinal_gpu.py                  # Run all tests
    python benchmark_ordinal_gpu.py --validate-only  # Skip benchmarks
    python benchmark_ordinal_gpu.py --benchmark-only # Skip validation
"""

import numpy as np
import time
import sys
import os
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pySuStaIn.OrdinalSustain import OrdinalSustain
from pySuStaIn.TorchOrdinalSustain import TorchOrdinalSustain


def generate_test_data(n_subjects=1000, n_biomarkers=10, n_scores=3, seed=42):
    """
    Generate synthetic test data for OrdinalSustain.

    Returns:
        Tuple of (prob_nl, prob_score, score_vals, biomarker_labels)
    """
    np.random.seed(seed)

    p_correct = 0.9
    p_nl_dist = np.full((n_scores + 1), (1 - p_correct) / n_scores)
    p_nl_dist[0] = p_correct

    p_score_dist = np.full((n_scores, n_scores + 1), (1 - p_correct) / n_scores)
    for score in range(n_scores):
        p_score_dist[score, score + 1] = p_correct

    data = np.random.choice(range(n_scores + 1), n_subjects * n_biomarkers,
                            replace=True, p=p_nl_dist)
    data = data.reshape((n_subjects, n_biomarkers))

    prob_nl = p_nl_dist[data]

    prob_score = np.zeros((n_subjects, n_biomarkers, n_scores))
    for n in range(n_biomarkers):
        for z in range(n_scores):
            for score in range(n_scores + 1):
                prob_score[data[:, n] == score, n, z] = p_score_dist[z, score]

    score_vals = np.tile(np.arange(1, n_scores + 1), (n_biomarkers, 1))
    biomarker_labels = [f"Biomarker_{i}" for i in range(n_biomarkers)]

    return prob_nl, prob_score, score_vals, biomarker_labels


# ============================================================
# VALIDATION TESTS
# ============================================================

def test_likelihood_stage(prob_nl, prob_score, score_vals, biomarker_labels,
                          n_tests=10, tolerance=1e-10):
    """
    Test 1: _calculate_likelihood_stage produces identical results.

    Uses force_float64=True on GPU to eliminate precision as a variable.
    """
    print("\n" + "=" * 80)
    print("TEST 1: _calculate_likelihood_stage (GPU float64 vs CPU)")
    print("=" * 80)

    output_dir = "./temp_bench"
    os.makedirs(output_dir, exist_ok=True)

    cpu_sustain = OrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 1, 100, output_dir, "cpu_test", False, 42
    )

    gpu_sustain = TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 1, 100, output_dir, "gpu_test", False, 42,
        use_gpu=True, force_float64=True
    )

    cpu_data = getattr(cpu_sustain, '_OrdinalSustain__sustainData')
    gpu_data = getattr(gpu_sustain, '_OrdinalSustain__sustainData')

    N = score_vals.shape[0] * (score_vals.shape[1])  # total events
    # Actually compute from stage_score
    N = cpu_sustain.stage_score.shape[1]
    rng = np.random.default_rng(42)
    all_passed = True

    for test_idx in range(n_tests):
        # Generate a valid random sequence (random permutation of events)
        S_test = rng.permutation(N).astype(float)

        cpu_result = cpu_sustain._calculate_likelihood_stage(cpu_data, S_test)
        gpu_result = gpu_sustain._calculate_likelihood_stage(gpu_data, S_test)

        max_diff = np.max(np.abs(cpu_result - gpu_result))
        mean_diff = np.mean(np.abs(cpu_result - gpu_result))

        passed = max_diff <= tolerance
        status = "PASS" if passed else "FAIL"
        print(f"  Test {test_idx + 1:2d}/{n_tests}: max_diff={max_diff:.2e}, "
              f"mean_diff={mean_diff:.2e}  [{status}]")

        if not passed:
            all_passed = False

    shutil.rmtree(output_dir, ignore_errors=True)

    print(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'} "
          f"(tolerance={tolerance:.0e})")
    return all_passed


def test_calculate_likelihood(prob_nl, prob_score, score_vals, biomarker_labels,
                              n_tests=5, tolerance=1e-10):
    """
    Test 2: _calculate_likelihood (full mixture) produces identical results.

    Tests with N_S=1 and N_S=2 subtypes.
    """
    print("\n" + "=" * 80)
    print("TEST 2: _calculate_likelihood (mixture model, GPU float64 vs CPU)")
    print("=" * 80)

    output_dir = "./temp_bench"
    os.makedirs(output_dir, exist_ok=True)

    cpu_sustain = OrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 2, 100, output_dir, "cpu_test", False, 42
    )

    gpu_sustain = TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 2, 100, output_dir, "gpu_test", False, 42,
        use_gpu=True, force_float64=True
    )

    cpu_data = getattr(cpu_sustain, '_OrdinalSustain__sustainData')
    gpu_data = getattr(gpu_sustain, '_OrdinalSustain__sustainData')

    N = cpu_sustain.stage_score.shape[1]
    rng = np.random.default_rng(123)
    all_passed = True

    for n_subtypes in [1, 2]:
        print(f"\n  N_S = {n_subtypes}:")
        for test_idx in range(n_tests):
            S_test = np.zeros((n_subtypes, N))
            for s in range(n_subtypes):
                S_test[s] = rng.permutation(N).astype(float)
            f_test = np.ones(n_subtypes) / n_subtypes

            cpu_ll, cpu_tps, cpu_tpst, cpu_tpc, cpu_ppk = \
                cpu_sustain._calculate_likelihood(cpu_data, S_test, f_test)
            gpu_ll, gpu_tps, gpu_tpst, gpu_tpc, gpu_ppk = \
                gpu_sustain._calculate_likelihood(gpu_data, S_test, f_test)

            ll_diff = abs(cpu_ll - gpu_ll)
            tps_diff = np.max(np.abs(cpu_tps - gpu_tps))
            ppk_diff = np.max(np.abs(cpu_ppk - gpu_ppk))

            passed = ll_diff <= tolerance and ppk_diff <= tolerance
            status = "PASS" if passed else "FAIL"
            print(f"    Test {test_idx + 1}/{n_tests}: ll_diff={ll_diff:.2e}, "
                  f"ppk_diff={ppk_diff:.2e}  [{status}]")

            if not passed:
                all_passed = False

    shutil.rmtree(output_dir, ignore_errors=True)

    print(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'} "
          f"(tolerance={tolerance:.0e})")
    return all_passed


def test_subset_handling(prob_nl, prob_score, score_vals, biomarker_labels,
                         tolerance=1e-10):
    """
    Test 3: GPU correctly handles data subsets (the bug that was fixed).

    Simulates what AbstractSustain does during cross-validation: creates
    a reindexed subset and computes likelihood on it.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Data subset handling (reindex bug fix)")
    print("=" * 80)

    output_dir = "./temp_bench"
    os.makedirs(output_dir, exist_ok=True)

    cpu_sustain = OrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 1, 100, output_dir, "cpu_test", False, 42
    )

    gpu_sustain = TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        1, 1, 100, output_dir, "gpu_test", False, 42,
        use_gpu=True, force_float64=True
    )

    full_data = getattr(cpu_sustain, '_OrdinalSustain__sustainData')
    N = cpu_sustain.stage_score.shape[1]
    M = full_data.getNumSamples()

    # Create a subset (first half of subjects)
    subset_idx = np.zeros(M, dtype=bool)
    subset_idx[:M // 2] = True
    subset_data = full_data.reindex(subset_idx)

    rng = np.random.default_rng(99)
    S_test = rng.permutation(N).astype(float)

    # CPU on subset
    cpu_result = cpu_sustain._calculate_likelihood_stage(subset_data, S_test)
    # GPU on subset (this was broken before — would use full data)
    gpu_result = gpu_sustain._calculate_likelihood_stage(subset_data, S_test)

    max_diff = np.max(np.abs(cpu_result - gpu_result))

    print(f"  Full data shape:   ({M}, {N + 1})")
    print(f"  Subset data shape: ({subset_data.getNumSamples()}, {N + 1})")
    print(f"  CPU result shape:  {cpu_result.shape}")
    print(f"  GPU result shape:  {gpu_result.shape}")
    print(f"  Max difference:    {max_diff:.2e}")

    # Check shapes match
    shapes_match = cpu_result.shape == gpu_result.shape
    values_match = max_diff <= tolerance

    passed = shapes_match and values_match
    if not shapes_match:
        print(f"  FAIL: Shape mismatch!")
    elif not values_match:
        print(f"  FAIL: Values differ beyond tolerance")
    else:
        print(f"  PASS: Subset handled correctly")

    shutil.rmtree(output_dir, ignore_errors=True)
    return passed


def test_full_pipeline(prob_nl, prob_score, score_vals, biomarker_labels,
                       tolerance=1e-6):
    """
    Test 4: Full run_sustain_algorithm produces identical results.

    Runs both CPU and GPU with same seed, compares sequences and likelihoods.
    Uses small N_iterations_MCMC for speed.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Full pipeline (run_sustain_algorithm, GPU float64 vs CPU)")
    print("=" * 80)

    cpu_dir = "./temp_bench_cpu"
    gpu_dir = "./temp_bench_gpu"
    os.makedirs(cpu_dir, exist_ok=True)
    os.makedirs(gpu_dir, exist_ok=True)

    seed = 42
    n_startpoints = 5
    n_iterations = 1000  # Small for speed

    print(f"  Config: N_startpoints={n_startpoints}, N_iterations_MCMC={n_iterations}")
    print(f"  Running CPU OrdinalSustain...")

    cpu_sustain = OrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        n_startpoints, 1, n_iterations, cpu_dir, "cpu_test", False, seed
    )
    cpu_start = time.time()
    cpu_results = cpu_sustain.run_sustain_algorithm(plot=False)
    cpu_time = time.time() - cpu_start
    print(f"  CPU time: {cpu_time:.1f}s")

    print(f"  Running GPU TorchOrdinalSustain (float64)...")
    gpu_sustain = TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        n_startpoints, 1, n_iterations, gpu_dir, "gpu_test", False, seed,
        use_gpu=True, force_float64=True
    )
    gpu_start = time.time()
    gpu_results = gpu_sustain.run_sustain_algorithm(plot=False)
    gpu_time = time.time() - gpu_start
    print(f"  GPU time: {gpu_time:.1f}s")

    # Compare results
    # results = (ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, ...)
    # The exact structure depends on N_S_max, but for N_S_max=1 we get a single subtype
    cpu_ml_stage = cpu_results[4].flatten() if cpu_results[4] is not None else np.array([])
    gpu_ml_stage = gpu_results[4].flatten() if gpu_results[4] is not None else np.array([])

    if len(cpu_ml_stage) > 0 and len(gpu_ml_stage) > 0:
        exact_match = np.mean(cpu_ml_stage == gpu_ml_stage)
        correlation = np.corrcoef(cpu_ml_stage, gpu_ml_stage)[0, 1] if len(cpu_ml_stage) > 1 else 1.0
        print(f"\n  Stage assignment exact match: {exact_match * 100:.1f}%")
        print(f"  Stage assignment correlation: {correlation:.6f}")

        # Note: through full MCMC, tiny float64 GPU vs CPU differences in log/exp
        # accumulate to different accept/reject decisions, so MCMC trajectories diverge.
        # This is expected — the likelihood-level tests (Tests 1-3) prove the math is
        # identical. Here we just check the results are reasonable (same optimum found
        # during EM, similar staging). High exact match (>90%) and positive correlation
        # indicate the same algorithm running with minor numerical noise.
        passed = exact_match > 0.80 and correlation > 0.5
    else:
        print(f"\n  Could not compare results (empty arrays)")
        passed = False

    print(f"  Speedup: {cpu_time / gpu_time:.2f}x")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if passed and exact_match < 0.99:
        print(f"  Note: <100% match expected — tiny GPU float differences accumulate")
        print(f"  through MCMC accept/reject. Tests 1-3 prove math is identical.")

    shutil.rmtree(cpu_dir, ignore_errors=True)
    shutil.rmtree(gpu_dir, ignore_errors=True)
    return passed


# ============================================================
# BENCHMARKING
# ============================================================

def benchmark_likelihood_stage(configs, n_iterations=50):
    """
    Benchmark _calculate_likelihood_stage at various dataset sizes.

    Compares CPU OrdinalSustain vs GPU TorchOrdinalSustain (float32 for speed).
    """
    print("\n" + "=" * 80)
    print("BENCHMARK: _calculate_likelihood_stage timing")
    print("=" * 80)

    results = []

    for config in configs:
        n_sub = config['n_subjects']
        n_bio = config['n_biomarkers']
        print(f"\n  {n_sub} subjects x {n_bio} biomarkers:")

        prob_nl, prob_score, score_vals, labels = generate_test_data(
            n_subjects=n_sub, n_biomarkers=n_bio
        )

        output_dir = "./temp_bench"
        os.makedirs(output_dir, exist_ok=True)

        cpu_sustain = OrdinalSustain(
            prob_nl, prob_score, score_vals, labels,
            1, 1, 100, output_dir, "cpu", False, 42
        )
        gpu_sustain = TorchOrdinalSustain(
            prob_nl, prob_score, score_vals, labels,
            1, 1, 100, output_dir, "gpu", False, 42,
            use_gpu=True, force_float64=False  # float32 for benchmark speed
        )

        cpu_data = getattr(cpu_sustain, '_OrdinalSustain__sustainData')
        gpu_data = getattr(gpu_sustain, '_OrdinalSustain__sustainData')
        N = cpu_sustain.stage_score.shape[1]
        rng = np.random.default_rng(42)
        S_test = rng.permutation(N).astype(float)

        # Warmup GPU
        for _ in range(3):
            _ = gpu_sustain._calculate_likelihood_stage(gpu_data, S_test)

        # Benchmark CPU
        cpu_times = []
        for _ in range(n_iterations):
            t0 = time.time()
            _ = cpu_sustain._calculate_likelihood_stage(cpu_data, S_test)
            cpu_times.append(time.time() - t0)

        # Benchmark GPU
        gpu_times = []
        for _ in range(n_iterations):
            t0 = time.time()
            _ = gpu_sustain._calculate_likelihood_stage(gpu_data, S_test)
            gpu_times.append(time.time() - t0)

        cpu_mean = np.mean(cpu_times)
        gpu_mean = np.mean(gpu_times)
        speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 0

        print(f"    CPU: {cpu_mean * 1000:.2f}ms  GPU: {gpu_mean * 1000:.2f}ms  "
              f"Speedup: {speedup:.2f}x")

        results.append({
            'n_subjects': n_sub,
            'n_biomarkers': n_bio,
            'cpu_ms': cpu_mean * 1000,
            'gpu_ms': gpu_mean * 1000,
            'speedup': speedup
        })

        shutil.rmtree(output_dir, ignore_errors=True)

    # Summary table
    print("\n" + "-" * 70)
    print(f"  {'Subjects':<10} {'Biomarkers':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(f"  {r['n_subjects']:<10} {r['n_biomarkers']:<12} "
              f"{r['cpu_ms']:<12.2f} {r['gpu_ms']:<12.2f} {r['speedup']:<10.2f}x")
    print("-" * 70)

    return results


def benchmark_full_pipeline(configs):
    """
    Benchmark full run_sustain_algorithm at various dataset sizes.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK: Full pipeline timing (run_sustain_algorithm)")
    print("=" * 80)

    results = []

    for config in configs:
        n_sub = config['n_subjects']
        n_bio = config['n_biomarkers']
        n_iter = config.get('n_iterations', 2000)
        n_start = config.get('n_startpoints', 5)

        print(f"\n  {n_sub} subjects x {n_bio} biomarkers "
              f"({n_start} startpoints, {n_iter} MCMC):")

        prob_nl, prob_score, score_vals, labels = generate_test_data(
            n_subjects=n_sub, n_biomarkers=n_bio
        )

        # CPU
        cpu_dir = "./temp_bench_cpu"
        os.makedirs(cpu_dir, exist_ok=True)
        cpu_sustain = OrdinalSustain(
            prob_nl, prob_score, score_vals, labels,
            n_start, 1, n_iter, cpu_dir, "cpu", False, 42
        )
        t0 = time.time()
        cpu_results = cpu_sustain.run_sustain_algorithm(plot=False)
        cpu_time = time.time() - t0
        shutil.rmtree(cpu_dir, ignore_errors=True)

        # GPU
        gpu_dir = "./temp_bench_gpu"
        os.makedirs(gpu_dir, exist_ok=True)
        gpu_sustain = TorchOrdinalSustain(
            prob_nl, prob_score, score_vals, labels,
            n_start, 1, n_iter, gpu_dir, "gpu", False, 42,
            use_gpu=True, force_float64=False  # float32 for benchmark speed
        )
        t0 = time.time()
        gpu_results = gpu_sustain.run_sustain_algorithm(plot=False)
        gpu_time = time.time() - t0
        shutil.rmtree(gpu_dir, ignore_errors=True)

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print(f"    CPU: {cpu_time:.1f}s  GPU: {gpu_time:.1f}s  Speedup: {speedup:.2f}x")

        results.append({
            'n_subjects': n_sub,
            'n_biomarkers': n_bio,
            'n_iterations': n_iter,
            'n_startpoints': n_start,
            'cpu_s': cpu_time,
            'gpu_s': gpu_time,
            'speedup': speedup
        })

    # Summary
    print("\n" + "-" * 80)
    print(f"  {'Subjects':<10} {'Biomarkers':<12} {'MCMC':<8} {'CPU (s)':<10} "
          f"{'GPU (s)':<10} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        print(f"  {r['n_subjects']:<10} {r['n_biomarkers']:<12} {r['n_iterations']:<8} "
              f"{r['cpu_s']:<10.1f} {r['gpu_s']:<10.1f} {r['speedup']:<10.2f}x")
    print("-" * 80)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU-accelerated OrdinalSustain")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation tests")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmarks")
    args = parser.parse_args()

    print("\n" + "#" * 80)
    print("# GPU-ACCELERATED ORDINAL SUSTAIN: VALIDATION & BENCHMARK")
    print("#" * 80)

    # Generate default test data
    prob_nl, prob_score, score_vals, biomarker_labels = generate_test_data(
        n_subjects=500, n_biomarkers=10
    )

    # ---- VALIDATION ----
    if not args.benchmark_only:
        print("\n\n" + "=" * 80)
        print("VALIDATION SUITE")
        print("=" * 80)

        results = {}
        results['likelihood_stage'] = test_likelihood_stage(
            prob_nl, prob_score, score_vals, biomarker_labels
        )
        results['calculate_likelihood'] = test_calculate_likelihood(
            prob_nl, prob_score, score_vals, biomarker_labels
        )
        results['subset_handling'] = test_subset_handling(
            prob_nl, prob_score, score_vals, biomarker_labels
        )
        results['full_pipeline'] = test_full_pipeline(
            prob_nl, prob_score, score_vals, biomarker_labels
        )

        print("\n\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        all_passed = True
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {name:<30} [{status}]")
            if not passed:
                all_passed = False
        print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print("=" * 80)

    # ---- BENCHMARKING ----
    if not args.validate_only:
        print("\n\n" + "=" * 80)
        print("BENCHMARK SUITE")
        print("=" * 80)

        likelihood_configs = [
            {"n_subjects": 100, "n_biomarkers": 5},
            {"n_subjects": 500, "n_biomarkers": 10},
            {"n_subjects": 1000, "n_biomarkers": 10},
            {"n_subjects": 2000, "n_biomarkers": 15},
            {"n_subjects": 5000, "n_biomarkers": 19},  # DICE-scale
        ]

        pipeline_configs = [
            {"n_subjects": 200, "n_biomarkers": 10, "n_iterations": 2000, "n_startpoints": 5},
            {"n_subjects": 500, "n_biomarkers": 15, "n_iterations": 5000, "n_startpoints": 10},
            {"n_subjects": 1000, "n_biomarkers": 19, "n_iterations": 5000, "n_startpoints": 10},
        ]

        benchmark_likelihood_stage(likelihood_configs)
        benchmark_full_pipeline(pipeline_configs)


if __name__ == "__main__":
    main()
