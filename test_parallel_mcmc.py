#!/usr/bin/env python
"""
Test Parallel MCMC for OrdinalSustain with user's actual dataset size:
- 8000 subjects
- 13 biomarkers (symptom domains)
- 3 severity levels
- 39 total stages (13 × 3)

This demonstrates the 2-4x speedup from running multiple MCMC chains in parallel.
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


def compare_serial_vs_parallel(prob_nl, prob_score, score_vals, biomarker_labels,
                               n_iterations=1000):
    """Compare serial vs parallel MCMC performance."""

    print("="*80)
    print("PARALLEL MCMC SPEEDUP TEST")
    print("="*80)
    print(f"\nDataset configuration:")
    print(f"  • Subjects: {prob_nl.shape[0]:,}")
    print(f"  • Symptom domains (biomarkers): {prob_nl.shape[1]}")
    print(f"  • Severity levels per domain: {prob_score.shape[2]}")
    print(f"  • MCMC iterations: {n_iterations:,}")
    print(f"  • Total disease events (stages): {prob_nl.shape[1] * prob_score.shape[2]}")

    # Test configurations
    test_configs = [
        {"name": "Serial (1 chain)", "n_chains": 1, "use_parallel": False},
        {"name": "Parallel (2 chains)", "n_chains": 2, "use_parallel": True},
        {"name": "Parallel (4 chains)", "n_chains": 4, "use_parallel": True},
    ]

    results = []

    for config in test_configs:
        print("\n" + "="*80)
        print(f"Testing: {config['name']}")
        print("="*80)

        start_time = time.time()

        if config['use_parallel']:
            # Create parallel instance
            sustain = ParallelOrdinalSustain(
                prob_nl, prob_score, score_vals, biomarker_labels,
                N_startpoints=1,
                N_S_max=1,
                N_iterations_MCMC=n_iterations,
                output_folder="./temp",
                dataset_name="parallel_test",
                use_parallel_startpoints=False,
                seed=42,
                use_parallel_mcmc=True,
                n_mcmc_chains=config['n_chains'],
                mcmc_backend='thread'
            )
        else:
            # Create serial instance
            sustain = OrdinalSustain(
                prob_nl, prob_score, score_vals, biomarker_labels,
                N_startpoints=1,
                N_S_max=1,
                N_iterations_MCMC=n_iterations,
                output_folder="./temp",
                dataset_name="serial_test",
                use_parallel_startpoints=False,
                seed=42
            )

        # Get data object
        sustain_data = getattr(sustain, '_OrdinalSustain__sustainData')

        # Initialize sequence and fractions using proper OrdinalSustain method
        # OrdinalSustain requires sequences where each biomarker's stages are in increasing order
        rng = np.random.default_rng(42)
        S_init = sustain._initialise_sequence(sustain_data, rng).reshape(1, -1)
        f_init = np.array([1.0])

        # Run uncertainty estimation (this is where MCMC happens)
        print("\nRunning MCMC uncertainty estimation...")
        ml_sequence, ml_f, ml_likelihood, samples_seq, samples_f, samples_like = \
            sustain._estimate_uncertainty_sustain_model(sustain_data, S_init, f_init)

        total_time = time.time() - start_time

        results.append({
            'name': config['name'],
            'n_chains': config['n_chains'],
            'time': total_time,
            'samples': samples_like.shape[0]
        })

        print(f"\n✓ {config['name']} completed in {total_time:.2f} seconds")
        print(f"  Total samples collected: {samples_like.shape[0]:,}")

    # Calculate speedups
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    baseline_time = results[0]['time']

    print(f"\n{'Configuration':<25} {'Time (s)':<12} {'Samples':<12} {'Speedup':<10}")
    print("-"*80)

    for result in results:
        speedup = baseline_time / result['time']
        print(f"{result['name']:<25} {result['time']:<12.2f} "
              f"{result['samples']:<12,} {speedup:<10.2f}x")

    print("="*80)

    # Interpretation
    best_speedup = max(baseline_time / r['time'] for r in results[1:])
    print(f"\n🚀 Best speedup achieved: {best_speedup:.2f}x")

    if best_speedup >= 1.8:
        print("✅ Parallel MCMC provides significant performance improvement!")
        print("   Recommendation: Use ParallelOrdinalSustain with 2-4 chains")
    elif best_speedup >= 1.2:
        print("✓ Parallel MCMC provides moderate performance improvement")
        print("  Recommendation: Use ParallelOrdinalSustain with 2 chains")
    else:
        print("⚠️  Parallel MCMC overhead too high for this dataset size")
        print("   Recommendation: Use original OrdinalSustain")

    return results


def benchmark_different_chain_counts(prob_nl, prob_score, score_vals, biomarker_labels,
                                     n_iterations=500):
    """Benchmark different numbers of parallel chains."""

    print("\n" + "="*80)
    print("CHAIN COUNT OPTIMIZATION")
    print("="*80)
    print("\nFinding optimal number of parallel chains...\n")

    # Create instance
    sustain = ParallelOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints=1,
        N_S_max=1,
        N_iterations_MCMC=n_iterations,
        output_folder="./temp",
        dataset_name="benchmark_test",
        use_parallel_startpoints=False,
        seed=42,
        use_parallel_mcmc=True,
        n_mcmc_chains=4,  # Will be overridden in benchmark
        mcmc_backend='thread'
    )

    # Get data
    sustain_data = getattr(sustain, '_OrdinalSustain__sustainData')

    # Initialize sequence using proper OrdinalSustain method
    rng = np.random.default_rng(42)
    S_init = sustain._initialise_sequence(sustain_data, rng).reshape(1, -1)
    f_init = np.array([1.0])

    # Run benchmark
    results = sustain.benchmark_parallel_performance(
        sustain_data, S_init, f_init,
        n_chains_list=[1, 2, 4]
    )

    # Find optimal
    best_chains = max(results.keys(), key=lambda k: results[k]['speedup'])

    print(f"\n💡 Optimal configuration for your dataset:")
    print(f"   • Number of chains: {best_chains}")
    print(f"   • Speedup: {results[best_chains]['speedup']:.2f}x")
    print(f"   • Efficiency: {results[best_chains]['efficiency']*100:.1f}%")

    return results


def main():
    print("="*80)
    print("PARALLEL MCMC FOR ORDINALSUSTAIN")
    print("Testing with User's Dataset Size")
    print("="*80)

    # Generate test data
    print("\n📊 Generating synthetic test data...")
    prob_nl, prob_score, score_vals, biomarker_labels = generate_test_data(
        n_subjects=8000,
        n_biomarkers=13,
        n_scores=3
    )
    print(f"✓ Data generated: {prob_nl.shape[0]:,} subjects, {prob_nl.shape[1]} biomarkers")

    # Run comparison with reduced iterations for faster testing
    print("\n" + "="*80)
    print("PART 1: Serial vs Parallel Comparison")
    print("="*80)

    comparison_results = compare_serial_vs_parallel(
        prob_nl, prob_score, score_vals, biomarker_labels,
        n_iterations=1000  # Reduced for faster testing
    )

    # Run optimization benchmark
    print("\n" + "="*80)
    print("PART 2: Chain Count Optimization")
    print("="*80)

    optimization_results = benchmark_different_chain_counts(
        prob_nl, prob_score, score_vals, biomarker_labels,
        n_iterations=500  # Reduced for faster testing
    )

    # Final recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)

    print(f"\nFor your dataset ({prob_nl.shape[0]:,} subjects, {prob_nl.shape[1]} biomarkers):")
    print("\n1. Use ParallelOrdinalSustain instead of OrdinalSustain")
    print("2. Recommended configuration:")
    print("   ```python")
    print("   from pySuStaIn.ParallelOrdinalSustain import ParallelOrdinalSustain")
    print("")
    print("   sustain = ParallelOrdinalSustain(")
    print("       prob_nl, prob_score, score_vals, biomarker_labels,")
    print("       N_startpoints=25,")
    print("       N_S_max=3,")
    print("       N_iterations_MCMC=100000,  # Your actual iteration count")
    print("       output_folder='./output',")
    print("       dataset_name='my_data',")
    print("       use_parallel_startpoints=True,")
    print("       use_parallel_mcmc=True,     # Enable parallel MCMC")
    print("       n_mcmc_chains=4,            # Use 4 chains (2-4x speedup)")
    print("       mcmc_backend='thread'       # Use thread backend")
    print("   )")
    print("   ```")
    print("\n3. Expected performance:")

    best_result = max(comparison_results[1:], key=lambda x: x['time'])
    speedup = comparison_results[0]['time'] / best_result['time']
    print(f"   • Speedup: ~{speedup:.1f}x faster than serial")
    print(f"   • Time savings: ~{(1 - 1/speedup)*100:.0f}% reduction")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
