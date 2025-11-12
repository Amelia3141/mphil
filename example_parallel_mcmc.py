#!/usr/bin/env python3
"""
Example script demonstrating parallel MCMC execution for SuStaIn algorithms.

This script shows how to use the parallel MCMC implementation to significantly
reduce computation time for SuStaIn algorithms.
"""

import numpy as np
import time
import sys
import os

# Add the pySuStaIn directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pySuStaIn'))

def create_sample_data(n_subjects=200, n_biomarkers=6):
    """Create sample data for testing."""
    # Generate synthetic z-score data
    data = np.random.randn(n_subjects, n_biomarkers)
    
    # Create Z_vals matrix (3 thresholds per biomarker)
    Z_vals = np.tile(np.array([1, 2, 3]), (n_biomarkers, 1))
    
    # Create Z_max vector
    Z_max = np.full(n_biomarkers, 5.0)
    
    # Create biomarker labels
    biomarker_labels = [f'biomarker_{i+1}' for i in range(n_biomarkers)]
    
    return data, Z_vals, Z_max, biomarker_labels


def benchmark_serial_vs_parallel():
    """Benchmark serial vs parallel MCMC execution."""
    print("=== SuStaIn Parallel MCMC Benchmark ===\n")
    
    # Create sample data
    print("Creating sample data...")
    data, Z_vals, Z_max, biomarker_labels = create_sample_data(n_subjects=150, n_biomarkers=5)
    print(f"Data shape: {data.shape}")
    
    try:
        from pySuStaIn.parallel_torch_sustain import create_parallel_torch_zscore_sustain_missing_data
        
        # Test parameters
        n_iterations = 2000  # Reduced for demo
        n_chains_list = [1, 2, 4]  # Test different numbers of chains
        
        print(f"\nTesting with {n_iterations} MCMC iterations per chain...")
        
        results = {}
        
        for n_chains in n_chains_list:
            print(f"\n--- Testing {n_chains} chain(s) ---")
            
            # Create parallel SuStaIn instance
            sustain = create_parallel_torch_zscore_sustain_missing_data(
                data=data,
                Z_vals=Z_vals,
                Z_max=Z_max,
                biomarker_labels=biomarker_labels,
                N_startpoints=5,  # Reduced for demo
                N_S_max=2,
                N_iterations_MCMC=n_iterations,
                output_folder="./temp_output",
                dataset_name=f"parallel_test_{n_chains}chains",
                use_parallel_startpoints=True,
                seed=42,
                use_gpu=False,  # Use CPU for compatibility
                use_parallel_mcmc=(n_chains > 1),
                n_mcmc_chains=n_chains,
                mcmc_backend='process'
            )
            
            # Run benchmark
            start_time = time.time()
            
            # Run just the MCMC part (not the full algorithm)
            from pySuStaIn.ZScoreSustainMissingData import ZScoreSustainData
            sustain_data = ZScoreSustainData(data, sustain._ZscoreSustainMissingData__sustainData.getNumStages())
            
            # Create test sequences
            N = sustain_data.getNumStages()
            seq_init = np.random.permutation(N).reshape(1, N)
            f_init = np.array([1.0])
            
            # Run MCMC
            if n_chains == 1:
                # Serial execution
                ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood = \
                    sustain._original_estimate_uncertainty_sustain_model(sustain_data, seq_init, f_init)
            else:
                # Parallel execution
                ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood = \
                    sustain._estimate_uncertainty_sustain_model(sustain_data, seq_init, f_init)
            
            total_time = time.time() - start_time
            
            results[n_chains] = {
                'time': total_time,
                'samples_shape': samples_sequence.shape,
                'likelihood_shape': samples_likelihood.shape
            }
            
            print(f"  Time: {total_time:.2f} seconds")
            print(f"  Samples shape: {samples_sequence.shape}")
            print(f"  Likelihood shape: {samples_likelihood.shape}")
        
        # Calculate speedup
        print(f"\n=== Performance Summary ===")
        serial_time = results[1]['time']
        
        for n_chains in n_chains_list:
            time_taken = results[n_chains]['time']
            speedup = serial_time / time_taken
            efficiency = speedup / n_chains
            
            print(f"{n_chains} chains: {time_taken:.2f}s, speedup: {speedup:.2f}x, efficiency: {efficiency:.2f}")
        
        return results
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install pathos concurrent.futures")
        return None
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_parallel_features():
    """Demonstrate parallel MCMC features."""
    print("\n=== Parallel MCMC Features Demo ===\n")
    
    try:
        from pySuStaIn.parallel_torch_sustain import create_parallel_torch_zscore_sustain_missing_data
        
        # Create sample data
        data, Z_vals, Z_max, biomarker_labels = create_sample_data(n_subjects=100, n_biomarkers=4)
        
        # Create parallel SuStaIn instance
        sustain = create_parallel_torch_zscore_sustain_missing_data(
            data=data,
            Z_vals=Z_vals,
            Z_max=Z_max,
            biomarker_labels=biomarker_labels,
            N_startpoints=3,
            N_S_max=2,
            N_iterations_MCMC=500,
            use_parallel_mcmc=True,
            n_mcmc_chains=4,
            mcmc_backend='process',
            use_gpu=False
        )
        
        print("Parallel SuStaIn instance created successfully!")
        
        # Get parallel stats
        stats = sustain.get_parallel_stats()
        print(f"Parallel stats: {stats}")
        
        # Demonstrate backend switching
        print("\nSwitching to thread backend...")
        sustain.switch_parallel_backend('thread', n_workers=2)
        
        # Demonstrate disabling parallel MCMC
        print("\nDisabling parallel MCMC...")
        sustain.disable_parallel_mcmc()
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run examples."""
    print("SuStaIn Parallel MCMC Examples")
    print("=" * 40)
    
    # Run benchmark
    benchmark_results = benchmark_serial_vs_parallel()
    
    if benchmark_results:
        # Run feature demo
        demo_parallel_features()
        
        print("\n=== Summary ===")
        print("Parallel MCMC implementation provides:")
        print("1. Multiple MCMC chains running in parallel")
        print("2. Process-based and thread-based parallelism")
        print("3. GPU acceleration support")
        print("4. Automatic result combination")
        print("5. Performance benchmarking")
        print("\nExpected speedup: 2-4x for 4 chains (depending on system)")
    else:
        print("Benchmark failed - check dependencies and try again")


if __name__ == "__main__":
    main()
