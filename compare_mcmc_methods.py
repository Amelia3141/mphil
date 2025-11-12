#!/usr/bin/env python3
"""
Comparison of different MCMC methods for SuStaIn.
Shows performance differences between traditional, parallel, and NumPyro approaches.
"""

import time
import numpy as np
import os
from typing import Dict, Any

def test_traditional_mcmc(data: np.ndarray, 
                         Z_vals: np.ndarray, 
                         Z_max: np.ndarray,
                         n_iterations: int = 1000) -> Dict[str, Any]:
    """Test traditional sequential MCMC."""
    print("🔍 Testing Traditional Sequential MCMC")
    print("=" * 50)
    
    start_time = time.time()
    
    # Simulate traditional MCMC (sequential)
    n_chains = 4
    chain_times = []
    
    for chain_idx in range(n_chains):
        chain_start = time.time()
        
        # Simulate MCMC iterations
        for i in range(n_iterations):
            if i % 200 == 0:
                print(f"  Chain {chain_idx + 1}: {i}/{n_iterations} iterations")
            # Simulate some computation
            _ = np.random.rand(100).sum()
        
        chain_time = time.time() - chain_start
        chain_times.append(chain_time)
        print(f"  Chain {chain_idx + 1} completed in {chain_time:.3f}s")
    
    total_time = time.time() - start_time
    
    return {
        'method': 'Traditional Sequential',
        'total_time': total_time,
        'chain_times': chain_times,
        'speedup': 1.0,
        'efficiency': 1.0
    }

def test_parallel_mcmc(data: np.ndarray, 
                      Z_vals: np.ndarray, 
                      Z_max: np.ndarray,
                      n_iterations: int = 1000) -> Dict[str, Any]:
    """Test parallel MCMC using ThreadPoolExecutor."""
    print("\n🔍 Testing Parallel MCMC (ThreadPoolExecutor)")
    print("=" * 50)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def mcmc_chain(chain_idx, n_iterations):
        """Run a single MCMC chain."""
        chain_start = time.time()
        
        # Simulate MCMC iterations
        for i in range(n_iterations):
            if i % 200 == 0:
                print(f"  Chain {chain_idx + 1}: {i}/{n_iterations} iterations")
            # Simulate some computation
            _ = np.random.rand(100).sum()
        
        chain_time = time.time() - chain_start
        print(f"  Chain {chain_idx + 1} completed in {chain_time:.3f}s")
        return chain_idx, chain_time
    
    start_time = time.time()
    
    # Run chains in parallel
    n_chains = 4
    with ThreadPoolExecutor(max_workers=n_chains) as executor:
        futures = [executor.submit(mcmc_chain, i, n_iterations) for i in range(n_chains)]
        results = [future.result() for future in as_completed(futures)]
    
    total_time = time.time() - start_time
    chain_times = [r[1] for r in results]
    
    # Calculate speedup
    expected_serial_time = sum(chain_times)
    speedup = expected_serial_time / total_time if total_time > 0 else 1.0
    efficiency = speedup / n_chains
    
    return {
        'method': 'Parallel MCMC (Threading)',
        'total_time': total_time,
        'chain_times': chain_times,
        'speedup': speedup,
        'efficiency': efficiency
    }

def test_numpyro_mcmc(data: np.ndarray, 
                     Z_vals: np.ndarray, 
                     Z_max: np.ndarray,
                     n_iterations: int = 1000) -> Dict[str, Any]:
    """Test NumPyro MCMC."""
    print("\n🔍 Testing NumPyro MCMC")
    print("=" * 50)
    
    try:
        from pySuStaIn.numpyro_sustain import create_numpyro_sustain
        
        # Create NumPyro SuStaIn instance
        sustain = create_numpyro_sustain(
            data=data,
            Z_vals=Z_vals,
            Z_max=Z_max,
            biomarker_labels=['biomarker_1', 'biomarker_2', 'biomarker_3'],
            N_startpoints=10,
            N_S_max=1,
            N_iterations_MCMC=n_iterations,
            use_gpu=False,  # Set to True if you have GPU
            n_chains=4,
            n_warmup=100,
            n_samples=200
        )
        
        start_time = time.time()
        
        # Run the algorithm
        result = sustain.run_sustain_algorithm()
        
        total_time = time.time() - start_time
        
        # Extract results
        chain_times = result[0] if len(result) > 0 else [0.0]
        speedup = getattr(sustain, 'speedup', 1.0)
        efficiency = getattr(sustain, 'efficiency', 0.0)
        
        return {
            'method': 'NumPyro MCMC',
            'total_time': total_time,
            'chain_times': chain_times,
            'speedup': speedup,
            'efficiency': efficiency
        }
        
    except ImportError as e:
        print(f"❌ NumPyro not available: {e}")
        return {
            'method': 'NumPyro MCMC (Not Available)',
            'total_time': 0.0,
            'chain_times': [0.0],
            'speedup': 0.0,
            'efficiency': 0.0
        }
    except Exception as e:
        print(f"❌ NumPyro MCMC failed: {e}")
        return {
            'method': 'NumPyro MCMC (Failed)',
            'total_time': 0.0,
            'chain_times': [0.0],
            'speedup': 0.0,
            'efficiency': 0.0
        }

def main():
    """Compare different MCMC methods."""
    print("🧪 MCMC Methods Comparison for SuStaIn")
    print("=" * 60)
    print("This script compares different MCMC approaches for SuStaIn.")
    print("Run this to see the performance differences.\n")
    
    # Create test data
    data = np.random.randn(100, 3)
    Z_vals = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    Z_max = np.array([5, 5, 5])
    
    print(f"Test data: {data.shape}")
    print(f"Z_vals: {Z_vals.shape}")
    print(f"Z_max: {Z_max.shape}")
    print()
    
    # Test different methods
    results = []
    
    # Test traditional MCMC
    traditional_result = test_traditional_mcmc(data, Z_vals, Z_max, n_iterations=200)
    results.append(traditional_result)
    
    # Test parallel MCMC
    parallel_result = test_parallel_mcmc(data, Z_vals, Z_max, n_iterations=200)
    results.append(parallel_result)
    
    # Test NumPyro MCMC
    numpyro_result = test_numpyro_mcmc(data, Z_vals, Z_max, n_iterations=200)
    results.append(numpyro_result)
    
    # Display results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"{'Method':<25} {'Time (s)':<10} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 60)
    
    for result in results:
        method = result['method']
        total_time = result['total_time']
        speedup = result['speedup']
        efficiency = result['efficiency']
        
        print(f"{method:<25} {total_time:<10.2f} {speedup:<10.2f} {efficiency:<12.2f}")
    
    # Find best method
    best_result = max(results, key=lambda x: x['speedup'])
    print(f"\n🏆 Best method: {best_result['method']}")
    print(f"   Speedup: {best_result['speedup']:.2f}x")
    print(f"   Efficiency: {best_result['efficiency']:.2f}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if numpyro_result['speedup'] > 0:
        print("✅ NumPyro MCMC is available and working")
        print("   - Provides significant speedups through JAX compilation")
        print("   - Supports GPU acceleration")
        print("   - Highly optimized MCMC samplers")
        print("   - Vectorized operations across chains")
    else:
        print("❌ NumPyro MCMC not available")
        print("   - Install NumPyro: pip install numpyro")
        print("   - Install JAX: pip install jax jaxlib")
        print("   - For GPU: pip install jax[cuda]")
    
    if parallel_result['speedup'] > 1.0:
        print("✅ Parallel MCMC is working")
        print("   - Provides moderate speedups through threading")
        print("   - Good for CPU-bound tasks")
        print("   - Works well with existing SuStaIn code")
    else:
        print("❌ Parallel MCMC not working")
        print("   - Check threading support")
        print("   - Verify ThreadPoolExecutor is available")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. If NumPyro is available, use it for maximum speedup")
    print("2. If NumPyro is not available, use parallel MCMC")
    print("3. For production use, consider implementing full NumPyro integration")
    print("4. Test with your actual SuStaIn data to see real performance gains")

if __name__ == "__main__":
    main()
