#!/usr/bin/env python3
"""
Example script demonstrating GPU-accelerated SuStaIn usage.

This script shows how to use the new PyTorch-based GPU acceleration
for ZScoreSustainMissingData computations.

Authors: GPU Migration Team
"""

import numpy as np
import time
import sys
import os

# Add the pySuStaIn directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pySuStaIn'))

from pySuStaIn.ZScoreSustainMissingData import ZscoreSustainMissingData
from pySuStaIn.TorchZScoreSustainMissingData import TorchZScoreSustainMissingData, benchmark_gpu_vs_cpu
from pySuStaIn.test_torch_validation import validate_numerical_accuracy


def create_example_data(n_subjects=500, n_biomarkers=8, missing_rate=0.15):
    """
    Create example data for testing GPU acceleration.
    
    Args:
        n_subjects: Number of subjects
        n_biomarkers: Number of biomarkers
        missing_rate: Fraction of missing values
        
    Returns:
        Tuple of (data, Z_vals, Z_max, biomarker_labels)
    """
    print(f"Creating example data: {n_subjects} subjects, {n_biomarkers} biomarkers")
    
    # Create synthetic z-score data
    np.random.seed(42)
    data = np.random.randn(n_subjects, n_biomarkers) * 1.5 + 0.5
    
    # Add missing values
    n_missing = int(n_subjects * n_biomarkers * missing_rate)
    missing_indices = np.random.choice(n_subjects * n_biomarkers, size=n_missing, replace=False)
    data.flat[missing_indices] = np.nan
    
    # Create Z-score thresholds (3 thresholds per biomarker)
    Z_vals = np.array([[1, 2, 3]] * n_biomarkers)
    Z_max = np.array([5.0] * n_biomarkers)
    biomarker_labels = [f"Biomarker_{i+1}" for i in range(n_biomarkers)]
    
    print(f"  - Data shape: {data.shape}")
    print(f"  - Missing values: {np.isnan(data).sum()} ({np.isnan(data).mean():.1%})")
    
    return data, Z_vals, Z_max, biomarker_labels


def run_performance_comparison(data, Z_vals, Z_max, biomarker_labels):
    """Run performance comparison between CPU and GPU implementations."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Run benchmark
    benchmark_results = benchmark_gpu_vs_cpu(
        data, Z_vals, Z_max, biomarker_labels, num_iterations=10
    )
    
    print(f"CPU mean time: {benchmark_results['cpu_mean_time']:.4f} ± {benchmark_results['cpu_std_time']:.4f} seconds")
    print(f"GPU mean time: {benchmark_results['gpu_mean_time']:.4f} ± {benchmark_results['gpu_std_time']:.4f} seconds")
    
    if benchmark_results['gpu_available']:
        print(f"🚀 GPU Speedup: {benchmark_results['speedup']:.2f}x")
        if benchmark_results['speedup'] > 2.0:
            print("   Excellent GPU acceleration!")
        elif benchmark_results['speedup'] > 1.5:
            print("   Good GPU acceleration!")
        else:
            print("   Modest GPU acceleration (may be due to small dataset)")
    else:
        print("⚠️  GPU not available - running on CPU only")
    
    return benchmark_results


def run_numerical_validation(data, Z_vals, Z_max, biomarker_labels):
    """Run numerical accuracy validation."""
    print("\n" + "="*60)
    print("NUMERICAL ACCURACY VALIDATION")
    print("="*60)
    
    validation_results = validate_numerical_accuracy(
        data, Z_vals, Z_max, biomarker_labels, tolerance=1e-5
    )
    
    if validation_results['tests_failed'] == 0:
        print("✅ All numerical accuracy tests passed!")
    else:
        print(f"❌ {validation_results['tests_failed']} test(s) failed")
        for error in validation_results['errors']:
            print(f"   - {error}")
    
    return validation_results


def demonstrate_gpu_usage(data, Z_vals, Z_max, biomarker_labels):
    """Demonstrate how to use the GPU-accelerated implementation."""
    print("\n" + "="*60)
    print("GPU USAGE DEMONSTRATION")
    print("="*60)
    
    # Create GPU-accelerated SuStaIn instance
    print("Creating GPU-accelerated SuStaIn instance...")
    gpu_sustain = TorchZScoreSustainMissingData(
        data, Z_vals, Z_max, biomarker_labels,
        N_startpoints=10,
        N_S_max=3,
        N_iterations_MCMC=5000,
        output_folder="./gpu_output",
        dataset_name="gpu_example",
        use_parallel_startpoints=True,
        seed=42,
        use_gpu=True  # Enable GPU acceleration
    )
    
    print(f"GPU acceleration: {'Enabled' if gpu_sustain.use_gpu else 'Disabled'}")
    
    # Get performance statistics
    if gpu_sustain.use_gpu:
        stats = gpu_sustain.get_performance_stats()
        print(f"Performance statistics:")
        for operation, time_taken in stats['computation_times'].items():
            print(f"  - {operation}: {time_taken:.4f} seconds")
        
        if 'device_info' in stats:
            device_info = stats['device_info']
            print(f"GPU memory usage:")
            print(f"  - Allocated: {device_info['allocated']:.2f} GB")
            print(f"  - Reserved: {device_info['reserved']:.2f} GB")
    
    # Demonstrate switching between CPU and GPU
    print("\nDemonstrating CPU/GPU switching...")
    
    # Switch to CPU
    gpu_sustain.switch_to_cpu()
    print("Switched to CPU mode")
    
    # Switch back to GPU
    gpu_sustain.switch_to_gpu()
    print("Switched back to GPU mode")
    
    return gpu_sustain


def main():
    """Main demonstration function."""
    print("🚀 GPU-Accelerated SuStaIn Demonstration")
    print("="*60)
    
    # Create example data
    data, Z_vals, Z_max, biomarker_labels = create_example_data(
        n_subjects=1000,  # Larger dataset for better GPU performance
        n_biomarkers=10,
        missing_rate=0.2
    )
    
    # Run performance comparison
    benchmark_results = run_performance_comparison(data, Z_vals, Z_max, biomarker_labels)
    
    # Run numerical validation
    validation_results = run_numerical_validation(data, Z_vals, Z_max, biomarker_labels)
    
    # Demonstrate usage
    gpu_sustain = demonstrate_gpu_usage(data, Z_vals, Z_max, biomarker_labels)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if benchmark_results['gpu_available']:
        print(f"✅ GPU acceleration is available and working")
        print(f"✅ Performance improvement: {benchmark_results['speedup']:.2f}x speedup")
    else:
        print("⚠️  GPU acceleration not available (CUDA not found)")
    
    if validation_results['tests_failed'] == 0:
        print("✅ Numerical accuracy validated")
    else:
        print(f"❌ {validation_results['tests_failed']} numerical accuracy issues found")
    
    print("\n🎉 GPU acceleration demonstration completed!")
    print("\nTo use GPU acceleration in your own code:")
    print("1. Import: from pySuStaIn.TorchZScoreSustainMissingData import TorchZScoreSustainMissingData")
    print("2. Create instance with use_gpu=True")
    print("3. Use the same API as the original ZScoreSustainMissingData")
    
    return benchmark_results, validation_results


if __name__ == "__main__":
    try:
        benchmark_results, validation_results = main()
        
        # Exit with appropriate code
        if validation_results['tests_failed'] == 0:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Validation failed
            
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
