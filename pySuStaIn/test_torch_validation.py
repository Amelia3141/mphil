###
# Validation tests for GPU-accelerated SuStaIn implementations
# 
# This module provides comprehensive tests to ensure numerical accuracy
# between the original CPU implementations and the new GPU implementations.
#
# Authors: GPU Migration Team
###

import numpy as np
import torch
import unittest
from typing import Tuple, Dict, Any
import time
import warnings

from .ZScoreSustainMissingData import ZscoreSustainMissingData
from .TorchZScoreSustainMissingData import TorchZScoreSustainMissingData, benchmark_gpu_vs_cpu
from .torch_backend import create_torch_backend
from .torch_data_classes import create_torch_zscore_data
from .torch_likelihood import create_zscore_missing_data_likelihood_calculator


class TestTorchValidation(unittest.TestCase):
    """Test suite for validating GPU implementations against CPU implementations."""
    
    def setUp(self):
        """Set up test data and parameters."""
        # Create synthetic test data
        np.random.seed(42)
        self.M = 100  # Number of subjects
        self.B = 5    # Number of biomarkers
        
        # Create test data with some missing values
        self.data = np.random.randn(self.M, self.B) * 2 + 1  # Z-scores around 1
        # Add some missing values (NaN)
        missing_indices = np.random.choice(self.M * self.B, size=int(0.1 * self.M * self.B), replace=False)
        self.data.flat[missing_indices] = np.nan
        
        # Create Z-score thresholds
        self.Z_vals = np.array([[1, 2, 3]] * self.B)  # 3 thresholds per biomarker
        self.Z_max = np.array([5.0] * self.B)
        self.biomarker_labels = [f"Biomarker_{i}" for i in range(self.B)]
        
        # Test parameters
        self.N_startpoints = 5
        self.N_S_max = 2
        self.N_iterations_MCMC = 1000
        
        # Create test sequences and fractions
        N_stages = np.sum(self.Z_vals > 0) + 1
        self.S_test = np.random.permutation(N_stages).reshape(1, N_stages)
        self.f_test = np.array([1.0])
        
        # Tolerance for numerical comparisons
        self.rtol = 1e-5  # Relative tolerance
        self.atol = 1e-8  # Absolute tolerance
    
    def test_torch_backend_initialization(self):
        """Test PyTorch backend initialization."""
        # Test GPU backend (if available)
        gpu_backend = create_torch_backend(use_gpu=True)
        self.assertIsNotNone(gpu_backend)
        
        # Test CPU backend
        cpu_backend = create_torch_backend(use_gpu=False)
        self.assertIsNotNone(cpu_backend)
        self.assertFalse(cpu_backend.use_gpu)
    
    def test_data_class_conversion(self):
        """Test conversion between numpy arrays and PyTorch tensors."""
        backend = create_torch_backend(use_gpu=False)  # Use CPU for testing
        
        # Test tensor conversion
        test_array = np.random.randn(10, 5)
        torch_tensor = backend.to_torch(test_array)
        numpy_array = backend.to_numpy(torch_tensor)
        
        np.testing.assert_allclose(test_array, numpy_array, rtol=self.rtol, atol=self.atol)
    
    def test_likelihood_stage_accuracy(self):
        """Test accuracy of GPU likelihood stage computation vs CPU."""
        # Create CPU version
        cpu_sustain = ZscoreSustainMissingData(
            self.data, self.Z_vals, self.Z_max, self.biomarker_labels,
            self.N_startpoints, self.N_S_max, self.N_iterations_MCMC,
            "./temp", "cpu_test", False, 42
        )
        
        # Create GPU version
        gpu_sustain = TorchZScoreSustainMissingData(
            self.data, self.Z_vals, self.Z_max, self.biomarker_labels,
            self.N_startpoints, self.N_S_max, self.N_iterations_MCMC,
            "./temp", "gpu_test", False, 42, use_gpu=True
        )
        
        # Test likelihood stage computation
        cpu_result = cpu_sustain._calculate_likelihood_stage(
            cpu_sustain._ZScoreSustainMissingData__sustainData, self.S_test[0]
        )
        
        gpu_result = gpu_sustain._calculate_likelihood_stage(
            gpu_sustain._ZScoreSustainMissingData__sustainData, self.S_test[0]
        )
        
        # Compare results
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=self.rtol, atol=self.atol,
                                 err_msg="GPU likelihood stage computation differs from CPU")
    
    def test_likelihood_accuracy(self):
        """Test accuracy of full likelihood computation vs CPU."""
        # Create CPU version
        cpu_sustain = ZscoreSustainMissingData(
            self.data, self.Z_vals, self.Z_max, self.biomarker_labels,
            self.N_startpoints, self.N_S_max, self.N_iterations_MCMC,
            "./temp", "cpu_test", False, 42
        )
        
        # Create GPU version
        gpu_sustain = TorchZScoreSustainMissingData(
            self.data, self.Z_vals, self.Z_max, self.biomarker_labels,
            self.N_startpoints, self.N_S_max, self.N_iterations_MCMC,
            "./temp", "gpu_test", False, 42, use_gpu=True
        )
        
        # Test full likelihood computation
        cpu_loglike, cpu_prob_subj, cpu_prob_stage, cpu_prob_cluster, cpu_p_perm_k = \
            cpu_sustain._calculate_likelihood(
                cpu_sustain._ZScoreSustainMissingData__sustainData, self.S_test, self.f_test
            )
        
        gpu_loglike, gpu_prob_subj, gpu_prob_stage, gpu_prob_cluster, gpu_p_perm_k = \
            gpu_sustain._calculate_likelihood(
                gpu_sustain._ZScoreSustainMissingData__sustainData, self.S_test, self.f_test
            )
        
        # Compare results
        self.assertAlmostEqual(cpu_loglike, gpu_loglike, places=5,
                             msg="GPU log-likelihood differs from CPU")
        
        np.testing.assert_allclose(cpu_prob_subj, gpu_prob_subj, rtol=self.rtol, atol=self.atol,
                                 err_msg="GPU subject probabilities differ from CPU")
        
        np.testing.assert_allclose(cpu_prob_stage, gpu_prob_stage, rtol=self.rtol, atol=self.atol,
                                 err_msg="GPU stage probabilities differ from CPU")
        
        np.testing.assert_allclose(cpu_prob_cluster, gpu_prob_cluster, rtol=self.rtol, atol=self.atol,
                                 err_msg="GPU cluster probabilities differ from CPU")
        
        np.testing.assert_allclose(cpu_p_perm_k, gpu_p_perm_k, rtol=self.rtol, atol=self.atol,
                                 err_msg="GPU p_perm_k differs from CPU")
    
    def test_missing_data_handling(self):
        """Test that missing data is handled correctly."""
        # Create data with specific missing patterns
        test_data = np.random.randn(10, 3) * 2 + 1
        test_data[0, 0] = np.nan  # Missing first subject, first biomarker
        test_data[1, 1] = np.nan  # Missing second subject, second biomarker
        test_data[2, :] = np.nan  # Missing all biomarkers for third subject
        
        # Create GPU version
        gpu_sustain = TorchZScoreSustainMissingData(
            test_data, self.Z_vals[:3], self.Z_max[:3], self.biomarker_labels[:3],
            self.N_startpoints, self.N_S_max, self.N_iterations_MCMC,
            "./temp", "gpu_test", False, 42, use_gpu=True
        )
        
        # Test that computation doesn't crash with missing data
        try:
            result = gpu_sustain._calculate_likelihood_stage(
                gpu_sustain._ZScoreSustainMissingData__sustainData, self.S_test[0]
            )
            self.assertIsNotNone(result)
            self.assertFalse(np.any(np.isnan(result)), "Result contains NaN values")
        except Exception as e:
            self.fail(f"Missing data handling failed: {e}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of GPU implementation."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")
        
        # Create large dataset
        large_data = np.random.randn(1000, 10) * 2 + 1
        large_data[np.random.choice(1000 * 10, size=1000, replace=False)] = np.nan
        
        gpu_sustain = TorchZScoreSustainMissingData(
            large_data, self.Z_vals, self.Z_max, self.biomarker_labels,
            self.N_startpoints, self.N_S_max, self.N_iterations_MCMC,
            "./temp", "gpu_test", False, 42, use_gpu=True
        )
        
        # Test memory usage
        initial_memory = gpu_sustain.torch_backend.device_manager.get_memory_info()
        
        # Perform computation
        _ = gpu_sustain._calculate_likelihood_stage(
            gpu_sustain._ZScoreSustainMissingData__sustainData, self.S_test[0]
        )
        
        final_memory = gpu_sustain.torch_backend.device_manager.get_memory_info()
        
        # Check that memory usage is reasonable
        memory_increase = final_memory['allocated'] - initial_memory['allocated']
        self.assertLess(memory_increase, 1.0, "Memory usage too high (>1GB)")
    
    def test_performance_benchmark(self):
        """Test performance benchmarking functionality."""
        # Run benchmark
        benchmark_results = benchmark_gpu_vs_cpu(
            self.data, self.Z_vals, self.Z_max, self.biomarker_labels, num_iterations=3
        )
        
        # Check that benchmark completed successfully
        self.assertIn('cpu_mean_time', benchmark_results)
        self.assertIn('gpu_mean_time', benchmark_results)
        self.assertIn('speedup', benchmark_results)
        
        # If GPU is available, check for speedup
        if benchmark_results['gpu_available']:
            self.assertGreater(benchmark_results['speedup'], 1.0,
                             "GPU should be faster than CPU")
            print(f"GPU speedup: {benchmark_results['speedup']:.2f}x")
        else:
            print("GPU not available, benchmark ran on CPU only")
    
    def test_fallback_mechanism(self):
        """Test fallback to CPU when GPU runs out of memory."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for fallback testing")
        
        # Create very large dataset to trigger memory issues
        huge_data = np.random.randn(10000, 20) * 2 + 1
        
        gpu_sustain = TorchZScoreSustainMissingData(
            huge_data, self.Z_vals, self.Z_max, self.biomarker_labels,
            self.N_startpoints, self.N_S_max, self.N_iterations_MCMC,
            "./temp", "gpu_test", False, 42, use_gpu=True
        )
        
        # This should either work or gracefully fall back to CPU
        try:
            result = gpu_sustain._calculate_likelihood_stage(
                gpu_sustain._ZScoreSustainMissingData__sustainData, self.S_test[0]
            )
            self.assertIsNotNone(result)
        except Exception as e:
            # If it fails, it should be a memory error that gets handled
            self.assertIn("memory", str(e).lower())


def run_validation_tests():
    """Run all validation tests."""
    print("Running GPU validation tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTorchValidation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All validation tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
            print(failure[1])
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(error[1])
    
    return result.wasSuccessful()


def validate_numerical_accuracy(data: np.ndarray, Z_vals: np.ndarray, Z_max: np.ndarray,
                              biomarker_labels: list, tolerance: float = 1e-5) -> Dict[str, Any]:
    """
    Comprehensive numerical accuracy validation.
    
    Args:
        data: Test data
        Z_vals: Z-score thresholds
        Z_max: Maximum z-scores
        biomarker_labels: Biomarker labels
        tolerance: Numerical tolerance for comparisons
        
    Returns:
        Dictionary with validation results
    """
    print("Running comprehensive numerical accuracy validation...")
    
    # Create test sequences
    N_stages = np.sum(Z_vals > 0) + 1
    S_test = np.random.permutation(N_stages).reshape(1, N_stages)
    f_test = np.array([1.0])
    
    # Create CPU and GPU versions
    cpu_sustain = ZscoreSustainMissingData(
        data, Z_vals, Z_max, biomarker_labels,
        5, 2, 1000, "./temp", "cpu_test", False, 42
    )
    
    gpu_sustain = TorchZScoreSustainMissingData(
        data, Z_vals, Z_max, biomarker_labels,
        5, 2, 1000, "./temp", "gpu_test", False, 42, use_gpu=True
    )
    
    validation_results = {
        'gpu_available': gpu_sustain.use_gpu,
        'tests_passed': 0,
        'tests_failed': 0,
        'errors': []
    }
    
    # Test 1: Likelihood stage computation
    try:
        cpu_result = cpu_sustain._calculate_likelihood_stage(
            cpu_sustain._ZScoreSustainMissingData__sustainData, S_test[0]
        )
        gpu_result = gpu_sustain._calculate_likelihood_stage(
            gpu_sustain._ZScoreSustainMissingData__sustainData, S_test[0]
        )
        
        if np.allclose(cpu_result, gpu_result, rtol=tolerance, atol=tolerance):
            validation_results['tests_passed'] += 1
            print("✅ Likelihood stage computation: PASSED")
        else:
            validation_results['tests_failed'] += 1
            validation_results['errors'].append("Likelihood stage computation failed")
            print("❌ Likelihood stage computation: FAILED")
    except Exception as e:
        validation_results['tests_failed'] += 1
        validation_results['errors'].append(f"Likelihood stage computation error: {e}")
        print(f"❌ Likelihood stage computation: ERROR - {e}")
    
    # Test 2: Full likelihood computation
    try:
        cpu_loglike, cpu_prob_subj, cpu_prob_stage, cpu_prob_cluster, cpu_p_perm_k = \
            cpu_sustain._calculate_likelihood(
                cpu_sustain._ZScoreSustainMissingData__sustainData, S_test, f_test
            )
        
        gpu_loglike, gpu_prob_subj, gpu_prob_stage, gpu_prob_cluster, gpu_p_perm_k = \
            gpu_sustain._calculate_likelihood(
                gpu_sustain._ZScoreSustainMissingData__sustainData, S_test, f_test
            )
        
        # Check log-likelihood
        loglike_diff = abs(cpu_loglike - gpu_loglike)
        if loglike_diff < tolerance:
            validation_results['tests_passed'] += 1
            print("✅ Log-likelihood computation: PASSED")
        else:
            validation_results['tests_failed'] += 1
            validation_results['errors'].append(f"Log-likelihood computation failed (diff: {loglike_diff})")
            print(f"❌ Log-likelihood computation: FAILED (diff: {loglike_diff})")
        
        # Check probability arrays
        prob_arrays = [
            (cpu_prob_subj, gpu_prob_subj, "Subject probabilities"),
            (cpu_prob_stage, gpu_prob_stage, "Stage probabilities"),
            (cpu_prob_cluster, gpu_prob_cluster, "Cluster probabilities"),
            (cpu_p_perm_k, gpu_p_perm_k, "p_perm_k")
        ]
        
        for cpu_arr, gpu_arr, name in prob_arrays:
            if np.allclose(cpu_arr, gpu_arr, rtol=tolerance, atol=tolerance):
                validation_results['tests_passed'] += 1
                print(f"✅ {name}: PASSED")
            else:
                validation_results['tests_failed'] += 1
                validation_results['errors'].append(f"{name} failed")
                print(f"❌ {name}: FAILED")
    
    except Exception as e:
        validation_results['tests_failed'] += 1
        validation_results['errors'].append(f"Full likelihood computation error: {e}")
        print(f"❌ Full likelihood computation: ERROR - {e}")
    
    # Test 3: Performance comparison
    if gpu_sustain.use_gpu:
        try:
            benchmark_results = benchmark_gpu_vs_cpu(
                data, Z_vals, Z_max, biomarker_labels, num_iterations=5
            )
            validation_results['performance'] = benchmark_results
            print(f"✅ Performance benchmark: GPU speedup {benchmark_results['speedup']:.2f}x")
        except Exception as e:
            validation_results['errors'].append(f"Performance benchmark error: {e}")
            print(f"❌ Performance benchmark: ERROR - {e}")
    
    # Summary
    total_tests = validation_results['tests_passed'] + validation_results['tests_failed']
    success_rate = validation_results['tests_passed'] / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 Validation Summary:")
    print(f"   Tests passed: {validation_results['tests_passed']}")
    print(f"   Tests failed: {validation_results['tests_failed']}")
    print(f"   Success rate: {success_rate:.1%}")
    
    if validation_results['errors']:
        print(f"   Errors: {len(validation_results['errors'])}")
        for error in validation_results['errors']:
            print(f"     - {error}")
    
    return validation_results


if __name__ == "__main__":
    # Run validation tests
    success = run_validation_tests()
    
    if success:
        print("\n🎉 All validation tests completed successfully!")
    else:
        print("\n⚠️  Some validation tests failed. Please check the output above.")
