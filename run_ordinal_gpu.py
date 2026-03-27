#!/usr/bin/env python3
"""
GPU-Accelerated OrdinalSustain Analysis Script

This standalone script runs OrdinalSustain with GPU acceleration for ordinal/categorical data.
Can be run on local machine, server, or cluster with GPU access.

Usage:
    python run_ordinal_gpu.py --config config.json
    python run_ordinal_gpu.py --quick-test  # Run quick validation test

Requirements:
    - PyTorch with CUDA
    - NumPy, SciPy, matplotlib, pandas, scikit-learn, tqdm

Authors: GPU Optimization Team
"""

import argparse
import json
import numpy as np
import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add pySuStaIn to path
sys.path.insert(0, str(Path(__file__).parent))

from pySuStaIn.TorchOrdinalSustain import TorchOrdinalSustain
from sustain_logger import (
    setup_logger, log_section, log_config, log_data_info,
    log_runtime_stats, log_gpu_info
)


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_test_data(n_subjects=8000, n_biomarkers=13, n_scores=3, seed=42):
    """
    Generate synthetic ordinal test data.

    Args:
        n_subjects: Number of subjects
        n_biomarkers: Number of biomarkers/domains
        n_scores: Number of severity levels (3 = mild, moderate, severe)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prob_nl, prob_score, score_vals, biomarker_labels)
    """
    print(f"Generating synthetic data: {n_subjects} subjects, {n_biomarkers} biomarkers, {n_scores} severity levels")
    np.random.seed(seed)

    # Probability distributions
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

    # Calculate probabilities
    prob_nl = p_nl_dist[data]

    prob_score = np.zeros((n_subjects, n_biomarkers, n_scores))
    for n in range(n_biomarkers):
        for z in range(n_scores):
            for score in range(n_scores + 1):
                prob_score[data[:, n] == score, n, z] = p_score_dist[z, score]

    score_vals = np.tile(np.arange(1, n_scores + 1), (n_biomarkers, 1))
    biomarker_labels = [f"Domain_{i+1}" for i in range(n_biomarkers)]

    print(f"   Data shape: prob_nl={prob_nl.shape}, prob_score={prob_score.shape}")
    return prob_nl, prob_score, score_vals, biomarker_labels


def load_user_data(data_config):
    """
    Load user's actual data from files.

    Args:
        data_config: Dictionary with paths to data files

    Returns:
        Tuple of (prob_nl, prob_score, score_vals, biomarker_labels)
    """
    print("Loading user data...")

    prob_nl = np.load(data_config['prob_nl_path'])
    prob_score = np.load(data_config['prob_score_path'])
    score_vals = np.load(data_config['score_vals_path'])

    # Load biomarker labels
    if 'biomarker_labels_path' in data_config:
        with open(data_config['biomarker_labels_path'], 'r') as f:
            biomarker_labels = json.load(f)
    else:
        biomarker_labels = [f"Biomarker_{i}" for i in range(prob_nl.shape[1])]

    print(f"   Loaded: {prob_nl.shape[0]} subjects, {prob_nl.shape[1]} biomarkers")
    return prob_nl, prob_score, score_vals, biomarker_labels


def run_quick_test(device_id=0):
    """
    Run quick test to validate GPU acceleration and estimate full runtime.

    Args:
        device_id: GPU device ID to use
    """
    print("="*70)
    print(" QUICK TEST - GPU Validation")
    print("="*70)

    # Generate small test dataset
    prob_nl, prob_score, score_vals, biomarker_labels = generate_test_data(
        n_subjects=1000, n_biomarkers=13, n_scores=3
    )

    # Create test instance
    print("\nInitializing TorchOrdinalSustain...")
    test_sustain = TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints=5,
        N_S_max=1,
        N_iterations_MCMC=1000,
        output_folder="./test_output",
        dataset_name="quicktest",
        use_parallel_startpoints=False,
        seed=42,
        use_gpu=True,
        device_id=device_id
    )

    # Check GPU status
    if test_sustain.use_gpu:
        print(f" GPU initialized: {test_sustain.torch_backend.device_manager.device}")
    else:
        print(" GPU not available - running on CPU")
        return False

    # Run test
    print("\n Running test...")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()
    test_sustain.run_sustain_algorithm()
    test_time = time.time() - start_time

    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Test completed in {test_time:.1f} seconds")

    # Estimate full run time
    full_iterations = 100000
    test_iterations = 1000
    estimated_time = test_time * (full_iterations / test_iterations)
    estimated_hours = estimated_time / 3600
    estimated_days = estimated_hours / 24

    print("\n" + "="*70)
    print(" PROJECTIONS FOR FULL RUN")
    print("="*70)
    print(f"Parameters: 100k MCMC iterations, 25 startpoints, 3 subtypes")
    print(f"\nEstimated runtime:")
    print(f"   {estimated_hours:.1f} hours = {estimated_days:.1f} days")

    if estimated_days < 30:
        speedup = 30 / estimated_days
        print(f"\n GPU Speedup: {speedup:.1f}x faster than CPU (30 days)")
        print(f"   Time saved: {30 - estimated_days:.1f} days")

    print("="*70)
    return True


def run_full_analysis(config):
    """
    Run full OrdinalSustain analysis with GPU acceleration.

    Args:
        config: Configuration dictionary
    """
    # Setup logger
    logger = setup_logger('ordinal_gpu', config.get('logging', {}))

    log_section(logger, "GPU-ACCELERATED ORDINAL SUSTAIN ANALYSIS")

    # Log configuration
    logger.info("Loading configuration...")
    log_config(logger, config)

    # Load data
    logger.info("Loading data...")
    sustain_params = config.get('sustain_parameters', {})

    if config.get('data', {}).get('use_test_data', False) or config.get('use_test_data', False):
        test_config = config.get('test_data', {})
        prob_nl, prob_score, score_vals, biomarker_labels = generate_test_data(
            n_subjects=test_config.get('n_subjects', 8000),
            n_biomarkers=test_config.get('n_biomarkers', 13),
            n_scores=test_config.get('n_scores', 3),
            seed=test_config.get('seed', 42)
        )
        logger.info("Using synthetic test data")
    else:
        prob_nl, prob_score, score_vals, biomarker_labels = load_user_data(config['data'])
        logger.info("Loaded user data from files")

    log_data_info(logger, prob_nl, prob_score, biomarker_labels)

    # Create output folder
    output_config = config.get('output', {})
    output_folder = output_config.get('output_folder', './sustain_output')
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    # Initialize SuStaIn
    logger.info("Initializing TorchOrdinalSustain...")
    gpu_config = config.get('gpu_settings', {})

    gpu_sustain = TorchOrdinalSustain(
        prob_nl, prob_score, score_vals, biomarker_labels,
        N_startpoints=sustain_params.get('N_startpoints', 25),
        N_S_max=sustain_params.get('N_S_max', 3),
        N_iterations_MCMC=sustain_params.get('N_iterations_MCMC', 100000),
        output_folder=output_folder,
        dataset_name=output_config.get('dataset_name', 'ordinal_analysis'),
        use_parallel_startpoints=sustain_params.get('use_parallel_startpoints', False),
        seed=sustain_params.get('seed', 42),
        use_gpu=gpu_config.get('use_gpu', True),
        device_id=gpu_config.get('device_id', 0)
    )

    # Verify GPU
    if not gpu_sustain.use_gpu:
        logger.error("GPU not available!")
        return None

    log_gpu_info(logger, gpu_sustain)

    # Run analysis
    log_section(logger, "STARTING ANALYSIS")
    start_time = time.time()

    logger.info("Running SuStaIn algorithm...")
    logger.info("This will take hours to days depending on parameters...")

    # RUN!
    samples_sequence, samples_f, ml_subtype, prob_ml_subtype, \
    ml_stage, prob_ml_stage, prob_subtype_stage = gpu_sustain.run_sustain_algorithm()

    # Calculate runtime
    end_time = time.time()

    log_section(logger, "ANALYSIS COMPLETE!")
    log_runtime_stats(logger, start_time, end_time)
    logger.info(f"Results saved to: {output_folder}")

    return {
        'samples_sequence': samples_sequence,
        'samples_f': samples_f,
        'ml_subtype': ml_subtype,
        'prob_ml_subtype': prob_ml_subtype,
        'ml_stage': ml_stage,
        'prob_ml_stage': prob_ml_stage,
        'prob_subtype_stage': prob_subtype_stage,
        'runtime': end_time - start_time,
        'output_folder': output_folder
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated OrdinalSustain Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test to validate GPU
  python run_ordinal_gpu.py --quick-test

  # Run full analysis with config file
  python run_ordinal_gpu.py --config config.json

  # Run on specific GPU device
  python run_ordinal_gpu.py --config config.json --device 1

  # Use test data (no real data needed)
  python run_ordinal_gpu.py --test-data --N-iterations 10000
        """
    )

    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test to validate GPU acceleration')
    parser.add_argument('--config', type=str,
                       help='Path to configuration JSON file')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--test-data', action='store_true',
                       help='Use synthetic test data instead of loading real data')
    parser.add_argument('--N-iterations', type=int, default=100000,
                       help='Number of MCMC iterations (default: 100000)')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        success = run_quick_test(device_id=args.device)
        sys.exit(0 if success else 1)

    # Full analysis mode
    if args.config:
        config = load_config(args.config)
    elif args.test_data:
        # Use default config with test data
        config = {
            'use_test_data': True,
            'n_subjects': 8000,
            'n_biomarkers': 13,
            'n_scores': 3,
            'N_startpoints': 25,
            'N_S_max': 3,
            'N_iterations_MCMC': args.N_iterations,
            'output_folder': './sustain_output',
            'dataset_name': 'test_analysis',
            'use_parallel_startpoints': False,
            'seed': 42,
            'device_id': args.device
        }
    else:
        print("ERROR: Must provide either --config or --test-data")
        parser.print_help()
        sys.exit(1)

    # Override device if specified
    if args.device:
        config['device_id'] = args.device

    # Run analysis
    try:
        results = run_full_analysis(config)
        if results:
            print("\n Analysis completed successfully!")
            sys.exit(0)
        else:
            print("\n Analysis failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
