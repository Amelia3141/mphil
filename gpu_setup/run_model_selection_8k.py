#!/usr/bin/env python3
"""
Run SuStaIn model selection for k=1 to k=5 on 8000 patients
Designed for RTX 3090 GPU execution

Usage:
    python run_model_selection_8k.py --k=3 --gpu=0 --output=results/k3
"""

import argparse
import time
import logging
import sys
from pathlib import Path
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [GPU%(gpu_id)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/model_selection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, help='Number of subtypes')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (0 or 1)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--n_iterations', type=int, default=1000000, help='MCMC iterations')
    parser.add_argument('--n_s_trained', type=int, default=10, help='Number of trained models')
    args = parser.parse_args()

    # Set GPU
    torch.cuda.set_device(args.gpu)
    logging.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    logging.info(f"VRAM: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Load data (you'll need to prepare this)
    logging.info("Loading data...")
    # TODO: Replace with your actual data loading
    # prob_nl = np.load('data/prob_nl_8000patients.npy')
    # prob_score = np.load('data/prob_score_8000patients.npy')

    # For now, placeholder to show structure
    logging.warning("Using placeholder data - replace with actual data files")

    # Import after logging setup
    from pySuStaIn.TorchOrdinalSustain import TorchOrdinalSustain

    logging.info(f"Starting model selection for k={args.k}")
    logging.info(f"  Iterations: {args.n_iterations:,}")
    logging.info(f"  Trained models: {args.n_s_trained}")

    start_time = time.time()

    # TODO: Initialize model with your data
    # model = TorchOrdinalSustain(
    #     prob_nl,
    #     prob_score,
    #     ...,
    #     use_gpu=True
    # )

    # TODO: Run MCMC
    # samples_sequence, samples_f, ll = model.run_sustain_algorithm(
    #     n_iterations=args.n_iterations,
    #     n_s=args.n_s_trained
    # )

    # Save results
    # np.save(f'{args.output}/samples_sequence.npy', samples_sequence)
    # np.save(f'{args.output}/samples_f.npy', samples_f)
    # np.save(f'{args.output}/log_likelihood.npy', ll)

    elapsed = time.time() - start_time
    logging.info(f"Completed k={args.k} in {elapsed/3600:.2f} hours")
    # logging.info(f"Final log-likelihood: {ll[-1]:.2f}")

    # GPU memory stats
    logging.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == '__main__':
    main()
