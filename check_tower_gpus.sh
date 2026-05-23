#!/bin/bash
# Quick script to check GPU specs on a tower

echo "=== GPU Hardware Info ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

echo -e "\n=== Detailed GPU List ==="
nvidia-smi --list-gpus

echo -e "\n=== Current Utilization ==="
nvidia-smi

echo -e "\n=== CUDA Version ==="
nvcc --version 2>/dev/null || echo "CUDA toolkit not installed"

echo -e "\n=== Quick PyTorch Test ==="
python3 << 'PYTHON'
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
PYTHON
