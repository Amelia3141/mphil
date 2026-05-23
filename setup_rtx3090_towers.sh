#!/bin/bash
# Setup script for RTX 3090 towers
# Run this on each tower after basic network + SSH is configured

set -e

echo "=== Installing NVIDIA Drivers ==="
sudo apt update
sudo apt install -y nvidia-driver-535
echo "Driver installed. Reboot required - run 'sudo reboot' then re-run this script"
read -p "Press enter if already rebooted, Ctrl+C if need to reboot first..."

echo "=== Verifying GPUs ==="
nvidia-smi
if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi failed. Reboot and check drivers."
    exit 1
fi

echo "=== Installing CUDA 11.8 ==="
if [ ! -f cuda_11.8.0_520.61.05_linux.run ]; then
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
fi
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit

echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "=== Installing Miniconda ==="
if [ ! -d ~/miniconda3 ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc
fi

echo "=== Creating SuStaIn Environment ==="
conda create -n sustain python=3.10 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sustain

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib pandas scikit-learn tqdm

echo "=== Cloning Repository ==="
cd ~
if [ ! -d mphil ]; then
    git clone https://github.com/Amelia3141/mphil.git
fi
cd mphil
git checkout claude/convert-to-jupyter-01GY8iZvAixjYs4t3VyLsWHf
git pull

echo "=== Testing GPU Access ==="
python << 'PYEOF'
import torch
print(f"\n{'='*50}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
print(f"{'='*50}\n")
PYEOF

echo "=== Creating Directories ==="
mkdir -p ~/mphil/results ~/mphil/logs

echo "✓ Setup complete! This tower is ready to run SuStaIn."
echo "  - 2× RTX 3090 GPUs detected"
echo "  - PyTorch with CUDA installed"
echo "  - Repository cloned"
echo ""
echo "To activate environment: conda activate sustain"
echo "To run a job: cd ~/mphil && python run_experiment.py"
