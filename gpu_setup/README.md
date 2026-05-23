# GPU Server Setup

Documentation and scripts for setting up and managing multi-user GPU towers.

## Overview

This folder contains everything needed to set up 2 RTX 3090 GPU towers for ~3 remote users to run GPU-accelerated experiments.

## Quick Start

1. **Read first:** [SERVER_SETUP_GUIDE.md](SERVER_SETUP_GUIDE.md) - Complete setup walkthrough
2. **Learn tmux:** [tmux_quick_reference.md](tmux_quick_reference.md) - Essential for remote work
3. **Multi-user guide:** [MULTI_USER_SETUP.md](MULTI_USER_SETUP.md) - SuStaIn-specific setup

## Files

### Documentation
- **SERVER_SETUP_GUIDE.md** - Main setup guide (user accounts, SSH, tmux, file systems)
- **MULTI_USER_SETUP.md** - Multi-user GPU sharing for SuStaIn workloads
- **tmux_quick_reference.md** - tmux cheat sheet for beginners

### Setup Scripts
- **setup_rtx3090_towers.sh** - Install drivers, CUDA, PyTorch on each tower
- **check_tower_gpus.sh** - Diagnostic script to check GPU hardware

### GPU Management
- **claim_gpu.sh** - Claim a GPU to prevent conflicts
- **release_gpu.sh** - Release a claimed GPU
- **gpu_status.sh** - Dashboard showing GPU usage and claims

### Job Dispatch
- **dispatch_to_towers.sh** - Distribute jobs across multiple towers
- **monitor_towers.sh** - Monitor all towers remotely
- **run_model_selection_8k.py** - Template for running SuStaIn on GPU

## Typical Workflow

**One-time setup:**
```bash
# On each tower
bash gpu_setup/setup_rtx3090_towers.sh
```

**Daily usage:**
```bash
# SSH to server
ssh tower1

# Start persistent session
tmux new -s work

# Check GPU availability
gpu_status

# Claim a GPU
claim_gpu 0

# Run your job
export CUDA_VISIBLE_DEVICES=0
python your_script.py

# Detach (Ctrl+a d)
# Job keeps running even if you disconnect

# Release when done
release_gpu 0
```

## Key Concepts

- **tmux** = Sessions persist when you disconnect
- **GPU claiming** = Prevents multiple users from conflicting
- **Shared storage** = `/shared/datasets/` and `/shared/results/`
- **SSH keys** = Password-free, secure login

## Support

- Check the documentation files in this folder
- Online resources linked in SERVER_SETUP_GUIDE.md
- For SuStaIn-specific help, see notebooks/ folder
