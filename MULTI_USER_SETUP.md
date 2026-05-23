# Multi-User GPU Tower Setup Guide

Setup for 3 users sharing 2 GPU towers with ~4 RTX 3090 GPUs total.

## Initial Setup (Run Once Per Tower)

### 1. Create User Accounts

```bash
# On each tower, create accounts for all users
sudo adduser alice
sudo adduser bob  
sudo adduser charlie

# Set strong passwords
sudo passwd alice
sudo passwd bob
sudo passwd charlie
```

### 2. Install GPU Management Scripts

```bash
# Copy scripts to shared location
sudo mkdir -p /usr/local/bin/gpu-tools
sudo cp claim_gpu.sh release_gpu.sh gpu_status.sh /usr/local/bin/gpu-tools/
sudo chmod +x /usr/local/bin/gpu-tools/*.sh

# Create symlinks for easy access
sudo ln -s /usr/local/bin/gpu-tools/claim_gpu.sh /usr/local/bin/claim_gpu
sudo ln -s /usr/local/bin/gpu-tools/release_gpu.sh /usr/local/bin/release_gpu
sudo ln -s /usr/local/bin/gpu-tools/gpu_status.sh /usr/local/bin/gpu_status

# Create lock directory
sudo mkdir -p /tmp/gpu_locks
sudo chmod 777 /tmp/gpu_locks
```

### 3. Set Up Each User's Environment

Run for each user:

```bash
# Switch to user
sudo su - alice

# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create SuStaIn environment
conda create -n sustain python=3.10 -y
conda activate sustain
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib pandas scikit-learn tqdm

# Clone repository
cd ~
git clone https://github.com/Amelia3141/mphil.git
cd mphil
git checkout claude/convert-to-jupyter-01GY8iZvAixjYs4t3VyLsWHf

# Create personal results directory
mkdir -p ~/results ~/data

# Exit back to admin
exit
```

Repeat for bob and charlie.

## Daily Usage

### User Workflow

**1. SSH to a tower:**
```bash
ssh alice@192.168.1.101
```

**2. Check GPU availability:**
```bash
gpu_status
```

**3. Claim an available GPU:**
```bash
claim_gpu 0   # Claims GPU 0
```

**4. Run your job:**
```bash
cd ~/mphil
conda activate sustain

# Use the GPU you claimed
export CUDA_VISIBLE_DEVICES=0
python run_model_selection_8k.py --k=3 --gpu=0 --output=~/results/k3
```

**5. Release GPU when done:**
```bash
release_gpu 0
```

## GPU Assignment Protocol

**Recommended default assignments:**

| User    | Tower | GPU | Notes |
|---------|-------|-----|-------|
| Alice   | Tower 1 | GPU 0 | Primary |
| Bob     | Tower 1 | GPU 1 | Primary |
| Charlie | Tower 2 | GPU 0 | Primary |
| (any)   | Tower 2 | GPU 1 | Shared/overflow |

**Rules:**
- Always check `gpu_status` before claiming
- Claim GPU before starting work
- Release GPU when job completes or if stepping away
- If you need to leave a long job running, note it in shared communication (Slack/email)
- Maximum claim time: 48 hours (then auto-release or coordinate with others)

## Monitoring

**Watch live GPU usage:**
```bash
watch -n 5 gpu_status
```

**From your laptop (remote monitoring):**
```bash
# Check Tower 1
ssh alice@192.168.1.101 gpu_status

# Check Tower 2
ssh alice@192.168.1.102 gpu_status
```

## File Organization

Each user's home directory structure:

```
/home/alice/
├── mphil/              # Code repository (same for all)
├── data/               # Your datasets
│   ├── prob_nl.npy
│   └── prob_score.npy
├── results/            # Your results (isolated)
│   ├── k1/
│   ├── k2/
│   └── ...
└── logs/               # Your logs
    ├── k1.log
    └── ...
```

**Never write to other users' directories!**

## Troubleshooting

**Someone left a GPU claimed but not in use:**
```bash
# Check who claimed it
gpu_status

# Contact them to release, or admin can force-release:
sudo release_gpu 0
```

**Out of memory error:**
```bash
# Check if someone else is using your GPU
gpu_status

# Make sure CUDA_VISIBLE_DEVICES matches your claimed GPU
echo $CUDA_VISIBLE_DEVICES
```

**GPU showing 100% usage but no process listed:**
```bash
# Zombie process - check with:
nvidia-smi

# Kill process:
kill <PID>
```

## Advanced: Simple Job Queue (Optional)

For fair automatic scheduling:

```bash
# Install SLURM (proper job scheduler)
sudo apt install slurm-wlm

# Or use simpler alternative like 'at' command
# Users submit jobs that wait for GPU availability
```

This is optional - manual claiming works fine for 3 users.

## Security Notes

- Each user can only see their own files (Linux permissions)
- Lock files prevent GPU conflicts
- Monitor with `gpu_status` for transparency
- Set up shared Slack/Discord channel for coordination
- Consider setting up login notifications:
  ```bash
  # Add to /etc/profile.d/login-notify.sh
  echo "$USER logged into $(hostname) at $(date)" >> /var/log/user-logins.log
  ```
