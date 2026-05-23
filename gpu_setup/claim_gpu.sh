#!/bin/bash
# GPU claiming system for multi-user access
# Prevents multiple users from using the same GPU

GPU_ID=$1
LOCKDIR="/tmp/gpu_locks"
LOCKFILE="$LOCKDIR/gpu_${GPU_ID}.lock"

if [ -z "$GPU_ID" ]; then
    echo "Usage: claim_gpu.sh <gpu_id>"
    echo ""
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    echo ""
    echo "Claimed GPUs:"
    for lock in $LOCKDIR/gpu_*.lock 2>/dev/null; do
        if [ -f "$lock" ]; then
            gpu=$(basename $lock | sed 's/gpu_//;s/.lock//')
            owner=$(cat $lock)
            echo "  GPU $gpu: $owner (since $(stat -c %y $lock | cut -d. -f1))"
        fi
    done
    exit 1
fi

# Create lock directory if needed
mkdir -p $LOCKDIR

# Check if GPU is already claimed
if [ -f "$LOCKFILE" ]; then
    owner=$(cat $LOCKFILE)
    echo "❌ GPU $GPU_ID is already claimed by $owner"
    echo "   Release it with: release_gpu.sh $GPU_ID"
    exit 1
fi

# Claim the GPU
echo "$USER" > $LOCKFILE
echo "✓ GPU $GPU_ID claimed by $USER"
echo ""
echo "To use this GPU, set:"
echo "  export CUDA_VISIBLE_DEVICES=$GPU_ID"
echo ""
echo "Remember to release when done:"
echo "  release_gpu.sh $GPU_ID"
