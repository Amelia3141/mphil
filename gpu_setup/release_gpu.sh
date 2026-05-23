#!/bin/bash
# Release a claimed GPU

GPU_ID=$1
LOCKDIR="/tmp/gpu_locks"
LOCKFILE="$LOCKDIR/gpu_${GPU_ID}.lock"

if [ -z "$GPU_ID" ]; then
    echo "Usage: release_gpu.sh <gpu_id>"
    exit 1
fi

if [ ! -f "$LOCKFILE" ]; then
    echo "GPU $GPU_ID is not claimed"
    exit 0
fi

owner=$(cat $LOCKFILE)

# Only allow owner or root to release
if [ "$USER" != "$owner" ] && [ "$USER" != "root" ]; then
    echo "❌ GPU $GPU_ID is claimed by $owner (you are $USER)"
    echo "   Only $owner or root can release it"
    exit 1
fi

rm $LOCKFILE
echo "✓ GPU $GPU_ID released"
