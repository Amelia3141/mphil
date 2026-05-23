#!/bin/bash
# Show GPU status across both towers for all users

LOCKDIR="/tmp/gpu_locks"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           Multi-User GPU Status Dashboard                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Show GPU hardware status
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
while IFS=, read -r idx name util mem_used mem_total temp; do
    # Check if GPU is claimed
    lockfile="$LOCKDIR/gpu_${idx}.lock"
    if [ -f "$lockfile" ]; then
        claimed="$(cat $lockfile)"
        claim_time=$(stat -c %y $lockfile | cut -d. -f1)
    else
        claimed="(available)"
        claim_time=""
    fi

    printf "GPU %s: %s\n" "$idx" "$name"
    printf "  Utilization: %3s%%  |  Memory: %s / %s MB  |  Temp: %s°C\n" \
        "$util" "$mem_used" "$mem_total" "$temp"
    printf "  Claimed by: %s\n" "$claimed"
    if [ -n "$claim_time" ]; then
        printf "  Since: %s\n" "$claim_time"
    fi
    echo ""
done

# Show running Python processes by user
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running SuStaIn Jobs:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use nvidia-smi to show GPU processes
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | \
while IFS=, read -r pid mem; do
    # Get process details
    ps_info=$(ps -p $pid -o user,cmd --no-headers 2>/dev/null)
    if [ -n "$ps_info" ]; then
        echo "  PID $pid: $ps_info (GPU mem: $mem MB)"
    fi
done

if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "  (no active GPU jobs)"
fi

echo ""
echo "Commands:"
echo "  claim_gpu.sh <id>   - Claim a GPU"
echo "  release_gpu.sh <id> - Release your GPU"
echo "  watch -n 5 gpu_status.sh - Live monitoring"
