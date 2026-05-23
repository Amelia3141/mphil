#!/bin/bash
# Monitor GPU usage and job progress across all towers

# Tower IPs
TOWER1="192.168.1.101"
TOWER2="192.168.1.102"
TOWER3="192.168.1.103"
USER="yourusername"

echo "=== RTX 3090 Tower Status ==="
echo ""

for TOWER in $TOWER1 $TOWER2 $TOWER3; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Tower: $TOWER"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # GPU status
    ssh $USER@$TOWER 'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits' 2>/dev/null | \
    while IFS=, read -r idx name gpu_util mem_util mem_used mem_total temp; do
        printf "  GPU %s: %3s%% util | %5s/%5s MB | %2s°C | %s\n" \
            "$idx" "$gpu_util" "$mem_used" "$mem_total" "$temp" "$name"
    done

    # Running processes
    echo ""
    echo "  Running Python processes:"
    ssh $USER@$TOWER 'pgrep -af "python.*run_model" || echo "    (none)"' 2>/dev/null

    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "View detailed log:"
echo "  ssh $USER@$TOWER1 'tail -f ~/mphil/logs/k1.log'"
echo ""
echo "Run continuously:"
echo "  watch -n 10 ./monitor_towers.sh"
