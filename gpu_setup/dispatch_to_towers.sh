#!/bin/bash
# Dispatch SuStaIn jobs across 2 RTX 3090 towers
# Each tower has 2 GPUs = 4 GPUs total

# Tower IPs (update these after setup)
TOWER1="192.168.1.101"
TOWER2="192.168.1.102"

# Your username on the towers
USER="yourusername"

# Commands to run (each uses 1 GPU)
declare -a JOBS=(
    # Model selection k=1 to k=4 (parallel on 4 GPUs)
    "cd ~/mphil && conda activate sustain && python run_model_selection_8k.py --k=1 --gpu=0 --output=results/k1"
    "cd ~/mphil && conda activate sustain && python run_model_selection_8k.py --k=2 --gpu=1 --output=results/k2"
    "cd ~/mphil && conda activate sustain && python run_model_selection_8k.py --k=3 --gpu=0 --output=results/k3"
    "cd ~/mphil && conda activate sustain && python run_model_selection_8k.py --k=4 --gpu=1 --output=results/k4"
)

# Map jobs to towers and GPUs
# Tower 1: GPU 0 (k=1), GPU 1 (k=2)
# Tower 2: GPU 0 (k=3), GPU 1 (k=4)
# k=5 can run afterwards on any GPU

echo "=== Dispatching jobs to 2 RTX 3090 towers (4 GPUs) ==="
echo ""

# Tower 1 - k=1 and k=2
echo "Tower 1 ($TOWER1):"
echo "  GPU 0: k=1"
ssh $USER@$TOWER1 "nohup ${JOBS[0]} > logs/k1.log 2>&1 &" && echo "    ✓ Started"
echo "  GPU 1: k=2"
ssh $USER@$TOWER1 "nohup ${JOBS[1]} > logs/k2.log 2>&1 &" && echo "    ✓ Started"

# Tower 2 - k=3 and k=4
echo "Tower 2 ($TOWER2):"
echo "  GPU 0: k=3"
ssh $USER@$TOWER2 "nohup ${JOBS[2]} > logs/k3.log 2>&1 &" && echo "    ✓ Started"
echo "  GPU 1: k=4"
ssh $USER@$TOWER2 "nohup ${JOBS[3]} > logs/k4.log 2>&1 &" && echo "    ✓ Started"

echo ""
echo "=== All 4 parallel jobs dispatched! ==="
echo ""
echo "To run k=5 after k=1-4 complete:"
echo "  ssh $USER@$TOWER1 'cd ~/mphil && conda activate sustain && python run_model_selection_8k.py --k=5 --gpu=0 --output=results/k5 > logs/k5.log 2>&1 &'"
echo ""
echo "Monitor progress:"
echo "  ./monitor_towers.sh"
echo ""
echo "View specific log:"
echo "  ssh $USER@$TOWER1 'tail -f ~/mphil/logs/k1.log'"
