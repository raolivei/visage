#!/bin/bash
# Script to restart worker with 50% throttling
cd /Users/roliveira/WORKSPACE/raolivei/visage/apps/worker

# Kill current worker
pkill -f "python.*src.main" 2>/dev/null || true
sleep 5

# Restart with throttling
source .venv/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
nice -n 10 python3 -m src.main >> worker.log 2>&1 &

echo "$(date): Worker restarted with 50% throttling" >> /tmp/visage-throttle.log
