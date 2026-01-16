#!/bin/bash
# Throttled worker - uses ~50% resources
# Use this script when you need your Mac for other work

cd "$(dirname "$0")"
source .venv/bin/activate

# Limit CPU threads (uses ~4 cores instead of all)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# NOTE: Do NOT set PYTORCH_MPS_HIGH_WATERMARK_RATIO - causes errors on Apple Silicon

# Run with low priority (nice 15 = yields to other apps)
echo "Starting throttled worker (nice 15, 4 threads)..."
echo "Training will be slower but your Mac stays responsive."
nice -n 15 python -m src.main "$@"
