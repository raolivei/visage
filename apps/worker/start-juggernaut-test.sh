#!/bin/bash
# Start Juggernaut XL A/B Test Training
# Run this after SDXL training completes

set -e

PACK_ID="b873d2c5-480c-4402-9204-626a41067e83"
API_URL="http://localhost:8004"
WORKER_DIR="$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║       JUGGERNAUT XL A/B TEST - TRAINING SETUP            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if current training is still running
if pgrep -f "python.*src.main" > /dev/null; then
    echo "⚠️  Worker is still running!"
    echo "   Current SDXL training must complete first."
    echo ""
    echo "   Check status: tail -f $WORKER_DIR/worker.log"
    echo ""
    read -p "Stop current worker and continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    pkill -f "python.*src.main" || true
    sleep 2
fi

# Backup current .env and set Juggernaut
echo "📝 Configuring worker for Juggernaut XL..."
cd "$WORKER_DIR"
cp .env .env.backup 2>/dev/null || true

# Add Juggernaut preset
if ! grep -q "MODEL_PRESET" .env 2>/dev/null; then
    echo "" >> .env
    echo "# Juggernaut XL Test" >> .env
    echo "MODEL_PRESET=juggernaut" >> .env
else
    sed -i '' 's/MODEL_PRESET=.*/MODEL_PRESET=juggernaut/' .env
fi

echo "✅ MODEL_PRESET=juggernaut set in .env"
echo ""

# Start worker
echo "🚀 Starting Juggernaut worker..."
source .venv/bin/activate
nohup python3 -m src.main > worker-juggernaut.log 2>&1 &
WORKER_PID=$!
echo "   Worker PID: $WORKER_PID"
sleep 3

# Check worker started
if ! ps -p $WORKER_PID > /dev/null 2>&1; then
    echo "❌ Worker failed to start! Check worker-juggernaut.log"
    exit 1
fi

echo "✅ Worker started successfully"
echo ""

# Trigger training for the test pack
echo "📤 Triggering training job for pack $PACK_ID..."
RESPONSE=$(curl -s -X POST "$API_URL/api/packs/$PACK_ID/generate" \
    -H "Content-Type: application/json" \
    -d '{"style_presets": ["corporate", "creative", "studio", "executive", "natural"]}')

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job']['id'])" 2>/dev/null || echo "")

if [ -n "$JOB_ID" ]; then
    echo "✅ Training job created: $JOB_ID"
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                    TRAINING STARTED!                      ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Monitor progress:"
    echo "   tail -f $WORKER_DIR/worker-juggernaut.log"
    echo ""
    echo "🖼️  View results when complete:"
    echo "   SDXL:       http://localhost:3004/packs/8579c2ad-b891-4e75-ad41-c13ab028e9cd"
    echo "   Juggernaut: http://localhost:3004/packs/$PACK_ID"
else
    echo "❌ Failed to create job. Response:"
    echo "$RESPONSE"
    exit 1
fi
