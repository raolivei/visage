#!/bin/bash
# Start Visage Worker in Production Mode
# Connects to ElderTree k8s cluster services

set -e

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKER_DIR"

echo "🚀 Visage Worker - Production Mode"
echo "==================================="
echo ""

# Check kubectl context
CONTEXT=$(kubectl config current-context)
echo "📍 Kubernetes context: $CONTEXT"

# Kill any existing port-forwards
echo ""
echo "🔄 Setting up port-forwards..."
pkill -f "kubectl port-forward.*visage" 2>/dev/null || true
pkill -f "kubectl port-forward.*pushgateway" 2>/dev/null || true
sleep 2

# Start Redis port-forward
echo "  → Redis (6379)..."
kubectl port-forward -n visage svc/visage-redis 6379:6379 > /tmp/pf-redis.log 2>&1 &
echo $! > /tmp/pf-redis.pid

# Start Pushgateway port-forward  
echo "  → Pushgateway (9091)..."
kubectl port-forward -n observability svc/observability-monitoring-stack-prometheus-pushgateway 9091:9091 > /tmp/pf-pushgateway.log 2>&1 &
echo $! > /tmp/pf-pushgateway.pid

# Wait for port-forwards to establish
sleep 3

# Verify connections
echo ""
echo "🔍 Verifying connections..."
redis-cli -p 6379 ping > /dev/null 2>&1 && echo "  ✅ Redis: Connected" || echo "  ❌ Redis: Failed"
curl -s http://localhost:9091/metrics > /dev/null 2>&1 && echo "  ✅ Pushgateway: Connected" || echo "  ❌ Pushgateway: Failed"

# Copy production config
echo ""
echo "📝 Using production configuration..."
cp .env.production .env

# Activate venv and start worker
echo ""
echo "🏃 Starting worker..."
source .venv/bin/activate

# Keep Mac awake
caffeinate -dims &
echo $! > /tmp/caffeinate.pid

# Start worker
exec python3 -m src.main
