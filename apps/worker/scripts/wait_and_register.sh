#!/bin/bash
# Wait for generation to complete and then register outputs

PACK_ID="${1:-8579c2ad-b891-4e75-ad41-c13ab028e9cd}"
JOB_ID="${2:-3c932873-3d79-468a-83cf-d5779c8e5aea}"
API_URL="http://localhost:8004"
REDIS_HOST="localhost"
REDIS_PORT="6383"

echo "📊 Monitoring generation job: $JOB_ID"
echo "📦 Pack: $PACK_ID"
echo ""

while true; do
    # Get job status from Redis
    STATUS=$(docker exec visage-redis redis-cli HGET "visage:jobs:$JOB_ID:data" status)
    PROGRESS=$(docker exec visage-redis redis-cli HGET "visage:jobs:$JOB_ID:data" progress)
    STEP=$(docker exec visage-redis redis-cli HGET "visage:jobs:$JOB_ID:data" current_step)
    
    echo "$(date '+%H:%M:%S') | Status: $STATUS | Progress: $PROGRESS% | $STEP"
    
    if [[ "$STATUS" == "completed" ]]; then
        echo ""
        echo "✅ Generation complete!"
        echo ""
        echo "🔧 Running output registration..."
        cd "$(dirname "$0")/.."
        source .venv/bin/activate
        python scripts/register_missing_outputs.py --pack-id "$PACK_ID" --job-id "$JOB_ID"
        
        echo ""
        echo "🔄 Syncing job status to database..."
        # Update job status in PostgreSQL
        PGPASSWORD=visage psql -h localhost -p 5436 -U visage -d visage -c "
            UPDATE jobs 
            SET status = 'completed', 
                progress = 100, 
                current_step = 'Generation complete',
                completed_at = NOW()
            WHERE id = '$JOB_ID';
        "
        
        echo ""
        echo "✅ All done! View results at: http://localhost:3004/packs/$PACK_ID"
        break
    elif [[ "$STATUS" == "failed" ]]; then
        echo ""
        echo "❌ Job failed!"
        ERROR=$(docker exec visage-redis redis-cli HGET "visage:jobs:$JOB_ID:data" error)
        echo "Error: $ERROR"
        break
    fi
    
    sleep 30
done
