#!/bin/bash
#
# Visage Pack Migration Script
# Migrates pack data from local docker-compose to k8s (ElderTree)
#
# Usage: ./migrate-pack-to-k8s.sh <pack_id>
#
# Prerequisites:
# - mc (MinIO client) installed and configured
# - kubectl configured for eldertree cluster
# - psql client installed
# - Port-forward to k8s postgres running

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PACK_ID="${1:-}"
LOCAL_PG_HOST="localhost"
LOCAL_PG_PORT="5436"
LOCAL_PG_USER="visage"
LOCAL_PG_PASSWORD="visage"
LOCAL_PG_DB="visage"

K8S_PG_HOST="localhost"
K8S_PG_PORT="5432"  # Via port-forward
K8S_PG_USER="visage"
K8S_PG_DB="visage"

LOCAL_MINIO_ALIAS="visage-local"
K8S_MINIO_ALIAS="visage-k8s"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check mc
    if ! command -v mc &> /dev/null; then
        log_error "MinIO client (mc) not found. Install with: brew install minio/stable/mc"
        exit 1
    fi
    
    # Check psql
    if ! command -v psql &> /dev/null; then
        log_error "psql not found. Install with: brew install postgresql"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found"
        exit 1
    fi
    
    log_info "Prerequisites OK"
}

setup_minio_aliases() {
    log_info "Setting up MinIO aliases..."
    
    # Local MinIO
    mc alias set $LOCAL_MINIO_ALIAS http://localhost:9000 minioadmin minioadmin 2>/dev/null || true
    
    # K8s MinIO (via ingress or port-forward)
    # Assumes you have minio.eldertree.local accessible
    mc alias set $K8S_MINIO_ALIAS https://minio.eldertree.local minioadmin minioadmin --insecure 2>/dev/null || true
    
    log_info "MinIO aliases configured"
}

setup_port_forwards() {
    log_info "Setting up port-forwards to k8s..."
    
    # Kill existing port-forwards
    pkill -f "port-forward.*visage-postgres" 2>/dev/null || true
    sleep 2
    
    # Start port-forward for postgres
    kubectl port-forward -n visage svc/visage-postgres 5432:5432 &
    PF_PID=$!
    
    sleep 3
    
    # Verify connection
    if ! nc -z localhost 5432; then
        log_error "Port-forward to postgres failed"
        exit 1
    fi
    
    log_info "Port-forward started (PID: $PF_PID)"
    echo $PF_PID > /tmp/visage-migration-pf.pid
}

export_pack_from_local() {
    log_info "Exporting pack $PACK_ID from local database..."
    
    EXPORT_DIR="/tmp/visage-migration-$PACK_ID"
    mkdir -p "$EXPORT_DIR"
    
    # Export pack record
    PGPASSWORD=$LOCAL_PG_PASSWORD psql -h $LOCAL_PG_HOST -p $LOCAL_PG_PORT -U $LOCAL_PG_USER -d $LOCAL_PG_DB \
        -c "COPY (SELECT * FROM packs WHERE id = '$PACK_ID') TO STDOUT WITH CSV HEADER" \
        > "$EXPORT_DIR/pack.csv"
    
    # Export photos
    PGPASSWORD=$LOCAL_PG_PASSWORD psql -h $LOCAL_PG_HOST -p $LOCAL_PG_PORT -U $LOCAL_PG_USER -d $LOCAL_PG_DB \
        -c "COPY (SELECT * FROM photos WHERE pack_id = '$PACK_ID') TO STDOUT WITH CSV HEADER" \
        > "$EXPORT_DIR/photos.csv"
    
    # Export jobs
    PGPASSWORD=$LOCAL_PG_PASSWORD psql -h $LOCAL_PG_HOST -p $LOCAL_PG_PORT -U $LOCAL_PG_USER -d $LOCAL_PG_DB \
        -c "COPY (SELECT * FROM jobs WHERE pack_id = '$PACK_ID') TO STDOUT WITH CSV HEADER" \
        > "$EXPORT_DIR/jobs.csv"
    
    # Export outputs
    PGPASSWORD=$LOCAL_PG_PASSWORD psql -h $LOCAL_PG_HOST -p $LOCAL_PG_PORT -U $LOCAL_PG_USER -d $LOCAL_PG_DB \
        -c "COPY (SELECT * FROM outputs WHERE pack_id = '$PACK_ID') TO STDOUT WITH CSV HEADER" \
        > "$EXPORT_DIR/outputs.csv"
    
    log_info "Database export complete: $EXPORT_DIR"
}

mirror_minio_data() {
    log_info "Mirroring MinIO data for pack $PACK_ID..."
    
    # Mirror pack data from local to k8s
    mc mirror --overwrite \
        "$LOCAL_MINIO_ALIAS/visage/packs/$PACK_ID/" \
        "$K8S_MINIO_ALIAS/visage/packs/$PACK_ID/" \
        || {
            log_warn "Mirror failed, trying with --insecure flag"
            mc mirror --overwrite --insecure \
                "$LOCAL_MINIO_ALIAS/visage/packs/$PACK_ID/" \
                "$K8S_MINIO_ALIAS/visage/packs/$PACK_ID/"
        }
    
    log_info "MinIO data mirrored successfully"
}

import_pack_to_k8s() {
    log_info "Importing pack $PACK_ID to k8s database..."
    
    EXPORT_DIR="/tmp/visage-migration-$PACK_ID"
    
    # Get k8s postgres password from secret
    K8S_PG_PASSWORD=$(kubectl get secret -n visage visage-postgres-secret -o jsonpath='{.data.password}' | base64 -d)
    
    # Import pack record
    PGPASSWORD=$K8S_PG_PASSWORD psql -h $K8S_PG_HOST -p $K8S_PG_PORT -U $K8S_PG_USER -d $K8S_PG_DB \
        -c "BEGIN;
            DELETE FROM packs WHERE id = '$PACK_ID';
            COPY packs FROM STDIN WITH CSV HEADER;
            COMMIT;" < "$EXPORT_DIR/pack.csv"
    
    # Import photos
    PGPASSWORD=$K8S_PG_PASSWORD psql -h $K8S_PG_HOST -p $K8S_PG_PORT -U $K8S_PG_USER -d $K8S_PG_DB \
        -c "BEGIN;
            DELETE FROM photos WHERE pack_id = '$PACK_ID';
            COPY photos FROM STDIN WITH CSV HEADER;
            COMMIT;" < "$EXPORT_DIR/photos.csv"
    
    # Import jobs
    PGPASSWORD=$K8S_PG_PASSWORD psql -h $K8S_PG_HOST -p $K8S_PG_PORT -U $K8S_PG_USER -d $K8S_PG_DB \
        -c "BEGIN;
            DELETE FROM jobs WHERE pack_id = '$PACK_ID';
            COPY jobs FROM STDIN WITH CSV HEADER;
            COMMIT;" < "$EXPORT_DIR/jobs.csv"
    
    # Import outputs
    PGPASSWORD=$K8S_PG_PASSWORD psql -h $K8S_PG_HOST -p $K8S_PG_PORT -U $K8S_PG_USER -d $K8S_PG_DB \
        -c "BEGIN;
            DELETE FROM outputs WHERE pack_id = '$PACK_ID';
            COPY outputs FROM STDIN WITH CSV HEADER;
            COMMIT;" < "$EXPORT_DIR/outputs.csv"
    
    log_info "Database import complete"
}

verify_migration() {
    log_info "Verifying migration..."
    
    K8S_PG_PASSWORD=$(kubectl get secret -n visage visage-postgres-secret -o jsonpath='{.data.password}' | base64 -d)
    
    # Verify pack exists in k8s
    PACK_COUNT=$(PGPASSWORD=$K8S_PG_PASSWORD psql -h $K8S_PG_HOST -p $K8S_PG_PORT -U $K8S_PG_USER -d $K8S_PG_DB \
        -t -c "SELECT COUNT(*) FROM packs WHERE id = '$PACK_ID'")
    
    if [ "$PACK_COUNT" -eq "1" ]; then
        log_info "Pack verified in k8s database"
    else
        log_error "Pack not found in k8s database!"
        exit 1
    fi
    
    # Verify MinIO files
    LOCAL_FILES=$(mc ls --recursive "$LOCAL_MINIO_ALIAS/visage/packs/$PACK_ID/" 2>/dev/null | wc -l)
    K8S_FILES=$(mc ls --recursive "$K8S_MINIO_ALIAS/visage/packs/$PACK_ID/" --insecure 2>/dev/null | wc -l)
    
    log_info "Local MinIO files: $LOCAL_FILES"
    log_info "K8s MinIO files: $K8S_FILES"
    
    if [ "$LOCAL_FILES" -eq "$K8S_FILES" ]; then
        log_info "MinIO file count matches"
    else
        log_warn "MinIO file count mismatch - verify manually"
    fi
}

cleanup() {
    log_info "Cleaning up..."
    
    # Kill port-forward
    if [ -f /tmp/visage-migration-pf.pid ]; then
        kill $(cat /tmp/visage-migration-pf.pid) 2>/dev/null || true
        rm /tmp/visage-migration-pf.pid
    fi
    
    log_info "Cleanup complete"
}

# Main
main() {
    if [ -z "$PACK_ID" ]; then
        echo "Usage: $0 <pack_id>"
        echo ""
        echo "Example: $0 8579c2ad-b891-4e75-ad41-c13ab028e9cd"
        exit 1
    fi
    
    log_info "Starting migration for pack: $PACK_ID"
    
    trap cleanup EXIT
    
    check_prerequisites
    setup_minio_aliases
    setup_port_forwards
    export_pack_from_local
    mirror_minio_data
    import_pack_to_k8s
    verify_migration
    
    log_info "Migration complete!"
    echo ""
    echo "Pack $PACK_ID has been migrated to k8s."
    echo "Access it at: https://visage.eldertree.local/packs/$PACK_ID"
}

main "$@"
