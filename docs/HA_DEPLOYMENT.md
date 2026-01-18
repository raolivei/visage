# Visage High Availability Deployment

This document describes how to deploy Visage with full high availability on the ElderTree k3s cluster.

> **Status:** HA infrastructure is planned but not yet fully implemented.
> See GitHub Issues for tracking:
> - [#55 - Epic: Full High Availability](https://github.com/raolivei/visage/issues/55)
> - [#51 - PostgreSQL HA](https://github.com/raolivei/visage/issues/51)
> - [#52 - Redis HA](https://github.com/raolivei/visage/issues/52)
> - [#53 - MinIO HA](https://github.com/raolivei/visage/issues/53)
> - [#54 - API/Web Multi-Replica](https://github.com/raolivei/visage/issues/54)

## Current State (January 2026)

| Component  | Replicas | Node   | HA Status |
|------------|----------|--------|-----------|
| PostgreSQL | 1        | node-1 | ❌ SPOF   |
| Redis      | 1        | node-1 | ❌ SPOF   |
| MinIO      | 1        | node-1 | ❌ SPOF   |
| API        | 1        | node-1 | ❌ SPOF   |
| Web        | 1        | node-1 | ❌ SPOF   |
| Worker     | 1        | Mac    | N/A (GPU) |

**Network:**
- VIP: 192.168.2.200 (Traefik Ingress via MetalLB L2)
- Nodes: 192.168.2.101-103

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         K8s Cluster (HA)                             │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Stateless Services                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
│  │  │ Web-1   │  │ Web-2   │  │ API-1   │  │ API-2   │         │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘         │   │
│  │       └───────┬────┴────────────┴────────────┘               │   │
│  │               │ Anti-affinity: different nodes               │   │
│  └───────────────┼──────────────────────────────────────────────┘   │
│                  │                                                   │
│  ┌───────────────┼──────────────────────────────────────────────┐   │
│  │               │       Stateful Services (HA)                  │   │
│  │  ┌────────────┴───────────┐   ┌──────────────────────────┐   │   │
│  │  │   CloudNativePG        │   │   Redis Sentinel         │   │   │
│  │  │  ┌─────┐ ┌─────┐      │   │  ┌─────┐ ┌─────┐ ┌─────┐ │   │   │
│  │  │  │ PG-1│→│ PG-2│      │   │  │ Sen │ │ Sen │ │ Sen │ │   │   │
│  │  │  │ Pri │ │ Rep │      │   │  │  1  │ │  2  │ │  3  │ │   │   │
│  │  │  └─────┘ └─────┘      │   │  └──┬──┘ └──┬──┘ └──┬──┘ │   │   │
│  │  │           │           │   │     └───────┼───────┘     │   │   │
│  │  │      ┌─────┐          │   │  ┌─────┐ ┌─────┐ ┌─────┐ │   │   │
│  │  │      │ PG-3│          │   │  │ Red │ │ Red │ │ Red │ │   │   │
│  │  │      │ Rep │          │   │  │ Mas │ │ Rep │ │ Rep │ │   │   │
│  │  │      └─────┘          │   │  └─────┘ └─────┘ └─────┘ │   │   │
│  │  └────────────────────────┘   └──────────────────────────┘   │   │
│  │                                                               │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │                    MinIO + Longhorn                     │  │   │
│  │  │         ┌─────────────────────────────────┐            │  │   │
│  │  │         │  MinIO (Longhorn 3-replica PVC) │            │  │   │
│  │  │         └─────────────────────────────────┘            │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ (Port-forward or VPN)
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU Worker (Mac)                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Worker with Checkpointing                    │ │
│  │  • LoRA Training (MPS GPU)                                     │ │
│  │  • Image Generation (SDXL)                                     │ │
│  │  • Checkpoint every 100 steps to MinIO                         │ │
│  │  • Auto-resume on restart                                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- ElderTree k3s cluster with Longhorn storage
- CloudNativePG operator
- Helm 3.x
- kubectl configured for eldertree

## Deployment Steps

### 1. Install CloudNativePG Operator

```bash
# Install CloudNativePG operator
kubectl apply -f https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/release-1.22/releases/cnpg-1.22.0.yaml

# Verify installation
kubectl get pods -n cnpg-system
```

### 2. Deploy PostgreSQL HA Cluster

```bash
# Create postgres secret first
kubectl apply -f k8s/postgres-secret.yaml -n visage

# Deploy CloudNativePG cluster
kubectl apply -f k8s/postgres-cluster.yaml

# Verify cluster
kubectl get cluster -n visage
kubectl get pods -n visage -l cnpg.io/cluster=visage-postgres-cluster
```

### 3. Deploy Redis Sentinel

```bash
# Add Bitnami repo
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Create Redis secret
kubectl apply -f k8s/redis-secret.yaml -n visage

# Install Redis with Sentinel
helm install visage-redis bitnami/redis \
  -n visage \
  -f helm/redis-values.yaml

# Verify
kubectl get pods -n visage -l app.kubernetes.io/name=redis
```

### 4. Deploy Application (HA Mode)

```bash
# Apply base resources
kubectl apply -k k8s/

# Verify deployments
kubectl get pods -n visage -o wide
```

### 5. Configure Worker for K8s

```bash
# Copy production config
cd apps/worker
cp .env.production .env

# Start worker
source .venv/bin/activate
python -m src.main
```

## Failover Testing

### PostgreSQL Failover

```bash
# Get current primary
kubectl get pods -n visage -l cnpg.io/cluster=visage-postgres-cluster

# Kill the primary
kubectl delete pod visage-postgres-cluster-1 -n visage

# Watch failover (should take ~15 seconds)
kubectl get pods -n visage -l cnpg.io/cluster=visage-postgres-cluster -w
```

### Redis Failover

```bash
# Get current master
kubectl exec -n visage visage-redis-node-1 -- redis-cli -a <password> SENTINEL masters

# Kill master
kubectl delete pod visage-redis-node-1 -n visage

# Watch sentinel elect new master
kubectl exec -n visage visage-redis-node-1 -- redis-cli -a <password> SENTINEL masters
```

### API Pod Failover

```bash
# Verify 2 replicas on different nodes
kubectl get pods -n visage -l app.kubernetes.io/name=api -o wide

# Kill one pod
kubectl delete pod <api-pod-name> -n visage

# Traffic should continue via other pod
curl https://visage.eldertree.local/health
```

## Backup and Recovery

### PostgreSQL Backups

Backups are automatically configured to MinIO:

```bash
# Check backup status
kubectl get backups -n visage

# Manual backup
kubectl apply -f - <<EOF
apiVersion: postgresql.cnpg.io/v1
kind: Backup
metadata:
  name: manual-backup-$(date +%Y%m%d)
  namespace: visage
spec:
  cluster:
    name: visage-postgres-cluster
EOF

# Restore from backup
kubectl apply -f - <<EOF
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: visage-postgres-restored
  namespace: visage
spec:
  instances: 3
  bootstrap:
    recovery:
      source: visage-postgres-cluster
      recoveryTarget:
        targetTime: "2026-01-15T10:00:00Z"
  externalClusters:
    - name: visage-postgres-cluster
      barmanObjectStore:
        destinationPath: s3://visage-backups/postgres/
        endpointURL: http://visage-minio:9000
        s3Credentials:
          accessKeyId:
            name: visage-minio-secret
            key: access-key
          secretAccessKey:
            name: visage-minio-secret
            key: secret-key
EOF
```

## Failure Tolerance Summary

| Component | Replicas | Failure Tolerance | Recovery Time |
|-----------|----------|-------------------|---------------|
| PostgreSQL | 3 (1 primary + 2 replicas) | 1 node | ~15 seconds |
| Redis | 3 masters + 3 sentinels | 1 node | ~10 seconds |
| MinIO | 1 (Longhorn 3-replica) | 1 node | Immediate |
| API | 2 | 1 pod | Immediate |
| Web | 2 | 1 pod | Immediate |
| Worker | 1 (GPU bound) | 0 | Manual restart + checkpoint resume |
| Storage | Longhorn 3-replica | 1 node | Auto rebuild |

## Monitoring

HA metrics are available in Grafana at `https://grafana.eldertree.local`:

- **PostgreSQL**: CloudNativePG dashboard
- **Redis**: Redis Sentinel dashboard
- **Longhorn**: Storage health dashboard
- **Visage**: Operations and Training dashboards

## Troubleshooting

### PostgreSQL Not Syncing

```bash
# Check replication status
kubectl exec -it visage-postgres-cluster-1 -n visage -- psql -c "SELECT * FROM pg_stat_replication;"

# Check CNPG logs
kubectl logs -n visage -l cnpg.io/cluster=visage-postgres-cluster --tail=100
```

### Redis Sentinel Issues

```bash
# Check sentinel status
kubectl exec -n visage visage-redis-node-1 -- redis-cli SENTINEL masters

# Check sentinel logs
kubectl logs -n visage -l app.kubernetes.io/component=sentinel --tail=100
```

### Worker Not Resuming from Checkpoint

```bash
# Check checkpoint file
mc ls visage-k8s/visage/checkpoints/

# Force clean start
rm -rf /tmp/visage-checkpoints
python -m src.main
```
