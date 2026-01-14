# Visage Helm Charts

Helm configuration for Visage infrastructure on ElderTree.

## MinIO

MinIO provides S3-compatible object storage for user photos and generated outputs.

### Prerequisites

1. Create MinIO credentials in Vault:
   ```bash
   vault kv put secret/visage/minio \
     access-key="visage-minio-user" \
     secret-key="<strong-random-password>"
   ```

2. Ensure the ExternalSecret is deployed:
   ```bash
   kubectl apply -f k8s/externalsecrets.yaml
   kubectl get secret visage-minio-secret -n visage
   ```

### Installation

```bash
# Add Bitnami repo
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install MinIO
helm upgrade --install visage-minio bitnami/minio \
  --namespace visage \
  --create-namespace \
  -f helm/minio-values.yaml

# Check deployment
kubectl get pods -n visage -l app.kubernetes.io/name=minio
```

### Accessing MinIO Console

Local access: https://minio.eldertree.local

Or via port-forward:
```bash
kubectl port-forward -n visage svc/visage-minio 9001:9001
# Open http://localhost:9001
```

### Storage Sizing

Default: 20Gi

For production, consider:
- Photos: ~5MB each × 20 photos × N packs
- Outputs: ~2MB each × 50 outputs × N packs
- LoRA weights: ~100MB per pack

Estimate: 500MB per pack → 20Gi supports ~40 packs

### Backup

MinIO data is stored on Longhorn, which provides:
- Replication across nodes
- Snapshot capability

For additional backup:
```bash
# Use mc (MinIO Client) to sync to external storage
mc alias set visage https://minio.eldertree.local <access-key> <secret-key>
mc mirror visage/visage /backup/visage
```
