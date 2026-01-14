# Visage Kubernetes Manifests

Kubernetes manifests for deploying Visage on ElderTree.

## Prerequisites

1. **Vault Secrets**: Create the following secrets in Vault:
   - `secret/visage/postgres` with `password`
   - `secret/visage/database` with `url` (full PostgreSQL connection string)
   - `secret/visage/minio` with `access-key` and `secret-key`
   - `secret/visage/app` with `secret-key`

2. **GHCR Secret**: Create a secret for pulling images from GitHub Container Registry:
   ```bash
   kubectl create secret docker-registry ghcr-secret \
     --namespace=visage \
     --docker-server=ghcr.io \
     --docker-username=<github-username> \
     --docker-password=<github-token>
   ```

3. **MinIO**: Deploy MinIO using the Helm chart (see `../helm/minio-values.yaml`)

## Deployment

### Using Kustomize

```bash
# Preview manifests
kubectl kustomize k8s/

# Apply manifests
kubectl apply -k k8s/

# Check deployment status
kubectl get all -n visage
```

### Manual Deployment

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create external secrets
kubectl apply -f k8s/externalsecrets.yaml

# Wait for secrets to sync
kubectl get externalsecrets -n visage

# Deploy infrastructure
kubectl apply -f k8s/postgres-pvc.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml

# Deploy application
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

## Vault Secret Setup

```bash
# Port forward to Vault
kubectl port-forward -n vault svc/vault 8200:8200

# Set Vault address
export VAULT_ADDR=https://localhost:8200

# Login to Vault
vault login

# Create secrets
vault kv put secret/visage/postgres password="<strong-password>"

vault kv put secret/visage/database \
  url="postgresql+asyncpg://visage:<password>@visage-postgres:5432/visage"

vault kv put secret/visage/minio \
  access-key="minioadmin" \
  secret-key="<strong-password>"

vault kv put secret/visage/app \
  secret-key="<random-secret-key>"
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| visage-web | 3000 | Next.js frontend |
| visage-api | 8000 | FastAPI backend |
| visage-postgres | 5432 | PostgreSQL database |
| visage-redis | 6379 | Redis job queue |
| visage-minio | 9000 | MinIO object storage |

## Accessing Locally

Add to `/etc/hosts` or configure Pi-hole:
```
192.168.2.201  visage.eldertree.local
```

Then access: https://visage.eldertree.local

## GPU Worker

The GPU worker runs **outside** the cluster on a Mac with Apple Silicon:

```bash
cd apps/worker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
export API_URL=https://visage.eldertree.local
export REDIS_URL=redis://visage-redis.eldertree.local:6379
export MINIO_ENDPOINT=minio.eldertree.local:9000
# ... set other vars from .env.example

python -m src.main
```

## Troubleshooting

### Check pod status
```bash
kubectl get pods -n visage
kubectl describe pod <pod-name> -n visage
kubectl logs <pod-name> -n visage
```

### Check secrets
```bash
kubectl get externalsecrets -n visage
kubectl get secrets -n visage
```

### Check ingress
```bash
kubectl get ingress -n visage
kubectl describe ingress visage-ingress -n visage
```
