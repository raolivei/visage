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

### Rollout after new API image

After CI pushes a new `ghcr.io/raolivei/visage-api:latest`, switch kubectl to the Eldertree cluster and restart the API so pods pull the new image:

```bash
kubectl rollout restart deployment/visage-api -n visage
kubectl rollout status deployment/visage-api -n visage
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

| Service         | Port | Description          |
| --------------- | ---- | -------------------- |
| visage-web      | 3000 | Next.js frontend     |
| visage-api      | 8000 | FastAPI backend      |
| visage-postgres | 5432 | PostgreSQL database  |
| visage-redis    | 6379 | Redis job queue      |
| visage-minio    | 9000 | MinIO object storage |

## Accessing visage

Use the same path as for all Eldertree services (vault, grafana, swimto, etc.):

1. **Tailscale (primary)** — Follow [pi-fleet TAILSCALE.md → Access all services from your Mac](https://github.com/raolivei/pi-fleet/blob/main/docs/TAILSCALE.md#access-all-services-from-your-mac): enable Tailscale, Accept Routes, add the full `/etc/hosts` block from [eldertree-local-hosts-block.txt](https://github.com/raolivei/pi-fleet/blob/main/docs/eldertree-local-hosts-block.txt) (replacing `TRAEFIK_LB_IP` with the Traefik EXTERNAL-IP).
2. Open **https://visage.eldertree.local** (accept self-signed cert if prompted).

**Fallback when Tailscale is not available** (e.g. Tailscale down or not set up):

1. Add to `/etc/hosts`: `127.0.0.1  visage.eldertree.local` (tunnel mode only; for direct access use the same Traefik IP as other services, e.g. `192.168.2.200  visage.eldertree.local`).
2. Run `./scripts/visage-tunnel.sh` (kubectl port-forward to Traefik; keeps running).
3. Open **https://visage.eldertree.local:8443** (accept self-signed cert).

Requires `kubectl` and a kubeconfig that can reach the Eldertree API server.

If another device is on the cluster LAN, it can use the Traefik LB IP in `/etc/hosts` or Pi-hole for `*.eldertree.local`.

## Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ElderTree Cluster                         │
│                                                               │
│  Nodes:                                                       │
│  ├── node-1: 192.168.2.101 (wlan0)                          │
│  ├── node-2: 192.168.2.102 (wlan0)                          │
│  └── node-3: 192.168.2.103 (wlan0)                          │
│                                                               │
│  VIPs (MetalLB L2):                                          │
│  ├── Traefik: 192.168.2.200 (HTTPS ingress)                 │
│  └── Pi-hole: 192.168.2.201 (DNS)                           │
│                                                               │
│  k3s Networks:                                               │
│  ├── Pods:     10.42.0.0/16                                 │
│  ├── Services: 10.43.0.0/16                                 │
│  └── Internal: 10.0.0.0/24 (eth0)                           │
└─────────────────────────────────────────────────────────────┘
```

## GPU Worker

The GPU worker runs **outside** the cluster on a Mac with Apple Silicon.

### Quick Start (Production)

```bash
cd apps/worker
source .venv/bin/activate

# Use production config (connects to k3s)
cp .env.production .env

# Start Redis port-forward (separate terminal)
kubectl port-forward -n visage svc/visage-redis 6379:6379

# Run worker
python -m src.main
```

### DNS Configuration

DNS is handled by Pi-hole (192.168.2.201) which resolves all `*.eldertree.local` domains.

If not using Pi-hole, add to `/etc/hosts`:

```
# ElderTree k8s Cluster VIP
192.168.2.200  visage.eldertree.local
192.168.2.200  minio.eldertree.local
192.168.2.200  pushgateway.eldertree.local
192.168.2.200  grafana.eldertree.local
192.168.2.200  prometheus.eldertree.local
```

### Monitoring

The worker pushes metrics to the central Prometheus Pushgateway:

- **Grafana**: https://grafana.eldertree.local
- **Dashboards**: Visage Operations, Visage Training

Metrics include:

- Training progress, loss, step duration, ETA
- Job status and duration
- Image generation and quality scores
- Queue depth

### Environment Options

| Variable          | Local Dev                | Production                              |
| ----------------- | ------------------------ | --------------------------------------- |
| `API_URL`         | `http://localhost:8004`  | `https://visage.eldertree.local`        |
| `REDIS_URL`       | `redis://localhost:6383` | `redis://localhost:6379` (port-forward) |
| `MINIO_ENDPOINT`  | `localhost:9000`         | `minio.eldertree.local`                 |
| `MINIO_SECURE`    | `false`                  | `true`                                  |
| `PUSHGATEWAY_URL` | (empty)                  | `https://pushgateway.eldertree.local`   |

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
