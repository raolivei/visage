# Visage

[![CI](https://github.com/raolivei/visage/actions/workflows/ci.yml/badge.svg)](https://github.com/raolivei/visage/actions/workflows/ci.yml)
[![Security Scanning](https://github.com/raolivei/visage/actions/workflows/security-scan.yml/badge.svg)](https://github.com/raolivei/visage/actions/workflows/security-scan.yml)
[![Build API](https://github.com/raolivei/visage/actions/workflows/build-api.yml/badge.svg)](https://github.com/raolivei/visage/actions/workflows/build-api.yml)
[![Build Web](https://github.com/raolivei/visage/actions/workflows/build-web.yml/badge.svg)](https://github.com/raolivei/visage/actions/workflows/build-web.yml)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](VERSION)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Self-hosted AI headshot generator that produces professional, LinkedIn-quality headshots using SDXL and per-user LoRA training.

## Overview

Visage is designed to match commercial AI headshot tools (C$70–C$150) without cloud costs by:

- Using existing SDXL diffusion models
- Training per-user LoRA for personalization
- Running orchestration on the ElderTree Raspberry Pi k3s cluster
- Offloading GPU compute to a local Mac with Apple Silicon
- Delivering only high-quality, curated results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              ElderTree Cluster (ARM64)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Web UI  │  │   API    │  │  Redis   │  │  MinIO   │   │
│  │ (Next.js)│  │(FastAPI) │  │ (Queue)  │  │(Storage) │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       └─────────────┴──────┬──────┴─────────────┘          │
│                            │                               │
└────────────────────────────┼───────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Mac GPU Worker │
                    │  (Apple Silicon)│
                    │  SDXL + LoRA    │
                    └─────────────────┘
```

## Features

- **Photo Upload**: Drag-and-drop upload with validation
- **Style Presets**: Corporate, Studio, Natural Light, Executive, Creative
- **LoRA Training**: Per-user model fine-tuning
- **Auto-Filtering**: Only delivers the best results
- **Gallery**: Browse and download generated headshots

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Next.js 14 (App Router) |
| Backend | FastAPI (Python) |
| Database | PostgreSQL |
| Job Queue | Redis |
| Object Storage | MinIO (S3-compatible) |
| AI Stack | SDXL + LoRA (PyTorch MPS) |
| Orchestration | Kubernetes (k3s on ElderTree) |

## Project Structure

```
visage/
├── .github/
│   ├── workflows/          # CI/CD workflows
│   │   ├── build-api.yml   # API Docker build
│   │   ├── build-web.yml   # Web Docker build
│   │   ├── ci.yml          # Lint, test, build verification
│   │   └── security-scan.yml # Security scanning
│   ├── ISSUE_TEMPLATE/     # Issue templates
│   ├── CODEOWNERS          # Code ownership
│   ├── dependabot.yml      # Dependency updates
│   └── SECURITY.md         # Security policy
├── github/
│   ├── branch-protection-config.json
│   ├── setup-branch-protection.sh
│   └── setup-labels.sh
├── apps/
│   ├── web/                # Next.js frontend
│   ├── api/                # FastAPI backend
│   └── worker/             # GPU worker (runs on Mac)
├── packages/
│   └── shared/             # Shared types, prompt templates
├── k8s/                    # Kubernetes manifests
├── helm/                   # Helm values (MinIO)
├── docker-compose.yml      # Local development
├── CHANGELOG.md            # Version history
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # MIT License
└── VERSION                 # Semantic version
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 20+
- Python 3.11+
- (For GPU worker) Mac with Apple Silicon

### Local Development

```bash
# Start infrastructure (Postgres, Redis, MinIO)
docker-compose up -d postgres redis minio

# Start API
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8004

# Start Web (in another terminal)
cd apps/web
npm install
npm run dev

# Start Worker (on Mac with GPU)
cd apps/worker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

### With Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

## Environment Variables

### API

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://visage:visage@localhost:5436/visage` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6383` |
| `MINIO_ENDPOINT` | MinIO endpoint | `localhost:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | `minioadmin` |
| `MINIO_SECRET_KEY` | MinIO secret key | `minioadmin` |
| `MINIO_BUCKET` | S3 bucket name | `visage` |

### Worker

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | Visage API endpoint | `http://localhost:8004` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6383` |
| `MINIO_ENDPOINT` | MinIO endpoint | `localhost:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | `minioadmin` |
| `MINIO_SECRET_KEY` | MinIO secret key | `minioadmin` |
| `DEVICE` | PyTorch device | `mps` (Apple Silicon) |

## Deployment

### ElderTree (Kubernetes)

```bash
# Apply namespace and secrets
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/externalsecrets.yaml

# Deploy MinIO
helm upgrade --install visage-minio bitnami/minio \
  -n visage -f helm/minio-values.yaml

# Deploy services
kubectl apply -f k8s/
```

### GPU Worker (Mac)

The GPU worker runs outside the cluster on a Mac with Apple Silicon:

```bash
cd apps/worker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure to connect to ElderTree services
export API_URL=https://visage.eldertree.local/api
export REDIS_URL=redis://visage-redis.eldertree.local:6379
export MINIO_ENDPOINT=minio.eldertree.local:9000

python -m src.main
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/packs` | Create new headshot pack |
| `GET` | `/api/packs/{id}` | Get pack details |
| `POST` | `/api/packs/{id}/photos` | Upload photos to pack |
| `POST` | `/api/packs/{id}/generate` | Start generation job |
| `GET` | `/api/packs/{id}/outputs` | Get generated outputs |
| `GET` | `/api/packs/{id}/download` | Download ZIP of outputs |

## User Flow

1. **Create Pack** - User creates a new headshot pack
2. **Upload Photos** - Upload 8-20 photos with validation
3. **Select Style** - Choose style preset(s)
4. **Training** - System trains LoRA on user's photos
5. **Generation** - Generate 20-40 images per style
6. **Filtering** - Auto-filter to keep only best results
7. **Delivery** - User receives gallery and ZIP download

## Quality Standards

To match commercial tools, Visage:

- ✅ Validates uploads (face detection, lighting, diversity)
- ✅ Uses SDXL base + per-user LoRA
- ✅ Applies curated prompt presets per style
- ✅ Generates many, delivers few (reroll + filtering)
- ✅ Upscales and polishes final outputs
- ✅ Hides failures from user

## Development

### Running Tests

```bash
# API tests
cd apps/api
pytest

# Web tests
cd apps/web
npm test
```

### Code Style

```bash
# Python (API/Worker)
ruff check .
ruff format .

# TypeScript (Web)
npm run lint
npm run format
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [.github/SECURITY.md](.github/SECURITY.md) for security policy and reporting vulnerabilities.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related

- [ElderTree Cluster](../pi-fleet/) - Kubernetes infrastructure
- [Workspace Config](../workspace-config/) - Port assignments and conventions
