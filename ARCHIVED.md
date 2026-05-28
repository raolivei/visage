# Visage — archived

**Status:** Read-only archive (2026-04-02)  
**GitHub:** Repository archived; no new features or production deploys.

## What this repo still contains

- Application source (`apps/api`, `apps/web`, `apps/worker`)
- Kubernetes manifests (`k8s/`) for historical Eldertree deployment
- Docker build workflows (manual `workflow_dispatch` only)
- Monitoring dashboard JSON under `monitoring/`

## What was retired on Eldertree

- Live `visage` namespace (optional delete when cluster reachable)
- Public `visage.eldertree.xyz` and local `visage.eldertree.local` routing
- Active Prometheus/Grafana provisioning (reference copies in `pi-fleet/docs/archive/visage/`)

## Assistants

Listed under **ORG_REPOS_ARCHIVED** in Ollie — not an active raolivei project.

## Restore

Not planned. For emergencies, un-archive on GitHub and follow `k8s/README.md` — expect outdated dependencies.
