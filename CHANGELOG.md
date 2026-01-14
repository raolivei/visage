# Changelog

All notable changes to Visage will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning Strategy

**Pre-1.0 (Current):** Minor version bumps may include breaking changes. Patch versions are for bug fixes only.

**Post-1.0:** Strict semantic versioning. Breaking changes only in major versions.

## [Unreleased]

### Added

#### Commercial Quality Pipeline
- **Photo Validator** (`apps/worker/src/pipeline/validator.py`)
  - Face detection using InsightFace/RetinaFace
  - Quality checks: resolution, blur, exposure
  - Face preprocessing with alignment
  - Background removal support (rembg)
  - Pose diversity checking

- **Quality Filter** (`apps/worker/src/pipeline/filter.py`)
  - Multi-factor scoring (face similarity, aesthetic, technical, artifacts)
  - ArcFace face embeddings for similarity scoring
  - CLIP-based aesthetic scoring
  - Artifact detection for eyes, teeth, composition
  - Configurable thresholds and weights

- **Post-Processor** (`apps/worker/src/pipeline/postprocess.py`)
  - Face restoration with CodeFormer/GFPGAN
  - Upscaling with Real-ESRGAN (2x)
  - Auto color correction
  - Sharpening pipeline

- **Real LoRA Training** (`apps/worker/src/pipeline/trainer.py`)
  - PEFT-based LoRA training for SDXL
  - Auto-captioning for training images
  - Configurable hyperparameters
  - MPS (Apple Silicon) support
  - Progress tracking

- **Expanded Prompt Library** (`packages/shared/prompts.py`)
  - 15+ professional headshot styles
  - 3-5 prompt variations per style
  - Categories: business, studio, natural, creative, professional, personal_brand, entertainment
  - Industry-based style recommendations

#### Repository Standards
- Repository template standards from repo-template
- `.github/CODEOWNERS` - Code ownership definitions
- `.github/dependabot.yml` - Automated dependency updates
- `.github/SECURITY.md` - Security policy
- `.github/PULL_REQUEST_TEMPLATE.md` - PR template
- `.github/ISSUE_TEMPLATE/` - Issue templates (bug, feature, infrastructure, security)
- `.github/labels.json` - Standardized labels
- `.github/workflows/ci.yml` - CI pipeline (lint, type-check, test, build)
- `.github/workflows/security-scan.yml` - Security scanning (CodeQL, Trivy, dependency audit)
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - MIT License
- `github/setup-labels.sh` - Label setup script
- Version extraction in build workflows
- Trivy vulnerability scanning in build workflows
- Updated branch protection configuration with CI/security checks

### Changed

- Build workflows use `GITHUB_TOKEN` for GHCR authentication (no expiring PAT needed)
- Updated README with badges and improved structure
- Worker requirements updated with all ML dependencies

## [0.1.0] - 2026-01-14

### Added

- Initial project scaffold
- FastAPI backend with health endpoint
- Next.js frontend with basic layout
- GPU worker stub with MPS support
- Docker Compose for local development
- Kubernetes manifests for ElderTree deployment
- MinIO Helm configuration for object storage
- GitHub Actions CI/CD workflows (build-api, build-web)
- Monorepo structure with apps/web, apps/api, apps/worker
- Database schema design (packs, photos, jobs, outputs)
- Architecture documentation
