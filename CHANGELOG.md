# Changelog

All notable changes to Visage will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning Strategy

**Pre-1.0 (Current):** Minor version bumps may include breaking changes. Patch versions are for bug fixes only.

**Post-1.0:** Strict semantic versioning. Breaking changes only in major versions.

## [Unreleased]

### Added

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

- Build workflows now use `CR_PAT` secret instead of `GITHUB_TOKEN`
- Updated README with badges and improved structure

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
