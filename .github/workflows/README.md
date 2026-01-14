# GitHub Actions Workflows

This directory contains CI/CD workflows for the Visage project.

## Workflows

### `build-api.yml` - Build and Push API Image

Builds and pushes the FastAPI backend Docker image.

**Triggers:**
- Push to `main` or `dev` branches (when `apps/api/**` changes)
- Git tags matching `v*`
- Pull requests to `main`
- Manual dispatch

**Features:**
- Version extraction from `VERSION` file
- ARM64-only build (for ElderTree cluster)
- Trivy vulnerability scanning
- Automatic tagging (semver, branch, SHA)

### `build-web.yml` - Build and Push Web Image

Builds and pushes the Next.js frontend Docker image.

**Triggers:**
- Push to `main` or `dev` branches (when `apps/web/**` changes)
- Git tags matching `v*`
- Pull requests to `main`
- Manual dispatch

**Features:**
- Version extraction from `VERSION` file
- ARM64-only build (for ElderTree cluster)
- Trivy vulnerability scanning
- Automatic tagging (semver, branch, SHA)

### `ci.yml` - Continuous Integration

Runs linting, type checking, and tests.

**Triggers:**
- Pull requests to `main` or `dev`
- Push to `dev`
- Manual dispatch

**Jobs:**
- `lint`: Python (ruff, mypy) and TypeScript linting
- `type-check`: Python and TypeScript type checking
- `test`: Python (pytest) and Node.js tests
- `build`: Docker build verification (no push)

### `security-scan.yml` - Security Scanning

Comprehensive security scanning for vulnerabilities.

**Triggers:**
- Push to `main` or `dev`
- Pull requests to `main`
- Weekly schedule (Mondays at midnight)
- Manual dispatch

**Jobs:**
- `codeql-analysis`: Static analysis for JavaScript and Python
- `dependency-scan`: npm audit and pip-audit
- `container-scan`: Trivy filesystem scan
- `secret-scan`: Pattern-based secret detection

## Image Tagging Strategy

| Trigger               | Tags Generated                  |
| --------------------- | ------------------------------- |
| Push to `main`        | `main`, `latest`, `main-<sha>`  |
| Push to `dev`         | `dev`, `dev-<sha>`              |
| Git tag `v1.2.3`      | `v1.2.3`, `v1.2`, `v1`, `<sha>` |
| Pull request #42      | `pr-42` (build only, no push)   |
| Manual with tag input | Custom tag                      |

## Required Secrets

| Secret   | Description                                    |
| -------- | ---------------------------------------------- |
| `CR_PAT` | GitHub Container Registry Personal Access Token |

**Creating `CR_PAT`:**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Create a new token with scopes: `write:packages`, `read:packages`
3. Add it as a repository secret named `CR_PAT`

## Published Images

Images are published to GitHub Container Registry:

- `ghcr.io/raolivei/visage-api`
- `ghcr.io/raolivei/visage-web`

## Platform Support

All images are built for **ARM64 only** to support the ElderTree Raspberry Pi k3s cluster.

## Troubleshooting

### Build Failures

1. Check that Dockerfiles exist and are valid
2. Verify build context paths are correct
3. Review workflow logs for specific errors

### Security Scan Issues

1. **CodeQL**: Ensure the language is in the matrix
2. **Dependency Scan**: Verify lock files exist
3. **Trivy**: Check that results are being generated

### Authentication Issues

1. Verify `CR_PAT` secret is set
2. Check token hasn't expired
3. Ensure token has correct permissions
