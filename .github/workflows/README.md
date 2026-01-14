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

## Authentication

The workflows use the built-in `GITHUB_TOKEN` for pushing images to GHCR. This token:
- Never expires
- Is automatically provided by GitHub Actions
- Has `packages:write` permission when workflow write access is enabled

**Required Repository Settings:**

1. Go to **Settings → Actions → General → Workflow permissions**
2. Select **"Read and write permissions"**
3. (Optional) Check "Allow GitHub Actions to create and approve pull requests"

No manual secrets or PAT rotation required.

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

1. Verify repository has workflow write permissions enabled (Settings → Actions → General)
2. Check that the workflow has `packages: write` permission in the `permissions` block
3. Ensure the image name matches the repository owner (`ghcr.io/raolivei/...`)
