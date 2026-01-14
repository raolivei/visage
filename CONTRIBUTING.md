# Contributing Guidelines

Thank you for your interest in contributing to Visage! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project
- Show empathy towards other contributors

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/visage.git
   cd visage
   ```
3. **Set up upstream remote**:
   ```bash
   git remote add upstream https://github.com/raolivei/visage.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker and Docker Compose
- (Optional) Mac with MPS for GPU worker development

### Local Development

```bash
# Start all services
docker-compose up -d

# Access services:
# - Web UI: http://localhost:3004
# - API: http://localhost:8004
# - MinIO Console: http://localhost:9001
```

### Running Tests

```bash
# Python tests (API)
cd apps/api
pip install -r requirements.txt
pytest

# Node.js tests (Web)
cd apps/web
npm install
npm test
```

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates
- `infra/description` - Infrastructure changes

### Making Changes

1. **Make your changes** in your feature branch
2. **Write/update tests** if applicable
3. **Update documentation** if needed
4. **Run tests locally**:
   ```bash
   # Python
   cd apps/api && pytest
   
   # Node.js
   cd apps/web && npm test
   ```
5. **Run linters**:
   ```bash
   # Python
   cd apps/api
   ruff check src/
   mypy src/ --ignore-missing-imports
   
   # Node.js
   cd apps/web
   npm run lint
   ```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks
- `infra`: Infrastructure changes

**Scopes:**
- `api`: Backend API changes
- `web`: Frontend changes
- `worker`: GPU worker changes
- `k8s`: Kubernetes/infrastructure changes
- `ci`: CI/CD changes

**Examples:**
```
feat(api): add user authentication endpoint

Implements JWT-based authentication for user login.
Adds /api/v1/auth/login endpoint with token generation.

Closes #123
```

```
fix(web): resolve memory leak in component cleanup

The useEffect hook was not properly cleaning up event listeners,
causing memory leaks on component unmount.

Fixes #456
```

### Pull Request Process

1. **Update your branch** with latest changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub:
   - Fill out the PR template
   - Link related issues
   - Request reviews from maintainers

4. **Ensure all checks pass**:
   - CI workflows must pass
   - Security scans must pass
   - Code review approval required

5. **Address review feedback**:
   - Make requested changes
   - Push updates to your branch
   - PR will update automatically

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where applicable
- Maximum line length: 100 characters
- Use `ruff` for linting
- Use `mypy` for type checking

**Example:**
```python
from typing import Optional

def process_data(data: list[str], limit: Optional[int] = None) -> dict[str, int]:
    """Process data and return statistics.
    
    Args:
        data: List of data items to process
        limit: Optional limit on number of items
        
    Returns:
        Dictionary with processing statistics
    """
    if limit:
        data = data[:limit]
    return {"count": len(data), "processed": True}
```

### TypeScript/JavaScript

- Use TypeScript for type safety
- Follow consistent formatting (Prettier if configured)
- Use ESLint for linting

**Example:**
```typescript
interface User {
  id: string;
  name: string;
  email: string;
}

async function fetchUser(id: string): Promise<User> {
  const response = await fetch(`/api/users/${id}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch user: ${response.statusText}`);
  }
  return response.json();
}
```

### Docker

- Use multi-stage builds when possible
- Minimize image size
- Use specific version tags for base images
- Build for ARM64 (ElderTree cluster)

## Documentation

- Update README.md for user-facing changes
- Update CHANGELOG.md for all changes
- Add code comments for complex logic
- Update API documentation if applicable

### CHANGELOG Format

Follow [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [1.2.0] - 2024-01-15

### Added
- New feature: User authentication endpoint
- API documentation in `/docs` directory

### Changed
- Updated dependency versions
- Improved error messages

### Fixed
- Memory leak in component cleanup
- Authentication token expiration issue

### Removed
- Deprecated API endpoint `/api/v1/old-endpoint`
```

## Security

- **Never commit secrets** (API keys, passwords, tokens)
- Use environment variables for sensitive data
- Report security vulnerabilities privately (see SECURITY.md)
- Follow secure coding practices

## Questions?

- Open an issue for bug reports or feature requests
- Check existing documentation
- Review closed issues/PRs for similar questions

Thank you for contributing! 🎉
