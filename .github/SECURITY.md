# Security Policy

## Supported Versions

We actively support security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue. Instead, please report it via one of the following methods:

1. **GitHub Security Advisory**: Use GitHub's private vulnerability reporting feature (if enabled)
2. **Direct Contact**: Contact @raolivei directly via GitHub

### What to Include

When reporting a vulnerability, please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)
- Your contact information

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on severity and complexity

### Security Best Practices

We follow these security practices:

- ✅ Regular dependency updates via Dependabot
- ✅ Automated security scanning in CI/CD
- ✅ Container image vulnerability scanning
- ✅ Code quality and security linting
- ✅ Branch protection and required reviews
- ✅ Secrets management via Vault/External Secrets
- ✅ No secrets committed to repository

### Known Security Measures

- All Docker images are scanned for vulnerabilities
- Dependencies are automatically updated for security patches
- Code is scanned for security issues via CodeQL
- Secrets are never committed to the repository
- All external dependencies are regularly audited
- AI model weights and checkpoints are isolated from user data

## Security Updates

Security updates are released as soon as possible after a vulnerability is confirmed and a fix is available. Critical security updates may be released outside of the normal release cycle.

## Acknowledgments

We appreciate responsible disclosure of security vulnerabilities. Contributors who report valid security issues will be acknowledged (with permission) in our security advisories.
