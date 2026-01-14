---
name: Infrastructure Task
about: Request infrastructure changes (deployment, scaling, monitoring, etc.)
title: "[INFRA] "
labels: type:infrastructure
assignees: ""
---

## Description

<!-- A clear and concise description of the infrastructure change or task -->

## Infrastructure Component

<!-- Which infrastructure component is affected? -->

- [ ] Kubernetes/Helm
- [ ] Docker/Container
- [ ] CI/CD Pipeline
- [ ] Monitoring/Logging
- [ ] Networking/DNS
- [ ] Storage/Volumes (Longhorn/MinIO)
- [ ] Secrets Management (Vault)
- [ ] GPU Worker Setup
- [ ] Other: <!-- specify -->

## Change Type

<!-- What type of infrastructure change is this? -->

- [ ] Deployment (new service, update existing)
- [ ] Scaling (horizontal/vertical)
- [ ] Monitoring/Alerting
- [ ] Security/Hardening
- [ ] Performance Optimization
- [ ] Disaster Recovery/Backup
- [ ] Documentation
- [ ] Other: <!-- specify -->

## Current State

<!-- Describe the current infrastructure setup -->

## Desired State

<!-- Describe what you want the infrastructure to look like after this change -->

## Impact Assessment

### Affected Services/Components

<!-- List services or components that will be affected -->

-
-
-

### Potential Risks

<!-- What risks or concerns should be considered? -->

-
-
-

### Rollback Plan

<!-- How can this change be rolled back if needed? -->

## Implementation Details

<!-- Provide specific implementation details, configurations, or code snippets if applicable -->

### Kubernetes/Helm Changes

```yaml
# Example Helm values or Kubernetes manifests
```

### Docker Changes

```dockerfile
# Example Dockerfile changes
```

### CI/CD Changes

<!-- Describe any CI/CD pipeline changes needed -->

## Testing Plan

<!-- How will this infrastructure change be tested? -->

- [ ] Test in development environment
- [ ] Test on ElderTree cluster
- [ ] Load testing
- [ ] Security scanning
- [ ] Rollback testing

## Dependencies

<!-- Are there any dependencies or prerequisites for this change? -->

-
-
-

## Related Issues

<!-- Link any related issues. Use "Closes #123" in your PR to automatically close this issue when merged. -->

- Related to #
- Blocked by #
- Closes #

## Additional Context

<!-- Add any other context, diagrams, or references about the infrastructure change here -->
