# Grafana Dashboard Migration

**Migration Date:** 2026-05-04  
**Status:** ✅ Completed

## Summary

Visage Grafana dashboards have been migrated from static file provisioning to Helm-managed Infrastructure-as-Code.

## What Changed

### Before (Static Provisioning)
- **Location:** `visage/monitoring/grafana/dashboards/*.json`
- **Provisioning:** Static YAML config (`provisioning/dashboards/dashboards.yml`)
- **Folder:** `"Visage"` (static, configured in provisioning file)
- **Not IaC:** Manual file management, not reconstructible

### After (Helm-Managed)
- **Location:** `pi-fleet/helm/monitoring-stack/dashboards/*.json`
- **Provisioning:** Helm templates + Grafana sidecar
- **Folder:** `"Applications/Visage"` (consistent with other apps)
- **Fully IaC:** Declarative folder mappings in `values.yaml`, fully reconstructible

## Migrated Dashboards

| Dashboard | Old Location | New Location | Folder |
|-----------|-------------|-------------|---------|
| Visage Operations | `visage/monitoring/grafana/dashboards/visage-operations.json` | `pi-fleet/helm/monitoring-stack/dashboards/visage-operations.json` | Applications/Visage |
| Visage Training | `visage/monitoring/grafana/dashboards/visage-training.json` | `pi-fleet/helm/monitoring-stack/dashboards/visage-training.json` | Applications/Visage |

## Changes Made

1. **Copied dashboards to Helm chart** (using newer version from Visage directory)
2. **Added folder mappings** in `pi-fleet/helm/monitoring-stack/values.yaml`:
   ```yaml
   dashboardFolders:
     visage-operations: "Applications/Visage"
     visage-training: "Applications/Visage"
   ```
3. **Deprecated static provisioning** - `provisioning/dashboards/dashboards.yml` commented out
4. **Created validation script** - `pi-fleet/helm/monitoring-stack/scripts/validate-dashboards.sh`
5. **Updated documentation** - See `pi-fleet/helm/monitoring-stack/README-DASHBOARDS.md`

## Dashboard URLs

No change to URLs:
- Visage Operations: `https://grafana.eldertree.local/d/visage-ops`
- Visage Training: `https://grafana.eldertree.local/d/visage-training`

Dashboard UIDs remain the same, so bookmarks and links continue to work.

## Folder Change

Dashboard folder changed from:
- **Old:** `Visage` (top-level)
- **New:** `Applications/Visage` (nested under Applications category)

This aligns with the overall folder structure:
```
Applications/
  ├── Visage/
  ├── SwimTO/
  └── Pitanga/

Platform/
  ├── Overview/
  ├── Cluster/
  └── ...
```

## Making Future Changes

### Option 1: Edit in Grafana UI (Quick Changes)
1. Make changes in Grafana
2. Export dashboard JSON (Share → Export)
3. Save to `pi-fleet/helm/monitoring-stack/dashboards/`
4. Commit to git
5. Deploy: `helm upgrade monitoring-stack . -n observability`

### Option 2: Edit JSON Directly
1. Edit `pi-fleet/helm/monitoring-stack/dashboards/visage-*.json`
2. Validate: `./pi-fleet/helm/monitoring-stack/scripts/validate-dashboards.sh`
3. Commit to git
4. Deploy Helm chart

**Important:** Changes made only in Grafana UI are ephemeral and will be lost on pod restart unless exported back to git.

## Verification

Verify the migration was successful:

```bash
# Check ConfigMaps were created
kubectl get configmap -n observability -l grafana_dashboard=1 | grep visage

# Check folder annotations
kubectl get configmap dashboard-visage-operations -n observability -o jsonpath='{.metadata.annotations.grafana_folder}'
# Should output: Applications/Visage

kubectl get configmap dashboard-visage-training -n observability -o jsonpath='{.metadata.annotations.grafana_folder}'
# Should output: Applications/Visage

# Verify in Grafana UI
# Navigate to: Dashboards → Applications → Visage
# Both dashboards should appear there
```

## Rollback (if needed)

If you need to rollback to static provisioning:

1. Uncomment the provisioning YAML:
   ```bash
   vi visage/monitoring/grafana/provisioning/dashboards/dashboards.yml
   ```

2. Restore dashboard JSON files to Visage directory

3. Remove from Helm chart:
   ```bash
   # Remove from pi-fleet/helm/monitoring-stack/dashboards/
   rm dashboards/visage-operations.json dashboards/visage-training.json
   
   # Remove from values.yaml dashboardFolders
   # Deploy to remove ConfigMaps
   helm upgrade monitoring-stack . -n observability
   ```

4. Restart Grafana with static provisioning

**Note:** Rollback should not be necessary as the Helm approach is more robust and maintainable.

## Old Files (Kept for Reference)

These files are kept for reference but are no longer active:

- `visage/monitoring/grafana/dashboards/visage-operations.json` (backup copy)
- `visage/monitoring/grafana/dashboards/visage-training.json` (backup copy)
- `visage/monitoring/grafana/provisioning/dashboards/dashboards.yml` (commented out)

You may delete these once you confirm the migration is working correctly.

## Benefits of New Approach

1. **✅ Single source of truth** - All dashboards in one location
2. **✅ Fully reconstructible** - Can rebuild entire Grafana config from git
3. **✅ Consistent folder structure** - Aligns with other applications
4. **✅ Validation** - Pre-deployment checks for JSON syntax and mappings
5. **✅ Git-tracked** - All changes versioned and auditable
6. **✅ Declarative** - Folder mappings in values.yaml, not hardcoded
7. **✅ Scalable** - Easy to add new dashboards following the same pattern

## Documentation

For complete documentation on managing dashboards, see:
- **Adding/editing dashboards:** `pi-fleet/helm/monitoring-stack/README-DASHBOARDS.md`
- **Dashboard inventory:** `pi-fleet/helm/monitoring-stack/DASHBOARDS.md`
- **Folder mappings:** `pi-fleet/helm/monitoring-stack/values.yaml` (lines 323-337)
- **Validation script:** `pi-fleet/helm/monitoring-stack/scripts/validate-dashboards.sh`

## Questions?

See the comprehensive documentation in `pi-fleet/helm/monitoring-stack/README-DASHBOARDS.md` or check the existing dashboards for examples.
