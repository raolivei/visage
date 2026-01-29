#!/usr/bin/env bash
# Tunnel visage to your Mac when Tailscale is not available (e.g. Tailscale not set up or down).
#
# Uses kubectl port-forward to Traefik. Run this, then open:
#   https://visage.eldertree.local:8443
#
# Prereqs (tunnel mode only):
#   - kubectl context eldertree (API server reachable from your Mac)
#   - /etc/hosts:  127.0.0.1  visage.eldertree.local
# For direct access (same LAN), use the same Traefik IP as other services (e.g. 192.168.2.200) instead of 127.0.0.1.

set -e
CONTEXT="${1:-eldertree}"
LOCAL_PORT="${2:-8443}"

echo "Visage tunnel (context=${CONTEXT}, local port=${LOCAL_PORT})"
echo ""

if ! kubectl --context "$CONTEXT" get svc traefik -n kube-system &>/dev/null; then
  echo "Error: cannot reach cluster (traefik not found). Check kubeconfig and context."
  exit 1
fi

if ! grep -q '127.0.0.1.*visage.eldertree.local' /etc/hosts 2>/dev/null; then
  echo "Add this to /etc/hosts for tunnel mode:"
  echo "  127.0.0.1  visage.eldertree.local"
  echo ""
fi

echo "Forwarding Traefik HTTPS to localhost:${LOCAL_PORT}"
echo "Open: https://visage.eldertree.local:${LOCAL_PORT}"
echo " (accept self-signed cert if prompted)"
echo ""
exec kubectl --context "$CONTEXT" port-forward -n kube-system svc/traefik "${LOCAL_PORT}:443"
