#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env
base="$(server_base_url)"
echo "=== liveness ==="
curl -sS -w '\nHTTP %{http_code}\n' "$base/health/live" || true
key="$(api_key)"
echo "=== readiness ==="
if [[ -n "$key" ]]; then
  curl -sS -w '\nHTTP %{http_code}\n' -H "Authorization: Bearer $key" "$base/health/ready?details=1" || true
else
  curl -sS -w '\nHTTP %{http_code}\n' "$base/health/ready" || true
fi
