#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"; load_env
file="$ROOT_DIR/state/tunnel.pid"
if pid="$(read_managed_pid "$file" "cloudflared" 2>/dev/null)"; then kill -TERM "$pid" 2>/dev/null || true; fi
rm -f "$file" data/tunnel_url.txt
