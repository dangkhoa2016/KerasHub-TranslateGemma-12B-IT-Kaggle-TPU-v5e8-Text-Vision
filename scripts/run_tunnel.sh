#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env

base_url="http://127.0.0.1:$PORT"
if ! curl -fsS --max-time 5 "$base_url/health/live" >/dev/null; then
  echo "Local coordinator is not live at $base_url. Run scripts/start.sh first." >&2
  exit 1
fi

if command -v cloudflared >/dev/null 2>&1; then
  bin="$(command -v cloudflared)"
elif [[ -x "$ROOT_DIR/bin/cloudflared" ]]; then
  bin="$ROOT_DIR/bin/cloudflared"
else
  echo "cloudflared is unavailable; install it or place it at bin/cloudflared." >&2
  exit 1
fi

pid_file="$ROOT_DIR/state/tunnel.pid"
if pid="$(read_managed_pid "$pid_file" "cloudflared" 2>/dev/null)"; then
  kill -TERM "$pid" 2>/dev/null || true
fi
: > "$ROOT_DIR/log/tunnel.log"
nohup "$bin" tunnel --no-autoupdate --url "$base_url" > "$ROOT_DIR/log/tunnel.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$pid_file"

for _ in $(seq 1 60); do
  url="$(grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "$ROOT_DIR/log/tunnel.log" | tail -n1 || true)"
  if [[ -n "$url" ]]; then
    printf '%s\n' "$url" > "$ROOT_DIR/data/tunnel_url.txt"
    echo "Tunnel URL: $url"
    echo "Authentication remains enabled; API key value is not printed."
    python3 "$ROOT_DIR/scripts/demo_info.py" --base-url "$url"
    exit 0
  fi
  kill -0 "$pid" 2>/dev/null || {
    echo "cloudflared exited before publishing a URL" >&2
    tail -n 80 "$ROOT_DIR/log/tunnel.log" >&2 || true
    exit 1
  }
  sleep 1
done

echo "Timed out waiting for a Cloudflare Quick Tunnel URL" >&2
exit 1
