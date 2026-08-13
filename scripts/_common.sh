#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p data log state .private
load_env() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
  fi
  export HOST="${HOST:-0.0.0.0}"
  export PORT="${PORT:-7860}"
}
api_key() {
  if [[ -n "${API_KEY:-}" ]]; then printf '%s' "$API_KEY";
  elif [[ -s "$ROOT_DIR/data/api_key.txt" ]]; then tr -d '\r\n' < "$ROOT_DIR/data/api_key.txt"; fi
}
restart_secret() {
  if [[ -n "${RESTART_SECRET:-}" ]]; then printf '%s' "$RESTART_SECRET";
  elif [[ -s "$ROOT_DIR/data/restart_secret.txt" ]]; then tr -d '\r\n' < "$ROOT_DIR/data/restart_secret.txt"; fi
}
server_base_url() { printf 'http://127.0.0.1:%s' "$PORT"; }
pid_matches_cmdline() {
  local pid="${1:-}" marker="${2:-}" cmdline
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *"$marker"* ]]
}
read_managed_pid() {
  local file="$1" marker="$2" pid
  [[ -s "$file" ]] || return 1
  read -r pid < "$file" || return 1
  if pid_matches_cmdline "$pid" "$marker"; then printf '%s' "$pid"; return 0; fi
  rm -f "$file"; return 1
}
