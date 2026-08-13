#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env
key="$(api_key)"
[[ -n "$key" ]] || { echo "API key unavailable" >&2; exit 1; }
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
cat > "$tmp" <<'JSON'
{
  "text": "Good morning! How are you?",
  "source_lang": "English",
  "target_lang": "Vietnamese",
  "max_new_tokens": 128
}
JSON
python3 scripts/api_smoke.py \
  --base-url "$(server_base_url)" \
  --api-key "$key" \
  --path /translate \
  --payload-file "$tmp" \
  --timeout "${SMOKE_TIMEOUT:-1200}"
