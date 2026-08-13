#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
load_env
image="${1:-$ROOT_DIR/assets/sample-image-with-text.png}"
[[ -f "$image" ]] || { echo "Missing image: $image" >&2; exit 1; }
key="$(api_key)"
[[ -n "$key" ]] || { echo "API key unavailable" >&2; exit 1; }
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
python3 - "$image" "$tmp" <<'PY'
import base64
import json
import sys
from pathlib import Path

raw = base64.b64encode(Path(sys.argv[1]).read_bytes()).decode("ascii")
payload = {
    "image_base64": raw,
    "source_lang": "English",
    "target_lang": "Vietnamese",
    "max_new_tokens": 128,
}
Path(sys.argv[2]).write_text(json.dumps(payload), encoding="utf-8")
PY
python3 scripts/api_smoke.py \
  --base-url "$(server_base_url)" \
  --api-key "$key" \
  --path /translate/image \
  --payload-file "$tmp" \
  --timeout "${SMOKE_TIMEOUT:-1200}"
