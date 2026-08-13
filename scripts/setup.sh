#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
[[ -f .env ]] || cp .env.example .env
load_env

python3 - <<'PY'
import sys
if sys.version_info < (3, 10): raise SystemExit("Python 3.10+ is required")
print("Python:", sys.version.split()[0])
PY

# TPU preflight policy:
#   required: real TPU validation must succeed (default and Kaggle validation mode)
#   auto:     require TPU only when enough TPU/VFIO device nodes are visible
#   skip:     CPU-only source/dependency/unit/static validation; never start a TPU worker
# The real Kaggle notebook explicitly uses required so a missing accelerator cannot PASS silently.
tpu_preflight_mode="${TPU_PREFLIGHT_MODE:-required}"
case "$tpu_preflight_mode" in
  required|auto|skip) ;;
  *) echo "Invalid TPU_PREFLIGHT_MODE=$tpu_preflight_mode (use required/auto/skip)" >&2; exit 2 ;;
esac

effective_tpu_preflight="$tpu_preflight_mode"
if [[ "$tpu_preflight_mode" == "auto" ]]; then
  expected="${EXPECTED_TPU_DEVICES:-8}"
  accel_count=0
  vfio_count=0
  if compgen -G '/dev/accel*' >/dev/null; then
    accel_count="$(find /dev -maxdepth 1 -type c -name 'accel*' 2>/dev/null | wc -l | tr -d ' ')"
  fi
  if [[ -d /dev/vfio ]]; then
    vfio_count="$(find /dev/vfio -maxdepth 1 \( -type c -o -type f \) -regex '.*/[0-9]+' 2>/dev/null | wc -l | tr -d ' ')"
  fi
  if (( accel_count >= expected || vfio_count >= expected )); then
    effective_tpu_preflight=required
    echo "[tpu-preflight] auto detected TPU device nodes (accel=$accel_count vfio=$vfio_count expected=$expected)"
  else
    effective_tpu_preflight=skip
    echo "[tpu-preflight] auto found no complete TPU device set; running CPU-only validation"
  fi
fi

if [[ "$effective_tpu_preflight" == "required" ]]; then
  # Configure Kaggle's metadata-less TPU discovery before any JAX/libtpu import.
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/configure_kaggle_tpu.sh"
fi

mode="${INSTALL_PYTHON_DEPS:-auto}"
needs=0
python3 scripts/check_dependency_hygiene.py --quiet || needs=1
case "$mode" in
  true|1|yes|auto) install_allowed=true ;;
  false|0|no) install_allowed=false ;;
  *) echo "Invalid INSTALL_PYTHON_DEPS=$mode (use auto/true/false)" >&2; exit 2 ;;
esac
if [[ "$install_allowed" == "true" && ( "$mode" != "auto" || "$needs" -ne 0 ) ]]; then
  # Do not use --upgrade: exact proven userspace versions are enough, and avoiding
  # blanket upgrades reduces churn in Kaggle's preinstalled JAX/JAXLIB image.
  python3 -m pip install \
    --disable-pip-version-check \
    --root-user-action=ignore \
    --no-warn-conflicts \
    --no-color \
    --progress-bar off \
    -r requirements.txt
elif [[ "$needs" -ne 0 ]]; then
  echo "Required userspace dependencies are missing and INSTALL_PYTHON_DEPS=false" >&2
  python3 scripts/check_dependency_hygiene.py || true
  exit 1
fi
python3 scripts/check_dependency_hygiene.py --pip-check

if [[ "$effective_tpu_preflight" == "required" ]]; then
  # Kaggle images may expose JAX/JAXLIB without shipping libtpu. Keep an existing
  # libtpu untouched; bootstrap only the proven 0.0.17 runtime when it is absent.
  python3 scripts/ensure_libtpu.py
  # Isolated TPU validation. The helper captures known Kaggle/libtpu startup noise
  # only after the child process proves that all expected TPU devices are visible.
  python3 scripts/check_tpu_preflight.py
else
  echo "[tpu-preflight] SKIP: CPU-only validation requested; TPU/libtpu was not initialized"
fi

bash scripts/test_unit.sh
python3 -m compileall -q src scripts
printf 'Setup checks passed.\n'
