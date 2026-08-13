#!/usr/bin/env bash
# Source this file before importing JAX/libtpu.
set -Eeuo pipefail

mode="${KAGGLE_TPU_FALLBACK:-auto}"
case "$mode" in
  auto|true|false) ;;
  *) echo "[tpu-config] invalid KAGGLE_TPU_FALLBACK=$mode (use auto|true|false)" >&2; return 2 2>/dev/null || exit 2 ;;
esac

is_kaggle=false
if [[ -d /kaggle/working || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -n "${KAGGLE_URL_BASE:-}" ]]; then
  is_kaggle=true
fi

metadata_available=false
if command -v curl >/dev/null 2>&1; then
  if curl -fsS --connect-timeout 1 --max-time 2 \
      -H 'Metadata-Flavor: Google' \
      'http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type' \
      >/dev/null 2>&1; then
    metadata_available=true
  fi
fi

apply=false
if [[ "$mode" == "true" ]]; then
  apply=true
elif [[ "$mode" == "auto" && "$is_kaggle" == "true" && "$metadata_available" == "false" ]]; then
  # Only auto-apply for the profile this project is explicitly designed for.
  if [[ "${REQUIRE_V5E8:-true}" == "true" && "${EXPECTED_TPU_DEVICES:-8}" == "8" ]]; then
    apply=true
  fi
fi

if [[ "$apply" == "true" ]]; then
  export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
  export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-${TPU_CHIPS_PER_HOST_BOUNDS_FALLBACK:-2,4,1}}"
  export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-${TPU_HOST_BOUNDS_FALLBACK:-1,1,1}}"
  export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-${TPU_ACCELERATOR_TYPE_FALLBACK:-v5e-8}}"
  export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-$(hostname)}"
  export TG_TPU_FALLBACK_APPLIED=true
  echo "[tpu-config] Kaggle TPU fallback applied: type=$TPU_ACCELERATOR_TYPE chips=$TPU_CHIPS_PER_HOST_BOUNDS host=$TPU_HOST_BOUNDS"
else
  export TG_TPU_FALLBACK_APPLIED=false
  echo "[tpu-config] normal TPU discovery retained (kaggle=$is_kaggle metadata_available=$metadata_available mode=$mode)"
fi
