#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"; load_env
if [[ "${_SUPERVISE_BG:-0}" != "1" ]]; then
  nohup env _SUPERVISE_BG=1 bash "$(realpath "$0")" > log/supervise.stdout.log 2>&1 &
  echo "Supervisor started (PID $!)."; exit 0
fi
max="${MAX_SERVER_RESTARTS:-10}"; count=0
bash "$ROOT_DIR/scripts/start.sh"
while (( count < max )); do
  if pid="$(read_managed_pid "$ROOT_DIR/state/server.pid" "src/server.py" 2>/dev/null)"; then sleep 5; continue; fi
  count=$((count+1)); echo "Coordinator stopped; restart $count/$max" >&2; bash "$ROOT_DIR/scripts/start.sh"; sleep 5
done
exit 1
