#!/usr/bin/env bash
# Outer resume loop: download_videos.py now exits(2) on rate-limit instead
# of sleeping in-process (that was fragile to session teardown). This loop
# owns the cooldown wait between short-lived download attempts.
set -uo pipefail
cd /projects/bhay/william/ruixin/vans_world_model/raw_data

PY=/u/yli8/.conda/envs/thinkjepa/bin/python3
COOLDOWN_SEC=$((70 * 60))

while true; do
  echo "[LOOP] $(date -Is) starting download_videos.py"
  "$PY" download_videos.py --workers 1 --delay 4.0
  code=$?
  echo "[LOOP] $(date -Is) download_videos.py exited with code $code"
  if [[ $code -eq 2 ]]; then
    echo "[LOOP] rate-limited, sleeping ${COOLDOWN_SEC}s before retry"
    sleep "$COOLDOWN_SEC"
    continue
  else
    echo "[LOOP] non-rate-limit exit ($code) -- stopping loop"
    break
  fi
done
