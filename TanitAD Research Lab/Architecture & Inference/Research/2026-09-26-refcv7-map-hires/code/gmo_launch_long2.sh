#!/usr/bin/env bash
# MAIN_long, DEFERRED (Master Mind 04:58): start only after my prereg run AND the box builder's
# GPU runs have exited (explicit PIDs, read-only); skip entirely if $R/SKIP_MAIN_LONG exists
# (the lift-lever arm has priority). Kills nothing, touches no process it did not start.
set -u
R=/home/nvidia/gmo_early_0327
PY=/home/nvidia/venvs/tanitad-train/bin/python
PIDS="$*"
echo "[long2] waiting for pids: $PIDS at $(date -u +%H:%M:%SZ)"
for i in $(seq 1 4320); do
  [ -e $R/SKIP_MAIN_LONG ] && { echo "[long2] SKIPPED by $R/SKIP_MAIN_LONG at $(date -u +%H:%M:%SZ)"; exit 0; }
  alive=0; for p in $PIDS; do [ -d /proc/$p ] && alive=1; done
  [ $alive = 0 ] && break
  sleep 10
done
[ $alive = 1 ] && { echo "[long2] REFUSED: still waiting after 12 h"; exit 3; }
[ -s $R/gmo_out/g_map_overfit.EARLY_NONBINDING.json ] || { echo "[long2] REFUSED: no prereg record"; exit 3; }
shared=""
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]')" != "0" ]; then
  names=$(for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
            ps -o args= -p $p | awk '{print $2}' | xargs -r basename; done | sort -u | tr '\n' ' ')
  shared="${names% } (GPU shared, Master Mind ruling 04:16)"
fi
echo "[long2] MAIN_long start $(date -u +%H:%M:%SZ) shared='$shared'"
cd $R
PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice -n 10 $PY $R/gmo_main_long.py \
  $R/code $R/gmo_out_long $R/weights/map_hires_class_weights_train_100x30.json "$shared"
echo "[long2] exit $? at $(date -u +%H:%M:%SZ)"
