#!/usr/bin/env bash
# The A12 lever arm's EARLY, NON-BINDING G-MAP-OVERFIT on Thor, queued behind the box builder's
# GPU job (explicit PIDs, read-only). Wakes on those PIDs' exit, then checks the GPU is EMPTY
# (one job at a time), then runs MAIN first. Kills nothing; touches no process it did not start.
# Skipped entirely if $R/HOLD_A12 exists (the Master Mind / box builder can hold it).
set -u
R=/home/nvidia/nb2r2_cef9
CODE=$R/code
W=/home/nvidia/gmo_early_0327/weights/map_hires_class_weights_train_100x30.json
OUT=$R/gmo_a12
PY=/home/nvidia/venvs/tanitad-train/bin/python
PIDS="$*"
echo "[a12] waiting for pids: ${PIDS:-none} at $(date -u +%H:%M:%SZ)"
for i in $(seq 1 4320); do
  alive=0; for p in $PIDS; do [ -d /proc/$p ] && alive=1; done
  [ $alive = 0 ] && break
  sleep 10
done
[ ${alive:-0} = 1 ] && { echo "[a12] REFUSED: still waiting after 12 h"; exit 3; }
while [ -e $R/HOLD_A12 ]; do sleep 30; done
sleep 20
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]')
if [ "$apps" != "0" ]; then
  echo "[a12] REFUSED: $apps GPU compute process(es) present -- ONE job at a time"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  exit 5
fi
[ "$(sha256sum $W | cut -c1-64)" = "d70dec8087ed73ee6d4129b6fc6e0a97a350f826907462b6b251413ede488b67" ] \
  || { echo "[a12] REFUSED: the TRAIN weights file is not the verified one"; exit 4; }
mkdir -p "$OUT"
echo "[a12] A12 arm start $(date -u +%H:%M:%SZ)"
cd $R
PYTHONPATH=$CODE/stack:$CODE/stack/scripts:$CODE/taniteval PYTHONIOENCODING=utf-8 \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice -n 10 $PY $R/gmo_r2_runner.py "$CODE" "$OUT" "$W" -- \
  --v2-cache /home/nvidia/data/refcv6-b1-416x1024-train \
  --gt-root /home/nvidia/data/sam3_gt_v3 \
  --extrinsics /home/nvidia/data/refcv6_train_eval139_extrinsics.json \
  --cam-ts-dir /home/nvidia/data/_b1stage416/r0/camera_front_wide \
  --arms healthy,s8_zeros,near_zeros,lane_w0,s8_detached
echo "[a12] exit $? at $(date -u +%H:%M:%SZ)"
