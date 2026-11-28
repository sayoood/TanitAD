#!/usr/bin/env bash
# Phase 2 of the EARLY, NON-BINDING G-MAP-OVERFIT: the TRAIN weights exist (PROVENANCE.json
# written by phase 1); phase 1 REFUSED because the GPU was held by another agent's job.
# Wake on THAT job's exit (explicit PID, read-only), re-check the GPU once, then run.
# Kills nothing, touches no process it did not start.
set -u
R=/home/nvidia/gmo_early_0327
CODE=$R/code
HOLD=${1:?pid of the GPU job to queue behind}
W=$R/weights/map_hires_class_weights_train_100x30.json
OUT=$R/gmo_out
PY=/home/nvidia/venvs/tanitad-train/bin/python
echo "[launch2] queued behind pid $HOLD at $(date -u +%H:%M:%SZ)"
for i in $(seq 1 1440); do [ -d /proc/$HOLD ] || break; sleep 10; done
if [ -d /proc/$HOLD ]; then echo "[launch2] REFUSED: pid $HOLD still running after 4 h"; exit 3; fi
echo "[launch2] pid $HOLD gone at $(date -u +%H:%M:%SZ)"
sleep 20
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]')
if [ "$apps" != "0" ]; then
  echo "[launch2] REFUSED: $apps GPU compute process(es) present after pid $HOLD exited"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  exit 5
fi
[ -s "$W" ] || { echo "[launch2] REFUSED: no weights"; exit 3; }
mkdir -p "$OUT"
echo "[launch2] G-MAP-OVERFIT start $(date -u +%H:%M:%SZ)"
cd $R
PYTHONPATH=$CODE/stack:$CODE/stack/scripts:$CODE/taniteval PYTHONIOENCODING=utf-8 \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice -n 10 $PY $R/gmo_early_runner.py "$CODE" "$OUT" "$W" \
  --v2-cache /home/nvidia/data/refcv6-b1-416x1024-train \
  --gt-root /home/nvidia/data/sam3_gt_v3 \
  --extrinsics /home/nvidia/data/refcv6_train_eval139_extrinsics.json \
  --cam-ts-dir /home/nvidia/data/_b1stage416/r0/camera_front_wide \
  --arms healthy,s8_zeros,lane_w0,s8_detached
echo "[launch2] G-MAP-OVERFIT exit $? at $(date -u +%H:%M:%SZ)"
