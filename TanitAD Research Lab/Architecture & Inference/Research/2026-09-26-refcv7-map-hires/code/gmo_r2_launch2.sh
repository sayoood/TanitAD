#!/usr/bin/env bash
# The A12 lever arm's EARLY, NON-BINDING G-MAP-OVERFIT on Thor -- v2 of the queue, agreed with the
# box-head builder (2026-09-27 ~10:00 Berlin): wake on the box LADDER's done-marker (its GPU job
# PID 3619335 writes /home/nvidia/bx_ladder2_0954/LADDER_DONE on exit); the box builder's
# refcv6-head control PID 3603614 is SIGSTOPped (State T) and stays listed on the GPU while
# computing nothing -- it is the ONLY compute process tolerated, and only while stopped. Any
# other compute process refuses (ONE job at a time). On exit -- PASS, FAIL or crash -- writes
# $OUT/A12_DONE so the box builder can SIGCONT its control. Kills nothing; touches no process it
# did not start. Held while $R/HOLD_A12 exists.
set -u
R=/home/nvidia/nb2r2_cef9
CODE=$R/code
W=/home/nvidia/gmo_early_0327/weights/map_hires_class_weights_train_100x30.json
OUT=$R/gmo_a12
PY=/home/nvidia/venvs/tanitad-train/bin/python
MARK=/home/nvidia/bx_ladder2_0954/LADDER_DONE
PAUSED=3603614
DONE=$OUT/A12_DONE
mkdir -p "$OUT"
echo "[a12] waiting for $MARK at $(date -u +%H:%M:%SZ)"
for i in $(seq 1 2160); do [ -e "$MARK" ] && break; sleep 20; done
[ -e "$MARK" ] || { echo "[a12] REFUSED: no ladder done-marker after 12 h"; exit 3; }
echo "[a12] ladder done-marker seen at $(date -u +%H:%M:%SZ): $(head -c 200 "$MARK")"
while [ -e $R/HOLD_A12 ]; do sleep 30; done
sleep 20
others=""
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  if [ "$p" = "$PAUSED" ] && grep -q '^State:[[:space:]]*T' /proc/$p/status 2>/dev/null; then
    continue                                   # known-paused box control: computes nothing
  fi
  others="$others $p"
done
if [ -n "$others" ]; then
  echo "[a12] REFUSED: GPU compute process(es) present:$others -- ONE job at a time"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  exit 5
fi
[ "$(sha256sum $W | cut -c1-64)" = "d70dec8087ed73ee6d4129b6fc6e0a97a350f826907462b6b251413ede488b67" ] \
  || { echo "[a12] REFUSED: the TRAIN weights file is not the verified one"; exit 4; }
echo "[a12] A12 arm start $(date -u +%H:%M:%SZ) (paused box control $PAUSED tolerated)"
cd $R
PYTHONPATH=$CODE/stack:$CODE/stack/scripts:$CODE/taniteval PYTHONIOENCODING=utf-8 \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice -n 10 $PY $R/gmo_r2_runner.py "$CODE" "$OUT" "$W" -- \
  --v2-cache /home/nvidia/data/refcv6-b1-416x1024-train \
  --gt-root /home/nvidia/data/sam3_gt_v3 \
  --extrinsics /home/nvidia/data/refcv6_train_eval139_extrinsics.json \
  --cam-ts-dir /home/nvidia/data/_b1stage416/r0/camera_front_wide \
  --arms healthy,s8_zeros,near_zeros,lane_w0,s8_detached
rc=$?
printf '{"exit_code": %s, "exit_utc": "%s", "record": "%s"}\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$OUT/g_map_overfit_A12.EARLY_NONBINDING.json" > "$DONE"
echo "[a12] exit $rc at $(date -u +%H:%M:%SZ); done-marker $DONE"
