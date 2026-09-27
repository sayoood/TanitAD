#!/usr/bin/env bash
# The A15 decoder arm's EARLY, NON-BINDING G-MAP-OVERFIT on Thor (SPEC_REFCV7 §20; stacked on
# A12's near lift). Wakes on a done-marker (arg 1; default: my paused A12 run's A12_DONE, the
# Master Mind's order: box G-BOX-OVERFIT -> A12 resumes and finishes -> A15). A STOPPED process
# (ps stat T) holds GPU memory and computes nothing -- tolerated; any compute process in another
# state refuses (ONE job at a time). On exit -- PASS, FAIL or crash -- writes $OUT/A15_DONE.
# Kills nothing; touches no process it did not start. Held while $R/HOLD_A15 exists.
set -u
R=/home/nvidia/nb2r3_2374
CODE=$R/code
W=/home/nvidia/gmo_early_0327/weights/map_hires_class_weights_train_100x30.json
OUT=$R/gmo_a15
PY=/home/nvidia/venvs/tanitad-train/bin/python
MARK=${1:-/home/nvidia/nb2r2_cef9/gmo_a12/A12_DONE}
PAUSED=any-stopped
DONE=$OUT/A15_DONE
mkdir -p "$OUT"
echo "[a15] waiting for $MARK at $(date -u +%H:%M:%SZ)"
for i in $(seq 1 2160); do [ -e "$MARK" ] && break; sleep 20; done
[ -e "$MARK" ] || { echo "[a15] REFUSED: no done-marker after 12 h"; exit 3; }
echo "[a15] done-marker seen at $(date -u +%H:%M:%SZ): $(head -c 200 "$MARK")"
# the box builder's chain (PID 3676134, /home/nvidia/bx_anch_1116/chain4.sh) must have exited
# too: it does not key on A12_DONE (box builder, ~12:20); explicit PID, read-only
BOX=3676134
while [ -d /proc/$BOX ]; do sleep 20; done
echo "[a15] box chain $BOX gone at $(date -u +%H:%M:%SZ)"
while [ -e $R/HOLD_A15 ]; do sleep 30; done
sleep 20
others=""
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  st=$(ps -o stat= -p "$p" 2>/dev/null | tr -d " "); if [ -n "$st" ] && [ "${st#T}" != "$st" ]; then
    continue                                   # STOPPED (state T): holds memory, computes nothing
  fi
  others="$others $p"
done
if [ -n "$others" ]; then
  echo "[a15] REFUSED: GPU compute process(es) present:$others -- ONE job at a time"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  exit 5
fi
[ "$(sha256sum $W | cut -c1-64)" = "d70dec8087ed73ee6d4129b6fc6e0a97a350f826907462b6b251413ede488b67" ] \
  || { echo "[a15] REFUSED: the TRAIN weights file is not the verified one"; exit 4; }
echo "[a15] A15 arm start $(date -u +%H:%M:%SZ) (stopped GPU processes tolerated)"
cd $R
PYTHONPATH=$CODE/stack:$CODE/stack/scripts:$CODE/taniteval PYTHONIOENCODING=utf-8 \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice -n 10 $PY $R/gmo_r3_runner.py "$CODE" "$OUT" "$W" -- \
  --v2-cache /home/nvidia/data/refcv6-b1-416x1024-train \
  --gt-root /home/nvidia/data/sam3_gt_v3 \
  --extrinsics /home/nvidia/data/refcv6_train_eval139_extrinsics.json \
  --cam-ts-dir /home/nvidia/data/_b1stage416/r0/camera_front_wide \
  --arms healthy,s8_zeros,near_block_zeros,lane_w0,s8_detached
rc=$?
printf '{"exit_code": %s, "exit_utc": "%s", "record": "%s"}\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$OUT/g_map_overfit_A15.EARLY_NONBINDING.json" > "$DONE"
echo "[a15] exit $rc at $(date -u +%H:%M:%SZ); done-marker $DONE"
