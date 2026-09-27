#!/usr/bin/env bash
# The A16 weights arm's EARLY, NON-BINDING G-MAP-OVERFIT on Thor (SPEC_REFCV7 §21; the mf weights, stacked on
# A15's near lift + near refine block). Wakes on a done-marker (arg 1; default: the A15 arm's
# A15_DONE, which exists -- A15 finished 2026-09-27 ~12:28Z). A STOPPED process
# (ps stat T) holds GPU memory and computes nothing -- tolerated; any compute process in another
# state refuses (ONE job at a time). On exit -- PASS, FAIL or crash -- writes $OUT/A16_DONE.
# Kills nothing; touches no process it did not start. Held while $R/HOLD_A16 exists.
set -u
R=/home/nvidia/nb2r4_1b81
CODE=$R/code
W=/home/nvidia/gmo_early_0327/weights_mf/map_hires_class_weights_train_100x30_MF.json
OUT=$R/gmo_a16
PY=/home/nvidia/venvs/tanitad-train/bin/python
MARK=${1:-/home/nvidia/nb2r3_2374/gmo_a15/A15_DONE}
PAUSED=any-stopped
DONE=$OUT/A16_DONE
mkdir -p "$OUT"
echo "[a16] waiting for $MARK at $(date -u +%H:%M:%SZ)"
for i in $(seq 1 2160); do [ -e "$MARK" ] && break; sleep 20; done
[ -e "$MARK" ] || { echo "[a16] REFUSED: no done-marker after 12 h"; exit 3; }
echo "[a16] done-marker seen at $(date -u +%H:%M:%SZ): $(head -c 200 "$MARK")"
# optional: an explicit PID to outwait (read-only), BOX_PID=<pid>; EMPTY = none. The box
# builder finished 2026-09-27 ~13:00Z with no Thor process left, so the default waits for
# nothing. (NOT a default of 1: /proc/1 always exists and the wait would never end.)
BOX=${BOX_PID:-}
if [ -n "$BOX" ]; then while [ -d /proc/$BOX ]; do sleep 20; done; fi
echo "[a16] box pid '$BOX' (empty = none) gone at $(date -u +%H:%M:%SZ)"
while [ -e $R/HOLD_A16 ]; do sleep 30; done
sleep 20
others=""
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  st=$(ps -o stat= -p "$p" 2>/dev/null | tr -d " "); if [ -n "$st" ] && [ "${st#T}" != "$st" ]; then
    continue                                   # STOPPED (state T): holds memory, computes nothing
  fi
  others="$others $p"
done
if [ -n "$others" ]; then
  echo "[a16] REFUSED: GPU compute process(es) present:$others -- ONE job at a time"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  exit 5
fi
[ "$(sha256sum $W | cut -c1-64)" = "8ff4fd6d8798031a4af59991833ff724db251618f32c794beb2e5b8575702c98" ] \
  || { echo "[a16] REFUSED: the TRAIN weights file is not the verified one"; exit 4; }
echo "[a16] A16 arm start $(date -u +%H:%M:%SZ) (stopped GPU processes tolerated)"
cd $R
PYTHONPATH=$CODE/stack:$CODE/stack/scripts:$CODE/taniteval PYTHONIOENCODING=utf-8 \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice -n 10 $PY $R/gmo_r4_runner.py "$CODE" "$OUT" "$W" -- \
  --v2-cache /home/nvidia/data/refcv6-b1-416x1024-train \
  --gt-root /home/nvidia/data/sam3_gt_v3 \
  --extrinsics /home/nvidia/data/refcv6_train_eval139_extrinsics.json \
  --cam-ts-dir /home/nvidia/data/_b1stage416/r0/camera_front_wide \
  --arms healthy,s8_zeros,edge_w0,lane_w0,s8_detached
rc=$?
printf '{"exit_code": %s, "exit_utc": "%s", "record": "%s"}\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$OUT/g_map_overfit_A16.EARLY_NONBINDING.json" > "$DONE"
echo "[a16] exit $rc at $(date -u +%H:%M:%SZ); done-marker $DONE"
