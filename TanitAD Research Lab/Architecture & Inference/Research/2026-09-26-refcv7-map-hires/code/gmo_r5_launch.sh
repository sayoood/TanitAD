#!/usr/bin/env bash
# The A17.1 lr-decay arm's EARLY, NON-BINDING G-MAP-OVERFIT on Thor (SPEC_REFCV7 §22.1, the PI's
# "same rule for the map"): A15's configuration (near lift 20 m + 1 near refine block, TRAIN
# sqrt_mf d70dec80) + lr cosine to 0 over steps 900-1,000; must-fail s8_zeros + near_block_zeros,
# lane_w0 kept. Candidate: tip 2ac0bfb + NEW-2 R5 blobs. ONE GPU job at a time: refuses if any
# GPU compute process is present that is not STOPPED (ps stat T). Held while $R/HOLD_A171 exists.
# Writes the python PID to $OUT/PYTHON_PID and $OUT/A171_DONE on exit (PASS, FAIL or crash).
# Kills nothing; touches no process it did not start.
set -u
R=/home/nvidia/nb2r5_2ac0
CODE=$R/code
W=/home/nvidia/gmo_early_0327/weights/map_hires_class_weights_train_100x30.json
OUT=$R/gmo_a171
PY=/home/nvidia/venvs/tanitad-train/bin/python
DONE=$OUT/A171_DONE
mkdir -p "$OUT"
[ -e "$DONE" ] && { echo "[a171] REFUSED: $DONE exists -- use a fresh directory"; exit 6; }
while [ -e $R/HOLD_A171 ]; do sleep 30; done
others=""
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  st=$(ps -o stat= -p "$p" 2>/dev/null | tr -d " ")
  if [ -n "$st" ] && [ "${st#T}" != "$st" ]; then
    continue                                   # STOPPED (state T): holds memory, computes nothing
  fi
  others="$others $p"
done
if [ -n "$others" ]; then
  echo "[a171] REFUSED: GPU compute process(es) present:$others -- ONE job at a time"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  exit 5
fi
[ "$(sha256sum $W | cut -c1-64)" = "d70dec8087ed73ee6d4129b6fc6e0a97a350f826907462b6b251413ede488b67" ] \
  || { echo "[a171] REFUSED: the TRAIN sqrt_mf weights file is not the verified one"; exit 4; }
echo "[a171] A17.1 arm start $(date -u +%H:%M:%SZ) (stopped GPU processes tolerated)"
cd $R
PYTHONPATH=$CODE/stack:$CODE/stack/scripts:$CODE/taniteval PYTHONIOENCODING=utf-8 \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice -n 10 $PY $R/gmo_r5_runner.py "$CODE" "$OUT" "$W" -- \
  --v2-cache /home/nvidia/data/refcv6-b1-416x1024-train \
  --gt-root /home/nvidia/data/sam3_gt_v3 \
  --extrinsics /home/nvidia/data/refcv6_train_eval139_extrinsics.json \
  --cam-ts-dir /home/nvidia/data/_b1stage416/r0/camera_front_wide \
  --arms healthy,s8_zeros,near_block_zeros,lane_w0,s8_detached < /dev/null 200>&- &
PYPID=$!
echo "$PYPID" > "$OUT/PYTHON_PID"
echo "[a171] python pid $PYPID"
wait $PYPID
rc=$?
printf '{"exit_code": %s, "exit_utc": "%s", "record": "%s"}\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$OUT/g_map_overfit_A171.EARLY_NONBINDING.json" > "$DONE"
echo "[a171] exit $rc at $(date -u +%H:%M:%SZ); done-marker $DONE"
