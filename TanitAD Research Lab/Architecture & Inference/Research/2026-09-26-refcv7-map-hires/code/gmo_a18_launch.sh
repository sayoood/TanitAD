#!/usr/bin/env bash
# The EARLY, NON-BINDING A18 MAIN (SPEC_REFCV7 §23, the PI's "Budget to 3,000 steps"): MAIN only,
# 3,000 steps, the decay from step 2,700, A15's configuration; runner gmo_a18_main.py on the R5
# candidate (tip 2ac0bfb + NEW-2 R5 blobs + the A18 spec; code-identical to 37086c3 + R5).
# Wakes on the A17.1 arm's done-marker (arg 1). ONE GPU job at a time: refuses if any GPU compute
# process is present that is not STOPPED (ps stat T). Held while $R/HOLD_A18 exists. Writes the
# python PID to $OUT/PYTHON_PID and $OUT/A18_DONE on exit (PASS, FAIL or crash) -- the Master
# Mind's cost probe chains on that marker. Kills nothing; touches no process it did not start.
set -u
R=/home/nvidia/nb2r5_2ac0
CODE=$R/code
W=/home/nvidia/gmo_early_0327/weights/map_hires_class_weights_train_100x30.json
OUT=$R/gmo_a18
PY=/home/nvidia/venvs/tanitad-train/bin/python
MARK=${1:-/home/nvidia/nb2r5_2ac0/gmo_a171/A171_DONE}
DONE=$OUT/A18_DONE
mkdir -p "$OUT"
[ -e "$DONE" ] && { echo "[a18] REFUSED: $DONE exists -- use a fresh directory"; exit 6; }
echo "[a18] waiting for $MARK at $(date -u +%H:%M:%SZ)"
for i in $(seq 1 2160); do [ -e "$MARK" ] && break; sleep 20; done
[ -e "$MARK" ] || { echo "[a18] REFUSED: no done-marker after 12 h"; exit 3; }
echo "[a18] done-marker seen at $(date -u +%H:%M:%SZ): $(head -c 200 "$MARK")"
while [ -e $R/HOLD_A18 ]; do sleep 30; done
sleep 20
others=""
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  st=$(ps -o stat= -p "$p" 2>/dev/null | tr -d " ")
  if [ -n "$st" ] && [ "${st#T}" != "$st" ]; then
    continue                                   # STOPPED (state T): holds memory, computes nothing
  fi
  others="$others $p"
done
if [ -n "$others" ]; then
  echo "[a18] REFUSED: GPU compute process(es) present:$others -- ONE job at a time"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  printf '{"exit_code": 5, "exit_utc": "%s", "refused": "GPU busy:%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$others" > "$DONE"
  exit 5
fi
[ "$(sha256sum $W | cut -c1-64)" = "d70dec8087ed73ee6d4129b6fc6e0a97a350f826907462b6b251413ede488b67" ] \
  || { echo "[a18] REFUSED: the TRAIN sqrt_mf weights file is not the verified one"; exit 4; }
echo "[a18] A18 MAIN start $(date -u +%H:%M:%SZ) (stopped GPU processes tolerated)"
cd $R
PYTHONPATH=$CODE/stack:$CODE/stack/scripts:$CODE/taniteval PYTHONIOENCODING=utf-8 \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice -n 10 $PY $R/gmo_a18_main.py "$CODE" "$OUT" "$W" \
  < /dev/null 200>&- &
PYPID=$!
echo "$PYPID" > "$OUT/PYTHON_PID"
echo "[a18] python pid $PYPID"
wait $PYPID
rc=$?
printf '{"exit_code": %s, "exit_utc": "%s", "record": "%s"}\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$OUT/g_map_overfit_A18_MAIN.EARLY_NONBINDING.json" > "$DONE"
echo "[a18] exit $rc at $(date -u +%H:%M:%SZ); done-marker $DONE"
