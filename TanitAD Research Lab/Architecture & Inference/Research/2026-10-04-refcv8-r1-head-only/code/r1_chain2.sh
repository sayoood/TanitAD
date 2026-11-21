#!/bin/bash
# R1 capture chain v2 on Thor: probe (3 windows, GPU identity) -> EVAL capture (identity on every window) ->
# identity report (CPU) -> TRAIN capture. Every GPU pass takes the shared lock for ITSELF only and stops itself before
# ~38 min (--budget-s 2280; the shared-lock courtesy), so a pass is re-run until its ARTIFACT exists -- the artifact
# (probe.json / record.json), never the exit code, is the evidence. A pass refuses to start with < 20 GB free.
set -u
R=/home/nvidia/refcv8_r1
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
export REFCV6_REPO=/home/nvidia/refcv7_run/fec3a0dccf REFCV6_KIT=/home/nvidia
export PYTHONPATH=/home/nvidia/refcv7_run/fec3a0dccf/stack OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd $R/code
free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
pass () {   # tag artifact max_attempts args...
  local tag=$1 art=$2 n=$3; shift 3
  local i
  for i in $(seq 1 $n); do
    if [ -s "$art" ]; then echo "[chain] $tag banked"; return 0; fi
    if [ "$(free_gb)" -lt 20 ]; then echo "[chain] $tag REFUSED: $(free_gb) GB free"; return 1; fi
    echo "[chain] $(date -u +%H:%M:%S) start $tag attempt $i (free $(free_gb) GB)"
    flock $LOCK $PY r1_capture.py "$@" >> $R/logs/$tag.log 2>&1 < /dev/null
    echo $? > $R/status/$tag.exit
    echo "[chain] $(date -u +%H:%M:%S) end $tag attempt $i rc=$(cat $R/status/$tag.exit) artifact=$( [ -s "$art" ] && echo yes || echo NO ) free $(free_gb) GB"
    if [ "$(cat $R/status/$tag.exit)" != "0" ] && [ "$(cat $R/status/$tag.exit)" != "3" ]; then return 1; fi
    sleep 20
  done
  [ -s "$art" ]
}
pass probe_gpu $R/out_probe/eval/probe.json 1 --split eval --out $R/out_probe --probe --max-windows 3 --workers 2 || exit 1
pass cap_eval $R/cache/eval/record.json 4 --split eval --out $R/cache --identity on --batch 1 --workers 4 || exit 1
$PY r1_identity_report.py --cache $R/cache > $R/logs/identity_report.log 2>&1 < /dev/null
echo "[chain] identity report rc=$? $(date -u +%H:%M:%S)"
export REFCV6_REMAP_OVERRIDES=$R/cache/remap_train_r1.json
pass cap_train $R/cache/train/record.json 4 --split train --out $R/cache --identity off --batch 8 --workers 6 || exit 1
echo "[chain] ALLDONE $(date -u +%H:%M:%S) cache $(du -sh $R/cache | cut -f1)"
