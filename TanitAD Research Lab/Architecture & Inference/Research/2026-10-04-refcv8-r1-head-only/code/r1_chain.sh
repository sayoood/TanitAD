#!/bin/bash
# R1 capture chain on Thor: probe (3 windows, GPU identity) -> EVAL capture with identity -> TRAIN capture.
# Each pass takes the shared GPU lock for ITSELF only. The ARTIFACT (record.json / probe.json) is the evidence a pass
# ran -- never the exit code. A pass refuses to start with < 20 GB free on / (the R1 budget is < 15 GB in total).
set -u
R=/home/nvidia/refcv8_r1
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
export REFCV6_REPO=/home/nvidia/refcv7_run/fec3a0dccf REFCV6_KIT=/home/nvidia
export PYTHONPATH=/home/nvidia/refcv7_run/fec3a0dccf/stack OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd $R/code
free_gb () { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
pass () {   # tag artifact args...
  local tag=$1 art=$2; shift 2
  if [ -s "$art" ]; then echo "[chain] $tag banked -- skip"; return 0; fi
  if [ "$(free_gb)" -lt 20 ]; then echo "[chain] $tag REFUSED: $(free_gb) GB free"; return 1; fi
  echo "[chain] $(date -u +%H:%M:%S) start $tag (free $(free_gb) GB)"
  flock $LOCK $PY r1_capture.py "$@" > $R/logs/$tag.log 2>&1 < /dev/null
  echo $? > $R/status/$tag.exit
  echo "[chain] $(date -u +%H:%M:%S) end $tag artifact=$( [ -s "$art" ] && echo yes || echo NO ) free $(free_gb) GB"
  [ -s "$art" ]
}
pass probe_gpu $R/out_probe/eval/probe.json --split eval --out $R/out_probe --probe --max-windows 3 --workers 2 || exit 1
pass cap_eval $R/cache/eval/record.json --split eval --out $R/cache --identity on --batch 1 --workers 4 || exit 1
export REFCV6_REMAP_OVERRIDES=$R/cache/remap_train_r1.json
pass cap_train $R/cache/train/record.json --split train --out $R/cache --identity off --batch 8 --workers 6 || exit 1
echo "[chain] ALLDONE $(date -u +%H:%M:%S) cache $(du -sh $R/cache | cut -f1)"
