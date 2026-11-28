#!/bin/bash
# mapcal_q.sh -- MM 2026-11-06 (Thor): PREREG_M1_FITTHR runbook steps 3-4. Takes the Thor GPU lock ONCE for all three FIT dumps (C0 / C1 / M1,
# model_ema, 139 TRAIN-DIAG clips x 8 = 1,112 windows; the chain cannot wedge a training run between them), checks the GPU is idle under it,
# then fits on CPU: thr_<A>_psw.json (DECISIVE: --planes psw) and thr_<A>_pswpedge.json (secondary). Status: $F/STATUS.txt; ends ZZMAPCAL-DONE / -FAIL-<why>.
export PATH=/usr/bin:/bin:$PATH; set -u
F=/home/nvidia/r8ns/r9e1/mapcal_fit; K=/home/nvidia/r8ns/r9e1/kit_mapcal_2408e921; PY=/home/nvidia/venvs/tanitad-train/bin/python
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $F/STATUS.txt; }
die() { ev "FAIL-$1 ${*:2}"; echo "ZZMAPCAL-FAIL-$1"; exit 3; }
penv() { local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO PYTHONPATH="$pp" \
  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token \
  REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES=$F/fit_remap.json "$@"; }
PP="$K/stack:$K/stack/scripts:$K/taniteval"; PPC="$PP:$K/stack/ops/r9e1_read/shim_scipy"
ev "QUEUE pid $$: waiting for $LOCK"
exec 9>$LOCK
flock -w 86400 9 || die LOCK-TIMEOUT "no lock in 24 h"
ev "LOCK-TAKEN"
napp=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
[ "$napp" = "0" ] || die GPU-BUSY "$napp compute app(s) under the lock"
for A in C0 C1 M1; do
  R=/home/nvidia/r8ns/r9e1/runs/r9e1-$A; O=$F/dump_$A
  [ -s $O/dump.json ] && { ev "DUMP-SKIP $A"; continue; }
  rm -rf $O; ev "DUMP-START $A"
  ( cd $K/stack && penv "$PP" $PY -X utf8 scripts/r9_e1_dump.py --ckpt $R/model_ema.pt --config $R/config.json --windows $F/fit_clips.json \
      --exclude-split $SPLIT --fit-k-per-ep 8 --require-ema --out $O > $F/dump_$A.log 2>&1 < /dev/null ); rc=$?
  n=$(ls $O/rast 2>/dev/null | grep -c 'npz$')
  [ -s $O/dump.json ] || die DUMP-$A "rc $rc (read $F/dump_$A.log)"
  ev "DUMP-DONE $A rc $rc rasters $n"
done
flock -u 9; exec 9>&-; ev "LOCK-RELEASED"
for A in C0 C1 M1; do
  for v in psw:psw pswpedge:psw,pedge; do t=${v%%:*}; pl=${v#*:}
    ( cd $K/stack && penv "$PPC" $PY -X utf8 scripts/r9_map_fit_thresholds.py fit --rast-dir $F/dump_$A/rast --dev-hold $SPLIT --arm $A \
        --planes $pl --out $F/thr_${A}_$t.json > $F/fit_${A}_$t.log 2>&1 < /dev/null ) || die FIT-$A-$t "(read $F/fit_${A}_$t.log)"
    ev "FIT-DONE $A $t $F/thr_${A}_$t.json"
  done
done
ev "DONE"; echo ZZMAPCAL-DONE
