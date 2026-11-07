#!/bin/bash
# mapcal_m2a.sh -- MM 2026-11-06 (Thor): PREREG_M2A_FITTHR.md. (1) M2a's FIT dump under the Thor GPU lock (idle-GPU check), the same command as
# mapcal_q.sh; (2) CPU fits thr_M2a_psw.json (DECISIVE) / thr_M2a_pswpedge.json; (3) once mapthr_read.sh has written both FIT floors AND FLOOR v3 has
# re-read M2a at 128 (read_v3/arm_M2a exists, its re-read finished: the two would share arm_M2a/instance.lock), r9e1_arm_read.sh M2a on read_v3
# with arm thr_M2a_<V>, reference thr_M1_<V>, floor floor_thr_<V>.  Status: $F/M2A_STATUS.txt; ends ZZMAPCALM2A-DONE / -FAIL-<why>.
export PATH=/usr/bin:/bin:$PATH; set -u
F=/home/nvidia/r8ns/r9e1/mapcal_fit; K=/home/nvidia/r8ns/r9e1/kit_mapcal_2408e921; PY=/home/nvidia/venvs/tanitad-train/bin/python
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock; RD=/home/nvidia/r8ns/r9e1/read_v3
AR=$K/stack/ops/r9e1_read/r9e1_arm_read.sh; LV=$RD/LIFT_VALID_M1.npz; A=M2a
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $F/M2A_STATUS.txt; }
die() { ev "FAIL-$1 ${*:2}"; echo "ZZMAPCALM2A-FAIL-$1"; exit 3; }
penv() { local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO PYTHONPATH="$pp" \
  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token \
  REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES=$F/fit_remap.json "$@"; }
PP="$K/stack:$K/stack/scripts:$K/taniteval"; PPC="$PP:$K/stack/ops/r9e1_read/shim_scipy"
R=/home/nvidia/r8ns/r9e1/runs/r9e1-$A; O=$F/dump_$A
[ -s $R/model_ema.pt ] && [ -s $R/config.json ] || die NO-CKPT "$R"
if [ ! -s $O/dump.json ]; then
  ev "QUEUE pid $$: waiting for $LOCK"
  exec 9>$LOCK
  flock -w 86400 9 || die LOCK-TIMEOUT "no lock in 24 h"
  ev "LOCK-TAKEN"
  napp=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  [ "$napp" = "0" ] || die GPU-BUSY "$napp compute app(s) under the lock"
  [ -e $O ] && { mv $O $O.partial_$(date +%s); ev "PARTIAL-MOVED $O"; }
  ev "DUMP-START $A"
  ( cd $K/stack && penv "$PP" $PY -X utf8 scripts/r9_e1_dump.py --ckpt $R/model_ema.pt --config $R/config.json --windows $F/fit_clips.json \
      --exclude-split $SPLIT --fit-k-per-ep 8 --require-ema --out $O > $F/dump_$A.log 2>&1 < /dev/null ); rc=$?
  n=$(ls $O/rast 2>/dev/null | grep -c 'npz$')
  flock -u 9; exec 9>&-; ev "LOCK-RELEASED"
  [ -s $O/dump.json ] || die DUMP-$A "rc $rc (read $F/dump_$A.log)"
  ev "DUMP-DONE $A rc $rc rasters $n"
fi
for v in psw:psw pswpedge:psw,pedge; do t=${v%%:*}; pl=${v#*:}
  [ -s $F/thr_${A}_$t.json ] && continue
  ( cd $K/stack && penv "$PPC" $PY -X utf8 scripts/r9_map_fit_thresholds.py fit --rast-dir $O/rast --dev-hold $SPLIT --arm $A \
      --planes $pl --out $F/thr_${A}_$t.json > $F/fit_${A}_$t.log 2>&1 < /dev/null ) || die FIT-$A-$t "(read $F/fit_${A}_$t.log)"
  ev "FIT-DONE $A $t $F/thr_${A}_$t.json"
done
# (3) wait: both FIT floors (mapthr_read.sh) and FLOOR v3's own 128 re-read of M2a (floor_v3.sh logs "ARM M2a:")
t0=$(date +%s)
until grep -q 'DONE all variants' $F/MAPTHR_STATUS.txt 2>/dev/null && grep -q 'ARM M2a:' $RD/READ_STATUS.txt 2>/dev/null; do
  grep -q "ARM-SKIP $A:" $RD/READ_STATUS.txt 2>/dev/null && die V3-SKIPPED "FLOOR v3 skipped $A (read $RD/READ_STATUS.txt)"
  grep -q 'ABORT' $F/MAPTHR_STATUS.txt 2>/dev/null && die FLOOR-ABORTED "mapthr_read.sh aborted (read $F/MAPTHR_STATUS.txt)"
  [ $(( $(date +%s) - t0 )) -gt 172800 ] && die WAIT-TIMEOUT "floors / FLOOR v3 M2a re-read not ready after 48 h"
  sleep 120
done
[ -e $RD/arm_$A/dump_$A/dump.json ] && [ -s $RD/arm_$A/STATUS_SNAPSHOT.txt ] || die NO-V3-ARM "$RD/arm_$A has no dump or snapshot"
ev "READY floors + FLOOR v3 M2a re-read: $(grep 'ARM M2a:' $RD/READ_STATUS.txt | tail -1 | cut -c1-120)"
for V in psw pswpedge; do
  ( cd $RD && R9E1_TREE="$K" R9E1_READ_DIR="$RD" R9E1_STATUS="$RD/arm_$A/STATUS_SNAPSHOT.txt" R9E1_LIFT_VALID="$LV" R9E1_LOCK_MAX_S=60 \
      R9E1_MAP_THR_ARM="$F/thr_${A}_$V.json" R9E1_MAP_THR_REF="$F/thr_M1_$V.json" R9E1_MAP_THR_FLOOR_DIR="$F/floor_thr_$V" \
      bash "$AR" $A > "$F/arm_${A}_thr_$V.out" 2>&1 < /dev/null )
  ev "ARM $V $A: $(grep -ohE 'ZZR9E1ARMREAD-[A-Z-]+ZZ' $F/arm_${A}_thr_$V.out | tail -1)"
done
ev "DONE"; echo ZZMAPCALM2A-DONE
