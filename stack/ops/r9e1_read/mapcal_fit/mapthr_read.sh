#!/usr/bin/env bash
# mapthr_read.sh -- MM 2026-11-06 (Thor-side, CPU ONLY, no GPU lock).  PREREG_M1_FITTHR.md runbook steps 5-6 (landed_hold_r9_mapcal/HANDOFF.md):
#   per variant V in psw (DECISIVE) then pswpedge (secondary):
#     5 FLOOR  EVAL_C0 / EVAL_C1 on read_v3's DEV dumps, each on its OWN FIT thresholds thr_<C>_<V>.json, WITH FLOOR v3's lift-valid mask;
#              r9_e1_floor.py --map-thresholds-c0/-c1 --lift-valid -> mapcal_fit/floor_thr_<V>/FLOOR.json (status OK required)
#     6 ARM    r9e1_arm_read.sh M1 (kit_mapcal_2408e921) with R9E1_MAP_THR_ARM=thr_M1_<V> R9E1_MAP_THR_REF=thr_C0_<V> R9E1_MAP_THR_FLOOR_DIR=floor_thr_<V>
#              on read_v3 (dumps exist => no GPU, no lock) -> read_v3/arm_M1/thr_<sha8>_<sha8>/VERDICT_M1.{json,md}
# Ends ZZMAPTHR-DONEZZ or ZZMAPTHR-ABORT-<why>ZZ.  Log: mapcal_fit/MAPTHR_STATUS.txt.
export PATH=/usr/bin:/bin:$PATH; set -u
R=/home/nvidia/r8ns/r9e1; RD=$R/read_v3; F=$R/mapcal_fit; K=$R/kit_mapcal_2408e921
PY=/home/nvidia/venvs/tanitad-train/bin/python; EV=$K/stack/scripts/r9_e1_eval.py; FL=$K/stack/scripts/r9_e1_floor.py; AR=$K/stack/ops/r9e1_read/r9e1_arm_read.sh
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; GATES=/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json; LV=$RD/LIFT_VALID_M1.npz
PYPATH_CPU="$K/stack:$K/stack/scripts:$K/taniteval:$K/stack/ops/r9e1_read/shim_scipy"     # the venv has no scipy (read_v2's SCIPY-SHIM line)
VARIANTS="${MAPTHR_VARIANTS:-psw pswpedge}"
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$F/MAPTHR_STATUS.txt"; echo "ZZMAPTHR-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
penv() { local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO \
  PYTHONPATH="$pp" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  HF_TOKEN_PATH=/nonexistent/hf_token REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES="$K/stack/ops/refcv8_eval_remap_a2_thor.json" CUDA_VISIBLE_DEVICES= "$@"; }
for f in "$EV" "$FL" "$AR" "$SPLIT" "$GATES" "$LV" "$RD/FLOOR.json" "$RD/EVAL_C1_vs_C0.json" "$RD/arm_M1/STATUS_SNAPSHOT.txt"; do [ -s "$f" ] || die NO-INPUT "$f"; done
[ -e "$RD/arm_M1/dump_M1/dump.json" ] || die NO-INPUT "$RD/arm_M1/dump_M1/dump.json"
ev START "kit $K; read dir $RD; lift-valid $LV ($(sha256sum $LV | cut -c1-12)); variants [$VARIANTS]"
for V in $VARIANTS; do
  RT=$F/floor_thr_$V; mkdir -p "$RT/logs"
  T0=$F/thr_C0_$V.json; T1=$F/thr_C1_$V.json; TM=$F/thr_M1_$V.json
  for f in "$T0" "$T1" "$TM"; do [ -s "$f" ] || die NO-THR "$f"; done
  EVC="--windows $SPLIT --split dev --gates-json $GATES --boot 2000 --seed 0 --lift-valid $LV"
  for C in C0 C1; do
    TC=$F/thr_${C}_$V.json; outj=$RT/EVAL_$C.json
    if [ -s "$outj" ]; then ev EVAL-SKIP "$V $C: $outj exists"; continue; fi
    ( cd "$RT" && penv "$PYPATH_CPU" "$PY" -X utf8 "$EV" --packs "$RD/dump_$C/perc_det.pkl" --rast-dir "$RD/dump_$C/rast" $EVC --map-thresholds "$TC" \
        --out "$outj" > "$RT/logs/eval_$C.out" 2> "$RT/logs/eval_$C.err" )
    [ -s "$outj" ] || die EVAL-FAILED "$V $C (read $RT/logs/eval_$C.err)"; ev EVAL-DONE "$V $C $outj"
  done
  if [ ! -s "$RT/FLOOR.json" ]; then
    ( cd "$RT" && penv "$PYPATH_CPU" "$PY" -X utf8 "$FL" --eval-c0 "$RT/EVAL_C0.json" --eval-c1 "$RT/EVAL_C1.json" --paired "$RD/EVAL_C1_vs_C0.json" \
        --rast-c0 "$RD/dump_C0/rast" --rast-c1 "$RD/dump_C1/rast" --lift-valid "$LV" --map-thresholds-c0 "$T0" --map-thresholds-c1 "$T1" \
        --boot 2000 --seed 0 --out-dir "$RT" > "$RT/logs/floor.out" 2> "$RT/logs/floor.err" )
  fi
  FST="$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("status"))' "$RT/FLOOR.json" 2>/dev/null)"
  [ "$FST" = OK ] || die FLOOR-NOT-OK "$V status ${FST:-MISSING} (read $RT/logs/floor.err)"
  ev FLOOR-DONE "$V $RT/FLOOR.json"
  ( cd "$RD" && R9E1_TREE="$K" R9E1_READ_DIR="$RD" R9E1_STATUS="$RD/arm_M1/STATUS_SNAPSHOT.txt" R9E1_LIFT_VALID="$LV" R9E1_LOCK_MAX_S=60 \
      R9E1_MAP_THR_ARM="$TM" R9E1_MAP_THR_REF="$T0" R9E1_MAP_THR_FLOOR_DIR="$RT" bash "$AR" M1 > "$F/arm_M1_thr_$V.out" 2>&1 < /dev/null )
  ev ARM "$V M1: $(grep -ohE 'ZZR9E1ARMREAD-[A-Z-]+ZZ' $F/arm_M1_thr_$V.out | tail -1) $(grep -ohE 'thr_[0-9a-f]{8}_[0-9a-f]{8}' $F/arm_M1_thr_$V.out | tail -1)"
done
ev DONE "all variants read"
