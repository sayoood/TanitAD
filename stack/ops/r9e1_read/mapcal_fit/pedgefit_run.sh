#!/usr/bin/env bash
# pedgefit_run.sh <kit dir> -- MM 2026-11-06 (Thor, CPU only, no GPU lock). PREREG_M1_PEDGEFIT.md (+ AMENDMENT A1):
#   1 FIT    thr_<A>_pswpedgeF.json for A in C0 C1 M1 M2a: --planes psw,pedge --plane-criterion pedge=f_tol on mapcal_fit/dump_<A>/rast;
#            A1 check: the re-fitted psw levels MUST equal thr_<A>_psw.json's, else ABORT (no read).
#   2 FLOOR  EVAL_C0 / EVAL_C1 (DEV, read_v3 dumps) on their own pswpedgeF files + FLOOR v3's lift-valid mask -> floor_thr_pswpedgeF/FLOOR.json (OK)
#   3 M1     waits for the secondary pswpedge M1 read to finish (it holds read_v3/arm_M1/instance.lock), then r9e1_arm_read.sh M1 vs C0 (DECISIVE)
#   4 M2a    waits for mapcal_m2a.sh DONE (it holds read_v3/arm_M2a's lock for its reads), then r9e1_arm_read.sh M2a vs M1 (secondary)
# Ends ZZPEDGEFIT-DONEZZ or ZZPEDGEFIT-ABORT-<why>ZZ.  Log: mapcal_fit/PEDGEFIT_STATUS.txt.
export PATH=/usr/bin:/bin:$PATH; set -u
K="$1"
R=/home/nvidia/r8ns/r9e1; RD=$R/read_v3; F=$R/mapcal_fit; PY=/home/nvidia/venvs/tanitad-train/bin/python
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; GATES=/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json; LV=$RD/LIFT_VALID_M1.npz
EV=$K/stack/scripts/r9_e1_eval.py; FL=$K/stack/scripts/r9_e1_floor.py; FT=$K/stack/scripts/r9_map_fit_thresholds.py; AR=$K/stack/ops/r9e1_read/r9e1_arm_read.sh
PPC="$K/stack:$K/stack/scripts:$K/taniteval:$K/stack/ops/r9e1_read/shim_scipy"
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$F/PEDGEFIT_STATUS.txt"; echo "ZZPEDGEFIT-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
penv() { local rm="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO PYTHONPATH="$PPC" \
  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token \
  REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES="$rm" CUDA_VISIBLE_DEVICES= "$@"; }
for f in "$EV" "$FL" "$FT" "$AR" "$SPLIT" "$GATES" "$LV" "$RD/EVAL_C1_vs_C0.json"; do [ -s "$f" ] || die NO-INPUT "$f"; done
[ "$(grep -c $'\r' $AR)" = 0 ] && bash -n "$AR" || die CRLF-KIT "$AR"
grep -q 'f_tol' "$FT" || die OLD-KIT "$FT has no f_tol criterion"
ev START "kit $K; lift-valid $LV"
# ---- 1 FIT (+ the A1 psw check) ----
for A in C0 C1 M1 M2a; do
  O=$F/thr_${A}_pswpedgeF.json
  if [ ! -s "$O" ]; then
    ( cd $K/stack && penv "$F/fit_remap.json" $PY -X utf8 "$FT" fit --rast-dir $F/dump_$A/rast --dev-hold $SPLIT --arm $A --planes psw,pedge \
        --plane-criterion pedge=f_tol --out "$O" > $F/fit_${A}_pswpedgeF.log 2>&1 < /dev/null ) || die FIT-$A "(read $F/fit_${A}_pswpedgeF.log)"
  fi
  CHK=$($PY -I -c 'import json,sys
a=json.load(open(sys.argv[1],encoding="utf-8"))["planes"]; b=json.load(open(sys.argv[2],encoding="utf-8"))["planes"]
print(("EQUAL" if a["psw"]==b["psw"] else "DIFFER"), "psw", a["psw"], b["psw"], "pedge", a.get("pedge"))' "$O" "$F/thr_${A}_psw.json")
  [ "${CHK%% *}" = EQUAL ] || die A1-PSW-$A "re-fitted psw differs from thr_${A}_psw.json: $CHK"
  ev FIT-DONE "$A $O ($CHK)"
done
# ---- 2 FLOOR ----
RT=$F/floor_thr_pswpedgeF; mkdir -p $RT/logs
EVC="--windows $SPLIT --split dev --gates-json $GATES --boot 2000 --seed 0 --lift-valid $LV"
RM=$K/stack/ops/refcv8_eval_remap_a2_thor.json
for C in C0 C1; do
  outj=$RT/EVAL_$C.json
  [ -s "$outj" ] && { ev EVAL-SKIP "$C"; continue; }
  ( cd "$RT" && penv "$RM" $PY -X utf8 "$EV" --packs "$RD/dump_$C/perc_det.pkl" --rast-dir "$RD/dump_$C/rast" $EVC --map-thresholds "$F/thr_${C}_pswpedgeF.json" \
      --out "$outj" > "$RT/logs/eval_$C.out" 2> "$RT/logs/eval_$C.err" )
  [ -s "$outj" ] || die EVAL-$C "(read $RT/logs/eval_$C.err)"; ev EVAL-DONE "$C $outj"
done
if [ ! -s "$RT/FLOOR.json" ]; then
  ( cd "$RT" && penv "$RM" $PY -X utf8 "$FL" --eval-c0 "$RT/EVAL_C0.json" --eval-c1 "$RT/EVAL_C1.json" --paired "$RD/EVAL_C1_vs_C0.json" \
      --rast-c0 "$RD/dump_C0/rast" --rast-c1 "$RD/dump_C1/rast" --lift-valid "$LV" --map-thresholds-c0 "$F/thr_C0_pswpedgeF.json" \
      --map-thresholds-c1 "$F/thr_C1_pswpedgeF.json" --boot 2000 --seed 0 --out-dir "$RT" > "$RT/logs/floor.out" 2> "$RT/logs/floor.err" )
fi
FST="$($PY -I -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("status"))' "$RT/FLOOR.json" 2>/dev/null)"
[ "$FST" = OK ] || die FLOOR-NOT-OK "status ${FST:-MISSING} (read $RT/logs/floor.err)"
ev FLOOR-DONE "$RT/FLOOR.json ($(grep -c 'nf_edge03' $RT/FLOOR.json) nf_edge03 lines)"
# ---- 3 M1 vs C0 (DECISIVE) ----
t0=$(date +%s)
until grep -q 'ARM pswpedge-rerun M1' $F/MAPTHR_STATUS.txt 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 21600 ] && die WAIT-M1 "the secondary pswpedge M1 read has not finished in 6 h"; sleep 60; done
armread() { local A="$1" REF="$2"
  ( cd "$RD" && R9E1_TREE="$K" R9E1_READ_DIR="$RD" R9E1_STATUS="$RD/arm_$A/STATUS_SNAPSHOT.txt" R9E1_LIFT_VALID="$LV" R9E1_LOCK_MAX_S=60 \
      R9E1_MAP_THR_ARM="$F/thr_${A}_pswpedgeF.json" R9E1_MAP_THR_REF="$F/thr_${REF}_pswpedgeF.json" R9E1_MAP_THR_FLOOR_DIR="$RT" \
      bash "$AR" $A > "$F/arm_${A}_thr_pswpedgeF.out" 2>&1 < /dev/null )
  ev ARM "pswpedgeF $A: $(grep -ohE 'ZZR9E1ARMREAD-[A-Z-]+ZZ' $F/arm_${A}_thr_pswpedgeF.out | tail -1) $(grep -ohE 'thr_[0-9a-f]{8}_[0-9a-f]{8}' $F/arm_${A}_thr_pswpedgeF.out | tail -1)"; }
armread M1 C0
# ---- 4 M2a vs M1 (secondary) ----
t0=$(date +%s)
until grep -qE 'Z DONE$|FAIL-' $F/M2A_STATUS.txt 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 172800 ] && die WAIT-M2A "mapcal_m2a.sh not finished in 48 h"; sleep 120; done
grep -q 'FAIL-' $F/M2A_STATUS.txt && die M2A-QUEUE-FAILED "mapcal_m2a.sh failed (read $F/M2A_STATUS.txt)"
armread M2a M1
ev DONE "all reads"
