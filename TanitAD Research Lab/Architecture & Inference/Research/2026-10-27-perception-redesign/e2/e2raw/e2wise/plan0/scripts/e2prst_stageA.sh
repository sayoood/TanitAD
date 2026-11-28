#!/usr/bin/env bash
# DERIVED by k2dc/make_prst_kit.py from the WISE-PLAN script of the same role: PLAN-RESET = S_plan at alpha 0 (model_wise_plan_a000.pt),
# pre-registered in e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM3.md (landed 93688696, before any PLAN-RESET number).
# e2prst_stageA.sh -- MM 4a3d05, 2026-11-17: E2-PLAN-RESET STAGE A (map only) = e2tail_stageA.sh with CK -> model_wise_plan_a000.pt (weight_interp.py a4faef50, 20ab9f54) and W -> runs/E2WISE_PLAN0/stageA.
# Pre-registered: e2/e2raw/PREREG_E2WISE_2026-11-17.md (ad9f1cbe) s3.1: full stage A, because reuse of the tail dumps was not proven tensor-by-tensor for the dump path.
# Arms: K2 = r9e2-K2DC, K0 = r9e2-K0. Read tree = K2DC's frozen E2 read tree. Per arm: FIT dump -> fit psw + psw/pedge(f_tol), A1 check -> DEV dump
# (the E2 roll's dump flags minus --require-ema) -> r9_e1_eval.py at the arm's own FIT thresholds (decides clause a) and at the default cut (reported).
# GUARD: every dump.json must record ckpt_md5 == md5 of the arm's $CK (model_wise_plan_a000.pt) and step 6000, else ABORT. Ends ZZE2PRSTA-DONEZZ / -ABORT-<why>ZZ.
export PATH=/usr/bin:/bin:$PATH; set -u
R=/home/nvidia/r8ns/r9e2; RT=$R/read_tree_9697fa4df004_k2dc; F1=/home/nvidia/r8ns/r9e1/mapcal_fit
W=$R/runs/E2WISE_PLAN0/stageA; PY=/home/nvidia/venvs/tanitad-train/bin/python; CK=model_wise_plan_a000.pt
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; S2=/home/nvidia/r8ns/perc8/inputs/s2_windows.json; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
LV=/home/nvidia/r8ns/r9e1/read_v3/LIFT_VALID_M1.npz; GATES=/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json
GATES_SHA=f4cb9a94f1dc3208599438e6748d19dfd12a0261dd7e476a8934b7ae53b94611
DEVREMAP=$RT/stack/ops/refcv8_eval_remap_a2_thor.json
FT=$RT/stack/scripts/r9_map_fit_thresholds.py; EV=$RT/stack/scripts/r9_e1_eval.py
mkdir -p $W; ST=$W/STATUS.txt
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZE2PRSTA-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
penv() { local pp="$1" rm="$2"; shift 2
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO PYTHONPATH="$pp" \
  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token \
  REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES="$rm" "$@"; }
PP="$RT/stack:$RT/stack/scripts:$RT/taniteval"; PPC="$PP:$RT/stack/ops/r9e1_read/shim_scipy"
for f in "$FT" "$EV" "$LV" "$GATES" "$SPLIT" "$S2" "$DEVREMAP" "$F1/fit_clips.json" "$F1/fit_remap.json"; do [ -s "$f" ] || die NO-INPUT "$f"; done
[ "$(sha256sum $GATES | cut -d' ' -f1)" = "$GATES_SHA" ] || die GATES-SHA "$GATES"
grep -q f_tol "$FT" || die OLD-KIT "$FT has no f_tol"
$PY -I -c 'import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
m.update({"--eval-labels": "/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz", "--speed-max-sidecar-v6-eval": "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl",
          "--r8-v9-labels-eval": "/home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz", "--r8-v9-eval-md5": "83e30bd3791f0767707eade5f4671759"})
json.dump(m, open(sys.argv[2], "w", encoding="utf-8"), indent=1)' $F1/fit_remap.json $W/fit_remap_e2.json || die REMAP "cannot write $W/fit_remap_e2.json"
cmp -s $W/fit_remap_e2.json $R/runs/E2_READ_K2M1N20/diag_fitthr/fit_remap_e2.json || die REMAP-DIFF "fit remap differs from the 11-12 diag's"
[ "$(md5sum /home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz | cut -c1-32)" = 83e30bd3791f0767707eade5f4671759 ] || die V9-MD5 "train v9 labels md5"
tailguard() {   # $1 = dump dir, $2 = the tail file: ckpt_md5 must equal it and step 6000
  $PY -I -c 'import json,sys,hashlib
d=json.load(open(sys.argv[1]+"/dump.json",encoding="utf-8"))
h=hashlib.md5(open(sys.argv[2],"rb").read()).hexdigest()
ok=(d.get("ckpt_md5")==h) and int(d.get("step",-1))==6000
print(("TAIL-OK" if ok else "TAIL-BAD"), "kind", d.get("ckpt_kind"), "ema", d.get("weights_are_ema"), "step", d.get("step"), "md5", d.get("ckpt_md5"), "file", h)
sys.exit(0 if ok else 1)' "$1" "$2"; }
gpu_dump() {   # $1 = arm, $2 = out dir, $3 = log, $4 = remap, rest = r9_e1_dump.py args
  local A="$1" O="$2" LG="$3" RM="$4"; shift 4
  if [ -s $O/dump.json ]; then ev DUMP-SKIP "$A $O exists"; return 0; fi
  exec 9>$LOCK; ev LOCK-WAIT "$A $LOCK"
  flock -w 43200 9 || die LOCK-TIMEOUT "no lock in 12 h"
  napp=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  [ "$napp" = "0" ] || { flock -u 9; die GPU-BUSY "$napp compute app(s) under the lock"; }
  [ -e $O ] && mv $O $O.partial_$(date +%s)
  ev DUMP-START "$A -> $O"
  ( cd $RT/stack && penv "$PP" "$RM" $PY -X utf8 scripts/r9_e1_dump.py "$@" --out $O > $LG 2>&1 < /dev/null ); rc=$?
  flock -u 9; exec 9>&-; ev LOCK-RELEASED "$A rc $rc"
  [ -s $O/dump.json ] || die DUMP "$A rc $rc (read $LG)"
}
ev START "E2-PLAN-RESET stage A arms K2=r9e2-K2DC K0=r9e2-K0 $CK tree $RT out $W (prereg e2/e2raw/PREREG_E2WISE_2026-11-17.md)"
for A in K2 K0; do
  case $A in K2) RUN=$R/runs/r9e2-K2DC;; K0) RUN=$R/runs/r9e2-K0;; esac
  [ -s $RUN/$CK ] && [ -s $RUN/config.json ] || die NO-CKPT "$RUN/$CK"
  FD=$W/fitdump_$A; DD=$W/devdump_$A
  gpu_dump $A $FD $W/fitdump_$A.log $W/fit_remap_e2.json --ckpt $RUN/$CK --config $RUN/config.json --windows $F1/fit_clips.json --exclude-split $SPLIT --fit-k-per-ep 8
  G=$(tailguard $FD $RUN/$CK) || die TAIL-GUARD "$A FIT dump: $G"; ev FITDUMP-DONE "$A rasters $(ls $FD/rast | grep -c 'npz$') ($G)"
  for pl in psw pswpedgeF; do
    out=$W/thr_${A}wise_$pl.json; [ -s $out ] && continue
    if [ $pl = psw ]; then extra=(--planes psw); else extra=(--planes psw,pedge --plane-criterion pedge=f_tol); fi
    ( cd $RT/stack && penv "$PPC" "$W/fit_remap_e2.json" $PY -X utf8 "$FT" fit --rast-dir $FD/rast --dev-hold $SPLIT --arm $A "${extra[@]}" --out $out > $out.log 2>&1 < /dev/null ) || die FIT "$A $pl (read $out.log)"
  done
  CHK=$($PY -I -c 'import json,sys
a=json.load(open(sys.argv[1],encoding="utf-8"))["planes"]; b=json.load(open(sys.argv[2],encoding="utf-8"))["planes"]
print(("EQUAL" if a["psw"]==b["psw"] else "DIFFER"), "psw", a["psw"], "pedge", b.get("pedge"))' $W/thr_${A}wise_psw.json $W/thr_${A}wise_pswpedgeF.json)
  [ "${CHK%% *}" = EQUAL ] || die A1-PSW "$A $CHK"
  ev FIT-DONE "$A ($CHK)"
  gpu_dump $A $DD $W/devdump_$A.log $DEVREMAP --ckpt $RUN/$CK --config $RUN/config.json --windows $SPLIT --split dev --s2-windows $S2
  G=$(tailguard $DD $RUN/$CK) || die TAIL-GUARD "$A DEV dump: $G"; ev DEVDUMP-DONE "$A ($G)"
  for mode in fitthr default; do
    E=$W/EVAL_${A}wise_$mode.json; [ -s $E ] && continue
    if [ $mode = fitthr ]; then mt=(--map-thresholds $W/thr_${A}wise_pswpedgeF.json); else mt=(); fi
    ( cd $W && penv "$PPC" "$DEVREMAP" CUDA_VISIBLE_DEVICES= $PY -X utf8 $EV --packs $DD/perc_det.pkl --rast-dir $DD/rast --windows $SPLIT --split dev \
        --gates-json $GATES --boot 2000 --seed 0 --lift-valid $LV ${mt[@]+"${mt[@]}"} --out $E > $W/eval_${A}wise_$mode.log 2>&1 < /dev/null )
    [ -s $E ] || die EVAL "$A $mode (read $W/eval_${A}wise_$mode.log)"
    P=$($PY -I -c 'import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))["nearfield"]["ci"]["nf_swedge_prec_lv_5_20"]; print(round(d["point"],6), [round(d["lo"],4), round(d["hi"],4)])' $E)
    ev EVAL-DONE "$A $mode nf_swedge_prec_lv_5_20 = $P ($E)"
  done
done
ev DONE "stage A: both arms read on PLAN-RESET (alpha 0.5) weights at their own FIT thresholds and the default cut: $W"
