#!/usr/bin/env bash
# e2_fitthr_diag.sh -- MM 4a3d05, 2026-11-12: DIAGNOSTIC for the E2 NO-GO's clause (a) -- NOT a verdict, nothing here changes E2_VERDICT.
# The E2 read decoded both arms' map heads at the fixed default cut (no --map-thresholds); REFCV9_DESIGN 2.1's decision rule is "per-channel, per-band
# thresholds fitted on a FIT split of TRAIN clips, never on eval", and E1's decisive map reads used exactly that (fitread_arm.sh: pswpedgeF).  This re-reads
# the map rows of the SAME E2 DEV dumps with each arm's OWN FIT thresholds, by E1's recipe step for step (fitread_arm.sh steps 2-3 + r9e1_arm_read.sh's
# --map-thresholds), with E2's own frozen read tree, lift-valid mask, gates, boot 2000 and seed 0.  Writes under runs/E2_READ_K2M1N20/diag_fitthr/.
#   per arm (K2 = r9e2-K2M1N20, K0 = r9e2-K0): FIT dump of E1's 1,112 TRAIN-DIAG windows under the Thor GPU lock (idle-GPU check) -> fit psw and
#   psw+pedge(f_tol), A1 check (psw levels EQUAL) -> r9_e1_eval.py on the E2 DEV dump with --map-thresholds -> EVAL_<A>_fitthr.json.
# Ends ZZE2FIT-DONEZZ or ZZE2FIT-ABORT-<why>ZZ (also in STATUS.txt).
export PATH=/usr/bin:/bin:$PATH; set -u
R=/home/nvidia/r8ns/r9e2; RT=$R/read_tree_cf653ca4bb47_lfix_e77c39d4_g55f14ef2; F1=/home/nvidia/r8ns/r9e1/mapcal_fit
OUT=$R/runs/E2_READ_K2M1N20; RD=$OUT/read; W=$OUT/diag_fitthr; PY=/home/nvidia/venvs/tanitad-train/bin/python
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
LV=/home/nvidia/r8ns/r9e1/read_v3/LIFT_VALID_M1.npz; GATES=/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json
FT=$RT/stack/scripts/r9_map_fit_thresholds.py; EV=$RT/stack/scripts/r9_e1_eval.py
mkdir -p $W; ST=$W/STATUS.txt
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZE2FIT-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
penv() { local pp="$1"; shift
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO PYTHONPATH="$pp" \
  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token \
  REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES=$W/fit_remap_e2.json "$@"; }
PP="$RT/stack:$RT/stack/scripts:$RT/taniteval"; PPC="$PP:$RT/stack/ops/r9e1_read/shim_scipy"
for f in "$FT" "$EV" "$LV" "$GATES" "$SPLIT" "$F1/fit_clips.json" "$F1/fit_remap.json" "$RD/dump_K2/perc_det.pkl" "$RD/dump_K0/perc_det.pkl" "$RD/EVAL_K2.json" "$RD/EVAL_K0.json"; do
  [ -s "$f" ] || die NO-INPUT "$f"; done
grep -q f_tol "$FT" || die OLD-KIT "$FT has no f_tol"
# E1's fit remap + the TRAIN-side files for the planner joins K2 / K0 carry (E1's rig arms had the planner off): the FIT clips are TRAIN clips
$PY -I -c 'import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
m.update({"--eval-labels": "/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz", "--speed-max-sidecar-v6-eval": "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl",
          "--r8-v9-labels-eval": "/home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz", "--r8-v9-eval-md5": "83e30bd3791f0767707eade5f4671759"})
json.dump(m, open(sys.argv[2], "w", encoding="utf-8"), indent=1)' $F1/fit_remap.json $W/fit_remap_e2.json || die REMAP "cannot write $W/fit_remap_e2.json"
[ "$(md5sum /home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz | cut -c1-32)" = 83e30bd3791f0767707eade5f4671759 ] || die V9-MD5 "train v9 labels md5"
ev START "arms K2 K0 tree $RT out $W"
for A in K2 K0; do
  case $A in K2) RUN=$R/runs/r9e2-K2M1N20;; K0) RUN=$R/runs/r9e2-K0;; esac
  [ -s $RUN/model_ema.pt ] && [ -s $RUN/config.json ] || die NO-CKPT "$RUN"
  O=$W/fitdump_$A
  if [ ! -s $O/dump.json ]; then
    exec 9>$LOCK; ev LOCK-WAIT "$A $LOCK"
    flock -w 43200 9 || die LOCK-TIMEOUT "no lock in 12 h"
    napp=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    [ "$napp" = "0" ] || { flock -u 9; die GPU-BUSY "$napp compute app(s) under the lock"; }
    [ -e $O ] && mv $O $O.partial_$(date +%s)
    ev DUMP-START "$A FIT windows (E1 fit_clips.json, 8 per episode, DEV/HOLD excluded)"
    ( cd $RT/stack && penv "$PP" $PY -X utf8 scripts/r9_e1_dump.py --ckpt $RUN/model_ema.pt --config $RUN/config.json --windows $F1/fit_clips.json \
        --exclude-split $SPLIT --fit-k-per-ep 8 --require-ema --out $O > $W/fitdump_$A.log 2>&1 < /dev/null ); rc=$?
    flock -u 9; exec 9>&-; ev LOCK-RELEASED "$A rc $rc"
    [ -s $O/dump.json ] || die DUMP "$A rc $rc (read $W/fitdump_$A.log)"
    ev DUMP-DONE "$A rasters $(ls $O/rast | grep -c 'npz$')"
  fi
  for pl in psw pswpedgeF; do
    out=$W/thr_${A}_$pl.json; [ -s $out ] && continue
    if [ $pl = psw ]; then extra=(--planes psw); else extra=(--planes psw,pedge --plane-criterion pedge=f_tol); fi
    ( cd $RT/stack && penv "$PPC" $PY -X utf8 "$FT" fit --rast-dir $O/rast --dev-hold $SPLIT --arm $A "${extra[@]}" --out $out > $out.log 2>&1 < /dev/null ) || die FIT "$A $pl (read $out.log)"
  done
  CHK=$($PY -I -c 'import json,sys
a=json.load(open(sys.argv[1],encoding="utf-8"))["planes"]; b=json.load(open(sys.argv[2],encoding="utf-8"))["planes"]
print(("EQUAL" if a["psw"]==b["psw"] else "DIFFER"), "psw", a["psw"], "pedge", b.get("pedge"))' $W/thr_${A}_psw.json $W/thr_${A}_pswpedgeF.json)
  [ "${CHK%% *}" = EQUAL ] || die A1-PSW "$A $CHK"
  ev FIT-DONE "$A ($CHK)"
  E=$W/EVAL_${A}_fitthr.json
  if [ ! -s $E ]; then
    ( cd $RD && penv "$PPC" CUDA_VISIBLE_DEVICES= $PY -X utf8 $EV --packs $RD/dump_$A/perc_det.pkl --rast-dir $RD/dump_$A/rast --windows $SPLIT --split dev \
        --gates-json $GATES --boot 2000 --seed 0 --lift-valid $LV --map-thresholds $W/thr_${A}_pswpedgeF.json --out $E > $W/eval_$A.log 2>&1 < /dev/null )
    [ -s $E ] || die EVAL "$A (read $W/eval_$A.log)"
  fi
  P=$($PY -I -c 'import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))["nearfield"]["ci"]["nf_swedge_prec_lv_5_20"]; print(round(d["point"],6), [round(d["lo"],4), round(d["hi"],4)])' $E)
  ev EVAL-DONE "$A fitted-threshold nf_swedge_prec_lv_5_20 = $P ($E)"
done
ev DONE "both arms re-read on their own FIT thresholds: $W"
