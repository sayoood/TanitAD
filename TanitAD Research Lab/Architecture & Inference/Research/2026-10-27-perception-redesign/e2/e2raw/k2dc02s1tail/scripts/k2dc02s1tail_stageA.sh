#!/usr/bin/env bash
# k2dc02s1tail_stageA.sh -- K2DC02s1-TAIL STAGE A (map only), pre-registered in e2/e2raw/PREREG_K2DC02S1_2026-11-20.md (d20cd6f3) BEFORE any K2DC02s1 tail number. Zero training.
# = e2tail/scripts/e2tail_stageA.sh for the K2 slot ONLY (K2 = r9e2-K2DC02s1 tail, built by k2dc02s1tail_build.sh into the staging run dir runs/E2TAIL_K2DC02s1/run_K2DC02s1),
# with K2DC02s1's own read tree (read_tree_e454e96c6a84_k2dc). K0's stage A outputs are NOT re-made: they are the banked E2-TAIL ones (e2tail/stageA/EVAL_K0tail_fitthr.json).
# Order: (0) the DELIBERATE REGRESSION (sec. 2.3): the tail guard must REFUSE a dump header whose ckpt_md5 is model_ema.pt's -- and must ACCEPT a header that is the tail's;
# if the guard does not go red the read does not start. (1) FIT dump -> fit psw + psw/pedge(f_tol), A1 check -> DEV dump -> r9_e1_eval.py at the FIT thresholds (clause a) and the default cut.
# GUARD: every dump.json must record ckpt_md5 == md5 of model_ema_tail5k6k.pt and step 6000, else ABORT. Ends ZZK2DC02S1TAILA-DONEZZ / -ABORT-<why>ZZ.
# Differences from k2dc02tail_stageA.sh (PREREG_K2DC02S1_2026-11-20.md sec. 2 item 3): ARM K2DC02s1 (run K2DC02s1, staging run dir runs/E2TAIL_K2DC02s1/run_K2DC02s1); the FIRST guard is that the trained
# arm exists; the read tree is read_tree_<K2S_TRAIN_COMMIT>_k2dc (the K2DC02s1 EMA read's tree); the deliberate-regression arm (the tail guard must REFUSE a model_ema.pt dump) is kept unchanged.
# DRY (K2S_DRY=1 or first arg DRY): every input guard runs, no file is written (the fit remap is compared on a pipe), no GPU lock, no dump; the commands are printed. Ends DRY-OK or the first ABORT.
export PATH=/usr/bin:/bin:$PATH; set -u
DRY=0; if [ "${K2S_DRY:-0}" = 1 ] || [ "${1:-}" = DRY ]; then DRY=1; fi
TC=${K2S_TRAIN_COMMIT:-}
R=/home/nvidia/r8ns/r9e2; RT=$R/read_tree_${TC}_k2dc; F1=/home/nvidia/r8ns/r9e1/mapcal_fit
W=$R/runs/E2TAIL_K2DC02s1/stageA; RUN=$R/runs/E2TAIL_K2DC02s1/run_K2DC02s1; RAWRUN=$R/runs/r9e2-K2DC02s1; PY=/home/nvidia/venvs/tanitad-train/bin/python; CK=model_ema_tail5k6k.pt
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; S2=/home/nvidia/r8ns/perc8/inputs/s2_windows.json; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
LV=/home/nvidia/r8ns/r9e1/read_v3/LIFT_VALID_M1.npz; GATES=/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json
GATES_SHA=f4cb9a94f1dc3208599438e6748d19dfd12a0261dd7e476a8934b7ae53b94611
DEVREMAP=$RT/stack/ops/refcv8_eval_remap_a2_thor.json
FT=$RT/stack/scripts/r9_map_fit_thresholds.py; EV=$RT/stack/scripts/r9_e1_eval.py
[ $DRY = 1 ] || mkdir -p $W; ST=$W/STATUS.txt
ev() { [ $DRY = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZK2DC02S1TAILA-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
# ---- guard 1 (the FIRST): the trained arm exists (K2DC02s1 has not trained when this kit is written) ----
[ -s $RAWRUN/model_ema.pt ] && [ -s $RAWRUN/ckpt_ema_5000.pt ] && [ -s $RAWRUN/config.json ] || die NO-RUN "$RAWRUN lacks model_ema.pt / ckpt_ema_5000.pt / config.json: K2DC02s1 has not trained"
[ ${#TC} = 12 ] || die NO-TRAIN-COMMIT "K2S_TRAIN_COMMIT (the 12-char commit K2DC02s1 trained from) is required, got '$TC'"
[ -d $RT/stack ] || die NO-TREE "$RT: build the EMA read tree first (e2read_launch_k2dc02s1.sh)"
penv() { local pp="$1" rm="$2"; shift 2
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO PYTHONPATH="$pp" \
  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token \
  REFCV6_KIT=/home/nvidia REFCV6_REMAP_OVERRIDES="$rm" "$@"; }
PP="$RT/stack:$RT/stack/scripts:$RT/taniteval"; PPC="$PP:$RT/stack/ops/r9e1_read/shim_scipy"
for f in "$FT" "$EV" "$LV" "$GATES" "$SPLIT" "$S2" "$DEVREMAP" "$F1/fit_clips.json" "$F1/fit_remap.json"; do [ -s "$f" ] || die NO-INPUT "$f"; done
[ "$(sha256sum $GATES | cut -d' ' -f1)" = "$GATES_SHA" ] || die GATES-SHA "$GATES"
grep -q f_tol "$FT" || die OLD-KIT "$FT has no f_tol"
REMAPPY='import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
m.update({"--eval-labels": "/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz", "--speed-max-sidecar-v6-eval": "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl",
          "--r8-v9-labels-eval": "/home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz", "--r8-v9-eval-md5": "83e30bd3791f0767707eade5f4671759"})
json.dump(m, open(sys.argv[2], "w", encoding="utf-8"), indent=1)'
if [ $DRY = 1 ]; then $PY -I -c "$REMAPPY" $F1/fit_remap.json /dev/stdout | cmp -s - $R/runs/E2_READ_K2M1N20/diag_fitthr/fit_remap_e2.json || die REMAP-DIFF "fit remap differs from the 11-12 diag's (dry: compared on a pipe)"
else
$PY -I -c "$REMAPPY" $F1/fit_remap.json $W/fit_remap_e2.json || die REMAP "cannot write $W/fit_remap_e2.json"
cmp -s $W/fit_remap_e2.json $R/runs/E2_READ_K2M1N20/diag_fitthr/fit_remap_e2.json || die REMAP-DIFF "fit remap differs from the 11-12 diag's"
fi
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
ev START "K2DC02s1-TAIL stage A, arm K2=r9e2-K2DC02s1 tail $CK in $RUN, tree $RT, out $W (prereg e2/e2raw/PREREG_K2DC02S1_2026-11-20.md)"
# ---- (0) DELIBERATE REGRESSION (prereg sec. 2.3): the guard must go RED on a dump made from model_ema.pt, and GREEN on the tail's ----
[ -s $RUN/$CK ] && [ -s $RUN/config.json ] && [ -s $RAWRUN/model_ema.pt ] || die NO-CKPT "$RUN/$CK or $RAWRUN/model_ema.pt"
if [ $DRY = 1 ]; then
  echo "[dry] would: (0) deliberate regression: the tail guard must REFUSE a header whose ckpt_md5 is $RAWRUN/model_ema.pt and ACCEPT the tail's, step 6000"
  echo "[dry] would: take $LOCK, require 0 GPU compute apps, then $PY -X utf8 scripts/r9_e1_dump.py --ckpt $RUN/$CK --config $RUN/config.json --windows $F1/fit_clips.json --exclude-split $SPLIT --fit-k-per-ep 8 --out $W/fitdump_K2"
  echo "[dry] would: r9_map_fit_thresholds.py fit (psw; psw,pedge f_tol), A1 check, DEV dump (--split dev), r9_e1_eval.py at the fit thresholds and the default cut -> $W/EVAL_K2tail_{fitthr,default}.json; tail guard (md5 == $RUN/$CK, step 6000) on both dumps"
  ev DRY-OK "all guards pass; nothing run"; exit 0
fi
RG=$W/regression_guard; mkdir -p $RG/bad $RG/good
$PY -I -c 'import json,sys,hashlib
md5=lambda p: hashlib.md5(open(p,"rb").read()).hexdigest()
json.dump({"ckpt_md5": md5(sys.argv[1]), "step": 6000, "ckpt_kind": "ema_model_only", "weights_are_ema": True, "synthetic": "DELIBERATE-REGRESSION: header of a dump made from model_ema.pt"}, open(sys.argv[3]+"/bad/dump.json","w"))
json.dump({"ckpt_md5": md5(sys.argv[2]), "step": 6000, "ckpt_kind": "ema_model_only", "weights_are_ema": True, "synthetic": "CONTROL: header of a dump made from the tail"}, open(sys.argv[3]+"/good/dump.json","w"))' $RAWRUN/model_ema.pt $RUN/$CK $RG || die REGRESSION "cannot write the synthetic headers"
GB=$(tailguard $RG/bad $RUN/$CK); RB=$?; GG=$(tailguard $RG/good $RUN/$CK); RG2=$?
[ "$RB" = 1 ] || die REGRESSION-NOT-RED "the guard did NOT refuse a model_ema.pt header (rc $RB: $GB)"
[ "$RG2" = 0 ] || die REGRESSION-CONTROL "the guard refused the tail's own header (rc $RG2: $GG)"
ev REGRESSION-RED "guard REFUSED the model_ema.pt header (rc 1: $GB); control ACCEPTED the tail header (rc 0: $GG)"
for A in K2; do
  [ -s $RUN/$CK ] && [ -s $RUN/config.json ] || die NO-CKPT "$RUN/$CK"
  FD=$W/fitdump_$A; DD=$W/devdump_$A
  gpu_dump $A $FD $W/fitdump_$A.log $W/fit_remap_e2.json --ckpt $RUN/$CK --config $RUN/config.json --windows $F1/fit_clips.json --exclude-split $SPLIT --fit-k-per-ep 8
  G=$(tailguard $FD $RUN/$CK) || die TAIL-GUARD "$A FIT dump: $G"; ev FITDUMP-DONE "$A rasters $(ls $FD/rast | grep -c 'npz$') ($G)"
  for pl in psw pswpedgeF; do
    out=$W/thr_${A}tail_$pl.json; [ -s $out ] && continue
    if [ $pl = psw ]; then extra=(--planes psw); else extra=(--planes psw,pedge --plane-criterion pedge=f_tol); fi
    ( cd $RT/stack && penv "$PPC" "$W/fit_remap_e2.json" $PY -X utf8 "$FT" fit --rast-dir $FD/rast --dev-hold $SPLIT --arm $A "${extra[@]}" --out $out > $out.log 2>&1 < /dev/null ) || die FIT "$A $pl (read $out.log)"
  done
  CHK=$($PY -I -c 'import json,sys
a=json.load(open(sys.argv[1],encoding="utf-8"))["planes"]; b=json.load(open(sys.argv[2],encoding="utf-8"))["planes"]
print(("EQUAL" if a["psw"]==b["psw"] else "DIFFER"), "psw", a["psw"], "pedge", b.get("pedge"))' $W/thr_${A}tail_psw.json $W/thr_${A}tail_pswpedgeF.json)
  [ "${CHK%% *}" = EQUAL ] || die A1-PSW "$A $CHK"
  ev FIT-DONE "$A ($CHK)"
  gpu_dump $A $DD $W/devdump_$A.log $DEVREMAP --ckpt $RUN/$CK --config $RUN/config.json --windows $SPLIT --split dev --s2-windows $S2
  G=$(tailguard $DD $RUN/$CK) || die TAIL-GUARD "$A DEV dump: $G"; ev DEVDUMP-DONE "$A ($G)"
  for mode in fitthr default; do
    E=$W/EVAL_${A}tail_$mode.json; [ -s $E ] && continue
    if [ $mode = fitthr ]; then mt=(--map-thresholds $W/thr_${A}tail_pswpedgeF.json); else mt=(); fi
    ( cd $W && penv "$PPC" "$DEVREMAP" CUDA_VISIBLE_DEVICES= $PY -X utf8 $EV --packs $DD/perc_det.pkl --rast-dir $DD/rast --windows $SPLIT --split dev \
        --gates-json $GATES --boot 2000 --seed 0 --lift-valid $LV ${mt[@]+"${mt[@]}"} --out $E > $W/eval_${A}tail_$mode.log 2>&1 < /dev/null )
    [ -s $E ] || die EVAL "$A $mode (read $W/eval_${A}tail_$mode.log)"
    P=$($PY -I -c 'import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))["nearfield"]["ci"]["nf_swedge_prec_lv_5_20"]; print(round(d["point"],6), [round(d["lo"],4), round(d["hi"],4)])' $E)
    ev EVAL-DONE "$A $mode nf_swedge_prec_lv_5_20 = $P ($E)"
  done
done
ev DONE "stage A: K2DC02s1 read on TAIL-EMA weights at their own FIT thresholds and the default cut: $W"
