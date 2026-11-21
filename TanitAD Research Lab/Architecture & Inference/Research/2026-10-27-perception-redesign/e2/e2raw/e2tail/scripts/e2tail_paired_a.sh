#!/usr/bin/env bash
# e2tail_paired_a.sh -- MM 4a3d05, 2026-11-17 (= e2raw_paired_a.sh on the TAIL-EMA stage A): DIAGNOSTIC (decides nothing; the prereg's clause (a) is the point rule). The PAIRED episode-cluster
# bootstrap of E2-TAIL clause (a): K2DC tail - K0 tail on nf_swedge_prec_lv_5_20 at each arm's OWN FIT thresholds, by the E1 floor tool
# (r9_e1_floor.py: c0 = K0tail, c1 = K2DCtail, delta = c1 - c0), on stage A's own DEV dumps. CPU only, nice 10. Ends ZZE2TAILP-DONEZZ / ZZE2TAILP-ABORT-<why>ZZ.
export PATH=/usr/bin:/bin:$PATH; set -u
R=/home/nvidia/r8ns/r9e2; RT=$R/read_tree_9697fa4df004_k2dc; A=$R/runs/E2TAIL_K2DC/stageA; O=$A/paired_diag; PY=/home/nvidia/venvs/tanitad-train/bin/python
SPLIT=/home/nvidia/r8ns/r9dump/DEV_HOLD_SPLIT.json; LV=/home/nvidia/r8ns/r9e1/read_v3/LIFT_VALID_M1.npz
GATES=/home/nvidia/r8ns/perc8/out/T_nosync/refcv8_det_presence_gates_train.json
mkdir -p $O; ST=$O/STATUS.txt
ev() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$ST"; echo "ZZE2TAILP-$1ZZ $(date -u +%H:%M:%SZ) ${*:2}"; }
die() { ev "ABORT-$1" "${*:2}"; exit 3; }
PPC="$RT/stack:$RT/stack/scripts:$RT/taniteval:$RT/stack/ops/r9e1_read/shim_scipy"
penv() { env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u REFCV7_KIT -u REFCV7_REMAP_OVERRIDES -u REFCV7_REPO PYTHONPATH="$PPC" \
  PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token \
  CUDA_VISIBLE_DEVICES= REFCV6_KIT=/home/nvidia "$@"; }
for f in $A/devdump_K2/perc_det.pkl $A/devdump_K0/perc_det.pkl $A/EVAL_K2tail_fitthr.json $A/EVAL_K0tail_fitthr.json $A/thr_K2tail_pswpedgeF.json $A/thr_K0tail_pswpedgeF.json $LV $GATES $SPLIT; do
  [ -s $f ] || die NO-INPUT "$f"; done
ev START "paired (a): c1 = K2DC tail, c0 = K0 tail, own FIT thresholds, out $O"
P=$O/EVAL_K2tail_vs_K0tail.json
if [ ! -s $P ]; then
  ( cd $O && penv nice -n 10 $PY -X utf8 $RT/stack/scripts/r9_e1_eval.py --packs $A/devdump_K2/perc_det.pkl --packs-b $A/devdump_K0/perc_det.pkl \
      --windows $SPLIT --split dev --gates-json $GATES --boot 2000 --seed 0 --out $P > $O/paired_eval.log 2>&1 < /dev/null )
  [ -s $P ] || die PAIRED-EVAL "read $O/paired_eval.log"
fi
ev PAIRED-EVAL-DONE "$P"
( cd $O && penv nice -n 10 $PY -X utf8 $RT/stack/scripts/r9_e1_floor.py --eval-c0 $A/EVAL_K0tail_fitthr.json --eval-c1 $A/EVAL_K2tail_fitthr.json --paired $P \
    --rast-c0 $A/devdump_K0/rast --rast-c1 $A/devdump_K2/rast --lift-valid $LV --map-thresholds-c0 $A/thr_K0tail_pswpedgeF.json \
    --map-thresholds-c1 $A/thr_K2tail_pswpedgeF.json --boot 2000 --seed 0 --out-dir $O > $O/floor.log 2>&1 < /dev/null ); rc=$?
[ -s $O/FLOOR.json ] || die FLOOR "rc $rc (read $O/floor.log)"
L=$($PY -I -c 'import json,sys
f=json.load(open(sys.argv[1],encoding="utf-8")); r=f.get("rows",{})
ks=[k for k in r if "nf_swedge_prec_lv_5_20" in k] or [k for k in r if "swedge_prec_lv" in k]
print(f.get("status"), "; ".join("%s: %s" % (k, json.dumps({x: r[k].get(x) for x in ("c0","c1","delta","lo","hi","separated") if x in r[k]})) for k in ks[:3]))' $O/FLOOR.json)
ev DONE "rc $rc $L"
