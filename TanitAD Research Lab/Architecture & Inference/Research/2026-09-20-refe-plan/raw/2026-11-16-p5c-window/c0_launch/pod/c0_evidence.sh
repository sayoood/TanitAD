#!/bin/bash
# C0 gate evidence, manual form of p5c_pod.sh stages lab + mem (egr is NOT run here: it needs an HF token file the pod does not hold).
set -u
D=/workspace/data
ROOT=$D/refe_p5c
PKG=$ROOT/code_33824420
RUN=$PKG/refe
PY=/workspace/venv-teacher/bin/python
export PYTHONPATH=$RUN
export REFE_BACKBONE_ROOT=$D/backbones PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export NUPLAN_MAPS_ROOT=$D/nuplan-maps/nuplan-maps-v1.0
cd "$RUN" || exit 9
ARGS="$(cat $ROOT/ARGS_C0.txt)"
OUT=$ROOT/evidence_c0
mkdir -p "$OUT"
echo "START $(date -u +%FT%T)" > $OUT/status.txt
# ---- lab
TOKS="$(env -u HF_TOKEN $PY p6_prep.py sample-tokens --cache $D/refe_navtrain/metric_cache --n 128 | sed -n 's/^ZZTOKENS //p')"
[ -n "$TOKS" ] || { echo "FAIL lab: no tokens" >> $OUT/status.txt; exit 1; }
env -u HF_TOKEN -u DRIVERL_EVAL_ROUTE_LANE_RANK $PY diag_p6_labeller_identity.py --cache $D/refe_navtrain/metric_cache --db-root $D/navtrain_dbs --map-root "$NUPLAN_MAPS_ROOT" --tokens "$TOKS" --workers 13 --chunks 2 --n-sets 256 --out $OUT/lab.json > $OUT/lab.log 2>&1
grep -q "^ZZLAB_IDENTITY_PASS" $OUT/lab.log || { echo "FAIL lab: identity (lab.log)" >> $OUT/status.txt; exit 1; }
echo "LAB_OK $(date -u +%FT%T) $(grep '^ZZLAB_THROUGHPUT' $OUT/lab.log | tail -n 1)" >> $OUT/status.txt
# ---- mem (C0's own argv: same model and shapes as C1; no table)
rm -rf $OUT/mem_run
M="$(env -u HF_TOKEN $PY p6_prep.py mem-argv --args "$ARGS" --out $OUT/mem_run | sed -n 's/^ZZMEM_ARGV //p')"
[ -n "$M" ] || { echo "FAIL mem: no mem argv" >> $OUT/status.txt; exit 1; }
echo "$M" > $OUT/mem_argv.txt
env -u HF_TOKEN $PY train.py $M > $OUT/mem.log 2>&1
grep -q "^TRAIN_DONE" $OUT/mem.log || { echo "FAIL mem: no TRAIN_DONE (mem.log)" >> $OUT/status.txt; exit 1; }
{ [ -s $OUT/mem_run/metrics.jsonl ] && [ -s $OUT/mem_run/ckpt_last.pt ]; } || { echo "FAIL mem: no metrics/ckpt" >> $OUT/status.txt; exit 1; }
cp $OUT/mem_run/metrics.jsonl $OUT/mem_metrics.jsonl
echo "MEM_OK $(date -u +%FT%T)" >> $OUT/status.txt
echo "ZZC0_EVIDENCE_DONE" >> $OUT/status.txt
