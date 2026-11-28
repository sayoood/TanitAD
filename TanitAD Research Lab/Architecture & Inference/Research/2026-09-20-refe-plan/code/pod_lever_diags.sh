#!/bin/bash
# The three lever A/Bs ON THE POD, in pod_dataprep.sh's environment, SEQUENTIALLY and small:
# the pod's CPU quota is saturated by the rank-0 shards, so this borrows at most 3 cores ~15 min.
# ⛔ They run on the pod because exactness of the scorer ego view rests on torch CPU reductions over
# DIFFERENT tensor shapes -- a property of the machine's kernels, not of the source; the dev-box
# verdict does not transfer.
WORK=/workspace; PKG=$WORK/refe-plan; DATA=$WORK/data; DBS=$DATA/navtrain_dbs
source "$WORK/teacher_env.sh"
export NUPLAN_DATA_ROOT="$DATA" REFE_NUPLAN_DB_ROOT="$DBS"
export REFE_NAVTRAIN_YAML="$PKG/splits/navtrain.yaml" REFE_BACKBONE_ROOT="$DATA/backbones"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 REFE_SIM_HZ=10
OUT=$WORK/lever_diags; mkdir -p "$OUT"
cd "$PKG/refe" || exit 1
G="$DATA/refe_navtrain/r0_s[0-6]/targets_rank0.jsonl"
echo "ZZDIAG1_START $(date -u +%T) ZZ" >> "$OUT/run.log"
"$DRIVERL_EVAL_PYTHON" diag_scorer_ego_view.py --bank-glob "$G" --logs 2 --frames-per-log 4 \
  --work "$OUT/egoview" > "$OUT/egoview.log" 2>&1
echo "ZZDIAG1_EXIT $? $(date -u +%T) ZZ" >> "$OUT/run.log"
# ⛔ PARSE the row, never grep its first bytes: MEASURED 2026-09-24, `head -c 400 | grep log_name`
# returned EMPTY (the key sits later in the row), the rank-0 A/B then ran on ZERO logs, wrote 0 rows,
# and only the diag's own `ref_rows_nonempty` check kept that from reading as "identical".
LOG1=$("$DRIVERL_EVAL_PYTHON" -c "import json,sys; print(json.loads(open(sys.argv[1]).readline())['log_name'])" \
       "$OUT/egoview/bank/targets_rank0.jsonl")
[ -n "$LOG1" ] || { echo "ZZDIAG2_NOLOG ZZ" >> "$OUT/run.log"; exit 1; }
"$DRIVERL_EVAL_PYTHON" diag_lever_ab.py --log "$LOG1" --frames 6 --out "$OUT/mapstep" --device cpu \
  --arm REF:REFE_MAP_STEP_FAST=0 --arm FAST:REFE_MAP_STEP_FAST=1 \
  --arm MUT:REFE_MAP_STEP_FAST=1,REFE_MAP_STEP_MUTATE=segments_1mm > "$OUT/mapstep.log" 2>&1
echo "ZZDIAG2_EXIT $? $(date -u +%T) ZZ" >> "$OUT/run.log"
"$DRIVERL_EVAL_PYTHON" diag_lever_ab_aug.py --bank-glob "$G" --logs 2 --frames-per-log 3 \
  --out "$OUT/mapstep_aug" --device cpu \
  --arm REF:REFE_MAP_STEP_FAST=0 --arm FAST:REFE_MAP_STEP_FAST=1 \
  --arm MUT:REFE_MAP_STEP_FAST=1,REFE_MAP_STEP_MUTATE=route_1mm > "$OUT/mapstep_aug.log" 2>&1
echo "ZZDIAG3_EXIT $? $(date -u +%T) ZZ" >> "$OUT/run.log"
