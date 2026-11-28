#!/usr/bin/env bash
# read_q1.sh : dev-box side of note PREREG_V8F_PROOF_ADD40_N7 (source cand-cvq). The form of read_i9.sh: pulls the dumps of the runs of both pairs (LS: bases 0 1, then LSD: bases 0 1), runs the headway and the
# family reader on the pairs the note names, then read_q1.py (controls q1 to q6 first; names only if all pass, per pair), then the combine step (-CONFIRMED only if the second pair reads the same name).
# After chain_q1.sh has ended (its deploy_q_*.out files carry the compare lines). Every Python job runs under the capped launcher (ops/pycap.sh, PYCAP_MB=2000, one job at a time).
# Starts nothing on the pod GPU. Deletes nothing. Last line: ZZ READ-Q1 DONE.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
PULL=$H/imagplan_q/pull_t1
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull,$W/imagplan_q/pull_t1"
mkdir -p "$PULL"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYCAP_MB=${PYCAP_MB:-2000}
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
one_pair() { # arm records-folder deploy-names(2)
  local A=$1 REC=$2 D0=$3 D1=$4 k B t r
  local R=$H/imagplan_q/read_q1/$A RW=$W/imagplan_q/read_q1/$A
  mkdir -p "$R"
  for k in 0 1; do
    B=${A}_s$k
    for t in IPo IPc IPl IPcq IPcqid IPcq0; do
      r=${t}q_$B
      [ -f "$PULL/$r/PULLED.txt" ] || Q=imagplan bash spatial/q3/pull_spatial_t1.sh "$r" 2>&1 | tail -n 1 | cut -c1-160
    done
  done
  local n; n() { echo "${1}q_$A"; }
  local PAIRS="$(n IPl):$(n IPo),$(n IPc):$(n IPo),$(n IPc):$(n IPl),$(n IPcq):$(n IPc),$(n IPcq):$(n IPl),$(n IPcq):$(n IPo),$(n IPcq0):$(n IPc),$(n IPcqid):$(n IPc)"
  ( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$RW/hw" > "$R/hw.log" 2>&1 ); echo "[$(ts)] $A headway rc $?"
  ( cd $H/spatial/q3 && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$RW/q3" > "$R/q3.log" 2>&1 ); echo "[$(ts)] $A families rc $?"
  "$PY" imagplan_q/read_q1.py --arm $A --sfx q --bases 0,1 --pull imagplan_q/pull --read-dir "imagplan_q/read_q1/$A" --seq-outs imagplan_q/deploy_q_$D0.out imagplan_q/deploy_q_$D1.out \
      --records "$REC" --out "imagplan_q/read_q1/$A/READ_Q1_$A.json" > "$R/read_q1.log" 2>&1
  tail -n 30 "$R/read_q1.log" | cut -c1-330
}
one_pair LS qhead_n1/dose/pulled_ls/heads q1ls0 q1ls1
one_pair LSD qhead_n1/dose/pulled/heads q1lsd0 q1lsd1
"$PY" imagplan_q/read_q1.py --combine imagplan_q/read_q1/LS/READ_Q1_LS.json imagplan_q/read_q1/LSD/READ_Q1_LSD.json --out imagplan_q/read_q1/READ_Q1_COMBINED.json 2>&1 | cut -c1-330
echo "ZZ READ-Q1 DONE $(ts)"
