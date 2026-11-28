#!/usr/bin/env bash
# read_q1b.sh : second pass of the dev-box read of note PREREG_V8F_PROOF_ADD40_N7 with read_q1b.py (one plumbing fix of control q2, see make_read_q1b.py).
# Pulls nothing and reruns neither the headway nor the family reader: their outputs of read_q1.sh are read as they are (the pair IPcq0 against IPc is in them).
# Refuses while read_q1.sh has not written its last line. Starts nothing on the pod. Deletes nothing. Last line: ZZ READ-Q1B DONE.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
H=/f/TanitAD-scratch/v8proof
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYCAP_MB=${PYCAP_MB:-2000}
cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
grep -aq 'ZZ READ-Q1 DONE' imagplan_q/read_q1.out || { echo "ZZ READ-Q1B STOP: read_q1.sh has not finished"; exit 3; }
[ -f imagplan_q/read_q1b.py ] || { echo "ZZ READ-Q1B STOP: read_q1b.py missing"; exit 3; }
one_pair() { # arm records-folder deploy-names(2)
  local A=$1 REC=$2 D0=$3 D1=$4
  local R=$H/imagplan_q/read_q1/$A
  "$PY" imagplan_q/read_q1b.py --arm $A --sfx q --bases 0,1 --pull imagplan_q/pull --read-dir "imagplan_q/read_q1/$A" --seq-outs imagplan_q/deploy_q_$D0.out imagplan_q/deploy_q_$D1.out \
      --records "$REC" --out "imagplan_q/read_q1/$A/READ_Q1B_$A.json" > "$R/read_q1b.log" 2>&1
  echo "[$(ts)] $A read rc $?"
  tail -n 34 "$R/read_q1b.log" | cut -c1-360
}
one_pair LS qhead_n1/dose/pulled_ls/heads q1ls0 q1ls1
one_pair LSD qhead_n1/dose/pulled/heads q1lsd0 q1lsd1
"$PY" imagplan_q/read_q1b.py --combine imagplan_q/read_q1/LS/READ_Q1B_LS.json imagplan_q/read_q1/LSD/READ_Q1B_LSD.json --out imagplan_q/read_q1/READ_Q1B_COMBINED.json 2>&1 | cut -c1-360
echo "ZZ READ-Q1B DONE $(ts)"
