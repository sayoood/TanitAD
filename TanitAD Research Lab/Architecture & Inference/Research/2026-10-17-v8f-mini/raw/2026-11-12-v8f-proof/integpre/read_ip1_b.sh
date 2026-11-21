#!/usr/bin/env bash
# read_ip1_b.sh : rerun of two steps of read_ip1.sh for card PREREG_V8F_PROOF_ADD44 (INTEG-PRE): the headway reader and the final read. Reason (logged in TIMES_NOTE.md before
# this ran): on 2026-11-19 the headway reader ended with ArrayMemoryError at a peak of 920 MB under its 2000 MB cap (the box's commit memory ran out), so RESULT_HEADWAY.json
# was absent and read_ip1.py gave no names and printed no difference. Commands, pairs, bases, readers and rules are those of read_ip1.sh; the family result of the first run stays.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
I=$H/integpre; IW=$W/integpre
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull,$W/imagplan/pull_t1,$IW/pull_t1"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYCAP_MB=${PYCAP_MB:-2000} PULL_ROOT=$I/pull_t1 RUNS_ROOT=$I/pull_runs
ts() { date -u +%H:%M:%SZ; }
cd "$I" || exit 1
SEEDS=0,1,2,3
PAIRS=$("$PY" "$IW/read_ip1.py" --pairs | tail -n 1 | tr -d '\r\n ')
[ -f "$I/read/q3/RESULT_Q3.json" ] || { echo "ZZ READ-IP1-B STOP no family result"; exit 3; }
( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds "$SEEDS" --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$IW/read/hw" > "$I/read/hw_b.log" 2>&1 ); echo "[$(ts)] headway rc $?"
[ -f "$I/read/hw/RESULT_HEADWAY.json" ] || { tail -n 3 "$I/read/hw_b.log" | cut -c1-300; echo "ZZ READ-IP1-B STOP no headway result"; exit 3; }
"$PY" "$IW/read_ip1.py" --read --plan-file "$IW/read/plan_ip1.json" --hw "$IW/read/hw/RESULT_HEADWAY.json" --q3 "$IW/read/q3/RESULT_Q3.json" --small "$IW/pull_small" --pull "$IW/pull_t1" \
  --out-json "$IW/read/READ_IP1.json" --out-md "$IW/read/READ_IP1.md" 2>&1 | cut -c1-330
echo "ZZ READ-IP1-B DONE $(ts)"
