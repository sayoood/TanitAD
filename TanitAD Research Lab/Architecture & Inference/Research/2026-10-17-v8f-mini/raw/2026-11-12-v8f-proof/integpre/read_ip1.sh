#!/usr/bin/env bash
# read_ip1.sh : dev-box side of card PREREG_V8F_PROOF_ADD44 (INTEG-PRE), after chain_ip1.sh has ended (its outputs are in integpre/pull_t1, pull_small). Numpy only, every Python job through ops/pycap.sh
# (PYCAP_MB=2000, one job at a time). Steps: plan (which bases can be read) -> headway reader r2 on the nine pairs of the card -> family reader r3 (same-sign rule of ADD13) -> read_ip1.py (controls c1 to c6 first, then D1 to D5,
# share reached, contact rates, families, names). Starts nothing on the pod, deletes nothing. Writes READ_IP1.json and READ_IP1.md in integpre/read/. Last line: ZZ READ-IP1 DONE.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
I=$H/integpre; IW=$W/integpre
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull,$W/imagplan/pull_t1,$IW/pull_t1"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYCAP_MB=${PYCAP_MB:-2000} PULL_ROOT=$I/pull_t1 RUNS_ROOT=$I/pull_runs
mkdir -p "$I/read/hw" "$I/read/q3"
ts() { date -u +%H:%M:%SZ; }
cd "$I" || exit 1
SEEDS=$("$PY" "$IW/read_ip1.py" --plan --small "$IW/pull_small" --pull "$IW/pull_t1" --arms "$IW/integpre_arms.txt" --out "$IW/read/plan_ip1.json" | tail -n 1 | tr -d '\r\n ')
echo "[$(ts)] bases that can be read: ${SEEDS:-none}"
PAIRS=$("$PY" "$IW/read_ip1.py" --pairs | tail -n 1 | tr -d '\r\n ')
if [ -n "$SEEDS" ] && [ "$(echo "$SEEDS" | tr ',' '\n' | grep -c .)" -ge 3 ]; then
  ( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds "$SEEDS" --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$IW/read/hw" > "$I/read/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
  ( cd $H/spatial/q3 && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds "$SEEDS" --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$IW/read/q3" > "$I/read/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
else
  echo "[$(ts)] fewer than three bases can be read: the readers are not run, nothing is named"
fi
"$PY" "$IW/read_ip1.py" --read --plan-file "$IW/read/plan_ip1.json" --hw "$IW/read/hw/RESULT_HEADWAY.json" --q3 "$IW/read/q3/RESULT_Q3.json" --small "$IW/pull_small" --pull "$IW/pull_t1" \
  --out-json "$IW/read/READ_IP1.json" --out-md "$IW/read/READ_IP1.md" 2>&1 | cut -c1-330
echo "ZZ READ-IP1 DONE $(ts)"
