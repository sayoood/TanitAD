#!/usr/bin/env bash
# read_i10k.sh : dev-box read of card PREREG_V8F_PROOF_ADD46 (INTEG10K-PRE) and its note N1, after chain_i10k.sh has ended (pull/ holds the run folders and pull/small).
# The headway reader and the family reader on the pairs of the card, then read_i10k.py (controls k1 to k10 first; names only if all pass). Numpy only, every Python job under ops/pycap.sh
# (PYCAP_MB 2000, one job at a time). Starts nothing on the pod. Deletes nothing. Last line: ZZ READ-I10K DONE.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof; I=$H/integ10k
PULL=$I/pull; RD=$I/read; RW=$W/integ10k/read
mkdir -p "$RD"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYCAP_MB=${PYCAP_MB:-2000}
ts() { date -u +%H:%M:%SZ; }
ROOTS="$W/integ10k/pull"
PAIRS="I10RR:I10R0,I10RR:I10RLb,I10RLb:I10R0,I10RR0:I10R0,I10RRid:I10R0,I10RRx:I10R0"
( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$RW/hw" > "$RD/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd $H/spatial/q3 && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$RW/q3" > "$RD/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$I/read_i10k.py" --read-dir "$RW" --pull "$W/integ10k/pull" --k5 "$W/integ10k/K5_BANKED.json" --out "$W/integ10k/read/READ_I10K.json" > "$RD/read_i10k.log" 2>&1
tail -n 30 "$RD/read_i10k.log" | cut -c1-330
echo "ZZ READ-I10K DONE $(ts)"
