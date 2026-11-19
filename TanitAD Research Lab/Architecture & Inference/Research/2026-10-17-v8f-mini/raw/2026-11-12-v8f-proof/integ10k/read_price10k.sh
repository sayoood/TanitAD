#!/usr/bin/env bash
# read_price10k.sh : dev-box read of card PREREG_V8F_PROOF_ADD47 (PRICE10K), after chain_price10k.sh has ended (pull/ holds the six new run folders and pull/small_price).
# The headway reader and the family reader on the pairs of the card (new runs against the reused dumps of card ADD46), then read_price10k.py (controls m1 to m7 first;
# the name only if all pass). Numpy only, every Python job under ops/pycap.sh (one job at a time). Starts nothing on the pod. Last line: ZZ READ-PRICE10K DONE.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof; I=$H/integ10k
PULL=$I/pull; RD=$I/read_price; RW=$W/integ10k/read_price
mkdir -p "$RD"
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYCAP_MB=${PYCAP_MB:-2000}
ts() { date -u +%H:%M:%SZ; }
ROOTS="$W/integ10k/pull"
PAIRS="I10RQ50:I10R0,I10RQ50:I10RLb,I10RQ50:I10RR,I10RQ35:I10R0,I10RQ35:I10RLb,I10RRaw:I10R0,I10RRaw:I10RLb,I10RR:I10R0,I10RR:I10RLb"
( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$RW/hw" > "$RD/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd $H/spatial/q3 && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$RW/q3" > "$RD/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$I/read_price10k.py" --read-dir "$RW" --pull "$W/integ10k/pull" --reused "$W/integ10k/PRICE10K_REUSED.json" --out "$W/integ10k/read_price/READ_PRICE10K.json" > "$RD/read_price10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 60 "$RD/read_price10k.log" | cut -c1-430
echo "ZZ READ-PRICE10K DONE $(ts)"
