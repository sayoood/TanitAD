#!/usr/bin/env bash
# read_close10k.sh : dev-box read of card PREREG_V8F_PROOF_ADD48 (CLOSE10K), after chain_close10k.sh has ended (pull/ holds the five new run folders and pull/small_close).
# The headway reader and the family reader on the pairs of the card (new runs against the reused dumps of cards ADD46/ADD47), then read_close10k.py (controls c1 to c7 first;
# the names only if all pass). Every Python job under ops/pycap.sh (one job at a time). Starts nothing on the pod. Last line: ZZ READ-CLOSE10K DONE.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof; I=$H/integ10k
RD=$I/read_close; RW=$W/integ10k/read_close
mkdir -p "$RD"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export PYCAP_MB=${PYCAP_MB:-2000}
ts() { date -u +%H:%M:%SZ; }
ROOTS="$W/integ10k/pull"
PAIRS="I10RLc:I10RLb,I10RLc:I10R0,I10RQ50c:I10RQ50,I10RQ50c:I10R0,I10RLb:I10R0,I10RQ50:I10R0"
( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --out-dir "$RW/hw" > "$RD/hw.log" 2>&1 ); echo "[$(ts)] headway rc $?"
( cd $H/spatial/q3 && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm I10R0 --spread-seeds 0,1 --same-sign --out-dir "$RW/q3" > "$RD/q3.log" 2>&1 ); echo "[$(ts)] families rc $?"
"$PY" "$I/read_close10k.py" --read-dir "$RW" --pull "$W/integ10k/pull" --reused "$W/integ10k/CLOSE10K_REUSED.json" --out "$W/integ10k/read_close/READ_CLOSE10K.json" > "$RD/read_close10k.log" 2>&1
echo "[$(ts)] read rc $?"
tail -n 60 "$RD/read_close10k.log" | cut -c1-430
echo "ZZ READ-CLOSE10K DONE $(ts)"
