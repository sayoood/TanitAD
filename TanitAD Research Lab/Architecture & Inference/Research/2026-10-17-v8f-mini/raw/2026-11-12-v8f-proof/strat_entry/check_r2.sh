#!/usr/bin/env bash
# check_r2.sh : dev-box side, CPU. Makes the reader versions r2 (make_r2.py) and shows that they give the numbers of the first versions on a banked read:
# the family read and the headway read of the pairs of row H-V8F-PROOF-80 (bases 0 and 1) and the imagination read of arm RLAm. Each r2 run writes into
# strat_entry/out/ and is compared value by value with the banked result of the first version (cmp_json.py). One Python job at a time; it waits while less than
# 5.5 GB of commit memory are free. Deletes nothing. Last line: ZZ CHECK-R2 PASS | FAIL.
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"
export PULL_ROOT=/c/Users/Admin/v8proof_c/pull RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd $H || exit 1
O=$W/strat_entry/out; mkdir -p $H/strat_entry/out
ts() { date -u +%H:%M:%SZ; }
mem() { local k=0 f; while :; do f=$(powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -cd '0-9'); [ "${f:-0}" -ge 6 ] && return 0; k=$((k+1)); [ $k -gt 60 ] && { echo "[$(ts)] commit memory stays low ($f GB), going on"; return 0; }; sleep 30; done; }
"$PY" strat_entry/make_r2.py || { echo "ZZ CHECK-R2 FAIL (make)"; exit 1; }
PAIRS="RLA:RLFT,RLAm:RLFT,RLFTb:RLFT"; bad=0
mem; ( cd $H/spatial/q3 && "$PY" q3_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$O/q3_rl" > "$H/strat_entry/out/q3_rl.log" 2>&1 ); echo "[$(ts)] families r2 rc $?"
"$PY" strat_entry/cmp_json.py "$W/actcons/read_rl/q3_rl/RESULT_Q3.json" "$O/q3_rl/RESULT_Q3.json" || bad=1
mem; ( cd $H/headway && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$O/hw_rl" > "$H/strat_entry/out/hw_rl.log" 2>&1 ); echo "[$(ts)] headway r2 rc $?"
"$PY" strat_entry/cmp_json.py "$W/actcons/read_rl/hw_rl/RESULT_HEADWAY.json" "$O/hw_rl/RESULT_HEADWAY.json" || bad=1
mem; ( cd $H/imaghead && "$PY" imag_read_r2.py read --dump "$W/imaghead/dump/DUMP_RLAm_s0.npz" "$W/imaghead/dump/DUMP_RLAm_s1.npz" --arm RLAm --seeds 0,1 --out-dir "$O/RLAm" > "$H/strat_entry/out/RLAm.log" 2>&1 ); echo "[$(ts)] imagination r2 rc $?"
"$PY" strat_entry/cmp_json.py "$W/actcons/read_rl/RLAm/RESULT_IMAG_RLAm.json" "$O/RLAm/RESULT_IMAG_RLAm.json" || bad=1
grep -c "NOT COMPUTED" "$H/strat_entry/out/q3_rl/RESULT_Q3.json" "$H/strat_entry/out/hw_rl/RESULT_HEADWAY.json" 2>/dev/null
[ $bad = 0 ] && echo "ZZ CHECK-R2 PASS $(ts)" || echo "ZZ CHECK-R2 FAIL $(ts)"
