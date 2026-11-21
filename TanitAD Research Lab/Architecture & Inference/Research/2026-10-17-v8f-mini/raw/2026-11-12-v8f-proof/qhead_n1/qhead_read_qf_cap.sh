#!/usr/bin/env bash
# qhead_read_qf.sh : dev-box side, note PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step, arms LSDQF_s0, LSDQF_s1). Written by the lead.
# The readers are called with the pairs of the LSD type only. It starts nothing on the pod, deletes nothing.
# Steps: headway_read_r2.py and q3_read_r3.py -> control h3 per arm (qhead_h3_n1.py) -> qhead_names_qf.py (controls h1 to h8, d1, d2 first; names only if they pass).
# Needed (override by env): PULL (rollouts), QH_PULLED (qhead_n1/dose/pulled: heads/<BASE>.qhead.json + .npz, controls/d1_pack.json, controls/d2_<BASE>.json),
# STATES (eval state files), BANKED (the *_eval_lead.npz.head.npz files), JOIN (join_eval141.npz), H6 (MUTATIONS_QHEAD_N1.json).
# Free commit memory of at least 2.5 GB before every Python job. Last line: ZZ READ-QHEAD-QF DONE.
PY=/f/TanitAD-scratch/v8proof/ops/pycap.sh
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
QHD=$H/qhead
N1=$H/qhead_n1
PULL=${PULL:-$H/imagplan/pull_t1}
ROOTS=${ROOTS:-"$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull,$W/imagplan/pull_t1"}
QH_PULLED=${QH_PULLED:-$N1/dose/pulled}
FIRST_LOG=${FIRST_LOG:-$QHD/pulled/qhead_queue.log}
TOOL=${TOOL:-$QHD/v8f_t1_eval_rule_qhead.py}
STATES=${STATES:-/c/Users/Admin/v8proof_c/s1states}
BANKED=${BANKED:-$H/spatial/leadsup/read/in}
JOIN=${JOIN:-"E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"}
H6=${H6:-$N1/MUTATIONS_QHEAD_N1.json}
OUT=${1:-$N1/dose/read}
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd "$H" || exit 1
ts() { date -u +%H:%M:%SZ; }
memwait() {
  local kb i
  for i in $(seq 1 30); do
    kb=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory" | tr -d '\r\n ')
    [ "${kb:-0}" -ge 2621440 ] && return 0
    sleep 60
  done
  echo "ZZ READ-QHEAD-QF STOP: free commit memory stayed under 2.5 GB"; exit 3
}
mkdir -p "$OUT/hw" "$OUT/q3" "$OUT/h3"
for b in LSD_s0 LSD_s1; do
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('h5_check_coverage_in_0.15_0.25') is True else 1)" "$QH_PULLED/heads/$b.qhead.json" \
    || { echo "ZZ READ-QHEAD-QF STOP: $b is outside the h5 band on the check clips; the LSD type is not named and no outcome number is computed"; exit 2; }
done
for a in LSQHI_s0 LSDQF_s0 LSDQF_s1; do
  [ -f "$PULL/$a/PULLED.txt" ] || { echo "ZZ READ-QHEAD-QF STOP: $a is not pulled ($PULL/$a/PULLED.txt)"; exit 3; }
done
PAIRS_HW="LSDQF:LSD,LSDQF:LSDRH,LSDQF:LSDRL,LSDQF:LSDMC,LSDQF:LSDQC"
PAIRS_Q3="LSDQF:LSD,LSDQF:LSDQC,LSDQF:LSDMC"
memwait
( cd "$H/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS_HW" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$OUT/hw" > "$OUT/hw.log" 2>&1 ); echo "[$(ts)] headway r2 rc $? : $(tail -n 1 "$OUT/hw.log" | cut -c1-200)"
memwait
( cd "$H/spatial/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS_Q3" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT/q3" > "$OUT/q3.log" 2>&1 ); echo "[$(ts)] families r3 rc $? : $(tail -n 1 "$OUT/q3.log" | cut -c1-200)"
for a in LSDQF_s0 LSDQF_s1; do
  base=${a/QF/}                                  # LSDQF_s0 -> LSD_s0
  memwait
  "$PY" "$N1/qhead_h3_n1.py" --tol "${H3_TOL:-0.02}" --arm "$a" --rule-log "$PULL/$a/ticks/rule_log_$a.npz" --join "$JOIN" --eval-states "$STATES/${base}_eval.npz" --banked-head "$BANKED/${base}_eval_lead.npz.head.npz" \
    --qhead "$QH_PULLED/heads/$base.qhead.npz" --out "$OUT/h3/H3_$a.json" 2>&1 | tail -n 1 | cut -c1-300
done
memwait
"$PY" "$N1/qhead_names_qf.py" --hw "$OUT/hw/RESULT_HEADWAY.json" --q3 "$OUT/q3/RESULT_Q3.json" --fit-dir "$QH_PULLED/heads" --h1-text "$FIRST_LOG" --tool-file "$TOOL" --rule-log-dir "$PULL" --h3 "$OUT/h3" --h6 "$H6" \
  --d1 "$QH_PULLED/controls/d1_pack.json" --d2-dir "$QH_PULLED/controls" --out "$OUT/READ_QHEAD_QF.json" 2>&1 | cut -c1-420
echo "ZZ READ-QHEAD-QF DONE $(ts)"
