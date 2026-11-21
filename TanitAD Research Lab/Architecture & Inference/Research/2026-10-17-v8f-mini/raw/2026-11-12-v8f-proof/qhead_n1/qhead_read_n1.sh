#!/usr/bin/env bash
# qhead_read_n1.sh : dev-box side, note PREREG_V8F_PROOF_ADD42_N1 (second version of qhead_read.sh). Run after the pod queue wrote QHEAD_DONE.txt and the lead pulled the rollouts (spatial/q3/pull_spatial_t1.sh with Q=distrule)
# and the small pod files (see HANDOFF.md, pull list). It starts nothing on the pod, deletes nothing, touches nothing under E:\Projects\TanitAD.
# Steps: the reader versions r2 (headway_read_r2.py, q3_read_r3.py) for the pairs of section 5 -> control h3 per arm (qhead_h3.py) -> qhead_names.py (controls h1 to h7 first; names only if every control passed).
# Needed (override by env): PULL (rollouts), QH_PULLED (folder with heads/<BASE>.qhead.json + .npz and qhead_queue.log), STATES (eval state files), BANKED (the *_eval_lead.npz.head.npz files), JOIN (join_eval141.npz),
# H6 (MUTATIONS_QHEAD.json of run_mutations.py). Free commit memory of at least 5.5 GB before every Python job (60 s steps, at most 30 min per wait). Last line: ZZ READ-QHEAD-N1 DONE.
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof; H=/f/TanitAD-scratch/v8proof
QHD=$H/qhead                       # the first-run tools (the rollout tool, its queue log, the pulled control rollout)
N1=$H/qhead_n1
PULL=${PULL:-/c/Users/Admin/v8proof_c/pull}
ROOTS=${ROOTS:-"$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull"}
QH_PULLED=${QH_PULLED:-$N1/pulled}
FIRST_LOG=${FIRST_LOG:-$QHD/pulled/qhead_queue.log}
TOOL=${TOOL:-$QHD/v8f_t1_eval_rule_qhead.py}
STATES=${STATES:-/c/Users/Admin/v8proof_c/s1states}
BANKED=${BANKED:-$H/spatial/leadsup/read/in}
JOIN=${JOIN:-"E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"}
H6=${H6:-$N1/MUTATIONS_QHEAD_N1.json}
OUT=${1:-$N1/read}
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd "$H" || exit 1
ts() { date -u +%H:%M:%SZ; }
memwait() {
  local kb i
  for i in $(seq 1 30); do
    kb=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory" | tr -d '\r\n ')
    [ "${kb:-0}" -ge 5767168 ] && return 0
    sleep 60
  done
  echo "ZZ READ-QHEAD-N1 STOP: free commit memory stayed under 5.5 GB"; exit 3
}
mkdir -p "$OUT/hw" "$OUT/q3" "$OUT/h3"
ARMS="LSQE_s0 LSQE_s1 LSDQE_s0 LSDQE_s1 LSQHI_s0"
for a in $ARMS; do
  [ -f "$PULL/$a/PULLED.txt" ] || { echo "ZZ READ-QHEAD-N1 STOP: $a is not pulled ($PULL/$a/PULLED.txt)"; exit 3; }
done
PAIRS_HW="LSQE:LS,LSQE:LSRH,LSQE:LSRL,LSQE:LSMC,LSQE:LSQC,LSDQE:LSD,LSDQE:LSDRH,LSDQE:LSDRL,LSDQE:LSDMC,LSDQE:LSDQC"
PAIRS_Q3="LSQE:LS,LSQE:LSQC,LSQE:LSMC,LSDQE:LSD,LSDQE:LSDQC,LSDQE:LSDMC"
memwait
( cd "$H/headway" && "$PY" headway_read_r2.py read --t1-root "$ROOTS" --pairs "$PAIRS_HW" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --out-dir "$OUT/hw" > "$OUT/hw.log" 2>&1 ); echo "[$(ts)] headway r2 rc $? : $(tail -n 1 "$OUT/hw.log" | cut -c1-200)"
memwait
( cd "$H/spatial/q3" && "$PY" q3_read_r3.py read --t1-root "$ROOTS" --pairs "$PAIRS_Q3" --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT/q3" > "$OUT/q3.log" 2>&1 ); echo "[$(ts)] families r2 rc $? : $(tail -n 1 "$OUT/q3.log" | cut -c1-200)"
for a in LSQE_s0 LSQE_s1 LSDQE_s0 LSDQE_s1; do
  base=${a/QE/}                                  # LSQE_s0 -> LS_s0 ; LSDQE_s0 -> LSD_s0
  memwait
  "$PY" "$N1/qhead_h3_n1.py" --tol "${H3_TOL:-0.02}" --arm "$a" --rule-log "$PULL/$a/ticks/rule_log_$a.npz" --join "$JOIN" --eval-states "$STATES/${base}_eval.npz" --banked-head "$BANKED/${base}_eval_lead.npz.head.npz" \
    --qhead "$QH_PULLED/heads/$base.qhead.npz" --out "$OUT/h3/H3_$a.json" 2>&1 | tail -n 1 | cut -c1-300
done
memwait
"$PY" "$N1/qhead_names_n1.py" --hw "$OUT/hw/RESULT_HEADWAY.json" --q3 "$OUT/q3/RESULT_Q3.json" --fit-dir "$QH_PULLED/heads" --h1-text "$FIRST_LOG" --tool-file "$TOOL" --rule-log-dir "$PULL" --h3 "$OUT/h3" --h6 "$H6" --out "$OUT/READ_QHEAD_N1.json" 2>&1 | cut -c1-420
echo "ZZ READ-QHEAD-N1 DONE $(ts)"
