#!/usr/bin/env bash
# pod_lookgap_dirs_queue.sh -- pod queue of card PREREG_V8F_PROOF_ADD108 (LOOKGAP-T0), stage A (direction files). Tier T0. No new encode.
# Order: folds -> five fold fits (locator_fit.py of card ADD107, seed 0, its settings, one run root per fold) -> predict for each fold's held-out clips (loc_dirs.py) -> predict for the 141 evaluation clips with the head
# of seed 0 of card ADD107 -> check (m0) -> score (S4, S8, S8_out, train held-out and evaluation; printed only, run only when the check passed).
# At most LG_MAX_JOBS (default 2) fits at a time; a job starts only with more than LG_MIN_FREE_GB (28) GB free in the container (memory.max minus anon) and fewer than 3 GPU compute processes on the card; it waits up to
# LG_WAIT_S (1800) s for a slot, then STOPs. Stop rule of the card: 1.5 pod-hours of the jobs of this queue (LG_STOP_S, 5400 s): the seconds of every finished job are written to $Q/jobs/<job>.SECS and summed before
# every start; at or above the limit no further job starts and the STOP marker is written (a running job is not touched).
# Markers in $O: START_DIRS.txt, DONE_DIRS.txt, STOP_DIRS.txt (RESUME=1 continues); per job in $O/queue/jobs: <job>.START/.DONE/.STOP/.SECS, logs in $O/queue/logs. ulimit -n 16384 (one memory map per clip).
# Writes only under $O. Never deletes, never moves, never stops a process.
# Usage: bash pod_lookgap_dirs_queue.sh   |   RESUME=1 bash pod_lookgap_dirs_queue.sh   |   DRY=1 bash pod_lookgap_dirs_queue.sh   (prints every command, makes nothing)
set -u
V=/workspace/v8proof
R1=${LG_RUN1:-$V/cropread/run1}
O=${LG_ROOT:-$V/lookgap}
K=${LG_KIT:-$O/kit_dirs}
KF=${LG_KIT_FIT:-$V/cropread/kit/fit}
KL=${LG_KIT_LOC:-$V/lookat/kit}
L107=${LG_ADD107:-$V/lookat}
BOX=${LG_BOX:-$V/cropread/enc_out/box}
PY=${LG_PY:-/workspace/venv/bin/python}
CG=${LG_CG:-/sys/fs/cgroup}
MAXJ=${LG_MAX_JOBS:-2}
MINGB=${LG_MIN_FREE_GB:-28}
WAITS=${LG_WAIT_S:-1800}
STOPS=${LG_STOP_S:-5400}
DRY=${DRY:-0}
Q=$O/queue
FO=$O/folds
D=$O/dirs
RC=$O/records
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
FITS=(fit_f0 fit_f1 fit_f2 fit_f3 fit_f4)
PREDS=(pred_f0 pred_f1 pred_f2 pred_f3 pred_f4 pred_eval)
LD="$PY $K/loc_dirs.py"
cmd_folds() { echo "$LD folds --root $R1 --out-root $FO --expect-train 500"; }
cmd_fit() { local k=${1#fit_f}; echo "$PY $KL/locator_fit.py --seed 0 --root $FO/f$k/run --box-dir $BOX --out-root $FO/f$k/loc --kit-fit $KF --device cuda"; }
cmd_pred() {
  if [ "$1" = pred_eval ]; then
    echo "$LD predict --root $R1 --head $L107/fits/LOC_s0/head.pt --fit-record $L107/fits/LOC_s0/FIT_RECORD.json --clips $R1/lists/eval141.txt --out-dir $D --record $RC/predict_eval.json --kit-fit $KF --locator-dir $KL --device cuda --expect-seed 0"
  else
    local k=${1#pred_f}
    echo "$LD predict --root $R1 --head $FO/f$k/loc/fits/LOC_s0/head.pt --fit-record $FO/f$k/loc/fits/LOC_s0/FIT_RECORD.json --clips $FO/f$k/heldout.txt --out-dir $D --record $RC/predict_f$k.json --kit-fit $KF --locator-dir $KL --device cuda --expect-seed 0"
  fi
}
check_out() { # the next unused check file name
  local n=1 p=$RC/CHECK_DIRS.json
  while [ -e "$p" ]; do n=$((n + 1)); p=$RC/CHECK_DIRS_$n.json; done
  echo "$p"
}
cmd_check() { echo "$LD check --root $R1 --dirs $D --folds-json $FO/FOLDS.json --fold-root $FO --records-dir $RC --add107-fit-record $L107/fits/LOC_s0/FIT_RECORD.json --out $1 --expect-total 641"; }
cmd_score() { # label, clips list, check file
  echo "$LD score --dirs $D --box-dir $BOX --clips $2 --check-json $3 --label $1"
}
if [ "$DRY" = 1 ]; then
  echo "CMD: ulimit -n 16384"
  echo "CMD: need $K/loc_dirs.py $KL/locator_fit.py $KF/crop_fit.py $R1/lists/train_final.txt $R1/lists/eval141.txt $R1/tok/D $L107/fits/LOC_s0/head.pt $L107/fits/LOC_s0/FIT_RECORD.json $BOX; memory free > $MINGB GB and fewer than 3 GPU compute processes before every job, else wait up to $WAITS s, then STOP; stop at $STOPS s of job time"
  echo "CMD: mkdir -p $FO $D $RC $Q/jobs $Q/logs"
  echo "JOB folds (first):"; echo "  CMD: $(cmd_folds)"
  for j in "${FITS[@]}"; do echo "JOB $j (after folds, at most $MAXJ at a time):"; echo "  CMD: $(cmd_fit "$j")"; done
  for j in "${PREDS[@]}"; do echo "JOB $j (after all fits):"; echo "  CMD: $(cmd_pred "$j")"; done
  echo "JOB check (after all predictions):"; echo "  CMD: $(cmd_check "$RC/CHECK_DIRS.json")"
  echo "JOB score_train (only after the check passed):"; echo "  CMD: $(cmd_score train_heldout "$R1/lists/train_final.txt" "$RC/CHECK_DIRS.json")"
  echo "JOB score_eval (only after the check passed):"; echo "  CMD: $(cmd_score evaluation "$R1/lists/eval141.txt" "$RC/CHECK_DIRS.json")"
  echo "MARKERS: $O/START_DIRS.txt $O/DONE_DIRS.txt $O/STOP_DIRS.txt"
  echo "DRY run complete: at most $MAXJ jobs at a time"; exit 0
fi
stop() { mkdir -p "$O"; echo "$(ts) $*" >> "$O/STOP_DIRS.txt"; echo "STOP $*"; exit 3; }
ulimit -n 16384 2>/dev/null || [ "${LG_ULIMIT_SOFT:-0}" = 1 ] || stop "ULIMIT-N-16384-REFUSED"      # LG_ULIMIT_SOFT=1 is for the dev-box tests only
[ -f "$O/DONE_DIRS.txt" ] && { echo "ALREADY-DONE"; exit 0; }
if [ -f "$O/STOP_DIRS.txt" ] && [ "${RESUME:-0}" != 1 ]; then echo "REFUSED: $O/STOP_DIRS.txt exists; RESUME=1 to continue"; exit 2; fi
for f in "$K/loc_dirs.py" "$KL/locator_fit.py" "$KF/crop_fit.py" "$R1/lists/train_final.txt" "$R1/lists/eval141.txt" "$L107/fits/LOC_s0/head.pt" "$L107/fits/LOC_s0/FIT_RECORD.json"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
for f in "$BOX" "$R1/tok/D"; do [ -d "$f" ] || stop "MISSING-INPUT $f"; done
command -v nvidia-smi >/dev/null 2>&1 || stop "NO-NVIDIA-SMI"
mkdir -p "$FO" "$D" "$RC" "$Q/jobs" "$Q/logs" || stop "CANNOT-MAKE-FOLDERS"
echo "$(ts) start pid $$ resume ${RESUME:-0} max jobs $MAXJ stop at $STOPS s" >> "$O/START_DIRS.txt"
blocked() { [ -f "$O/STOP_DIRS.txt" ] && [ "${RESUME:-0}" != 1 ]; }
free_gb() { local mm an; mm=$(tr -d ' \r\n' < "$CG/memory.max" 2>/dev/null); an=$(awk '$1=="anon"{print $2}' "$CG/memory.stat" 2>/dev/null | head -n 1)
  if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then echo $(( (mm - an) / 1073741824 )); else echo 999; fi; }
gpu_procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
used_s() { local t=0 x f; for f in "$Q"/jobs/*.SECS; do [ -f "$f" ] || continue; x=$(tr -d ' \r\n' < "$f"); [[ "$x" =~ ^[0-9]+$ ]] && t=$((t + x)); done; echo "$t"; }
over_budget() { [ "$(used_s)" -ge "$STOPS" ]; }
admit() { # waits for a slot (memory and GPU rule); returns 1 after the wait is used up
  local t=0 g n
  while :; do
    g=$(free_gb); n=$(gpu_procs); n=${n:-0}
    if [ "$g" -gt "$MINGB" ] && [ "$n" -lt 3 ]; then return 0; fi
    [ "$t" -ge "$WAITS" ] && return 1
    sleep "${LG_POLL_S:-30}"; t=$((t + ${LG_POLL_S:-30}))
  done
}
run_job() { # job, command line (split on blanks)
  local j=$1 c=$2 t0=$SECONDS rc
  echo "$(ts) START $j" > "$Q/jobs/$j.START"
  $c > "$Q/logs/$j.log" 2>&1; rc=$?
  echo $((SECONDS - t0)) > "$Q/jobs/$j.SECS"
  if [ "$rc" -ne 0 ]; then echo "$(ts) rc $rc" > "$Q/jobs/$j.STOP"; return 1; fi
  echo "$(ts) DONE $j" > "$Q/jobs/$j.DONE"
}
run_fit() { # a finished fit (its record exists) is not run again
  local k=${1#fit_f}
  if [ -f "$FO/f$k/loc/fits/LOC_s0/FIT_RECORD.json" ]; then echo "$(ts) DONE $1 (record found)" > "$Q/jobs/$1.DONE"; return 0; fi
  run_job "$1" "$(cmd_fit "$1")"
}
fail() { echo "$(ts) $*" >> "$O/STOP_DIRS.txt"; echo "STOP $*"; exit 3; }
# 1. folds
if [ ! -f "$Q/jobs/folds.DONE" ]; then
  if [ -f "$FO/FOLDS.json" ]; then echo "$(ts) DONE folds (FOLDS.json found)" > "$Q/jobs/folds.DONE"
  else over_budget && fail "BUDGET-BEFORE-folds"; run_job folds "$(cmd_folds)" || fail "job folds failed"; fi
fi
# 2. fits, at most MAXJ at a time
for j in "${FITS[@]}"; do
  [ -f "$Q/jobs/$j.DONE" ] && continue
  [ -f "$Q/jobs/$j.STOP" ] && echo "$(ts) retry $j (an old .STOP stays on disk)" >> "$O/START_DIRS.txt"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
  blocked && break
  over_budget && { echo "$(ts) BUDGET-BEFORE-$j used $(used_s) s" >> "$O/STOP_DIRS.txt"; break; }
  admit || { echo "$(ts) no slot for $j" >> "$O/STOP_DIRS.txt"; break; }
  ( run_fit "$j" || echo "$(ts) job $j failed" >> "$O/STOP_DIRS.txt" ) &
  sleep "${LG_SPACING_S:-20}"
done
wait
blocked && { echo "STOPPED: see $O/STOP_DIRS.txt"; exit 3; }
for j in "${FITS[@]}"; do [ -f "$Q/jobs/$j.DONE" ] || fail "fit $j not done"; done
# 3. direction files, one job at a time
for j in "${PREDS[@]}"; do
  [ -f "$Q/jobs/$j.DONE" ] && continue
  over_budget && fail "BUDGET-BEFORE-$j used $(used_s) s"
  admit || fail "no slot for $j"
  run_job "$j" "$(cmd_pred "$j")" || fail "job $j failed"
done
# 4. check (m0), then the scores
if [ ! -f "$Q/jobs/check.DONE" ]; then
  over_budget && fail "BUDGET-BEFORE-check used $(used_s) s"
  CK=$(check_out)
  echo "$CK" > "$Q/jobs/check.OUT"
  run_job check "$(cmd_check "$CK")" || fail "control m0 did not pass ($CK)"
fi
CK=$(tr -d '\r\n' < "$Q/jobs/check.OUT")
for s in "score_train train_heldout $R1/lists/train_final.txt" "score_eval evaluation $R1/lists/eval141.txt"; do
  set -- $s
  [ -f "$Q/jobs/$1.DONE" ] && continue
  run_job "$1" "$(cmd_score "$2" "$3" "$CK")" || fail "job $1 failed"
done
echo "$(ts) all steps done" > "$O/DONE_DIRS.txt"
cat "$Q/logs/check.log" "$Q/logs/score_train.log" "$Q/logs/score_eval.log"
echo "ZZ LOOKGAP-DIRS DONE"
