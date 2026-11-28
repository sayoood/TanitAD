#!/usr/bin/env bash
# pod_lookat_queue.sh -- pod queue of card PREREG_V8F_PROOF_ADD107 (LOOKAT-T0), stage FITS. Tier T0.
# Order: the identity control k5 first (locator_fit.py --identity, no fit; a run that does not pass stops the queue), then the fits of seeds 0, 1, 2 and the permuted fit (seed 0, --permute-targets);
# each fit writes its prediction file in the same run. At most LK_MAX_JOBS (default 2) jobs at a time; a job starts only with more than LK_MIN_FREE_GB (28) GB free in the container (memory.max minus anon)
# and fewer than 3 GPU compute processes on the card (the lead's rule: never more than three GPU jobs); it waits up to LK_WAIT_S (1800) s for a slot, then STOPs.
# Markers in $O: START_LOOKAT.txt, DONE_LOOKAT.txt, STOP_LOOKAT.txt (RESUME=1 continues); per job in $O/queue/jobs: <job>.START/.DONE/.STOP, logs in $O/queue/logs. A fit with its FIT_RECORD.json skips the fit
# (a missing prediction is then made with --predict); a job that failed writes the stop marker and no further job starts. ulimit -n 16384 (one memory map per clip). Writes only under $O. Never deletes, never moves, never stops a process.
# Usage: bash pod_lookat_queue.sh        |   RESUME=1 bash pod_lookat_queue.sh   |   DRY=1 bash pod_lookat_queue.sh   (prints every command, makes nothing)
set -u
V=/workspace/v8proof
R1=${LK_RUN1:-$V/cropread/run1}
O=${LK_ROOT:-$V/lookat}
K=${LK_KIT:-$O/kit}
KF=${LK_KIT_FIT:-$V/cropread/kit/fit}
BOX=${LK_BOX:-$V/cropread/enc_out/box}
JOIN=${LK_JOIN_TRAIN:-$V/data/inputs/join_train.npz}
PY=${LK_PY:-/workspace/venv/bin/python}
CG=${LK_CG:-/sys/fs/cgroup}
MAXJ=${LK_MAX_JOBS:-2}
MINGB=${LK_MIN_FREE_GB:-28}
WAITS=${LK_WAIT_S:-1800}
DRY=${DRY:-0}
Q=$O/queue
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
JOBS=(fit_s0 fit_s1 fit_s2 fit_s0_perm)
identcmd() { echo "$PY $K/locator_fit.py --identity --root $R1 --out-root $O --kit-fit $KF --join-train $JOIN --device cuda"; }
fitcmd() { # job
  local s=${1#fit_s}; s=${s%_perm}; local p=""; [[ "$1" == *_perm ]] && p=" --permute-targets"
  echo "$PY $K/locator_fit.py --seed $s$p --root $R1 --box-dir $BOX --out-root $O --kit-fit $KF --device cuda"
}
predcmd() { echo "$(fitcmd "$1") --predict"; }
if [ "$DRY" = 1 ]; then
  echo "CMD: ulimit -n 16384"
  echo "CMD: need $K/locator_fit.py $KF/crop_fit.py $JOIN $R1/lists/train_final.txt $R1/lists/eval141.txt $R1/fits/D_H5_s0/head.pt $BOX; memory free > $MINGB GB and fewer than 3 GPU compute processes before every job, else wait up to $WAITS s, then STOP"
  echo "CMD: mkdir -p $O/fits $O/pred $O/read $Q/jobs $Q/logs"
  echo "JOB ident_k5 (first, alone):"; echo "  CMD: $(identcmd)"
  for j in "${JOBS[@]}"; do echo "JOB $j (after ident_k5, at most $MAXJ at a time):"; echo "  CMD: $(fitcmd "$j")   (a finished fit without prediction: $(predcmd "$j"))"; done
  echo "MARKERS: $O/START_LOOKAT.txt $O/DONE_LOOKAT.txt $O/STOP_LOOKAT.txt"
  echo "DRY run complete: at most $MAXJ jobs at a time"; exit 0
fi
stop() { mkdir -p "$O"; echo "$(ts) $*" >> "$O/STOP_LOOKAT.txt"; echo "STOP $*"; exit 3; }
ulimit -n 16384 2>/dev/null || [ "${LK_ULIMIT_SOFT:-0}" = 1 ] || stop "ULIMIT-N-16384-REFUSED"      # LK_ULIMIT_SOFT=1 is for the dev-box tests only
[ -f "$O/DONE_LOOKAT.txt" ] && { echo "ALREADY-DONE"; exit 0; }
if [ -f "$O/STOP_LOOKAT.txt" ] && [ "${RESUME:-0}" != 1 ]; then echo "REFUSED: $O/STOP_LOOKAT.txt exists; RESUME=1 to continue"; exit 2; fi
for f in "$K/locator_fit.py" "$KF/crop_fit.py" "$JOIN" "$R1/lists/train_final.txt" "$R1/lists/eval141.txt" "$R1/fits/D_H5_s0/head.pt" "$R1/fits/D_H5_s0/FIT_RECORD.json"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
[ -d "$BOX" ] || stop "MISSING-INPUT $BOX"
command -v nvidia-smi >/dev/null 2>&1 || stop "NO-NVIDIA-SMI"
mkdir -p "$O/fits" "$O/pred" "$O/read" "$Q/jobs" "$Q/logs" || stop "CANNOT-MAKE-FOLDERS"
echo "$(ts) start pid $$ resume ${RESUME:-0} max jobs $MAXJ" >> "$O/START_LOOKAT.txt"
blocked() { [ -f "$O/STOP_LOOKAT.txt" ] && [ "${RESUME:-0}" != 1 ]; }
free_gb() { local mm an; mm=$(tr -d ' \r\n' < "$CG/memory.max" 2>/dev/null); an=$(awk '$1=="anon"{print $2}' "$CG/memory.stat" 2>/dev/null | head -n 1)
  if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then echo $(( (mm - an) / 1073741824 )); else echo 999; fi; }
gpu_procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
admit() { # waits for a slot (memory and GPU rule); returns 1 after the wait is used up
  local t=0 g n
  while :; do
    g=$(free_gb); n=$(gpu_procs); n=${n:-0}
    if [ "$g" -gt "$MINGB" ] && [ "$n" -lt 3 ]; then return 0; fi
    [ "$t" -ge "$WAITS" ] && return 1
    sleep "${LK_POLL_S:-30}"; t=$((t + ${LK_POLL_S:-30}))
  done
}
run_ident() {
  echo "$(ts) START ident_k5" > "$Q/jobs/ident_k5.START"
  if [ ! -f "$O/IDENTITY_K5.json" ]; then
    $(identcmd) > "$Q/logs/ident_k5.log" 2>&1 || { echo "$(ts) identity rc $? (k5 did not pass)" > "$Q/jobs/ident_k5.STOP"; return 1; }
  fi
  echo "$(ts) DONE ident_k5" > "$Q/jobs/ident_k5.DONE"
}
run_fit() { # job
  local j=$1 s=${1#fit_s}; s=${s%_perm}; local nm="LOC_s$s"; [[ "$1" == *_perm ]] && nm="${nm}_perm"
  echo "$(ts) START $j" > "$Q/jobs/$j.START"
  if [ ! -f "$O/fits/$nm/FIT_RECORD.json" ]; then
    $(fitcmd "$j") > "$Q/logs/$j.log" 2>&1 || { echo "$(ts) fit rc $?" > "$Q/jobs/$j.STOP"; return 1; }
  elif [ ! -f "$O/pred/${nm}_meta.json" ]; then
    $(predcmd "$j") > "$Q/logs/$j.predict.log" 2>&1 || { echo "$(ts) predict rc $?" > "$Q/jobs/$j.STOP"; return 1; }
  fi
  [ -f "$O/pred/${nm}_meta.json" ] || { echo "$(ts) no prediction meta" > "$Q/jobs/$j.STOP"; return 1; }
  echo "$(ts) DONE $j" > "$Q/jobs/$j.DONE"
}
if [ ! -f "$Q/jobs/ident_k5.DONE" ]; then
  admit || stop "NO-SLOT-FOR-ident_k5"
  run_ident || { echo "$(ts) job ident_k5 failed" >> "$O/STOP_LOOKAT.txt"; echo "STOP identity control k5 did not pass"; exit 3; }
fi
for j in "${JOBS[@]}"; do
  [ -f "$Q/jobs/$j.DONE" ] && continue
  [ -f "$Q/jobs/$j.STOP" ] && echo "$(ts) retry $j (an old .STOP stays on disk)" >> "$O/START_LOOKAT.txt"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
  blocked && break
  admit || { echo "$(ts) no slot for $j" >> "$O/STOP_LOOKAT.txt"; break; }
  ( run_fit "$j" || echo "$(ts) job $j failed" >> "$O/STOP_LOOKAT.txt" ) &
  sleep "${LK_SPACING_S:-20}"
done
wait
blocked && { echo "STOPPED: see $O/STOP_LOOKAT.txt"; exit 3; }
ok=1
for j in "${JOBS[@]}"; do [ -f "$Q/jobs/$j.DONE" ] || ok=0; done
if [ "$ok" = 1 ]; then echo "$(ts) all steps done" > "$O/DONE_LOOKAT.txt"; echo "ZZ LOOKAT-FITS DONE"; else echo "ZZ LOOKAT-FITS INCOMPLETE"; exit 3; fi
