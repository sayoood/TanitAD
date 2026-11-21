#!/usr/bin/env bash
# pod_cropread_fit_queue.sh -- pod queue of card PREREG_V8F_PROOF_ADD99 (CROPREAD): the eight fits (arm D and B, head H5 and H1, fit seeds 0 and 1) and the c5 fit (D, H5, seed 0, permuted labels),
# each followed by its dump on the eval frames of the present reader's dump. At most 3 fits at a time on the one GPU. Tier T0.
# Order (so that a stop still leaves value): D-H5 s0, D-H5 s0 perm (c5), D-H5 s1, D-H1 s0, D-H1 s1, B-H5 s0, B-H5 s1, B-H1 s0, B-H1 s1.
# Markers in $Q ($R/queue): START_FITQ.txt (a line per start, appended), jobs/<name>.START / .DONE / .STOP, DONE_FITQ.txt (all jobs done), STOP_FITQ.txt (a job failed; new jobs stop).
# Resumable: a job with fits/<name>/FIT_RECORD.json skips its fit, with its dump file skips its dump, with jobs/<name>.DONE is skipped. A start after a STOP needs RESUME=1.
# Never deletes, never moves, never kills. A job that ran into B-arm skip (ctl/P1.json with pass false) is skipped and logged. DRY=1 prints every command and runs nothing.
set -u
R=${CR_ROOT:-/workspace/v8proof/cropread}
K=$R/kit/fit
Q=$R/queue
PY=${CR_PY:-/workspace/venv/bin/python}
PRESENT=${CR_PRESENT:-/workspace/v8proof/closeread/dump/both}
JOIN=${CR_JOIN_TRAIN:-/workspace/v8proof/data/inputs/join_train.npz}
MAXJ=${CR_MAX_JOBS:-3}
MINGB=${CR_MIN_FREE_GB:-16}
DRY=${DRY:-0}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
JOBS=("D H5 0 0" "D H5 0 1" "D H5 1 0" "D H1 0 0" "D H1 1 0" "B H5 0 0" "B H5 1 0" "B H1 0 0" "B H1 1 0")     # arm head seed perm
jname() { if [ "$4" = 1 ]; then echo "$1_$2_s$3_perm"; else echo "$1_$2_s$3"; fi; }
dname() { if [ "$4" = 1 ]; then echo "$1_$2_perm"; else echo "$1_$2"; fi; }
fitcmd() { echo "$PY $K/crop_fit.py --arm $1 --head $2 --seed $3 --root $R --join-train $JOIN --device cuda$([ "$4" = 1 ] && echo ' --permute-labels')"; }
dumpcmd() { echo "$PY $K/crop_dump.py --arm $1 --head $2 --seed $3 --root $R --present-dump $PRESENT --device cuda$([ "$4" = 1 ] && echo ' --permute-labels')"; }
if [ "$DRY" = 1 ]; then
  echo "CMD: memory free > $MINGB GB (memory.max minus anon), else STOP"
  for j in "${JOBS[@]}"; do set -- $j; echo "JOB $(jname "$@"):"; echo "  CMD: $(fitcmd "$@")"; echo "  CMD: $(dumpcmd "$@")"; done
  echo "DRY run complete: at most $MAXJ jobs at a time"; exit 0
fi
stop() { echo "$(ts) $*" >> "$Q/STOP_FITQ.txt"; echo "STOP $*"; exit 3; }
[ -d "$K" ] || { echo "REFUSED: no $K"; exit 2; }
mkdir -p "$Q/jobs" "$R/fits" "$R/dump" "$R/queue/logs"
[ -f "$Q/DONE_FITQ.txt" ] && { echo "ALREADY-DONE"; exit 0; }
if [ -f "$Q/STOP_FITQ.txt" ] && [ "${RESUME:-0}" != 1 ]; then echo "REFUSED: $Q/STOP_FITQ.txt exists; RESUME=1 to continue"; exit 2; fi
for f in "$K/crop_fit.py" "$K/crop_dump.py" "$R/lists/train_final.txt" "$R/lists/eval141.txt" "$R/lists/LISTS_RECORD.json" "$R/tok/D/MANIFEST.json" "$R/tok/B/MANIFEST.json" "$JOIN" "$PRESENT/reader_s0_score.npz" "$PRESENT/reader_s1_score.npz"; do
  [ -f "$f" ] || stop "MISSING-INPUT $f"
done
mmax=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat | head -n 1)
if [[ "$mmax" =~ ^[0-9]+$ ]] && [[ "$anon" =~ ^[0-9]+$ ]]; then [ $(( (mmax - anon) / 1073741824 )) -gt "$MINGB" ] || stop "MEMORY-FREE-TOO-LOW"; fi
echo "$(ts) start pid $$ resume ${RESUME:-0}" >> "$Q/START_FITQ.txt"
P1OK=1
if [ -f "$R/ctl/P1.json" ]; then "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$R/ctl/P1.json')).get('pass') is True else 1)" || P1OK=0; fi
run_job() {  # arm head seed perm
  local n; n=$(jname "$@")
  { echo "$(ts) START $n" > "$Q/jobs/$n.START"; } 2>/dev/null
  if [ ! -f "$R/fits/$n/FIT_RECORD.json" ]; then
    $(fitcmd "$@") > "$Q/logs/fit_$n.log" 2>&1 || { echo "$(ts) fit rc $?" > "$Q/jobs/$n.STOP"; return 1; }
  fi
  local d; d=$(dname "$@")
  if [ ! -f "$R/dump/$d/reader_s$3_score.npz" ]; then
    $(dumpcmd "$@") > "$Q/logs/dump_$n.log" 2>&1 || { echo "$(ts) dump rc $?" > "$Q/jobs/$n.STOP"; return 1; }
  fi
  echo "$(ts) DONE $n" > "$Q/jobs/$n.DONE"
}
pids=()
for j in "${JOBS[@]}"; do
  set -- $j
  n=$(jname "$@")
  [ -f "$Q/jobs/$n.DONE" ] && continue
  if [ "$1" = B ] && [ "$P1OK" = 0 ]; then echo "$(ts) SKIP-B-P1-FAILED $n" >> "$Q/START_FITQ.txt"; continue; fi
  [ -f "$Q/jobs/$n.STOP" ] && echo "$(ts) retry $n (an old .STOP stays on disk)" >> "$Q/START_FITQ.txt"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
  [ -f "$Q/STOP_FITQ.txt" ] && [ "${RESUME:-0}" != 1 ] && break
  ( run_job "$@" || echo "$(ts) job $n failed" >> "$Q/STOP_FITQ.txt" ) &
done
wait
if [ -f "$Q/STOP_FITQ.txt" ] && [ "${RESUME:-0}" != 1 ]; then echo "STOPPED: see $Q/STOP_FITQ.txt"; exit 3; fi
ok=1
for j in "${JOBS[@]}"; do set -- $j; n=$(jname "$@"); if [ "$1" = B ] && [ "$P1OK" = 0 ]; then continue; fi; [ -f "$Q/jobs/$n.DONE" ] || ok=0; done
if [ "$ok" = 1 ]; then echo "$(ts) all jobs done" > "$Q/DONE_FITQ.txt"; echo "ZZ CROPREAD-FITQ DONE"; else echo "ZZ CROPREAD-FITQ INCOMPLETE"; exit 3; fi
