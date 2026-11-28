#!/usr/bin/env bash
# pod_steer_fit_queue.sh -- pod queue of card PREREG_V8F_PROOF_ADD100 (STEERREAD). Two stages, called one after the other by the lead:
#   loc    the six localiser fits L0..L4, LA (at most 3 at a time), then loc_steer.py (steering files of the 641 clips, ctl/S1.json, ctl/S2.json).
#   heads  needs tok/S/MANIFEST.json and ctl/S8.json with pass true; the five head fits S_H5 s0, S_H5d s0, S_H5d s0 perm (control s6), S_H5 s1, S_H5d s1, each followed by its
#          dump on the present reader's score rows; at most 3 at a time.
# Usage: bash pod_steer_fit_queue.sh loc|heads     (DRY=1 prints every command and runs nothing; RESUME=1 continues after a STOP)
# Markers in $Q ($S/queue): START_<STAGE>.txt (a line per start, appended), jobs/<name>.START / .DONE / .STOP, DONE_<STAGE>.txt, STOP_<STAGE>.txt (a job failed; new jobs stop).
# Resumable: a job whose output record exists skips that step, a job with jobs/<name>.DONE is skipped. Never deletes, never moves, never kills.
set -u
STAGE=${1:-}
S=${S_ROOT:-/workspace/v8proof/cropread/s_out}
R1=${CR_RUN1:-/workspace/v8proof/cropread/run1}
BOX=${CR_BOX:-/workspace/v8proof/cropread/enc_out/box}
K=${CR_KIT:-/workspace/v8proof/cropread/kit/steer_fit}
Q=$S/queue
PY=${CR_PY:-/workspace/venv/bin/python}
PRESENT=${CR_PRESENT:-/workspace/v8proof/closeread/dump/both}
JOIN=${CR_JOIN_TRAIN:-/workspace/v8proof/data/inputs/join_train.npz}
MAXJ=${CR_MAX_JOBS:-3}
MINGB=${CR_MIN_FREE_GB:-28}
DRY=${DRY:-0}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
case "$STAGE" in loc|heads) ;; *) echo "ZZ STEERQ STOP usage: loc|heads"; exit 2;; esac
UP=$(echo "$STAGE" | tr 'a-z' 'A-Z')
LOCJOBS=(L0 L1 L2 L3 L4 LA)
HEADJOBS=("H5 0 0" "H5d 0 0" "H5d 0 1" "H5 1 0" "H5d 1 0")        # head seed perm
hname() { if [ "$3" = 1 ]; then echo "S_$1_s$2_perm"; else echo "S_$1_s$2"; fi; }
dname() { if [ "$3" = 1 ]; then echo "S_$1_perm"; else echo "S_$1"; fi; }
loccmd() { echo "$PY $K/loc_fit.py --set $1 --run1 $R1 --box-dir $BOX --s-root $S --device cuda"; }
steercmd() { echo "$PY $K/loc_steer.py --run1 $R1 --s-root $S --device cuda"; }
fitcmd() { echo "$PY $K/crop_fit_s.py --head $1 --seed $2 --run1 $R1 --s-root $S --join-train $JOIN --device cuda$([ "$3" = 1 ] && echo ' --permute-labels')"; }
dumpcmd() { echo "$PY $K/crop_dump_s.py --head $1 --seed $2 --s-root $S --present-dump $PRESENT --device cuda$([ "$3" = 1 ] && echo ' --permute-labels')"; }
if [ "$DRY" = 1 ]; then
  echo "CMD: memory free > $MINGB GB (memory.max minus anon), else STOP"
  if [ "$STAGE" = loc ]; then
    for j in "${LOCJOBS[@]}"; do echo "JOB $j:"; echo "  CMD: $(loccmd "$j")"; done
    echo "JOB steer (after all six localisers):"; echo "  CMD: $(steercmd)"
  else
    echo "CMD: needs $S/tok/S/MANIFEST.json and $S/ctl/S8.json with pass true"
    for j in "${HEADJOBS[@]}"; do set -- $j; echo "JOB $(hname "$@"):"; echo "  CMD: $(fitcmd "$@")"; echo "  CMD: $(dumpcmd "$@")"; done
  fi
  echo "DRY run complete: stage $STAGE, at most $MAXJ jobs at a time"; exit 0
fi
stop() { mkdir -p "$Q"; echo "$(ts) $*" >> "$Q/STOP_$UP.txt"; echo "STOP $*"; exit 3; }
[ -d "$K" ] || { echo "REFUSED: no $K"; exit 2; }
mkdir -p "$Q/jobs" "$Q/logs" "$S/loc" "$S/fits" "$S/dump"
[ -f "$Q/DONE_$UP.txt" ] && { echo "ALREADY-DONE $STAGE"; exit 0; }
if [ -f "$Q/STOP_$UP.txt" ] && [ "${RESUME:-0}" != 1 ]; then echo "REFUSED: $Q/STOP_$UP.txt exists; RESUME=1 to continue"; exit 2; fi
need=("$K/steer_common.py" "$R1/lists/train_final.txt" "$R1/lists/eval141.txt" "$R1/tok/D/MANIFEST.json")
if [ "$STAGE" = loc ]; then need+=("$K/loc_fit.py" "$K/loc_steer.py" "$BOX"); else need+=("$K/crop_fit_s.py" "$K/crop_dump_s.py" "$JOIN" "$PRESENT/reader_s0_score.npz" "$PRESENT/reader_s1_score.npz" "$S/tok/S/MANIFEST.json" "$S/ctl/S8.json" "$S/ctl/S1.json" "$S/ctl/S2.json"); fi
for f in "${need[@]}"; do [ -e "$f" ] || stop "MISSING-INPUT $f"; done
if [ "$STAGE" = heads ]; then
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$S/ctl/S8.json')).get('pass') is True else 1)" || stop "S8-NOT-PASS"
  "$PY" -c "import json,sys; sys.exit(0 if (json.load(open('$S/ctl/S1.json')).get('pass') is True and json.load(open('$S/ctl/S2.json')).get('pass') is True) else 1)" || stop "S1-OR-S2-NOT-PASS"
fi
mmax=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max); anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat | head -n 1)
if [[ "$mmax" =~ ^[0-9]+$ ]] && [[ "$anon" =~ ^[0-9]+$ ]]; then [ $(( (mmax - anon) / 1073741824 )) -gt "$MINGB" ] || stop "MEMORY-FREE-TOO-LOW"; fi
echo "$(ts) start pid $$ resume ${RESUME:-0}" >> "$Q/START_$UP.txt"
run_loc() {  # name
  local n=$1
  echo "$(ts) START $n" > "$Q/jobs/$n.START"
  if [ ! -f "$S/loc/$n/LOC_RECORD.json" ]; then
    $(loccmd "$n") > "$Q/logs/loc_$n.log" 2>&1 || { echo "$(ts) loc rc $?" > "$Q/jobs/$n.STOP"; return 1; }
  fi
  echo "$(ts) DONE $n" > "$Q/jobs/$n.DONE"
}
run_head() {  # head seed perm
  local n; n=$(hname "$@")
  echo "$(ts) START $n" > "$Q/jobs/$n.START"
  if [ ! -f "$S/fits/$n/FIT_RECORD.json" ]; then
    $(fitcmd "$@") > "$Q/logs/fit_$n.log" 2>&1 || { echo "$(ts) fit rc $?" > "$Q/jobs/$n.STOP"; return 1; }
  fi
  local d; d=$(dname "$@")
  if [ ! -f "$S/dump/$d/reader_s$2_score.npz" ]; then
    $(dumpcmd "$@") > "$Q/logs/dump_$n.log" 2>&1 || { echo "$(ts) dump rc $?" > "$Q/jobs/$n.STOP"; return 1; }
  fi
  echo "$(ts) DONE $n" > "$Q/jobs/$n.DONE"
}
blocked() { [ -f "$Q/STOP_$UP.txt" ] && [ "${RESUME:-0}" != 1 ]; }
if [ "$STAGE" = loc ]; then
  for n in "${LOCJOBS[@]}"; do
    [ -f "$Q/jobs/$n.DONE" ] && continue
    [ -f "$Q/jobs/$n.STOP" ] && echo "$(ts) retry $n (an old .STOP stays on disk)" >> "$Q/START_$UP.txt"
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
    blocked && break
    ( run_loc "$n" || echo "$(ts) job $n failed" >> "$Q/STOP_$UP.txt" ) &
  done
  wait
  blocked && { echo "STOPPED: see $Q/STOP_$UP.txt"; exit 3; }
  for n in "${LOCJOBS[@]}"; do [ -f "$Q/jobs/$n.DONE" ] || { echo "ZZ STEERQ loc INCOMPLETE"; exit 3; }; done
  if [ ! -f "$Q/jobs/steer.DONE" ]; then
    echo "$(ts) START steer" > "$Q/jobs/steer.START"
    if [ -f "$S/ctl/S1.json" ] && [ -f "$S/ctl/S2.json" ]; then
      echo "$(ts) steer: S1 and S2 exist, step skipped" >> "$Q/START_$UP.txt"
    else
      # a half-written earlier run leaves steering files: --resume keeps them (checked for shape and finiteness), the S1/S2 records are written only at the end
      RES=""; [ -d "$S/steer" ] && RES="--resume"
      $(steercmd) $RES > "$Q/logs/steer.log" 2>&1 || { echo "$(ts) steer rc $?" > "$Q/jobs/steer.STOP"; echo "$(ts) job steer failed" >> "$Q/STOP_$UP.txt"; echo "STOPPED: steer"; exit 3; }
    fi
    echo "$(ts) DONE steer" > "$Q/jobs/steer.DONE"
  fi
  echo "$(ts) all jobs done" > "$Q/DONE_$UP.txt"; echo "ZZ STEERQ loc DONE"
else
  for j in "${HEADJOBS[@]}"; do
    set -- $j
    n=$(hname "$@")
    [ -f "$Q/jobs/$n.DONE" ] && continue
    [ -f "$Q/jobs/$n.STOP" ] && echo "$(ts) retry $n (an old .STOP stays on disk)" >> "$Q/START_$UP.txt"
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n || true; done
    blocked && break
    ( run_head "$@" || echo "$(ts) job $n failed" >> "$Q/STOP_$UP.txt" ) &
  done
  wait
  blocked && { echo "STOPPED: see $Q/STOP_$UP.txt"; exit 3; }
  ok=1
  for j in "${HEADJOBS[@]}"; do set -- $j; n=$(hname "$@"); [ -f "$Q/jobs/$n.DONE" ] || ok=0; done
  if [ "$ok" = 1 ]; then echo "$(ts) all jobs done" > "$Q/DONE_$UP.txt"; echo "ZZ STEERQ heads DONE"; else echo "ZZ STEERQ heads INCOMPLETE"; exit 3; fi
fi
