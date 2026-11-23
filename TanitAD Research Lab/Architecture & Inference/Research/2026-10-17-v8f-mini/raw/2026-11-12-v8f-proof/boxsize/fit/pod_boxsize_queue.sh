#!/usr/bin/env bash
# pod_boxsize_queue.sh -- pod queue of card PREREG_V8F_PROOF_ADD124 (BOXSIZE-T0). Tier T0.
# Steps in this order, one at a time: ident (boxsize_fit.py --identity), fit_s0, fit_s1, fit_s2, fit_s0_perm (fit, then prediction, in the same call).
# A GPU step starts only with fewer than 3 GPU compute processes on the card and more than BS_MIN_FREE_GB (28) GB free in the container (memory.max minus anon);
# the wait counts into the stop rule BS_STOP_S (default 5400 s since the start marker): when it is used up the queue writes the stop marker with the reason and ends.
# Markers in $B: BOXSIZE_START.txt (<epoch> <UTC time> pid <pid>), BOXSIZE_DONE.txt (last line: ZZ BOXSIZE DONE <UTC time> all steps done), BOXSIZE_STOP.txt (the reason). RESUME=1 continues after a stop.
# A finished fit with its prediction is skipped; a fit without prediction gets --predict; nothing is overwritten. Per step: $Q/jobs/<step>.START / .DONE / .STOP, logs in $Q/logs.
# The queue contains no delete and no relocate command and does not stop any process. Writes only under $B.
# Usage: bash pod_boxsize_queue.sh   |   RESUME=1 bash pod_boxsize_queue.sh   |   DRY=1 bash pod_boxsize_queue.sh   (prints every command, creates nothing)
set -u
ulimit -n 16384 2>/dev/null || ULIM_FAIL=1
V=/workspace/v8proof
B=${BS_ROOT:-$V/boxsize}
O=${BS_OUT:-$B/out}
Q=${BS_QUEUE:-$B/queue}
K=${BS_KIT:-$B/kit}
R1=${BS_RUN1:-$V/cropread/run1}
KF=${BS_KIT_FIT:-$V/cropread/kit/fit}
LOC=${BS_LOCATOR:-$V/lookat/kit/locator_fit.py}
BOX=${BS_BOX:-$V/cropread/enc_out/box}
JOIN=${BS_JOIN_TRAIN:-$V/data/inputs/join_train.npz}
PY=${BS_PY:-/workspace/venv/bin/python}
CG=${BS_CG:-/sys/fs/cgroup}
MINGB=${BS_MIN_FREE_GB:-28}
STOPS=${BS_STOP_S:-5400}
DRY=${DRY:-0}
START=$B/BOXSIZE_START.txt
DONE=$B/BOXSIZE_DONE.txt
STOPF=$B/BOXSIZE_STOP.txt
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} PYTHONIOENCODING=utf-8
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
STEPS=(fit_s0 fit_s1 fit_s2 fit_s0_perm)
identcmd() { echo "$PY $K/boxsize_fit.py --locator $LOC --identity --root $R1 --out-root $O --kit-fit $KF --join-train $JOIN --device cuda"; }
fitcmd() { # step
  local s=${1#fit_s}; s=${s%_perm}; local p=""; [[ "$1" == *_perm ]] && p=" --permute-targets"
  echo "$PY $K/boxsize_fit.py --locator $LOC --seed $s$p --root $R1 --box-dir $BOX --out-root $O --kit-fit $KF --device cuda"
}
predcmd() { echo "$(fitcmd "$1") --predict"; }
if [ "$DRY" = 1 ]; then
  echo "CMD: ulimit -n 16384"
  echo "CMD: need $K/boxsize_fit.py $LOC $KF/crop_fit.py $JOIN $R1/lists/train_final.txt $R1/lists/eval141.txt $R1/fits/D_H5_s0/head.pt $BOX; free memory > $MINGB GB and fewer than 3 GPU compute processes before every step, the wait counts into the stop rule of $STOPS s since the start"
  echo "CMD: mkdir -p $O/fits $O/pred $Q/jobs $Q/logs"
  echo "STEP 1 ident:"; echo "  CMD: $(identcmd)"
  n=2
  for j in "${STEPS[@]}"; do echo "STEP $n $j:"; echo "  CMD: $(fitcmd "$j")   (a finished fit without prediction: $(predcmd "$j"))"; n=$((n + 1)); done
  echo "MARKERS: $START $DONE $STOPF"
  echo "DRY run complete: 5 steps, one at a time, nothing created"; exit 0
fi
stop() { mkdir -p "$B"; echo "$(ts) $*" >> "$STOPF"; echo "STOP $*"; exit 3; }
[ -z "${ULIM_FAIL:-}" ] || [ "${BS_ULIMIT_SOFT:-0}" = 1 ] || stop "ULIMIT-N-16384-REFUSED"      # BS_ULIMIT_SOFT=1 is for the dev-box tests only
[ -f "$DONE" ] && { echo "ALREADY-DONE"; exit 0; }
if [ -f "$STOPF" ] && [ "${RESUME:-0}" != 1 ]; then echo "REFUSED: $STOPF exists; RESUME=1 to continue"; exit 2; fi
for f in "$K/boxsize_fit.py" "$LOC" "$KF/crop_fit.py" "$JOIN" "$R1/lists/train_final.txt" "$R1/lists/eval141.txt" "$R1/fits/D_H5_s0/head.pt" "$R1/fits/D_H5_s0/FIT_RECORD.json"; do [ -f "$f" ] || stop "MISSING-INPUT $f"; done
[ -d "$BOX" ] || stop "MISSING-INPUT $BOX"
command -v nvidia-smi >/dev/null 2>&1 || stop "NO-NVIDIA-SMI"
mkdir -p "$O/fits" "$O/pred" "$Q/jobs" "$Q/logs" || stop "CANNOT-MAKE-FOLDERS"
if [ -f "$START" ]; then T0=$(awk 'NR==1{print $1}' "$START"); else T0=$(date +%s); echo "$T0 $(ts) pid $$" > "$START"; fi
[[ "$T0" =~ ^[0-9]+$ ]] || T0=$(date +%s)
over() { [ $(( $(date +%s) - T0 )) -ge "$STOPS" ]; }
free_gb() { local mm an; mm=$(tr -d ' \r\n' < "$CG/memory.max" 2>/dev/null); an=$(awk '$1=="anon"{print $2}' "$CG/memory.stat" 2>/dev/null | head -n 1)
  if [[ "$mm" =~ ^[0-9]+$ ]] && [[ "$an" =~ ^[0-9]+$ ]]; then echo $(( (mm - an) / 1073741824 )); else echo 999; fi; }
gpu_procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
admit() { # waits for a slot; returns 1 when the stop rule is used up
  local g n
  while :; do
    over && return 1
    g=$(free_gb); n=$(gpu_procs); n=${n:-0}
    if [ "$g" -gt "$MINGB" ] && [ "$n" -lt 3 ]; then return 0; fi
    sleep "${BS_POLL_S:-30}"
  done
}
step_ident() {
  echo "$(ts) START ident" > "$Q/jobs/ident.START"
  if ! grep -q '"PASS": true' "$O/IDENTITY_K5.json" 2>/dev/null; then
    $(identcmd) > "$Q/logs/ident.log" 2>&1 || { echo "$(ts) identity rc $? (the identity run did not pass)" > "$Q/jobs/ident.STOP"; return 1; }
  fi
  echo "$(ts) DONE ident" > "$Q/jobs/ident.DONE"
}
step_fit() { # step
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
if [ ! -f "$Q/jobs/ident.DONE" ]; then
  admit || stop "STOP-RULE-${STOPS}s-USED-UP-WAITING-FOR-A-SLOT-ident"
  step_ident || stop "identity run did not pass"
fi
for j in "${STEPS[@]}"; do
  [ -f "$Q/jobs/$j.DONE" ] && continue
  admit || stop "STOP-RULE-${STOPS}s-USED-UP-WAITING-FOR-A-SLOT-$j"
  step_fit "$j" || stop "step $j failed"
done
echo "ZZ BOXSIZE DONE $(ts) all steps done" >> "$DONE"
echo "ZZ BOXSIZE DONE"
