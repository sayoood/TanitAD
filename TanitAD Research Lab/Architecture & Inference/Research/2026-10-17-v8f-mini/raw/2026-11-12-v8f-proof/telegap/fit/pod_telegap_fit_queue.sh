#!/usr/bin/env bash
# pod_telegap_fit_queue.sh -- FIT queue of card PREREG_V8F_PROOF_ADD122 (TELEGAP-T0), note ADD122_N1. Pod, detached, resumable. Tier T0.
# Deletes nothing, moves nothing, kills nothing. One GPU job at a time. Head H5 (the head of the fixed-window arm of card ADD99).
# Gate (exit 2, nothing created): $ROOT/TELEGAP_ENC_DONE.txt must exist, $ROOT/TELEGAP_ENC_STOP.txt must not, and every input must exist.
# Steps in this order (fit then dump each): D2 s0 (identity), F s0 permuted labels, M s0 s1 s2, F s0 s1 s2.
# Markers in $ROOT: TELEGAP_FIT_START.txt (first line "<epoch s> <ISO>Z pid <pid>", written with noclobber; a later line per resume), TELEGAP_FIT_DONE.txt (last line
# "ZZ TELEGAP_FIT DONE <ISO>Z all steps done"), TELEGAP_FIT_STOP.txt (a line per stop; a start after a stop needs RESUME=1). Step files: $ROOT/fit_queue/jobs/<name>.DONE, logs in fit_queue/logs.
# Before each GPU job: wait until fewer than 3 GPU processes run and more than 28 GB of container memory are free (memory.max minus the anon line of memory.stat). The waiting counts into the
# stop rule: TG_FIT_STOP_S (default 10800 s) of wall time since this invocation started; a stop is a stop of the queue, not a result. Resume-safe: a fit with TELE_FIT_RECORD.json, a dump with its
# meta file, a step with its .DONE are skipped. DRY=1 prints "CMD: ..." lines and creates nothing.
# Start (lead): nohup setsid bash $K/pod_telegap_fit_queue.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=${TG_ROOT:-/workspace/v8proof/telegap}
R1=${TG_CROPREAD:-/workspace/v8proof/cropread}
CK=${TG_CROPKIT:-$R1/kit/fit}
K=${TG_KIT:-$(cd "$(dirname "$0")" && pwd)}
PY=${TG_PY:-/workspace/venv/bin/python}
WIDE=${TG_WIDE:-$R1/enc_out/tok/D}
TELE=${TG_TELE:-$ROOT/enc_out/tok_T}
MATCH=${TG_MATCH:-$ROOT/enc_out/match}
OUT=${TG_OUT:-$ROOT/fit_out}
LISTS=${TG_LISTS:-$R1/lists}
JOIN=${TG_JOIN:-/workspace/v8proof/data/inputs/join_train.npz}
PRES=${TG_PRESENT:-/workspace/v8proof/closeread/dump/both}
HEAD=H5
STOP_S=${TG_FIT_STOP_S:-10800}
POLL_S=${TG_POLL_S:-60}
MINGB=28
MAXGPU=3
DRY=${DRY:-0}
RESUME=${RESUME:-0}
Q=$ROOT/fit_queue
START=$ROOT/TELEGAP_FIT_START.txt
DONE=$ROOT/TELEGAP_FIT_DONE.txt
STOPF=$ROOT/TELEGAP_FIT_STOP.txt
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
JOBS=("D2 0 0" "F 0 1" "M 0 0" "M 1 0" "M 2 0" "F 0 0" "F 1 0" "F 2 0")     # arm seed perm
jname() { if [ "$3" = 1 ]; then echo "$1_${HEAD}_s$2_perm"; else echo "$1_${HEAD}_s$2"; fi; }
dname() { if [ "$3" = 1 ]; then echo "$1_${HEAD}_perm"; else echo "$1_${HEAD}"; fi; }
presfile() { if [ "$1" -le 1 ]; then echo "$PRES/reader_s$1_score.npz"; else echo "$PRES/reader_s0_score.npz"; fi; }   # seed 2 reads the present dump of seed 0
common() { echo "--arm $1 --head $HEAD --seed $2 --wide-tok-dir $WIDE --tele-tok-dir $TELE --match-dir $MATCH --out-root $OUT --kit-fit $CK --device cuda$([ "$3" = 1 ] && echo ' --permute-labels')"; }
fitcmd() { echo "$PY $K/tele_fit.py $(common "$@") --lists-dir $LISTS --join-train $JOIN"; }
dumpcmd() { echo "$PY $K/tele_dump.py $(common "$@") --present-file $(presfile "$2")"; }

if [ "$DRY" = 1 ]; then
  echo "DRY=1: nothing is created; commands are printed"
  echo "CMD: gate: REFUSED (exit 2) unless $ROOT/TELEGAP_ENC_DONE.txt exists and $ROOT/TELEGAP_ENC_STOP.txt does not; inputs $WIDE $TELE $MATCH $LISTS $JOIN $PRES must exist"
  echo "CMD: before each GPU job wait: GPU processes < $MAXGPU and container memory free > $MINGB GB (memory.max minus anon); stop rule $STOP_S s (waiting counts)"
  for j in "${JOBS[@]}"; do
    set -- $j
    echo "JOB $(jname "$@"):"
    echo "  CMD: $(fitcmd "$@")"
    echo "  CMD: $(dumpcmd "$@")"
  done
  echo "DRY run complete: one GPU job at a time"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ gate: refusals create nothing
[ -f "$ROOT/TELEGAP_ENC_DONE.txt" ] || { echo "REFUSED: no $ROOT/TELEGAP_ENC_DONE.txt"; exit 2; }
[ -f "$ROOT/TELEGAP_ENC_STOP.txt" ] && { echo "REFUSED: $ROOT/TELEGAP_ENC_STOP.txt exists"; exit 2; }
[ -f "$DONE" ] && { echo "ALREADY-DONE"; exit 0; }
if [ -f "$STOPF" ] && [ "$RESUME" != 1 ]; then echo "REFUSED: $STOPF exists; RESUME=1 to continue"; exit 2; fi
if [ -f "$START" ] && [ ! -f "$STOPF" ] && [ "$RESUME" != 1 ]; then echo "REFUSED: $START exists (a run may be live); RESUME=1 to continue"; exit 2; fi
for f in "$K/tele_fit.py" "$K/tele_dump.py" "$CK/crop_fit.py" "$CK/crop_dump.py" "$LISTS/train_final.txt" "$LISTS/eval141.txt" "$JOIN" "$PRES/reader_s0_score.npz" "$PRES/reader_s1_score.npz"; do
  [ -f "$f" ] || { echo "REFUSED: missing input $f"; exit 2; }
done
for d in "$WIDE" "$TELE" "$MATCH"; do
  [ -d "$d" ] || { echo "REFUSED: missing folder $d"; exit 2; }
done
mkdir -p "$Q/jobs" "$Q/logs" "$OUT" || { echo "REFUSED: cannot make $Q"; exit 2; }
if [ ! -f "$START" ]; then
  ( set -C; echo "$(date -u +%s) $(ts) pid $$" > "$START" ) 2>/dev/null || { echo "REFUSED: cannot create $START"; exit 2; }
else
  echo "$(date -u +%s) $(ts) resume pid $$" >> "$START"
fi
if command -v flock > /dev/null 2>&1; then
  exec 9> "$Q/queue.lock"
  flock -n 9 || { echo "REFUSED: another queue holds $Q/queue.lock"; exit 2; }
fi
T0=$(date -u +%s)

stop() { echo "$(ts) $*" >> "$STOPF"; echo "STOP $*"; exit 3; }
free_gb() {
  local mmax anon
  mmax=$(tr -d ' \r\n' < /sys/fs/cgroup/memory.max 2>/dev/null); anon=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat 2>/dev/null | head -n 1)
  if [[ "$mmax" =~ ^[0-9]+$ ]] && [[ "$anon" =~ ^[0-9]+$ ]]; then echo $(( (mmax - anon) / 1073741824 )); else echo 9999; fi
}
gpu_procs() {
  if command -v nvidia-smi > /dev/null 2>&1; then nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'; else echo 0; fi
}
wait_slot() {
  while true; do
    [ -f "$STOPF" ] && [ "$RESUME" != 1 ] && stop "STOP-MARKER-APPEARED"
    [ $(( $(date -u +%s) - T0 )) -gt "$STOP_S" ] && stop "STOP-RULE ${STOP_S}s reached (waiting counts) before $1"
    g=$(gpu_procs); m=$(free_gb)
    if [ "$g" -lt "$MAXGPU" ] && [ "$m" -gt "$MINGB" ]; then return 0; fi
    echo "$(ts) wait before $1: gpu_procs $g free_gb $m" >> "$Q/wait.log"
    sleep "$POLL_S"
  done
}
for j in "${JOBS[@]}"; do
  set -- $j
  n=$(jname "$@"); d=$(dname "$@")
  [ -f "$Q/jobs/$n.DONE" ] && continue
  fd=$OUT/fits/$n
  if [ ! -f "$fd/TELE_FIT_RECORD.json" ]; then
    [ -f "$fd/FIT_RECORD.json" ] && stop "FIT-WITHOUT-TELE-RECORD $fd (crop_fit finished, tele record missing; look before going on)"
    wait_slot "fit $n"
    echo "$(ts) START fit $n" > "$Q/jobs/$n.START"
    $(fitcmd "$@") > "$Q/logs/fit_$n.log" 2>&1 || stop "fit $n rc $?"
  fi
  if [ ! -f "$OUT/dump/$d/reader_s$2_meta.json" ]; then
    [ -f "$OUT/dump/$d/reader_s$2_score.npz" ] && stop "DUMP-WITHOUT-META $OUT/dump/$d/reader_s$2_score.npz"
    wait_slot "dump $n"
    $(dumpcmd "$@") > "$Q/logs/dump_$n.log" 2>&1 || stop "dump $n rc $?"
  fi
  echo "$(ts) DONE $n" > "$Q/jobs/$n.DONE"
done
echo "$(ts) all jobs finished" >> "$DONE"
echo "ZZ TELEGAP_FIT DONE $(ts) all steps done" >> "$DONE"
echo "ZZ TELEGAP_FIT DONE"
