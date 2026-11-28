#!/usr/bin/env bash
# wait_seq2.sh <NAME> <BASEARM> "<run tags>" [responds yes|no|unknown] : dev-box side, script polling (no model turns). Waits for the pod sequence <NAME>
# (pod_imagplan_seq2.sh; card ADD40, notes N1 and N2) to end, prints its log, pulls the small logs of the listed runs (imag log, action log, run record)
# with a sha256 compare into imagplan/pull/<run>_<BASEARM>/, then reads p4b, p5b and p7 (plumb_read2.py; no unsafe rate is read).
# Deletes nothing. Last line: ZZ WAITSEQ <NAME> DONE or STOP.
NAME=${1:?name}; BASE=${2:?base arm}; TAGS=${3:?tags}; RESP=${4:-unknown}
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
H=/f/TanitAD-scratch/v8proof/imagplan; R=/workspace/v8proof/runs/imagplan
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
ts() { date -u +%H:%M:%SZ; }
k=0
while :; do
  k=$((k+1))
  s=$(timeout 60 bash "$P" "ls $R/SEQ_${NAME}_DONE.txt 2>/dev/null | wc -l; grep -c 'ZZ SEQ STOP' $R/seq_$NAME.log 2>/dev/null" 2>/dev/null | tr -d '\r' | tr '\n' ' ')
  set -- $s
  [ "${1:-0}" = 1 ] && break
  [ "${2:-0}" -ge 1 ] 2>/dev/null && { echo "[$(ts)] sequence $NAME stopped"; timeout 60 bash "$P" "cut -c1-260 $R/seq_$NAME.log" 2>/dev/null | sed '/hf_/d'; echo "ZZ WAITSEQ $NAME STOP"; exit 2; }
  [ "$k" -gt 144 ] && { echo "[$(ts)] sequence $NAME not ended after 12 h"; echo "ZZ WAITSEQ $NAME STOP"; exit 3; }
  sleep 300
done
echo "[$(ts)] sequence $NAME done"
timeout 60 bash "$P" "cut -c1-260 $R/seq_$NAME.log" 2>/dev/null | sed '/hf_/d'
for tag in $TAGS; do
  new=${tag}_$BASE
  fs=$(timeout 60 bash "$P" "cd $R/t1/$new/ticks 2>/dev/null && ls imag_log_$new.npz p45_$new.npz imagplan_run_$new.txt 2>/dev/null" 2>/dev/null | tr -d '\r')
  [ -n "$fs" ] || continue
  mkdir -p "$H/pull/$new"
  for f in $fs; do
    bash "$C" "$R/t1/$new/ticks/$f" "$H/pull/$new/$f" < /dev/null > /dev/null 2>&1
    a=$(timeout 60 bash "$P" "sha256sum $R/t1/$new/ticks/$f | cut -d' ' -f1" 2>/dev/null | tr -cd 'a-f0-9')
    b=$(sha256sum "$H/pull/$new/$f" 2>/dev/null | cut -d' ' -f1)
    if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "[$(ts)] pulled $new/$f sha256 ${a:0:16}"; else echo "[$(ts)] PULL-MISMATCH $new/$f pod ${a:0:16} local ${b:0:16}"; fi
  done
done
I=$H/pull/IPib_$BASE; CV=$H/pull/IPc_$BASE/imag_log_IPc_$BASE.npz
if [ -f "$I/p45_IPib_$BASE.npz" ] && [ -f "$I/imag_log_IPib_$BASE.npz" ]; then
  cvarg=""; [ -f "$CV" ] && cvarg="--cv-log $CV"
  ( cd "$H" && "$PY" plumb_read2.py --p45 "$I/p45_IPib_$BASE.npz" --imag-log "$I/imag_log_IPib_$BASE.npz" $cvarg --responds "$RESP" --out "$H/read_plumb/PLUMB2_$BASE.json" 2>&1 | cut -c1-700 )
else
  echo "[$(ts)] no IPib logs for $BASE: plumbing read not run"
fi
echo "ZZ WAITSEQ $NAME DONE $(ts)"
