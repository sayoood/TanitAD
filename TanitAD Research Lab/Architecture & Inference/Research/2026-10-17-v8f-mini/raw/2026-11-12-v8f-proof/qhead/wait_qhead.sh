#!/bin/bash
# wait_qhead.sh -- dev box, card PREREG_V8F_PROOF_ADD42 (QHEAD). Written by the lead. Polls the pod every 300 s for the end marker of pod_qhead_queue.sh
# (qhead_run/QHEAD_DONE.txt or QHEAD_STOP.txt; bound 16 h). On DONE: pulls the small files (heads, fit records, queue log) into qhead/pulled/ and the five
# rollouts (Q=distrule) into F: (imagplan/pull_t1; C: is at its floor), then runs qhead/qhead_read.sh (readers r2, control h3 per arm, names with controls
# first). On STOP: pulls the queue log only. Starts nothing on the pod. Deletes nothing. Last line: ZZ QHEAD-WAIT <DONE|STOP|TIMEOUT>.
set -u
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof
cd $H || exit 9
P=/c/Users/Admin/v8pod/pssh.sh
C=/c/Users/Admin/v8pod/pcat.sh
RUNF=/workspace/v8proof/qhead_run
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
pod() { timeout 60 bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
pull1() {  # remote local
  local a b
  a=$(pod "sha256sum $1 2>/dev/null | cut -d' ' -f1")
  [ -n "$a" ] || { echo "[$(ts)] not on the pod: $1"; return 1; }
  mkdir -p "$(dirname "$2")"
  bash "$C" "$1" "$2" < /dev/null > /dev/null 2>&1
  b=$(sha256sum "$2" 2>/dev/null | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "[$(ts)] pulled $(basename "$1") sha256 ${a:0:16}"; else echo "[$(ts)] PULL-MISMATCH $1"; return 1; fi
}
n=0; st=""
while true; do
  s=$(pod "cd $RUNF 2>/dev/null && { [ -f QHEAD_DONE.txt ] && echo DONE; [ -f QHEAD_STOP.txt ] && echo STOP; echo MARK; tail -n 1 qhead_queue.log 2>/dev/null | cut -c1-170; }")
  case "$s" in DONE*) st=DONE; break ;; STOP*) st=STOP; break ;; esac
  n=$((n+1))
  [ $((n % 6)) = 1 ] && echo "[$(ts)] poll $n: $(echo "$s" | tr '\n' ';' | cut -c1-240)"
  if [ "$n" -gt 192 ]; then st=TIMEOUT; break; fi
  sleep 300
done
echo "[$(ts)] queue state $st"
if [ "$st" = TIMEOUT ]; then echo "ZZ QHEAD-WAIT TIMEOUT $(ts)"; exit 8; fi
pull1 $RUNF/qhead_queue.log qhead/pulled/qhead_queue.log
[ "$st" = STOP ] && pull1 $RUNF/QHEAD_STOP.txt qhead/pulled/QHEAD_STOP.txt
[ "$st" = DONE ] && pull1 $RUNF/QHEAD_DONE.txt qhead/pulled/QHEAD_DONE.txt
for b in LS_s0 LS_s1 LSD_s0 LSD_s1; do
  pull1 $RUNF/heads/$b.qhead.json qhead/pulled/heads/$b.qhead.json
  pull1 $RUNF/heads/$b.qhead.npz qhead/pulled/heads/$b.qhead.npz
done
if [ "$st" != DONE ]; then tail -n 6 qhead/pulled/qhead_queue.log 2>/dev/null | cut -c1-240; echo "ZZ QHEAD-WAIT STOP $(ts)"; exit 0; fi
export PULL_ROOT=$H/imagplan/pull_t1 RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
mkdir -p "$PULL_ROOT"
for a in LSQHI_s0 LSQH_s0 LSQH_s1 LSDQH_s0 LSDQH_s1; do
  [ -f "$PULL_ROOT/$a/PULLED.txt" ] || Q=distrule bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-160
done
PULL=$PULL_ROOT ROOTS="$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull,$W/imagplan/pull_t1" bash qhead/qhead_read.sh > qhead/qhead_read.out 2>&1
echo "[$(ts)] qhead_read.sh rc $?"
tail -n 40 qhead/qhead_read.out | cut -c1-330
echo "ZZ QHEAD-WAIT DONE $(ts)"
