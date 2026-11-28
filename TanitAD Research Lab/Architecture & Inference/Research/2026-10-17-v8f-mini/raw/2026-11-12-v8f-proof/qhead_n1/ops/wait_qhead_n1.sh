#!/bin/bash
# wait_qhead_n1.sh -- dev box, notes PREREG_V8F_PROOF_ADD42_N1 and _N2. Written by the lead. Polls the pod every 180 s for the end marker of
# pod_qhead_n1_rollouts.sh (qhead_run_n1/ROLLOUTS_DONE.txt or ROLLOUTS_STOP.txt; bound 9 h). Then pulls the four head records and heads into qhead_n1/pulled/heads,
# the rollout log, and on DONE the rollouts named on the command line (Q=distrule) into imagplan/pull_t1, and runs qhead_n1/qhead_read_n2.sh.
# Starts nothing on the pod. Deletes nothing. Last line: ZZ QHEAD-N1-WAIT <DONE|STOP|TIMEOUT>.
set -u
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof
cd $H || exit 9
P=/c/Users/Admin/v8pod/pssh.sh
C=/c/Users/Admin/v8pod/pcat.sh
RUNF=/workspace/v8proof/qhead_run_n1
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
  s=$(pod "cd $RUNF 2>/dev/null && { [ -f ROLLOUTS_DONE.txt ] && echo DONE; [ -f ROLLOUTS_STOP.txt ] && echo STOP; echo MARK; tail -n 1 $RUNF/../qhead_n1_rollouts.out 2>/dev/null | cut -c1-170; }")
  case "$s" in DONE*) st=DONE; break ;; STOP*) st=STOP; break ;; esac
  n=$((n+1))
  [ $((n % 10)) = 1 ] && echo "[$(ts)] poll $n: $(echo "$s" | tr '\n' ';' | cut -c1-240)"
  if [ "$n" -gt 180 ]; then st=TIMEOUT; break; fi
  sleep 180
done
echo "[$(ts)] rollout queue state $st"
if [ "$st" = TIMEOUT ]; then echo "ZZ QHEAD-N1-WAIT TIMEOUT $(ts)"; exit 8; fi
pull1 /workspace/v8proof/qhead_n1_rollouts.out qhead_n1/pulled/qhead_n1_rollouts.out
pull1 /workspace/v8proof/qhead_n1_fits.out qhead_n1/pulled/qhead_n1_fits.out
for b in LS_s0 LS_s1 LSD_s0 LSD_s1; do
  pull1 $RUNF/heads/$b.qhead.json qhead_n1/pulled/heads/$b.qhead.json
  pull1 $RUNF/heads/$b.qhead.npz qhead_n1/pulled/heads/$b.qhead.npz
done
if [ "$st" != DONE ]; then tail -n 6 qhead_n1/pulled/qhead_n1_rollouts.out 2>/dev/null | cut -c1-240; echo "ZZ QHEAD-N1-WAIT STOP $(ts)"; exit 0; fi
export PULL_ROOT=$H/imagplan/pull_t1 RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
mkdir -p "$PULL_ROOT"
for a in "$@"; do
  [ -f "$PULL_ROOT/$a/PULLED.txt" ] || Q=distrule bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-160
done
bash qhead_n1/qhead_read_n2.sh > qhead_n1/qhead_read_n2.out 2>&1
echo "[$(ts)] qhead_read_n2.sh rc $?"
tail -n 40 qhead_n1/qhead_read_n2.out | cut -c1-330
echo "ZZ QHEAD-N1-WAIT DONE $(ts)"
