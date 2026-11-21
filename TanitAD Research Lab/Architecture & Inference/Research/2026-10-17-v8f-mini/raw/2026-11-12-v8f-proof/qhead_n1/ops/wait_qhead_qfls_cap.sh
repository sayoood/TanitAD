#!/bin/bash
# wait_qhead_qfls.sh -- dev box, note PREREG_V8F_PROOF_ADD42_N2 section 3 (dose arms LSQF_s0, LSQF_s1). Written by the lead. Polls the pod every 180 s for the end
# marker of pod_qhead_qx_rollouts.sh in qhead_dose/run_roll (ROLLOUTS_DONE.txt or ROLLOUTS_STOP.txt; bound 9 h). Then pulls the rollout log and, on DONE, the two
# rollouts (Q=distrule) into imagplan/pull_t1, and runs qhead_n1/qhead_read_qfls_cap.sh. Starts nothing on the pod. Deletes nothing.
# Last line: ZZ QHEAD-QF-WAIT <DONE|STOP|TIMEOUT>.
set -u
H=/f/TanitAD-scratch/v8proof; W=F:/TanitAD-scratch/v8proof
cd $H || exit 9
P=/c/Users/Admin/v8pod/pssh.sh
C=/c/Users/Admin/v8pod/pcat.sh
RUNF=/workspace/v8proof/qhead_dose_ls/run_roll
ROUT=/workspace/v8proof/qhead_dose_ls/rollouts.out
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
  s=$(pod "cd $RUNF 2>/dev/null && { [ -f ROLLOUTS_DONE.txt ] && echo DONE; [ -f ROLLOUTS_STOP.txt ] && echo STOP; echo MARK; tail -n 1 $ROUT 2>/dev/null | cut -c1-170; }")
  case "$s" in DONE*) st=DONE; break ;; STOP*) st=STOP; break ;; esac
  n=$((n+1))
  [ $((n % 10)) = 1 ] && echo "[$(ts)] poll $n: $(echo "$s" | tr '\n' ';' | cut -c1-240)"
  if [ "$n" -gt 180 ]; then st=TIMEOUT; break; fi
  sleep 180
done
echo "[$(ts)] rollout queue state $st"
if [ "$st" = TIMEOUT ]; then echo "ZZ QHEAD-QF-WAIT TIMEOUT $(ts)"; exit 8; fi
pull1 $ROUT qhead_n1/dose/pulled_ls/rollouts.out
if [ "$st" != DONE ]; then tail -n 6 qhead_n1/dose/pulled_ls/rollouts.out 2>/dev/null | cut -c1-240; echo "ZZ QHEAD-QF-WAIT STOP $(ts)"; exit 0; fi
export PULL_ROOT=$H/imagplan/pull_t1 RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
mkdir -p "$PULL_ROOT"
for a in LSQF_s0 LSQF_s1; do
  [ -f "$PULL_ROOT/$a/PULLED.txt" ] || Q=distrule bash spatial/q3/pull_spatial_t1.sh "$a" 2>&1 | tail -n 1 | cut -c1-160
done
bash qhead_n1/qhead_read_qfls_cap.sh > qhead_n1/qhead_read_qfls.out 2>&1
echo "[$(ts)] qhead_read_qfls.sh rc $?"
tail -n 40 qhead_n1/qhead_read_qfls.out | cut -c1-330
echo "ZZ QHEAD-QF-WAIT DONE $(ts)"
