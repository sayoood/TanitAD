#!/bin/bash
# SAFEHEAD queue (PREREG_SAFEHEAD.md s.7).  ONE run at a time, 6 threads, below-normal priority (inherited from the launcher).  Gated on the identity controls.
# Order: banked P controls -> P K=50 (SH_SG, SH_F, SH_A) -> T K=250 (SH_SG, SH_A, SH_F).  sh_summarize.py re-run after every arm (cheap; missing inputs read PENDING).
# Completion lines: logs/queue.txt.  Final line: ZZSH_QUEUE_DONE.
export PYTHONUTF8=1 TMP=C:/Users/Admin/AppData/Local/Temp TEMP=C:/Users/Admin/AppData/Local/Temp
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd "$(dirname "$0")"
Q=logs/queue.txt
echo "$(date -u +%FT%TZ) queue start pid $$" >> $Q
while ! grep -q "ZZCONTROLS" logs/controls.log 2>/dev/null; do sleep 30; done
if ! grep -q "ZZCONTROLS ALL_PASS" logs/controls.log; then echo "$(date -u +%FT%TZ) CONTROLS FAILED -- queue stops, no safety-head number is produced" >> $Q; exit 3; fi
echo "$(date -u +%FT%TZ) controls ALL_PASS" >> $Q
t0=$(date +%s); SH_THREADS=6 $PY sh_banked_eval.py > logs/banked_P.log 2>&1; echo "$(date -u +%FT%TZ) banked_P rc $? $(( $(date +%s) - t0 ))s" >> $Q
run() {  # design regime K snaps... [extra]
  d=$1; rg=$2; K=$3; shift 3
  t0=$(date +%s)
  $PY sh_closed.py --design $d --lam 1.0 --regime $rg --K $K "$@" --out-dir out --threads 6 > "logs/sh_${rg}_${d}_K${K}.log" 2>&1
  echo "$(date -u +%FT%TZ) $rg $d K$K rc $? $(( $(date +%s) - t0 ))s" >> $Q
  $PY sh_summarize.py > logs/summarize_latest.log 2>&1
}
run SH_SG P 50 --snaps 25 50
run SH_F  P 50 --snaps 25 50
run SH_A  P 50 --snaps 25 50 --identity-vs-banked
echo "$(date -u +%FT%TZ) ZZSH_P_DONE" >> $Q
run SH_SG T 250 --snaps 50 100 250
run SH_A  T 250 --snaps 50 100 250
run SH_F  T 250 --snaps 50 100 250
echo "$(date -u +%FT%TZ) ZZSH_QUEUE_DONE" >> $Q
