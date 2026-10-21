#!/bin/bash
# chain_l7cap_yield_thor.sh -- Master Mind 2026-10-22 00:40 Berlin.  H-L7-1 Thor capture, QUEUED BEHIND the R8-PICKER rolls and YIELDING to
# NT-DECIDE (H-NT-DECIDE-1 has priority on Thor; PI-authorised 2026-10-21).
#  1. waits (pure bash, no python) until /home/nvidia/r8ns/picker/out/waiter.log carries the picker's done or give-up marker (30 h cap);
#  2. runs the four capture steps of chain_l7cap_thor.sh, REORDERED so the K3 control capture (eval139 seed 1, the seed of the banked Thor
#     fanceil fan) runs FIRST: eval139 s1 -> train139 s0 -> train139 s1 -> eval139 s0;
#  3. YIELD RULE: before EACH step, if /home/nvidia/r8ns/nt/NT_LAUNCH_PENDING exists, the chain stops (the NT launch then waits at most one
#     running step for the GPU lock). Each step takes the GPU lock with `flock -o` (fd closed in the child) and waits at most 1 h for it.
# Opaque markers in $L/out/chain_yield.log: ZZL7YSTARTZZ | ZZL7YPOLLZZ | ZZL7YPKZZ <done|giveup> | ZZL7YYIELDZZ <step> | ZZL7YSTEPZZ <step> rc=N |
#   ZZL7YENDZZ | ZZL7YGIVEUPZZ <why>.  Launch detached:  nohup setsid bash chain_l7cap_yield_thor.sh > /dev/null 2>&1 < /dev/null 200>&- &
L=/home/nvidia/r8ns/l7cap; LOCK=/home/nvidia/refcv7_post/thor_gpu.lock; CH=$L/out/chain_yield.log; mkdir -p "$L/out"
PK=/home/nvidia/r8ns/picker/out/waiter.log; NTP=/home/nvidia/r8ns/nt/NT_LAUNCH_PENDING
echo "ZZL7YSTARTZZ $(date -Is) pid $$" >> "$CH"
t0=$(date +%s); n=0
while ! grep -qE 'ZZPKW(DONE|GIVEUP)ZZ' "$PK" 2>/dev/null; do
  if [ $(( $(date +%s) - t0 )) -gt 108000 ]; then echo "ZZL7YGIVEUPZZ picker-not-done-after-30h $(date -Is)" >> "$CH"; exit 9; fi
  n=$((n+1)); [ $((n % 6)) = 0 ] && echo "ZZL7YPOLLZZ $(date -Is)" >> "$CH"
  sleep 300 200>&-
done
echo "ZZL7YPKZZ $(grep -oE 'ZZPKW(DONE|GIVEUP)ZZ' "$PK" | tail -n 1) $(date -Is)" >> "$CH"
for step in "1 eval139" "0 train139" "1 train139" "0 eval139"; do
  set -- $step
  if [ -e "$NTP" ]; then echo "ZZL7YYIELDZZ seed=$1 mode=$2 $(date -Is)" >> "$CH"; exit 0; fi
  flock -o -w 3600 "$LOCK" env L7C_LOCKED=1 bash "$L/code/run_l7cap_thor.sh" "$1" "$2" 200>&-
  rc=$?
  echo "ZZL7YSTEPZZ seed=$1 mode=$2 rc=$rc $(date -Is)" >> "$CH"
done
echo "ZZL7YENDZZ $(date -Is)" >> "$CH"
