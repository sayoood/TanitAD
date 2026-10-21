#!/bin/bash
# Wait for the Thor lever pool (18 jobs) to drain, pull the scores, apply the pre-fixed reading rule. Markers ZZLVW<STAGE>ZZ.
set -u
D=/c/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag; LOG=$D/lever_waiter.log; L=/home/nvidia/r8ns/navtrain_sel/lever
mark() { echo "ZZLVW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START
while :; do
  z=$(ssh -n tanitad-thor "tail -c 40 $L/pool.out" 2>/dev/null | grep -oE 'ZZ[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+ZZ' | tail -1)
  al=$(echo "$z" | tr -d Z | cut -d- -f3); q=$(echo "$z" | tr -d Z | cut -d- -f4)
  if [ -n "$z" ] && [ "$al" = "0" ] && [ "$q" = "0" ]; then mark POOLDONE "$z"; break; fi
  sleep 120
done
mkdir -p $D/thor_scores
ssh -n tanitad-thor "cd $L && tar -cf - scores pool_state.json" | tar -xf - -C $D/thor_scores
n=$(ls -d $D/thor_scores/scores/r8sel_lever_* 2>/dev/null | wc -l); mark PULLED "dirs=$n"
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 $D/read_levers.py $D/thor_scores/scores $D/LEVER_READ.json > $D/read_levers.out 2>&1
[ -s $D/LEVER_READ.json ] && mark READOK "$(grep -o 'QQLEVERS.*' $D/read_levers.out | cut -c1-400)" || mark READBAD
mark END
