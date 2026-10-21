#!/bin/bash
# Dev-box reader for R8-WISEFT / R8-CKPT-TRAJ: wait for Thor's scoring loop to finish, pull the scores, apply the registered rule.
# Markers ZZWWW<STAGE>ZZ.
set -u
W=/c/Users/Admin/refcv8_pf/navsim_ckpt/wiseft; LOG=$W/wiseft_waiter.log; O=/home/nvidia/r8ns/navtrain_sel/wiseft
mark() { echo "ZZWWW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START; t0=$(date +%s)
while :; do
  r=$(ssh -n tanitad-thor "grep -c 'ZZWALLSCORE[D]' $O/score_loop.out; grep -cE 'ZZWSCORE(ABOR[T]|TIMEOU[T])' $O/score_loop.out; grep -c 'ZZWSCORE[D]_' $O/score_loop.out" 2>/dev/null | tr '\n' ' ')
  set -- $r
  [ "${2:-0}" != "0" ] && [ -n "${2:-}" ] && { mark ABORTED "$r"; exit 1; }
  [ "${1:-0}" != "0" ] && [ -n "${1:-}" ] && break
  [ $(( $(date +%s) - t0 )) -gt 25200 ] && { mark TIMEOUT "$r"; exit 1; }
  sleep 120
done
mark ALLSCORED "$r"
ssh -n tanitad-thor "cd $O && tar -cf - scores score_loop.out PREREG_ACK.txt wise_l0p00/bridge/*.manifest.json wise_l0p25/bridge/*.manifest.json wise_l0p50/bridge/*.manifest.json wise_l0p75/bridge/*.manifest.json traj_ema5000/bridge/*.manifest.json traj_ema15000/bridge/*.manifest.json traj_ema30000/bridge/*.manifest.json */cuda_controls.json" | tar -xf - -C $W
ssh -n tanitad-thor "cat /home/nvidia/r8ns/navtrain_sel/wiseft.out" > $W/wiseft_launch.out 2>/dev/null
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/code/wiseft_read.py > $W/WISEFT_READ.out 2>&1
[ -s $W/WISEFT_READ.json ] && mark READOK "$(grep -o 'QQWISE.*' $W/WISEFT_READ.out | cut -c1-700)" || mark READBAD
mark END
