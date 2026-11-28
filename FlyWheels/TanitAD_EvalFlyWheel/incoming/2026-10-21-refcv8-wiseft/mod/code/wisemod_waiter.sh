#!/bin/bash
# Dev-box reader for R8-WISEFT-MOD (7ec285d): wait for Thor's mod scoring loop, pull its four score dirs, apply the registered rule.  ZZWMW<STAGE>ZZ.
set -u
W=/c/Users/Admin/refcv8_pf/navsim_ckpt/wiseft; LOG=$W/wisemod_waiter.log; O=/home/nvidia/r8ns/navtrain_sel/wiseft
mark() { echo "ZZWMW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START; t0=$(date +%s)
while :; do
  r=$(ssh -n tanitad-thor "grep -c 'ZZMALLSCORE[D]' $O/score_loop_mod.out; grep -cE 'ZZMSCORE(ABOR[T]|TIMEOU[T])' $O/score_loop_mod.out" 2>/dev/null | tr '\n' ' ')
  set -- $r
  [ "${2:-0}" != "0" ] && [ -n "${2:-}" ] && { mark ABORTED "$r"; exit 1; }
  [ "${1:-0}" != "0" ] && [ -n "${1:-}" ] && break
  [ $(( $(date +%s) - t0 )) -gt 28800 ] && { mark TIMEOUT "$r"; exit 1; }
  sleep 120
done
mark ALLSCORED
ssh -n tanitad-thor "cd $O && tar -cf - scores/r8sel_wmod_PLAN_R8_LEGAL scores/r8sel_wmod_PLAN_R8_LEGAL_s1 scores/r8sel_wmod_PERC_R8_LEGAL scores/r8sel_wmod_PERC_R8_LEGAL_s1 score_loop_mod.out PREREG_ACK_MOD.txt wmod_PLAN/bridge/*.manifest.json wmod_PERC/bridge/*.manifest.json wmod_PLAN/cuda_controls.json wmod_PERC/cuda_controls.json" | tar -xf - -C $W
ssh -n tanitad-thor "cat /home/nvidia/r8ns/navtrain_sel/wisemod.out" > $W/wisemod_launch.out 2>/dev/null
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/code/wisemod_read.py > $W/WISEMOD_READ.out 2>&1
[ -s $W/WISEMOD_READ.json ] && mark READOK "$(grep -o 'QQMOD.*' $W/WISEMOD_READ.out | cut -c1-700)" || mark READBAD
mark END
