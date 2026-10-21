#!/bin/bash
# Dev-box reader for R8-WISEFT-MOD2 (fc67bf6): wait for Thor's mod scoring loop, pull its six score dirs, apply the registered rule.  ZZWM2W<STAGE>ZZ.
set -u
W=/c/Users/Admin/refcv8_pf/navsim_ckpt/wiseft; LOG=$W/wisemod2_waiter.log; O=/home/nvidia/r8ns/navtrain_sel/wiseft
mark() { echo "ZZWM2W$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START; t0=$(date +%s)
while :; do
  r=$(ssh -n tanitad-thor "grep -c 'ZZM2ALLSCORE[D]' $O/score_loop_mod2.out; grep -cE 'ZZM2SCORE(ABOR[T]|TIMEOU[T])' $O/score_loop_mod2.out" 2>/dev/null | tr '\n' ' ')
  set -- $r
  [ "${2:-0}" != "0" ] && [ -n "${2:-}" ] && { mark ABORTED "$r"; exit 1; }
  [ "${1:-0}" != "0" ] && [ -n "${1:-}" ] && break
  [ $(( $(date +%s) - t0 )) -gt 28800 ] && { mark TIMEOUT "$r"; exit 1; }
  sleep 120
done
mark ALLSCORED
ssh -n tanitad-thor "cd $O && tar -cf - scores/r8sel_wmod2_DEC_R8_LEGAL scores/r8sel_wmod2_DEC_R8_LEGAL_s1 scores/r8sel_wmod2_TAC_R8_LEGAL scores/r8sel_wmod2_TAC_R8_LEGAL_s1 scores/r8sel_wmod2_PLAN75_R8_LEGAL scores/r8sel_wmod2_PLAN75_R8_LEGAL_s1 score_loop_mod2.out PREREG_ACK_MOD2.txt ckpt_mod2/WISEFT_MOD2_BUILD.json wmod2_*/bridge/*.manifest.json wmod2_*/cuda_controls.json" | tar -xf - -C $W
ssh -n tanitad-thor "cat /home/nvidia/r8ns/navtrain_sel/wisemod2.out" > $W/wisemod2_launch.out 2>/dev/null
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/code/wisemod2_read.py > $W/WISEMOD2_READ.out 2>&1
[ -s $W/WISEMOD2_READ.json ] && mark READOK "$(grep -o 'QQMOD2.*' $W/WISEMOD2_READ.out | cut -c1-700)" || mark READBAD
mark END
