#!/bin/bash
# Dev-box reader for R8-WISEFT-DECLAM: wait for Thor's mod scoring loop, pull its six score dirs, apply the registered rule.  ZZWDLW<STAGE>ZZ.
set -u
W=/c/Users/Admin/refcv8_pf/navsim_ckpt/wiseft; LOG=$W/wisedeclam_waiter.log; O=/home/nvidia/r8ns/navtrain_sel/wiseft
mark() { echo "ZZWDLW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START; t0=$(date +%s)
while :; do
  r=$(ssh -n tanitad-thor "grep -c 'ZZDLALLSCORE[D]' $O/score_loop_declam.out; grep -cE 'ZZDLSCORE(ABOR[T]|TIMEOU[T])' $O/score_loop_declam.out" 2>/dev/null | tr '\n' ' ')
  set -- $r
  [ "${2:-0}" != "0" ] && [ -n "${2:-}" ] && { mark ABORTED "$r"; exit 1; }
  [ "${1:-0}" != "0" ] && [ -n "${1:-}" ] && break
  [ $(( $(date +%s) - t0 )) -gt 28800 ] && { mark TIMEOUT "$r"; exit 1; }
  sleep 120
done
mark ALLSCORED
ssh -n tanitad-thor "cd $O && tar -cf - scores/r8sel_wdecl_DEC625_R8_LEGAL scores/r8sel_wdecl_DEC625_R8_LEGAL_s1 scores/r8sel_wdecl_DEC75_R8_LEGAL scores/r8sel_wdecl_DEC75_R8_LEGAL_s1 scores/r8sel_wdecl_DEC875_R8_LEGAL scores/r8sel_wdecl_DEC875_R8_LEGAL_s1 score_loop_declam.out PREREG_ACK_DECLAM.txt ckpt_declam/WISEFT_DECLAM_BUILD.json wdecl_*/bridge/*.manifest.json wdecl_*/cuda_controls.json" | tar -xf - -C $W
ssh -n tanitad-thor "cat /home/nvidia/r8ns/navtrain_sel/wisedeclam.out" > $W/wisedeclam_launch.out 2>/dev/null
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/code/wisedeclam_read.py > $W/WISEDECLAM_READ.out 2>&1
[ -s $W/WISEDECLAM_READ.json ] && mark READOK "$(grep -o 'QQDECLAM.*' $W/WISEDECLAM_READ.out | cut -c1-700)" || mark READBAD
mark END
