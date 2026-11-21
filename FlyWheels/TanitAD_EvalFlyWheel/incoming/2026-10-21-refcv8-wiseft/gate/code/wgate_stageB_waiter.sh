#!/bin/bash
# R8-WISEFT-GATE stage B (f0ae643): waits for the stage-2 read (theta_0.5 navtest rows pulled) and the family waiter's seam pull, then
# runs the ONE gate read and the composite's paired four-family reads (beside).  ZZWGB<STAGE>ZZ.
set -u
G=/c/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/gate; LOG=$G/stageB_waiter.log; S2=/c/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/wiseft_stage2.log
FW=/c/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest/fam_wise050.log; FT=/c/Users/Admin/refcv8_pf/navsim_ckpt/families_navtest
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; export OMP_NUM_THREADS=4 PYTHONPATH="D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval"
mark() { echo "ZZWGB$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START; t0=$(date +%s)
until grep -q 'ZZWS2READOKZZ' "$S2" 2>/dev/null && grep -q 'ZZFWPULLEDZZ' "$FW" 2>/dev/null; do
  grep -qE 'ZZWS2(READBAD|REFUSED|TIMEOUT|S1FAIL|SHIPBAD|NOSTAGE2|POOLFAIL|POOLTIMEOUT|FINALIZEBAD|JOBSBAD)ZZ' "$S2" 2>/dev/null && { mark UPSTREAM "$(tail -1 $S2)"; exit 1; }
  grep -qE 'ZZFW(MD5BAD|UPSTREAM|TIMEOUT)ZZ' "$FW" 2>/dev/null && { mark UPSTREAM_FW "$(tail -1 $FW)"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 36000 ] && { mark TIMEOUT; exit 1; }
  sleep 60
done
mark READY
$PY -X utf8 $G/code/wgate_stageB_read.py > $G/stageB_read.out 2>&1
[ -s $G/stageB/WGATE_STAGE_B.json ] && mark READOK "$(grep -o 'QQWGB.*' $G/stageB_read.out | cut -c1-700)" || { mark READBAD; exit 2; }
cd $FT
for P in "seams_w050/seam_R8_LEGAL.npz W050_s0 paired_navtest_GWARC_minus_W050_s0" "seams_w050/seam_PRIOR_ha0p.npz PRIOR_ha0p paired_navtest_GWARC_minus_PRIOR_s0"; do
  set -- $P
  $PY -X utf8 code/paired_families_navsim.py --inputs inputs/navtest_inputs.json.gz --seam-a $1 --name-a $2 --seam-b $G/stageB/seam_GW_ARC_s0.npz \
      --name-b GW_ARC_s0 --label "navtest paired families: GW_ARC_s0 minus $2 (R8-WISEFT-GATE stage B beside; never a bar)" --out out/$3.json > out/$3.log 2>&1
  [ -s out/$3.json ] && mark PF_OK "$3 $(grep -o 'QQPF.*' out/$3.log | cut -c1-160)" || mark PF_BAD "$3"
done
mark END
