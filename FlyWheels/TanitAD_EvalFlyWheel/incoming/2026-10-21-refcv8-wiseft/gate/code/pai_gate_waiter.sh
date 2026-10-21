#!/bin/bash
# R8-WISEFT-GATE-PAI (ad0cbbf): wait for the theta_0.5 battery to end (ZZBWSEND) with 139 S6 dumps per seed, then run the registered read.
set -u
G=/c/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/gate; LOG=$G/pai_waiter.log; BL=/c/Users/Admin/refcv8_pf/battery_wise050_pf/bwise.log
TH=C:/Users/Admin/refcv8_pf/battery_wise050_pf/stepwise050_pf; T48=C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf
export OMP_NUM_THREADS=4 PYTHONPATH="D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval"
mark() { echo "ZZPGW$1ZZ $(date '+%F %T') ${2:-}" >> "$LOG"; }
mark START; t0=$(date +%s)
until grep -q 'ZZBWSENDZZ' "$BL" 2>/dev/null; do
  grep -qE 'ZZBWS(MD5BAD|NOLOCK)ZZ' "$BL" 2>/dev/null && { mark UPSTREAM "$(tail -1 $BL)"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 28800 ] && { mark TIMEOUT; exit 1; }
  sleep 60
done
n0=$(ls /c/Users/Admin/refcv8_pf/battery_wise050_pf/stepwise050_pf/s6_s0/ep*.npz 2>/dev/null | wc -l); n1=$(ls /c/Users/Admin/refcv8_pf/battery_wise050_pf/stepwise050_pf/s6_s1/ep*.npz 2>/dev/null | wc -l)
[ "$n0" = 139 ] && [ "$n1" = 139 ] || { mark DUMPSBAD "s0=$n0 s1=$n1"; exit 2; }
mark READY
mkdir -p $G/pai
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 $G/code/pai_gate_families.py --tree-48852 $T48 --tree-theta $TH --out $G/pai/PAI_GATE.json > $G/pai/pai_gate.out 2>&1
[ -s $G/pai/PAI_GATE.json ] && mark READOK "$(grep -o 'QQPAIG.*' $G/pai/pai_gate.out | cut -c1-500)" || mark READBAD
mark END
