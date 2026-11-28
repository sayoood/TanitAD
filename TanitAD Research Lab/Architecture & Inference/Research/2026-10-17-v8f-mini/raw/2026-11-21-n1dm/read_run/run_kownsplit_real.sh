#!/bin/bash
# run_kownsplit_real.sh -- K-OWN-SPLIT real runs, ONE python at a time, each gated on commit >= 5.5 GB. N1DH_s0/s1 then N1DM_s0/s1 (eval-141 probe windows, 480). Last line: ZZ KOWNSPLIT DONE ZZ / FAIL.
export PATH="/usr/bin:/bin:$PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
H=F:/TanitAD-scratch/n1dc/read_n1dm
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$H/kownsplit_real.log
export TMP=$H/tmp TEMP=$H/tmp
cd $H || exit 1
for spec in "N1DH_s0 F:/TanitAD-scratch/n1dc/read_n1dh/eval141/ckpts/N1DH_s0/ckpt_last.pt" "N1DH_s1 F:/TanitAD-scratch/n1dc/read_n1dh/eval141/ckpts/N1DH_s1/ckpt_last.pt" "N1DM_s0 $H/ckpts/N1DM_s0/ckpt_last.pt" "N1DM_s1 $H/ckpts/N1DM_s1/ckpt_last.pt"; do
  set -- $spec; arm=$1; ck=$2
  [ -s $H/RESULT_${arm}_KOWNSPLIT.json ] && continue
  $PY -I $WC 5.5 90 >> $LOG 2>&1 || { echo "ZZ KOWNSPLIT FAIL ZZ gate $arm" >> $LOG; exit 1; }
  fk=$(df -k /f | awk 'NR==2{print int($4/1048576)}'); [ "${fk:-0}" -ge 6 ] || { echo "ZZ KOWNSPLIT FAIL ZZ F: ${fk} GB" >> $LOG; exit 1; }
  echo "$(date -u +%FT%TZ) run $arm" >> $LOG
  $PY -I -u n1dm_kownsplit.py --tree $H/tree --ckpt $ck --profile e141 --max-windows 480 --work $H/kown_work_$arm --out $H/RESULT_${arm}_KOWNSPLIT.json >> $LOG 2>&1
  echo "$arm rc $?" >> $LOG
done
echo "ZZ KOWNSPLIT DONE ZZ" >> $LOG
