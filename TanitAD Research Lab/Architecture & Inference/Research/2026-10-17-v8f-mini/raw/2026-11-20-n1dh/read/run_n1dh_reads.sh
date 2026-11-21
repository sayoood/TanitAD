#!/bin/bash
# run_n1dh_reads.sh -- MM 2026-11-21: the dev-box N1DH reads, ONE process at a time, each gated on free commit >= 5.5 GB (wait_commit.py) and F: >= 6 GB.
#   1 n1dh_ho_read.py (deciding cell D1-HO TAC-LAT N1DH - N1D; POST-WEAK part waits for Z-POST, which runs on the pod CPU after RHp3)  -> out_real/
#   2 n1dh_klim_run.py K-LIM on the eval-141 probe batch, s0 then s1                                                                   -> KLIM_s0.json, KLIM_s1.json
#   3 eval141/run_n1dh141.sh (the eval-141 guard, four families, STRATEGIC)                                                            -> eval141/out/
# Launch only through run_n1dh_reads.cmd by WMI. Last log line: "ZZ N1DH-READS DONE ZZ" or "ZZ N1DH-READS FAIL ZZ <step>".
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
H=F:/TanitAD-scratch/n1dc/read_n1dh
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$H/reads_driver.log
export TMP=$H/tmp TEMP=$H/tmp
mkdir -p $H/tmp
cd $H || exit 1
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
stop() { echo "ZZ N1DH-READS FAIL ZZ $1" >> $LOG; exit 1; }
gate() { $PY -I $WC 5.5 60 >> $LOG 2>&1 || stop "commit below 5.5 GB for 60 min before $1"
         fk=$(df -k /f | awk 'NR==2{print int($4/1048576)}'); [ "${fk:-0}" -ge 6 ] || stop "F: ${fk} GB < 6 before $1"; }
say "start (heartbeat)"
if [ ! -f $H/out_real/.ok ]; then
  gate ho_read; say "step ho_read"
  $PY -I -u n1dh_ho_read.py --out out_real > $H/ho_read.log 2>&1 || stop "ho_read rc $? (ho_read.log)"
  echo ok > $H/out_real/.ok; say "ho_read ok"
fi
for s in 0 1; do
  [ -s $H/KLIM_s$s.json ] && continue
  gate klim_s$s; say "step klim_s$s"
  $PY -I -u n1dh_klim_run.py --tree $H/tree --ckpt $H/eval141/ckpts/N1DH_s$s/ckpt_last.pt --profile e141 --max-windows 480 --out $H/KLIM_s$s.json > $H/klim_s$s.log 2>&1 || stop "klim_s$s rc $? (klim_s$s.log)"
  say "klim_s$s ok"
done
say "step eval141 chain"
bash $H/eval141/run_n1dh141.sh < /dev/null
tail -n 1 $H/eval141/out/n1dh141_chain.log | grep -q "ZZ N1DH141 DONE ZZ" || stop "eval141 chain: $(tail -n 1 $H/eval141/out/n1dh141_chain.log | cut -c1-200)"
say "eval141 ok"
echo "ZZ N1DH-READS DONE ZZ" >> $LOG
