#!/bin/bash
# PREREG_N1D_TCODE driver. CPU only, ONE process at a time, nice 10.
# usage: nohup setsid bash run_n1dwtcode.sh [smoke|full] > /dev/null 2>&1 &    ; log = /root/strat1/n1dwtcode/n1dwtcode_<mode>.log ; last line = the sentinel
W=/root/strat1/n1dwtcode
MODE=${1:-full}
LOG=$W/n1dwtcode_$MODE.log
cd $W || exit 1
echo $$ > $W/driver_$MODE.pid
export PYTHONPATH=/root/strat1/n1dwprobe/pylib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= N1DWPROBE_PYLIB=/root/strat1/n1dwprobe/pylib N1DWPROBE_READ_OTBAND=/root/strat1/n1dwprobe/lib/read_otband.py
PY="nice -n 10 /root/venv/bin/python -u"
fail() { echo "ZZ N1DWTCODE FAIL ZZ $1" >> $LOG; exit 1; }
echo "start $(date -u +%FT%TZ) mode $MODE" > $LOG
MW=""
SM=""
if [ "$MODE" = smoke ]; then MW="--max-windows 200"; SM="--smoke"; fi
O=$W/out_$MODE
mkdir -p $O/feat1 $O/feat2 $O/result
for pass in feat1 feat2; do
  for a in N1D_s0 N1D_s1; do
    if [ -f $O/$pass/feat_$a.npz.json ]; then echo "skip extract $a $pass (done)" >> $LOG; continue; fi
    $PY code/n1dwtcode_extract.py --arm $a --out $O/$pass $MW >> $LOG 2>&1 || fail "extract $a $pass rc=$?"
  done
done
$PY code/n1dwtcode_probe_run.py --feats $O/feat1 --det-feats $O/feat2 --banked-feats /root/strat1/n1dwprobe/out/feat1 --out $O/result --workers 1 $SM >> $LOG 2>&1 || fail "probe rc=$?"
$PY code/n1dwtcode_read.py $O/result/RESULT_TCODE.json > $O/result/READ_TCODE.txt 2>&1; rc=$?
echo "reader rc=$rc (1 = a card control did not read; see READ_TCODE.txt)" >> $LOG
echo "ZZ N1DWTCODE DONE ZZ" >> $LOG
