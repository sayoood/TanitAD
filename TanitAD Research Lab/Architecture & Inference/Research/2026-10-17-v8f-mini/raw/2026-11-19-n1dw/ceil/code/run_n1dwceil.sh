#!/bin/bash
# PREREG_N1D_CEIL driver. CPU only, ONE process at a time, nice 10.
# usage: nohup setsid bash run_n1dwceil.sh [smoke|full] > /dev/null 2>&1 &    ; log = /root/strat1/n1dwceil/n1dwceil_<mode>.log ; last line = the sentinel
W=/root/strat1/n1dwceil
MODE=${1:-full}
LOG=$W/n1dwceil_$MODE.log
cd $W || exit 1
echo $$ > $W/driver_$MODE.pid
export PYTHONPATH=/root/strat1/n1dwprobe/pylib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= N1DWPROBE_PYLIB=/root/strat1/n1dwprobe/pylib N1DWPROBE_READ_OTBAND=/root/strat1/n1dwprobe/lib/read_otband.py
PY="nice -n 10 /root/venv/bin/python -u"
fail() { echo "ZZ N1DWCEIL FAIL ZZ $1" >> $LOG; exit 1; }
echo "start $(date -u +%FT%TZ) mode $MODE" > $LOG
MW=""
SM=""
if [ "$MODE" = smoke ]; then MW="--max-windows 200"; SM="--smoke"; fi
O=$W/out_$MODE
BANK=/root/strat1/n1dwtcode/out_full
mkdir -p $O/ego1 $O/ego2 $O/result
for pass in ego1 ego2; do
  for a in N1D_s0 N1D_s1; do
    if [ -f $O/$pass/feat_$a.npz.json ]; then echo "skip extract $a $pass (done)" >> $LOG; continue; fi
    $PY code/n1dwceil_extract.py --arm $a --out $O/$pass $MW >> $LOG 2>&1 || fail "extract $a $pass rc=$?"
  done
done
$PY code/n1dwceil_probe_run.py --banked-feats $BANK/feat1 --ego1 $O/ego1 --ego2 $O/ego2 --banked-oof $BANK/result/oof_correct.npz --out $O/result --workers 1 $SM >> $LOG 2>&1 || fail "probe rc=$?"
$PY code/n1dwceil_read.py $O/result/RESULT_CEIL.json > $O/result/READ_CEIL.txt 2>&1; rc=$?
echo "reader rc=$rc (1 = a card control did not read; see READ_CEIL.txt)" >> $LOG
echo "ZZ N1DWCEIL DONE ZZ" >> $LOG
