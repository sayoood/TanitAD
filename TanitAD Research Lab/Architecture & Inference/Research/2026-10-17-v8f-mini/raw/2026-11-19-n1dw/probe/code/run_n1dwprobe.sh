#!/bin/bash
# PROBE-N1D-FEAT full run (PREREG_V8F_N1DW.md sec 10). CPU only, cores 16-47, nice 10. Log: /root/strat1/n1dwprobe/n1dwprobe.log ; last line = the sentinel.
# usage: nohup setsid bash run_n1dwprobe.sh > /dev/null 2>&1 &     (writes its PID to /root/strat1/n1dwprobe/driver.pid)
W=/root/strat1/n1dwprobe
LOG=$W/n1dwprobe.log
cd $W || exit 1
echo $$ > $W/driver.pid
export PYTHONPATH=$W/pylib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
PY="taskset -c 16-47 nice -n 10 /root/venv/bin/python -u"
fail() { echo "ZZ N1DWPROBE FAIL ZZ $1" >> $LOG; exit 1; }
echo "start $(date -u +%FT%TZ)" > $LOG
mkdir -p out/feat1 out/feat2 out/result
for pass in feat1 feat2; do
  for a in N1D_s0 N1D_s1; do
    $PY code/n1dwprobe_extract.py --arm $a --out out/$pass >> $LOG 2>&1 || fail "extract $a $pass rc=$?"
  done
done
$PY code/n1dwprobe_probe.py --feats out/feat1 --det-feats out/feat2 --dump-dir /root/strat1/navrbd/out/t1 --out out/result --workers 8 >> $LOG 2>&1 || fail "probe rc=$?"
$PY code/n1dwprobe_read.py out/result/PROBE_RESULT.json > out/result/READ.txt 2>&1; rc=$?
echo "reader rc=$rc (1 = a card control did not read; see out/result/READ.txt)" >> $LOG
echo "ZZ N1DWPROBE DONE ZZ" >> $LOG
