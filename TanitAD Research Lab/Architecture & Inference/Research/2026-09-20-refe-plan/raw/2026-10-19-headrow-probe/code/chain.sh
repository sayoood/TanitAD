#!/bin/bash
# After `main` finishes: export (additive, Amendment 1), cpos, noise, ares, nullrate sequentially (one Python process at a time, so with the c2 process the total stays at 2); then, once c2 is done,
# metrics and the export verification.  A failure of `export` is logged and does NOT stop the chain (the deliverable stages go on); any other failure stops it.
RUN=C:/Users/Admin/par18/probe/run/f10
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
HP=C:/Users/Admin/par18/probe/code/headrow_probe.py
EX=C:/Users/Admin/par18/probe/code/export_oof.py
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8
cd C:/Users/Admin/par18/probe/code
until [ -f $RUN/oof/oof_ttc_M_s1.npz ]; do sleep 20; done
sleep 20
echo "[$(date +%H:%M:%S)] chain: export start" >> $RUN/chain.log
$PY -I $EX export >> $RUN/export.log 2>&1 < /dev/null
echo "[$(date +%H:%M:%S)] chain: export rc=$?" >> $RUN/chain.log
for st in cpos noise ares nullrate; do
  echo "[$(date +%H:%M:%S)] chain: $st start" >> $RUN/chain.log
  $PY -I $HP $st >> $RUN/$st.log 2>&1 < /dev/null
  rc=$?
  echo "[$(date +%H:%M:%S)] chain: $st rc=$rc" >> $RUN/chain.log
  if [ $rc -ne 0 ]; then echo "[$(date +%H:%M:%S)] chain: STOP after failure of $st" >> $RUN/chain.log; exit $rc; fi
done
until [ -f $RUN/c2/c2_ttc_M_rep5.npz ]; do sleep 20; done
sleep 20
echo "[$(date +%H:%M:%S)] chain: metrics start" >> $RUN/chain.log
$PY -I $HP metrics >> $RUN/metrics.log 2>&1 < /dev/null
echo "[$(date +%H:%M:%S)] chain: metrics rc=$?" >> $RUN/chain.log
$PY -I $EX verify >> $RUN/export.log 2>&1 < /dev/null
echo "[$(date +%H:%M:%S)] chain: verify rc=$?" >> $RUN/chain.log
