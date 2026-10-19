#!/bin/bash
# pass 2 of the regression comparison, serial (ONE test process at a time), CPU only.  Waits (script, not model turns) for the pass-1 base batch.
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
W=/c/Users/Admin/hn34/work
L=/c/Users/Admin/hn34/logs
while [ ! -f $L/batch_all_done.txt ]; do sleep 20; done
T=$(cat $W/tests_common.txt)
# 1. the whole list on the FINAL edit tree (stage-2 fixes applied), long timeout so test_aux_g12_bf16 (CPU 2-step determinism runs) can finish
$PY $W/run_batch.py new C:/Users/Admin/hn34/logs/batch_new 3600 test_hn_bank.py test_ctx_cache.py test_hn_screen_wiring.py $T > $L/batch_new.out 2>&1
# 2. the base tree's test_aux_g12_bf16 timed out at 900 s in pass 1: rerun with the same long timeout
$PY $W/run_batch.py base C:/Users/Admin/hn34/logs/batch_base_long 3600 test_aux_g12_bf16.py > $L/batch_base_long.out 2>&1
# 3. eval/test_hn_screen_read.py needs TANITEVAL_DIR (the export has no taniteval/): point both trees at the read-only D: copy
export TANITEVAL_DIR=D:/Projects/TanitAD/taniteval
$PY $W/run_batch.py new C:/Users/Admin/hn34/logs/batch_new_env 1800 eval/test_hn_screen_read.py > $L/batch_new_env.out 2>&1
$PY $W/run_batch.py base C:/Users/Admin/hn34/logs/batch_base_env 1800 eval/test_hn_screen_read.py > $L/batch_base_env.out 2>&1
echo PASS2_DONE > $L/pass2_done.txt
