#!/bin/bash
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
T=$(cat /c/Users/Admin/hn34/work/tests_common.txt)
$PY /c/Users/Admin/hn34/work/run_batch.py new C:/Users/Admin/hn34/logs/batch_new 900 test_hn_bank.py test_ctx_cache.py test_hn_screen_wiring.py $T > /c/Users/Admin/hn34/logs/batch_new.out 2>&1
$PY /c/Users/Admin/hn34/work/run_batch.py base C:/Users/Admin/hn34/logs/batch_base 900 test_hn_bank.py test_ctx_cache.py $T > /c/Users/Admin/hn34/logs/batch_base.out 2>&1
echo ALL_DONE > /c/Users/Admin/hn34/logs/batch_all_done.txt
