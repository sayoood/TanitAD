#!/usr/bin/env bash
# Mutation runs in the SEPARATE tree C:/lgt/i3mut (never the landing tree, never a repo).
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
MUT="$(dirname "$0")/mutate.py"   # this package's raw/scripts/mutate.py
export PYTHONPATH="C:/lgt/i3mut/stack;C:/lgt/i3mut/stack/scripts;C:/lgt/i3mut/taniteval" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=4
LOG=/c/lgt/i3logs/mutations.log
: > $LOG
$PY -c "import tanitad, taniteval; print('TREE', tanitad.__file__, taniteval.__file__)" >> $LOG 2>&1
run () {
  echo "=== $1 ===" >> $LOG
  (cd /c/lgt/i3mut && $PY -m pytest -p no:cacheprovider -q -rfE stack/tests/test_loader_stamped_queries.py taniteval/tests/test_bench_stamped_queries.py >> $LOG 2>&1; echo "PYTEST_EXIT_$1=$?" >> $LOG)
}
run CONTROL_unmutated
$PY "$MUT" M1 apply >> $LOG 2>&1 && run M1_agent_rule_removed
$PY "$MUT" M1 revert >> $LOG 2>&1
cmp /c/lgt/i3mut/stack/scripts/refc_v3_train.py /c/lgt/i3tree/stack/scripts/refc_v3_train.py && echo "M1 REVERTED byte-identical" >> $LOG
$PY "$MUT" M2 apply >> $LOG 2>&1 && run M2_box_stamp_ignored
$PY "$MUT" M2 revert >> $LOG 2>&1
cmp /c/lgt/i3mut/taniteval/tools/refcv3_arm.py /c/lgt/i3tree/taniteval/tools/refcv3_arm.py && echo "M2 REVERTED byte-identical" >> $LOG
echo "ALL_MUTATIONS_DONE" >> $LOG
