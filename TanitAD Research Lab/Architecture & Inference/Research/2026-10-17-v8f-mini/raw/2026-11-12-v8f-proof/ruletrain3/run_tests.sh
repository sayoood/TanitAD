#!/bin/bash
# serial pytest of the new test file plus the trainer tests of the earlier builder (the ruletrain tests.log list); TMP/TEMP/basetemp under ruletrain3/work
cd /f/TanitAD-scratch/v8proof/ruletrain3/work/tree_patch || exit 1
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
R=F:/TanitAD-scratch/v8proof/ruletrain3/work/tree_patch
LOG=${LOG:-F:/TanitAD-scratch/v8proof/ruletrain3/tests.log}
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONIOENCODING=utf-8 TMP=F:/TanitAD-scratch/v8proof/ruletrain3/work/tmp TEMP=F:/TanitAD-scratch/v8proof/ruletrain3/work/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$R/stack;$R/taniteval;$R/stack/scripts"
$PY -c "import tanitad,os;f=os.path.abspath(tanitad.__file__).lower().replace(chr(92),'/');assert f.startswith('f:/tanitad-scratch/v8proof/ruletrain3/work/tree_patch'),f;print('tanitad from',tanitad.__file__)" > $LOG 2>&1
$PY -m pytest -v stack/tests/test_v8f_rule_own.py stack/tests/test_v8f_rule_own_l1.py stack/tests/test_v8f_rule_relabel.py stack/tests/test_v8f_lead_sup.py stack/tests/test_v8f_target_levers.py stack/tests/test_v8f_cell_predictor.py stack/tests/test_v8f_motion_levers.py stack/tests/test_v8f_spatial_state.py --basetemp=F:/TanitAD-scratch/v8proof/ruletrain3/work/pt/run_all -p no:cacheprovider >> $LOG 2>&1
echo "PYTEST_RC $?" >> $LOG
