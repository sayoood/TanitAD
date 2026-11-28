#!/bin/bash
# serial pytest of the new test file plus the base test list of the previous builder (ruletrain3/run_tests.sh); TMP/TEMP/basetemp under leadcond/work
cd /f/TanitAD-scratch/v8proof/leadcond/work/tree_patch || exit 1
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
R=F:/TanitAD-scratch/v8proof/leadcond/work/tree_patch
LOG=${LOG:-F:/TanitAD-scratch/v8proof/leadcond/tests.log}
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONIOENCODING=utf-8 TMP=F:/TanitAD-scratch/v8proof/leadcond/work/tmp TEMP=F:/TanitAD-scratch/v8proof/leadcond/work/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$R/stack;$R/taniteval;$R/stack/scripts"
mkdir -p F:/TanitAD-scratch/v8proof/leadcond/work/pt
$PY -c "import tanitad,os;f=os.path.abspath(tanitad.__file__).lower().replace(chr(92),'/');assert f.startswith('f:/tanitad-scratch/v8proof/leadcond/work/tree_patch'),f;print('tanitad from',tanitad.__file__)" > $LOG 2>&1
$PY -m pytest -v stack/tests/test_v8f_lead_cond.py stack/tests/test_v8f_rule_own.py stack/tests/test_v8f_rule_own_l1.py stack/tests/test_v8f_rule_relabel.py stack/tests/test_v8f_lead_sup.py stack/tests/test_v8f_target_levers.py stack/tests/test_v8f_cell_predictor.py stack/tests/test_v8f_motion_levers.py stack/tests/test_v8f_spatial_state.py --basetemp=F:/TanitAD-scratch/v8proof/leadcond/work/pt/run_all -p no:cacheprovider >> $LOG 2>&1
echo "PYTEST_RC $?" >> $LOG
