#!/bin/bash
# runs the final checks one after the other (CPU timing needs the machine quiet): tests, m2345, m6, m1, m7. Logs in checks/.
cd /c/Users/Admin/v8proof_c/motion || exit 1
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
R=C:/Users/Admin/v8proof_c/motion/tree_patch
export CUDA_VISIBLE_DEVICES=-1 TMP=C:/Users/Admin/v8proof_c/motion/tmp TEMP=C:/Users/Admin/v8proof_c/motion/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$R/stack;$R/taniteval;$R/stack/scripts"
cd tree_patch
$PY -c "import tanitad,sys;assert tanitad.__file__.lower().startswith('c:\\\\users\\\\admin\\\\v8proof_c\\\\motion\\\\tree_patch'),tanitad.__file__;print('tanitad from',tanitad.__file__)" > ../checks/tests.log 2>&1
$PY -m pytest -q stack/tests/test_v8f_motion_levers.py stack/tests/test_v8f_cell_predictor.py stack/tests/test_v8f_spatial_state.py --basetemp=C:/Users/Admin/v8proof_c/motion/tmp/pt/m -p no:cacheprovider >> ../checks/tests.log 2>&1
echo "PYTEST_RC $?" >> ../checks/tests.log
cd ..
$PY checks/m2345_checks.py $R > checks/m2345_checks.log 2>&1
$PY checks/m6_selfcheck.py > checks/m6_selfcheck.log 2>&1
$PY checks/m1_identity.py 200 > checks/m1_identity.log 2>&1
$PY checks/m7_cost.py > checks/m7_cost.log 2>&1
echo ALLDONE > checks/ALLDONE.txt
