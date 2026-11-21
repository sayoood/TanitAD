#!/bin/bash
cd /c/Users/Admin/v8proof_c/target/tree_patch || exit 1
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
R=C:/Users/Admin/v8proof_c/target/tree_patch
LOG=F:/TanitAD-scratch/v8proof/spatial/target/tests.log
export CUDA_VISIBLE_DEVICES=-1 TMP=C:/Users/Admin/v8proof_c/target/tmp TEMP=C:/Users/Admin/v8proof_c/target/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$R/stack;$R/taniteval;$R/stack/scripts"
$PY -c "import tanitad,sys;assert tanitad.__file__.lower().startswith('c:\\\\users\\\\admin\\\\v8proof_c\\\\target\\\\tree_patch'),tanitad.__file__;print('tanitad from',tanitad.__file__)" > $LOG 2>&1
$PY -m pytest -v stack/tests/test_v8f_target_levers.py stack/tests/test_v8f_motion_levers.py stack/tests/test_v8f_cell_predictor.py stack/tests/test_v8f_spatial_state.py --basetemp=C:/Users/Admin/v8proof_c/target/tmp/pt/all -p no:cacheprovider >> $LOG 2>&1
echo "PYTEST_RC $?" >> $LOG
