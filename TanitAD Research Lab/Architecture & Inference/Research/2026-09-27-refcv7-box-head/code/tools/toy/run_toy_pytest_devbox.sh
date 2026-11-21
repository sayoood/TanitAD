#!/bin/bash
# the ADDENDUM test file on the dev box (tree C:/Users/Admin/bxh_tree_toy): tanitad asserted from the tree, then the
# whole file at the default thread count. usage: run_toy_pytest_devbox.sh <log>
T=C:/Users/Admin/bxh_tree_toy
LOG="$1"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONPATH="$T/stack;$T/taniteval;$T"
cd "$T" || exit 3
$PY -c "import tanitad,os,sys; f=os.path.normcase(os.path.abspath(tanitad.__file__)); t=os.path.normcase(os.path.abspath(sys.argv[1])); print('tanitad from', tanitad.__file__); sys.exit(0 if f.startswith(t) else 7)" "$T" > "$LOG" 2>&1 || { echo "TANITAD NOT FROM TREE" >> "$LOG"; exit 7; }
$PY -m pytest -q --no-header -p no:cacheprovider -o addopts="" -rfEs stack/tests/test_g_box_overfit.py >> "$LOG" 2>&1
echo "EXIT $?" >> "$LOG"
