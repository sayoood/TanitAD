#!/usr/bin/env bash
# The gate's unit tests on a CLEAN tree (tip + the gate files), CPU only, behind the brief's
# 8 GB free-RAM floor. Verdict = the junit xml + the tanitad.__file__ assertion, never the rc.
#   TREE=<tree> OUT=<dir> bash run_unit_tests.sh
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
: "${TREE:?}" "${OUT:?}"
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1
WTREE=$(cygpath -w "$TREE")
export PYTHONPATH="$WTREE\\stack;$WTREE\\taniteval"
mkdir -p "$OUT"
until "$PY" -c "import psutil,sys; sys.exit(0 if psutil.virtual_memory().available/2**30 >= 8 else 1)"; do
  sleep 60
done
"$PY" -c "import tanitad,sys; f=tanitad.__file__; print('tanitad from', f); sys.exit(0 if f.lower().startswith(sys.argv[1].lower()) else 3)" "$WTREE" > "$OUT/import_check.txt" 2>&1
echo "import_check_rc=$?" >> "$OUT/import_check.txt"
cd "$TREE/stack" || exit 1
"$PY" -m pytest -q -p no:cacheprovider tests/test_launch_gate.py --junitxml="$(cygpath -w "$OUT")\\junit_launch_gate.xml" -o junit_family=xunit2 > "$OUT/pytest_launch_gate.log" 2>&1
echo "ZZUNIT-DONE-$?ZZ" >> "$OUT/pytest_launch_gate.log"
