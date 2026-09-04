#!/bin/sh
# The paired scoped-suite evidence, run with NO concurrent edits.
# Scope: every test file that imports refc_v3 / refc / refc_v3_train /
# echo_gate / goal_provenance -- i.e. everything the v4 change can reach.
# The BASELINE half is the same tree with HEAD versions of the three modified
# files and the two new files removed, so the failure SETS are comparable and a
# clone-environment failure cannot be mistaken for a regression.
set -u
W=/c/Users/Admin/run_refcv4
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
SEL='refc_v3|refc_v3_train|from tanitad.refs import refc|refs\.refc|echo_gate|goal_provenance'
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""

echo "=== WITH the v4 changes ==="
cd "$W/repo/stack" || exit 1
export PYTHONPATH="C:\\Users\\Admin\\run_refcv4\\repo\\stack;C:\\Users\\Admin\\run_refcv4\\repo\\taniteval;C:\\Users\\Admin\\run_refcv4\\repo;C:\\Users\\Admin\\run_refcv4\\repo\\tools"
# shellcheck disable=SC2046
$PY -m pytest -q --tb=line $(grep -lE "$SEL" tests/*.py | tr '\n' ' ') 2>&1 | tail -12

echo
echo "=== BASELINE (HEAD versions of the 3 modified files, the 2 new files removed) ==="
cd "$W/base/stack" || exit 1
export PYTHONPATH="C:\\Users\\Admin\\run_refcv4\\base\\stack;C:\\Users\\Admin\\run_refcv4\\base\\taniteval;C:\\Users\\Admin\\run_refcv4\\base;C:\\Users\\Admin\\run_refcv4\\base\\tools"
# shellcheck disable=SC2046
$PY -m pytest -q --tb=no $(grep -lE "$SEL" tests/*.py | tr '\n' ' ') 2>&1 | tail -12
echo "ZZSUITE-DONE-ZZ"
