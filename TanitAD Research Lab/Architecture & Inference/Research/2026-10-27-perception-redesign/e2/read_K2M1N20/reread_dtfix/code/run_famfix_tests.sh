#!/bin/bash
# run_famfix_tests.sh -- MM 4a3d05, 2026-11-12: the r9_e2_families waypoint-clock fix.  1) the NEW test file against the TIP's reader (must be RED: the
# proof the tests see the defect); 2) the fixed reader + new tests + the neighbours that import it (distkeep, verdict, roll) -- scipy BLOCKED, stack/scripts
# on PYTHONPATH; 3) three mutants of the fixed reader that must turn the new test file RED.
export PATH="/usr/bin:/bin:$PATH"
W=F:/refcv9_e3/famfix_mm; TR=$W/tree; PYW=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
S=C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad
NS=$S/run_pytest_noscipy.py; FX=$S/famfix
cd $TR || exit 3
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
export PYTHONPATH="$(cygpath -w "$TR/stack");$(cygpath -w "$TR/taniteval");$(cygpath -w "$TR/stack/scripts")"
mkdir -p $W/bt
run() { $PYW $NS -q -p no:cacheprovider --basetemp=$W/bt/$1 "${@:2}" < /dev/null 2>&1 | tail -n 4; echo "RC ${PIPESTATUS[0]}"; }
R=stack/scripts/r9_e2_families.py; TT=stack/tests/test_r9_e2_families.py
cp $R $W/tip_reader.py
cp $FX/$TT $TT
echo "== 1. new tests vs the TIP reader (must be RED)"; run t1 $TT
cp $FX/$R $R
echo "== 2. fixed reader"; run t2 $TT
for t in test_r9_e2_distkeep.py test_r9_e2_verdict.py test_r9e2_roll.py; do echo "== 2. $t"; run "n_${t%.py}" stack/tests/$t; done
cp $R $W/fixed_reader.py
mutate() {   # $1 name, $2 old, $3 new
  $PYW -I $W/mut.py "$R" "$2" "$3" < /dev/null || { echo "MUTATION $1: not applied"; cp $W/fixed_reader.py $R; return; }
  echo "== 3. MUTANT $1 (must be RED)"; run "m_$1" $TT
  cp $W/fixed_reader.py $R
}
mutate const_dt '    dt = A["dt_s"]' '    dt = 0.1'
mutate no_guard '    if rel > DT_CHECK_REL_TOL:' '    if False:'
mutate no_arm_check '    if A["dt_s"] != B["dt_s"]:' '    if False:'
cmp -s $W/fixed_reader.py $R && echo "reader restored"
echo DONE
