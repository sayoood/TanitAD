#!/bin/bash
# Mutation controls for tests/test_v8f_t1_heads.py. Every mutant is a SCRATCH copy (mut/<name>/stack/scripts/v8f_t1_eval.py): the real tree is never edited.
# mut/<name>/stack/{tanitad,tests-deps} are junctions to the real read-only tree so the test file resolves REPO = the mutant dir and imports the mutated tool.
# usage: mutate.sh <name> <sed-expression-on-the-tool>      (the expression must change exactly the intended line; the script asserts a diff of exactly 1 line pair)
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8
H=/c/Users/Admin/t1heads; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
N=$1; EXPR=$2
M=$H/mut/$N
rm -rf "$M"; mkdir -p "$M/stack/scripts" "$M/stack/tests"
cmd //c "mklink /J $(cygpath -w $M/stack/tanitad) $(cygpath -w $H/tree/stack/tanitad)" > /dev/null
cmd //c "mklink /J $(cygpath -w $M/taniteval) $(cygpath -w $H/tree/taniteval)" > /dev/null
cp $H/tree/stack/tests/test_v8f_t1_heads.py $M/stack/tests/
cp $H/tree/stack/tests/conftest.py $M/stack/tests/
sed -e "$EXPR" $H/tree/stack/scripts/v8f_t1_eval.py > $M/stack/scripts/v8f_t1_eval.py
echo "=== MUTANT $N :: sed '$EXPR'"
diff $H/tree/stack/scripts/v8f_t1_eval.py $M/stack/scripts/v8f_t1_eval.py
echo "--- (diff rc=$?; 1 = differs, as intended)"
cd $M/stack && V8F_T1_TIP_TOOL=$H/ref/v8f_t1_eval.tip.lf.py $PY -m pytest -q -p no:cacheprovider --maxfail=50 --tb=line --basetemp $H/pt/mut_$N tests/test_v8f_t1_heads.py 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed|test_v8f_t1_heads.py:[0-9]+:|Error"
echo "=== END $N"
# remove the junctions (link only), never recursively
cmd //c "rmdir $(cygpath -w $M/stack/tanitad)" > /dev/null; cmd //c "rmdir $(cygpath -w $M/taniteval)" > /dev/null
