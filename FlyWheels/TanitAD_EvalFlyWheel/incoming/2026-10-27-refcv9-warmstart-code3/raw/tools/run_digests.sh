#!/bin/bash
# flags-absent bit identity, pristine (aeb5446; stack/ tree object == 9506371's) vs modified; every verdict is read from
# the report JSON afterwards, never from an exit code through a pipe
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export TMP=F:/c_overflow_2026-10-26/tmp TEMP=F:/c_overflow_2026-10-26/tmp CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1
D=F:/refcv9_code3/evidence/digests
W=F:/refcv9_code3/work
for t in pristine tree; do
  export PYTHONPATH="F:/refcv9_code3/$t/stack;F:/refcv9_code3/$t/taniteval"
  $PY -X utf8 $W/ref/stepid_digest.py --tree F:/refcv9_code3/$t --out $D/stepid_$t.json > $D/stepid_$t.log 2>&1
  $PY -X utf8 $W/map_digest.py --tree F:/refcv9_code3/$t --out $D/map_$t.json > $D/map_$t.log 2>&1
  $PY -X utf8 $W/pin_digest.py --tree F:/refcv9_code3/$t --out $D/pin_$t.json > $D/pin_$t.log 2>&1
done
export PYTHONPATH="F:/refcv9_code3/tree/stack;F:/refcv9_code3/tree/taniteval"
$PY -X utf8 $W/ref/stepid_digest.py --tree F:/refcv9_code3/tree --out $D/stepid_tree_rerun.json > $D/stepid_tree_rerun.log 2>&1
$PY -X utf8 $W/map_digest.py --tree F:/refcv9_code3/tree --out $D/map_tree_rerun.json > $D/map_tree_rerun.log 2>&1
$PY -X utf8 $W/pin_digest.py --tree F:/refcv9_code3/tree --out $D/pin_tree_rerun.json > $D/pin_tree_rerun.log 2>&1
$PY -X utf8 $W/map_digest.py --tree F:/refcv9_code3/tree --out $D/map_tree_mut_fp.json --mutate fp > $D/map_tree_mut_fp.log 2>&1
$PY -X utf8 $W/map_digest.py --tree F:/refcv9_code3/tree --out $D/map_tree_mut_r9.json --mutate r9 > $D/map_tree_mut_r9.log 2>&1
C="$PY -X utf8 $W/ref/stepid_digest.py --compare"
$C $D/stepid_pristine.json $D/stepid_tree.json --report $D/CMP_stepid_pristine_vs_modified.json > $D/CMP_stepid_pristine_vs_modified.log 2>&1
$C $D/stepid_tree.json $D/stepid_tree_rerun.json --report $D/CMP_stepid_modified_rerun_control.json > $D/CMP_stepid_modified_rerun_control.log 2>&1
$C $D/map_pristine.json $D/map_tree.json --report $D/CMP_map_pristine_vs_modified.json > $D/CMP_map_pristine_vs_modified.log 2>&1
$C $D/map_tree.json $D/map_tree_rerun.json --report $D/CMP_map_modified_rerun_control.json > $D/CMP_map_modified_rerun_control.log 2>&1
$C $D/map_pristine.json $D/map_tree_mut_fp.json --report $D/CMP_map_MUTATION_fp.json > $D/CMP_map_MUTATION_fp.log 2>&1
$C $D/map_pristine.json $D/map_tree_mut_r9.json --report $D/CMP_map_MUTATION_r9.json > $D/CMP_map_MUTATION_r9.log 2>&1
$C $D/pin_pristine.json $D/pin_tree.json --report $D/CMP_pin_pristine_vs_modified.json > $D/CMP_pin_pristine_vs_modified.log 2>&1
$C $D/pin_tree.json $D/pin_tree_rerun.json --report $D/CMP_pin_modified_rerun_control.json > $D/CMP_pin_modified_rerun_control.log 2>&1
echo DIGESTS_DONE > $D/DONE.txt
