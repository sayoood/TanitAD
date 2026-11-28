#!/bin/bash
# MM 2026-11-16 (replaces strat1_pilot.sh): the tree the tokens bind is commit 22999538 (= b9250158 + the launch checker + its token tests, LF).
# G-SUITE: base cdbff971 vs new b9250158 for every file, with the files that the two added/changed files touch (test_v8f_strat1.py,
# test_v8f_strat1_navbuilder.py, test_strat1_token.py) re-run on the 22999538 tree itself; their junit overrides the b9250158 rows (later file wins in
# strat1_check.junit_results). Only if 0 regressions: mint + run the three 200-step timing pilots (C0, N1, S1 s0) through the runner, one at a time.
# Detached: nohup setsid bash /root/strat1/strat1_pilot2.sh < /dev/null > /root/strat1/logs/pilot2.log 2>&1 &
# Ends with `ZZ STRAT1 PILOT DONE ZZ` or `ZZ STRAT1 PILOT FAIL ZZ <why>`. Never starts the 7-arm chain.
set -u
R=/root/strat1; TREE=$R/tree; INP=$R/inputs; RUNS=$R/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
BASE_COMMIT=cdbff971fa6dad4d20b937fe8abda6cb9f622adb
COMMIT=22999538fb6503ebf0656fa336a5ba354d5ca522
SHA_CHK=46b3bb21f9c3e3c8361753ecf1d986a52faa80fbf3a3d6a01eb220b1d23e328e
SHA_TOK=55db41362ae4621e062793e9367016d1e9e6f86598fa6d511d964ef25129d8c3
echo "$COMMIT" > $R/COMMIT
SUITE=$R/gsuite_result.json
export PYTHONPATH="$TREE/stack:$TREE" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $RUNS
fail() { echo "ZZ STRAT1 PILOT FAIL ZZ $1"; exit 1; }
# 0a. the tree is commit 22999538: the two files byte-equal to the commit's blobs, and no other difference from the b9250158 export
[ "$(sha256sum < $TREE/stack/scripts/strat1_check.py | cut -c1-64)" = "$SHA_CHK" ] || fail "tree strat1_check.py is not the 22999538 blob"
[ "$(sha256sum < $TREE/stack/tests/test_strat1_token.py | cut -c1-64)" = "$SHA_TOK" ] || fail "tree test_strat1_token.py is not the 22999538 blob"
D=$(diff -rq -x __pycache__ $R/new/stack $TREE/stack; diff -rq -x __pycache__ $R/new/taniteval $TREE/taniteval)
D=$(printf '%s\n' "$D" | grep -v "stack/scripts/strat1_check.py differ" | grep -v "Only in $TREE/stack/tests: test_strat1_token.py" | grep -v '^$')
[ -z "$D" ] || fail "tree differs from b9250158 beyond the two files: $(printf '%s' "$D" | head -3 | tr '\n' ' ')"
echo "$(date -u +%T) tree == 22999538 (two files by sha256, rest equal to the b9250158 export)"
# 0b. inputs complete and byte-equal to their sources
until [ -z "$(find $INP -name '*.part' | head -1)" ] && [ -f $INP/dagger_bank/s1/bank.json ] && [ -f $INP/data/v9/v9_labels_train.npz ] && [ -f $INP/dagger_bank/s1/shard_003of004.npz ]; do sleep 30; done
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify.log 2>&1 || fail "inputs sha256 mismatch"
# 1. wait for the suite and both shard-2 reruns
until [ -e $R/logs/gsuite_all.done ] && grep -q "ZZ RERUN" $R/junit/base/shard2.log 2>/dev/null && grep -q "ZZ RERUN" $R/junit/new/shard2.log 2>/dev/null; do sleep 30; done
echo "$(date -u +%T) suite shards done"
# 1b. targeted re-run on the 22999538 tree (same cwd form as gsuite.sh, so the test ids match)
mkdir -p $R/junit/c2 $R/tmp/c2 && rm -rf $R/junit/new_c2 && mkdir -p $R/junit/new_c2
( cd $TREE && TMPDIR=$R/tmp/c2 TMP=$R/tmp/c2 TEMP=$R/tmp/c2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= timeout 2h $PY -m pytest -q -p no:cacheprovider \
    --basetemp=$R/tmp/c2/bt --junitxml=$R/junit/c2/zz_c2_targeted.xml \
    $TREE/stack/tests/test_v8f_strat1.py $TREE/stack/tests/test_v8f_strat1_navbuilder.py $TREE/stack/tests/test_strat1_token.py ) > $R/junit/c2/targeted.log 2>&1
echo "targeted rc=$? :: $(tail -1 $R/junit/c2/targeted.log)"
[ -s $R/junit/c2/zz_c2_targeted.xml ] || fail "targeted re-run wrote no junit"
cp $R/junit/new/shard*.xml $R/junit/new_c2/ && cp $R/junit/c2/zz_c2_targeted.xml $R/junit/new_c2/
# 2. compare per test id
$PY $CHK suite --base-junit $R/junit/base --new-junit $R/junit/new_c2 --base-commit $BASE_COMMIT --new-commit $COMMIT --out $SUITE > $R/logs/suite_compare.log 2>&1 || fail "G-SUITE regressions (see $SUITE)"
echo "suite: $(cat $R/logs/suite_compare.log)"
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
for kind in C0 N1 S1; do
  tag=pilot200_${kind}_s0
  for i in $(seq 1 60); do [ "$(gpu_used)" -lt 3500 ] && break; sleep 30; done
  [ "$(gpu_used)" -lt 3500 ] || fail "GPU busy for 30 min before $tag ($(gpu_used) MiB)"
  echo "$(date -u +%T) mint $tag"
  $PY $CHK mint $kind s0 $COMMON --steps 200 > $RUNS/mint_$tag.log 2>&1 || fail "mint $tag: $(tail -c 400 $RUNS/mint_$tag.log)"
  ( while true; do gpu_used >> $RUNS/$tag.gpu_mib; sleep 2; done ) & SAMP=$!
  echo "$(date -u +%T) run $tag"
  $PY $CHK run $kind s0 $COMMON --steps 200 > $RUNS/run_$tag.log 2>&1; rc=$?
  kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
  [ $rc -eq 0 ] || fail "run $tag rc=$rc: $(tail -c 400 $RUNS/run_$tag.log)"
done
$PY $R/pilot_summary.py $RUNS > $R/pilot_summary.json 2>&1 || fail "pilot_summary"
echo "ZZ STRAT1 PILOT DONE ZZ"
