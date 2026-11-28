#!/bin/bash
# DRY debug of `mint` for C0 s0 (own runs dir; a throwaway suite result made by the checker from two tiny synthetic junit dirs; NOT a launch token, never used by the chain).
set -u
R=/root/strat1; TREE=$R/tree; D=$R/dry; PY=/root/venv/bin/python; CHK=$TREE/stack/scripts/strat1_check.py
export PYTHONPATH="$TREE/stack:$TREE" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
until [ -z "$(find $R/inputs/dagger_bank/s0 -name '*.part' | head -1)" ] && [ -f $R/inputs/dagger_bank/s0/shard_003of004.npz ] && [ -f $R/inputs/data/v9/v9_labels_train.npz ] && [ -z "$(find $R/inputs/data -name '*.part' | head -1)" ]; do sleep 20; done
rm -rf $D; mkdir -p $D/jb $D/jn
cat > $D/jb/s.xml <<'X'
<testsuites><testsuite><testcase classname="t" name="a"/></testsuite></testsuites>
X
cp $D/jb/s.xml $D/jn/s.xml
$PY $CHK suite --base-junit $D/jb --new-junit $D/jn --base-commit dry --new-commit dry --out $D/gsuite_result.json || { echo "ZZ DRY FAIL ZZ suite"; exit 1; }
$PY $CHK mint C0 s0 --root $TREE --inputs $R/inputs --cache-dir /root/data/cache_pod --runs $D/runs --commit dry --suite $D/gsuite_result.json --py $PY --gpu 0 --steps 200 > $D/mint.log 2>&1
echo "rc=$?" >> $D/mint.log
echo "ZZ DRY DONE ZZ" >> $D/mint.log
