#!/bin/bash
# After G-SUITE: compare, and ONLY if 0 regressions mint + run the three 200-step timing pilots (C0 s0, N1 s0, S1 s0) through the runner, one at a time.
# Detached:  nohup setsid bash /root/strat1/strat1_pilot.sh < /dev/null > /root/strat1/logs/pilot.log 2>&1 &
# Ends with `ZZ STRAT1 PILOT DONE ZZ` or `ZZ STRAT1 PILOT FAIL ZZ <why>`. Never starts the 7-arm chain.
set -u
R=/root/strat1; TREE=$R/tree; INP=$R/inputs; RUNS=$R/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
BASE_COMMIT=cdbff971fa6dad4d20b937fe8abda6cb9f622adb
COMMIT=b9250158fc6c41aefac50a3e7459d46a2e5c2f83
echo "$COMMIT" > $R/COMMIT
SUITE=$R/gsuite_result.json
export PYTHONPATH="$TREE/stack:$TREE" HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $RUNS
fail() { echo "ZZ STRAT1 PILOT FAIL ZZ $1"; exit 1; }
# 0. inputs complete and byte-equal to their box-2 / dev-box sources (sha256 list taken from the source side)
until [ -z "$(find $INP -name '*.part' | head -1)" ] && [ -f $INP/dagger_bank/s1/bank.json ] && [ -f $INP/data/v9/v9_labels_train.npz ] && [ -f $INP/dagger_bank/s1/shard_003of004.npz ]; do sleep 30; done
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify.log 2>&1 || fail "inputs sha256 mismatch ($(grep -v ': OK' $R/logs/inputs_verify.log | head -3 | tr '
' ' '))"
# 1. wait for the suite and both shard-2 reruns
until [ -e $R/logs/gsuite_all.done ] && grep -q "ZZ RERUN" $R/junit/base/shard2.log 2>/dev/null && grep -q "ZZ RERUN" $R/junit/new/shard2.log 2>/dev/null; do sleep 30; done
# 2. compare per test id
$PY $CHK suite --base-junit $R/junit/base --new-junit $R/junit/new --base-commit $BASE_COMMIT --new-commit $COMMIT --out $SUITE > $R/logs/suite_compare.log 2>&1 || fail "G-SUITE regressions (see $SUITE)"
echo "suite: $(cat $R/logs/suite_compare.log)"
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
for kind in C0 N1 S1; do
  tag=pilot200_${kind}_s0
  [ "$(gpu_used)" -lt 3500 ] || fail "GPU busy before $tag ($(gpu_used) MiB)"
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
