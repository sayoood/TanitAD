#!/bin/bash
# STRAT1 SEEDS-DISAGREE chain (MM 2026-11-16; PI go "Both"): C0 and S1 at seeds 10 (init 0) and 11 (init 1), PREREG_V8F_STRAT1 outcome table.
# Tree new4 = commit a4357e79 (e81bde01 + strat1_check.py SEEDS s10/s11 + one test). Steps: 0 tree check, 1 full G-SUITE with the base run's shard
# composition (as pilot5), 2 inputs check, 3 the 4 arms one at a time (mint, run through the runner, post check, ARM_DONE).
# usage: nohup setsid bash /root/strat1/strat1_s10.sh < /dev/null > /root/strat1/logs/s10.log 2>&1 &
# Ends with `ZZ STRAT1 S10 DONE ZZ` or `ZZ STRAT1 S10 FAIL ZZ <why>`. A STOP file at /root/strat1/chains/STOP ends it between arms.
set -u
R=/root/strat1; TREE=$R/new4; INP=$R/inputs; RUNS=$R/runs; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
COMMIT=a4357e7986f3aa30dab2ed83fde2351fda8d5662
BASE_COMMIT=cdbff971fa6dad4d20b937fe8abda6cb9f622adb
SUITE=$R/gsuite_result6.json
ARMS="C0:s10 S1:s10 C0:s11 S1:s11"
fail() { echo "ZZ STRAT1 S10 FAIL ZZ $1"; exit 1; }
# 0. new4 == new3 except the 2 files of a4357e79, which equal the commit's blobs
sha256sum -c $R/fix6.sha256 > $R/logs/fix6_verify.log 2>&1 || fail "new4 files are not the a4357e79 blobs"
D=$(diff -rq -x __pycache__ $R/new3/stack $TREE/stack; diff -rq -x __pycache__ $R/new3/taniteval $TREE/taniteval)
D=$(printf '%s\n' "$D" | grep -vE "stack/scripts/strat1_check\.py differ|stack/tests/test_v8f_strat1\.py differ" | grep -v '^$')
[ -z "$D" ] || fail "new4 differs from new3 beyond the 2 files: $(printf '%s' "$D" | head -3 | tr '\n' ' ')"
echo "$(date -u +%T) tree new4 == $COMMIT"
# 1. full G-SUITE on new4 with base's shard composition
J=$R/junit/new6; T=$R/tmp/new6; rm -rf "$J" "$T"; mkdir -p "$J" "$T"
cd "$TREE" || fail "cd"
export PYTHONPATH="$TREE/stack:$TREE" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
$PY -c "import tanitad, os; assert os.path.realpath(tanitad.__file__).startswith('$TREE/stack/'), tanitad.__file__" || fail "import resolves outside new4"
$PY $R/shards_like_base.py $R/junit/base $TREE $J || fail "shard lists"
pids=()
for k in 0 1 2 3 4 5 6 7; do
  ( mkdir -p "$T/s$k"
    TMPDIR="$T/s$k" TMP="$T/s$k" TEMP="$T/s$k" timeout 4h $PY -m pytest -q -p no:cacheprovider --continue-on-collection-errors \
      --basetemp="$T/s$k/bt" --junitxml="$J/shard$k.xml" $(cat "$J/shard$k.files" | tr '\n' ' ') > "$J/shard$k.log" 2>&1
    echo "rc=$?" >> "$J/shard$k.log" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "$(date -u +%T) suite shards done"
for k in 0 1 2 3 4 5 6 7; do [ -s "$J/shard$k.xml" ] || fail "shard $k wrote no junit"; done
$PY $CHK suite --base-junit $R/junit/base --new-junit $J --base-commit $BASE_COMMIT --new-commit $COMMIT --out $SUITE > $R/logs/suite_compare6.log 2>&1 || fail "G-SUITE regressions (see $SUITE)"
echo "suite: $(cat $R/logs/suite_compare6.log)"
rm -rf "$T"
# 2. inputs still byte-equal to their sources
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify6.log 2>&1 || fail "inputs sha256 mismatch"
# 3. the 4 arms, one at a time
unset CUDA_VISIBLE_DEVICES
mkdir -p $R/chains $RUNS
COMMON="--root $TREE --inputs $INP --cache-dir $CACHE --runs $RUNS --commit $COMMIT --suite $SUITE --py $PY --gpu 0"
gpu_idle() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-99999}" -lt 3500 ]; }
echo "$(date -u +%FT%TZ) chain start commit=$COMMIT"
for arm in $ARMS; do
  kind=${arm%%:*}; seed=${arm##*:}; tag=${kind}_${seed}
  [ -e $R/chains/STOP ] && fail "STOP file before $tag"
  [ -e $RUNS/$tag/ARM_DONE ] && { echo "skip $tag (ARM_DONE)"; continue; }
  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle (>= 3500 MiB used) before $tag"; sleep 10; done
  if [ ! -f $RUNS/$tag/gate/gate_token.json ]; then
    echo "$(date -u +%T) mint $tag"
    $PY $CHK mint $kind $seed $COMMON --steps 3000 > $RUNS/mint_$tag.log 2>&1 || fail "mint $tag refused: $(tail -c 300 $RUNS/mint_$tag.log)"
  fi
  echo "$(date -u +%T) run $tag"
  $PY $CHK run $kind $seed $COMMON --steps 3000 > $RUNS/run_$tag.log 2>&1; rc=$?
  [ $rc -eq 0 ] || fail "run $tag rc=$rc: $(tail -c 300 $RUNS/run_$tag.log)"
  init=$INP/M2b_s0/ckpt_last.pt; case $seed in s1|s11) init=$INP/M2b_s1/ckpt_last.pt;; esac
  $PY $CHK post $kind --init $init --final $RUNS/$tag/train/ckpt_last.pt > $RUNS/post_$tag.log 2>&1 || fail "post check $tag: $(tail -c 300 $RUNS/post_$tag.log)"
  echo "ZZ ARM $tag DONE ZZ" | tee $RUNS/$tag/ARM_DONE
done
echo "ZZ STRAT1 S10 DONE ZZ"
