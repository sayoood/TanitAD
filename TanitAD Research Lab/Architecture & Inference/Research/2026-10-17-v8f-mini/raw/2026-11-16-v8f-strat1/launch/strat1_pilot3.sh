#!/bin/bash
# MM 2026-11-16 (follows strat1_pilot2.sh, which ended "G-SUITE regressions": 87 = 51 nav_source G-DVB on info dicts without the key, 7 otband
# (nav probe autograd on a frozen e_nav), 28 speedsub (the T1 tool pin), 1 kindx_s1 (a data digest of a seed-built synthetic world)).
# The fix commit COMMIT3 (on 22999538) changes 4 files; the tree /root/strat1/new3 = /root/strat1/tree + those 4 files.
# G-SUITE: base cdbff971 (junit/base, unchanged) vs a FULL run of the new3 tree (8 shards, --continue-on-collection-errors), then mint + the three
# 200-step timing pilots (C0, N1, S1 s0), one at a time. Never starts the 7-arm chain.
# usage: strat1_pilot3.sh <COMMIT3 sha>   (detached: nohup setsid bash ... < /dev/null > /root/strat1/logs/pilot3.log 2>&1 &)
# Ends with `ZZ STRAT1 PILOT3 DONE ZZ` or `ZZ STRAT1 PILOT3 FAIL ZZ <why>`.
set -u
COMMIT=$1
R=/root/strat1; TREE=$R/new3; INP=$R/inputs; RUNS=$R/runs3; CACHE=/root/data/cache_pod
PY=/root/venv/bin/python; CHK="$TREE/stack/scripts/strat1_check.py"
BASE_COMMIT=cdbff971fa6dad4d20b937fe8abda6cb9f622adb
SUITE=$R/gsuite_result3.json
fail() { echo "ZZ STRAT1 PILOT3 FAIL ZZ $1"; exit 1; }
# 0. the tree is COMMIT3: the 4 changed files byte-equal to the commit's blobs (sha list shipped as fix3.sha256), nothing else differs from the 22999538 tree
sha256sum -c $R/fix3.sha256 > $R/logs/fix3_verify.log 2>&1 || fail "new3 files are not the COMMIT3 blobs"
D=$(diff -rq -x __pycache__ $R/tree/stack $TREE/stack; diff -rq -x __pycache__ $R/tree/taniteval $TREE/taniteval)
D=$(printf '%s\n' "$D" | grep -vE "stack/scripts/(v8f_mini_train|v8f_t1_speedsub|v8f_mini_eval_kindx)\.py differ|stack/tests/(test_v8f_strat1|test_v8f_kindx_s1)\.py differ" | grep -v '^$')
[ -z "$D" ] || fail "new3 differs from the 22999538 tree beyond the 5 files: $(printf '%s' "$D" | head -3 | tr '\n' ' ')"
echo "$COMMIT" > $R/COMMIT3
echo "$(date -u +%T) tree new3 == $COMMIT (5 files by sha256, rest equal to the 22999538 tree)"
# 1. full G-SUITE on new3 (same env as gsuite.sh; every shard with --continue-on-collection-errors, as base shard 2 was re-run)
J=$R/junit/new3; T=$R/tmp/new3; rm -rf "$J" "$T"; mkdir -p "$J" "$T"
cd "$TREE" || fail "cd"
export PYTHONPATH="$TREE/stack:$TREE" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
$PY -c "import tanitad, os; assert os.path.realpath(tanitad.__file__).startswith('$TREE/stack/'), tanitad.__file__" || fail "import resolves outside new3"
$PY - <<EOF
import glob, os
fs = sorted(glob.glob("$TREE/stack/tests/test_*.py"), key=os.path.getsize, reverse=True)
load = [0]*8; sh = [[] for _ in range(8)]
for f in fs:
    k = load.index(min(load)); sh[k].append(f); load[k] += os.path.getsize(f) + 20000
for k, l in enumerate(sh):
    open("$J/shard%d.files" % k, "w").write("\n".join(l) + "\n")
print(len(fs), "files in 8 shards")
EOF
pids=()
for k in 0 1 2 3 4 5 6 7; do
  ( mkdir -p "$T/s$k"
    TMPDIR="$T/s$k" TMP="$T/s$k" TEMP="$T/s$k" timeout 4h $PY -m pytest -q -p no:cacheprovider --continue-on-collection-errors \
      --basetemp="$T/s$k/bt" --junitxml="$J/shard$k.xml" $(cat "$J/shard$k.files" | tr '\n' ' ') > "$J/shard$k.log" 2>&1
    echo "rc=$?" >> "$J/shard$k.log" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "$(date -u +%T) suite shards done: $(grep -h 'passed\|failed' $J/shard*.log | tail -8 | tr '\n' ' ')"
for k in 0 1 2 3 4 5 6 7; do [ -s "$J/shard$k.xml" ] || fail "shard $k wrote no junit"; done
$PY $CHK suite --base-junit $R/junit/base --new-junit $J --base-commit $BASE_COMMIT --new-commit $COMMIT --out $SUITE > $R/logs/suite_compare3.log 2>&1 || fail "G-SUITE regressions (see $SUITE)"
echo "suite: $(cat $R/logs/suite_compare3.log)"
rm -rf "$T"
# 2. inputs still byte-equal to their sources
sha256sum -c $R/inputs.sha256 > $R/logs/inputs_verify3.log 2>&1 || fail "inputs sha256 mismatch"
# 3. mint + the three 200-step pilots, one at a time, GPU-free first
mkdir -p $RUNS
export PYTHONPATH="$TREE/stack:$TREE"
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
$PY $R/pilot_summary.py $RUNS > $R/pilot_summary3.json 2>&1 || fail "pilot_summary"
echo "ZZ STRAT1 PILOT3 DONE ZZ"
