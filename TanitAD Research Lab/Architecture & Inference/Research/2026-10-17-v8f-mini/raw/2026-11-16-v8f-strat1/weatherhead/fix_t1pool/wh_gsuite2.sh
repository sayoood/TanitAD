#!/bin/bash
# wh_gsuite2.sh -- G-SUITE (full) for the weather-head commit c02bf3a6 (agent/v8f-whead-20261116) on the pod.
# tree new8 = new4 (the a4357e79 tree) + 5 files (sha256 in wh5b.sha256): v8f_mini.py, v8f_mini_eval.py, v8f_t1_eval_pool.py (new),
# test_v8f_weatherhead.py, v8f_mini_tip_snapshot_whead.txt.  v8f_t1_eval.py is new4's and must keep sha256 180044eb (the speedsub pin).
# 8 shards in the base run's composition, compared with junit/new6 (the a4357e79 run) by strat1_check.py suite.  CPU only.
# Sentinel (last line of logs/wh_gsuite2.log): ZZ WH GSUITE2 DONE|FAIL ZZ.
R=/root/strat1
TREE=$R/new8
PY=/root/venv/bin/python
LOG=$R/logs/wh_gsuite2.log
J=$R/junit/new8
T=$R/tmp/new8
S=$R/tmp/whsuite2
fail() { echo "ZZ WH GSUITE2 FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "$(date -u +%FT%TZ) start" > "$LOG"
[ -e "$TREE" ] && fail "$TREE exists"
RSS=$(grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{print int($2/1073741824)}')
[ "${RSS:-99}" -lt 30 ] || fail "container anon memory ${RSS} GB >= 30 GB: not started beside it"
cp -a $R/new4 "$TREE" || fail "copy new4"
for f in stack/tanitad/models/v8f_mini.py stack/scripts/v8f_mini_eval.py stack/scripts/v8f_t1_eval_pool.py stack/tests/test_v8f_weatherhead.py stack/tests/v8f_mini_tip_snapshot_whead.txt; do
  cp "$S/$f" "$TREE/$f" || fail "copy $f"
done
(cd "$TREE" && sha256sum -c $S/wh5b.sha256) >> "$LOG" 2>&1 || fail "new8 files are not the c02bf3a6 blobs"
echo "180044eb28246ba94a18366ed337f061ec659d6667496c0f88a4a325a0d65c93  $TREE/stack/scripts/v8f_t1_eval.py" | sha256sum -c >> "$LOG" 2>&1 || fail "v8f_t1_eval.py is not at its pin"
D=$(diff -rq -x __pycache__ $R/new4/stack $TREE/stack; diff -rq -x __pycache__ $R/new4/taniteval $TREE/taniteval)
D=$(printf '%s\n' "$D" | grep -vE "v8f_mini\.py differ|v8f_mini_eval\.py differ|Only in $TREE/stack/scripts: v8f_t1_eval_pool\.py|Only in $TREE/stack/tests: (test_v8f_weatherhead\.py|v8f_mini_tip_snapshot_whead\.txt)" | grep -v '^$')
[ -z "$D" ] || fail "new8 differs from new4 beyond the 5 files: $(printf '%s' "$D" | head -3 | tr '\n' ' ')"
echo "$(date -u +%T) tree new8 == c02bf3a6 (5 files over new4, v8f_t1_eval.py at its pin)" >> "$LOG"
rm -rf "$J" "$T"; mkdir -p "$J" "$T"
cd "$TREE" || fail "cd"
export PYTHONPATH="$TREE/stack:$TREE" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
$PY -c "import tanitad, os; assert os.path.realpath(tanitad.__file__).startswith('$TREE/stack/'), tanitad.__file__" || fail "import resolves outside new8"
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
echo "$(date -u +%T) suite shards done" >> "$LOG"
for k in 0 1 2 3 4 5 6 7; do [ -s "$J/shard$k.xml" ] || fail "shard $k wrote no junit"; done
$PY $R/new4/stack/scripts/strat1_check.py suite --base-junit $R/junit/new6 --new-junit $J --base-commit a4357e7986f3aa30dab2ed83fde2351fda8d5662 \
    --new-commit c02bf3a6e77fdcd316a9cadc04224d3e3154abf7 --out $R/runs/gsuite_wh2.json > $R/logs/suite_compare_wh2.log 2>&1 || fail "G-SUITE regressions (see $R/runs/gsuite_wh2.json)"
echo "suite: $(cat $R/logs/suite_compare_wh2.log)" >> "$LOG"
rm -rf "/root/strat1/tmp/new8" "/root/strat1/new7"
echo "ZZ WH GSUITE2 DONE ZZ" >> "$LOG"
