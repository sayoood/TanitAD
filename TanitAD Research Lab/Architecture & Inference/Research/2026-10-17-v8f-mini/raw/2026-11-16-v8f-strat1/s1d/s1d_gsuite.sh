#!/bin/bash
# s1d_gsuite.sh -- G-SUITE (full) for the S1d checker commit 4dedac45 (agent/v8f-strat1-s1d-20261117) on the pod.
# tree new9 = new4 (the a4357e79 tree) + 2 files (sha256 in s1d2.sha256): stack/scripts/strat1_check.py, stack/tests/test_strat1_s1d.py (new).
# Waits first for the weather-head suite to end (ZZ WH GSUITE2 DONE|FAIL ZZ), so one suite runs at a time beside the DSCALE bank.
# 8 shards in the base run's composition, compared with junit/new6 (the a4357e79 run) by strat1_check.py suite -> /root/strat1/gsuite_result_s1d.json
# (the --suite file the S1d mint binds).  CPU only.  Sentinel (last line of logs/s1d_gsuite.log): ZZ S1D GSUITE DONE|FAIL ZZ.
R=/root/strat1
TREE=$R/new9
PY=/root/venv/bin/python
LOG=$R/logs/s1d_gsuite.log
J=$R/junit/new9
T=$R/tmp/new9
S=$R/tmp/s1dsuite
fail() { echo "ZZ S1D GSUITE FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
echo "$(date -u +%FT%TZ) waiting for the weather-head suite" > "$LOG"
n=0; until tail -c 600 $R/logs/wh_gsuite2.log 2>/dev/null | grep -qE "ZZ WH GSUITE2 (DONE|FAIL) ZZ"; do n=$((n+1)); [ $n -gt 360 ] && fail "no WH GSUITE2 sentinel after 6 h"; sleep 60; done
echo "$(date -u +%FT%TZ) start" >> "$LOG"
[ -e "$TREE" ] && fail "$TREE exists"
RSS=$(grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{print int($2/1073741824)}')
[ "${RSS:-99}" -lt 30 ] || fail "container anon memory ${RSS} GB >= 30 GB: not started beside it"
cp -a $R/new4 "$TREE" || fail "copy new4"
for f in stack/scripts/strat1_check.py stack/tests/test_strat1_s1d.py; do cp "$S/$f" "$TREE/$f" || fail "copy $f"; done
(cd "$TREE" && sha256sum -c $S/s1d2.sha256) >> "$LOG" 2>&1 || fail "new9 files are not the 4dedac45 blobs"
D=$(diff -rq -x __pycache__ $R/new4/stack $TREE/stack; diff -rq -x __pycache__ $R/new4/taniteval $TREE/taniteval)
D=$(printf '%s\n' "$D" | grep -vE "strat1_check\.py differ|Only in $TREE/stack/tests: test_strat1_s1d\.py" | grep -v '^$')
[ -z "$D" ] || fail "new9 differs from new4 beyond the 2 files: $(printf '%s' "$D" | head -3 | tr '\n' ' ')"
echo "$(date -u +%T) tree new9 == 4dedac45 (2 files over new4)" >> "$LOG"
rm -rf "$J" "$T"; mkdir -p "$J" "$T"
cd "$TREE" || fail "cd"
export PYTHONPATH="$TREE/stack:$TREE" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
$PY -c "import tanitad, os; assert os.path.realpath(tanitad.__file__).startswith('$TREE/stack/'), tanitad.__file__" || fail "import resolves outside new9"
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
    --new-commit 4dedac45dc8d4f6922711ed9af63e0ab54e435be --out $R/gsuite_result_s1d.json > $R/logs/suite_compare_s1d.log 2>&1 || fail "G-SUITE regressions (see $R/gsuite_result_s1d.json)"
echo "suite: $(cat $R/logs/suite_compare_s1d.log)" >> "$LOG"
rm -rf "/root/strat1/tmp/new9"
echo "ZZ S1D GSUITE DONE ZZ" >> "$LOG"
