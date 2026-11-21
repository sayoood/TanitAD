#!/bin/bash
# n1dm_gsuite.sh -- G-SUITE (full stack/tests) for the N1DM code of record (1f217bf0 + the N1DM delta files, NFILES entries vs the N1DH tree) on the pod, CPU ONLY. Pattern: n1dh_gsuite.sh (which ran: 0 regressions, 15,476 tests).
# LINEAGE: N1D trained at 0c3b374f, an ancestor of 1f217bf0 (N1DH's code of record); M2b is an ancestor of it too. The N1DM delta is checked against 1f217bf0 by the dev-box patch (mkpatch_n1dm.py: base blobs + patch == tree).
# Base = junit/n1dh: the 1f217bf0 run on THIS pod (same environment; tree /root/strat1/n1dh/tree = 1f217bf0 + the eval-141 grid, re-verified blob by blob below). Same shard files, tree path swapped to n1dm/tree; new test files go to the shard with the fewest files.
# Pinned to cores 48-95 (taskset), nice 10. Waves of 3, 3, 2 shards (coordinator: at most ~3 busy processes, pod capped at 15.3 cores) (OMP/MKL 2 threads). Memory guard: OUR added container anon memory (total_rss minus the baseline read before the wave) < 20 GB;
# the watchdog fails the suite at >= 20 GB added, shards killed by PID.
# PASS = 0 regressions AND 0 new-test failures AND 0 missing.   Result: /root/strat1/n1dm/gsuite_result_n1dm.json
# Sentinel (last line of logs/n1dm_gsuite.log): ZZ N1DM GSUITE DONE ZZ or ZZ N1DM GSUITE FAIL ZZ <why>
# LAUNCH (the chain runs it as a step, under the GPU lock): bash /root/strat1/n1dm/n1dm_gsuite.sh
R=/root/strat1
N=$R/n1dm
TREE=$N/tree
BT=$R/n1dh/tree
PY=/root/venv/bin/python
LOG=$R/logs/n1dm_gsuite.log
JB=$R/junit/n1dh
JN=$R/junit/n1dm
T=$R/tmp/n1dmsuite
COMMIT=@@N1DM_COMMIT@@
BASE=1f217bf015ebd6d69b888e424a2675970ff72075
NFILES=@@N1DM_NFILES@@
fail() { echo "ZZ N1DM GSUITE FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
rssb() { grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{printf "%d", $2}'; }
case "$COMMIT$NFILES" in *@@*) echo "ZZ N1DM GSUITE FAIL ZZ placeholders not substituted (run n1dm_stage.sh)"; exit 1;; esac
mkdir -p $R/logs
echo "$(date -u +%FT%TZ) start" > "$LOG"
[ -d "$TREE" ] || fail "no $TREE"
(cd "$TREE" && sha256sum -c $N/n1dm_tree.sha256) >> "$LOG" 2>&1 || fail "the N1DM delta files are not the landed commit's blobs (n1dm_tree.sha256)"
D=$(diff -rq -x __pycache__ $BT/stack $TREE/stack; diff -rq -x __pycache__ $BT/taniteval $TREE/taniteval)
n_d=$(printf '%s\n' "$D" | grep -v '^$' | wc -l)
[ "$n_d" -eq "$NFILES" ] || fail "the tree differs from n1dh/tree (1f217bf0) in $n_d entries, expected exactly the $NFILES N1DM files"
echo "$(date -u +%T) tree == 1f217bf0 + $NFILES N1DM files" >> "$LOG"
/root/venv/bin/python $N/blobcheck.py $BT $R/n1dh/all_blobs.txt >> "$LOG" 2>&1 || fail "the base tree $BT is not the 1f217bf0 blobs (junit/n1dh was run on it)"
for k in 0 1 2 3 4 5 6 7; do [ -s "$JB/shard$k.xml" ] && [ -s "$JB/shard$k.files" ] || fail "base junit (junit/n1dh) shard $k missing"; done
rm -rf "$JN" "$T"; mkdir -p "$JN" "$T"
for k in 0 1 2 3 4 5 6 7; do sed "s#^$BT/#$TREE/#" "$JB/shard$k.files" > "$JN/shard$k.files"; done
miss=0; for f in $(cat $JN/shard*.files); do [ -e "$f" ] || miss=$((miss+1)); done; [ $miss = 0 ] || fail "$miss shard files missing in the new tree"
for f in $(ls $TREE/stack/tests/test_*.py); do
  grep -qxF "$f" $JN/shard*.files && continue
  k=0; m=999999; for i in 0 1 2 3 4 5 6 7; do c=$(wc -l < $JN/shard$i.files); [ "$c" -lt "$m" ] && { m=$c; k=$i; }; done
  echo "$f" >> $JN/shard$k.files; echo "new-only -> $k $(basename $f)" >> "$LOG"
done
echo "$(cat $JN/shard*.files | wc -l) test files in 8 shards (junit/n1dm composition)" >> "$LOG"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
cd "$TREE" || fail "cd $TREE"
export PYTHONPATH="$TREE/stack:$TREE"
$PY -c "import tanitad, os; assert os.path.realpath(tanitad.__file__).startswith('$TREE/stack/'), tanitad.__file__" || fail "import resolves outside the tree"
for wave in "0 1 2" "3 4 5" "6 7"; do
  R0=$(rssb); [ "${R0:-0}" -lt 45000000000 ] || fail "container anon memory $R0 B >= 45 GB: wave $wave not started"
  echo "$(date -u +%T) wave $wave start (container rss $((R0/1073741824)) GB baseline)" >> "$LOG"
  pids=()
  for k in $wave; do
    ( mkdir -p "$T/s$k"
      TMPDIR="$T/s$k" TMP="$T/s$k" TEMP="$T/s$k" taskset -c 48-95 nice -n 10 timeout 4h $PY -m pytest -q -p no:cacheprovider --continue-on-collection-errors \
        --basetemp="$T/s$k/bt" --junitxml="$JN/shard$k.xml" $(cat "$JN/shard$k.files" | tr '\n' ' ') > "$JN/shard$k.log" 2>&1
      echo "rc=$?" >> "$JN/shard$k.log" ) &
    pids+=($!)
  done
  alive() { for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
  peak=0
  while alive; do
    r=$(rssb); d=$((r-R0)); [ "$d" -gt "$peak" ] && peak=$d
    if [ "$d" -ge 21474836480 ]; then
      for p in "${pids[@]}"; do pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done
      fail "memory guard: container anon memory grew by $((d/1073741824)) GB (>= 20 GB) during wave $wave; shards killed by PID"
    fi
    sleep 15
  done
  echo "$(date -u +%T) wave $wave done (peak container rss growth $((peak/1073741824)) GB)" >> "$LOG"
done
for k in 0 1 2 3 4 5 6 7; do [ -s "$JN/shard$k.xml" ] || fail "shard $k wrote no junit"; done
cd "$R" || fail "cd $R"
$PY $TREE/stack/scripts/strat1_check.py suite --base-junit $JB --new-junit $JN --base-commit $BASE --new-commit $COMMIT --out $N/gsuite_result_n1dm.json > $R/logs/suite_compare_n1dm.log 2>&1 \
  || fail "G-SUITE regressions (see $N/gsuite_result_n1dm.json)"
$PY -c "
import json,sys
r=json.load(open('$N/gsuite_result_n1dm.json'))
print('n_regressions',r['n_regressions'],'new_test_failures',len(r['new_test_failures']),'n_new_only',r['n_new_only'],'missing',len(r['missing_in_new']),'base',r['base'],'new',r['new'])
sys.exit(1 if (r['n_regressions'] or r['new_test_failures'] or r['missing_in_new']) else 0)" >> "$LOG" 2>&1 || fail "regressions, new-test failures or missing tests (see $N/gsuite_result_n1dm.json)"
rm -rf "$T"
echo "ZZ N1DM GSUITE DONE ZZ" >> "$LOG"
