#!/bin/bash
# n1dw_gsuite.sh -- G-SUITE (full stack/tests) for 134f5da1 (0c3b374f + the 10 N1DW files) on the pod, CPU ONLY. Pattern: navrbd_gsuite_fix.sh.
# Base = junit/navrbd: the 0c3b374f run on THIS pod (same environment). Same shard files, tree path swapped; new test files go to the shard with the fewest files.
# Pinned to cores 48-95 (taskset), nice 10. Waves of 3, 3, 2 shards (coordinator: at most ~3 busy processes, pod capped at 15.3 cores) (OMP/MKL 2 threads). Memory guard: OUR added container anon memory (total_rss minus the baseline read before the wave) < 20 GB;
# the watchdog fails the suite at >= 20 GB added, shards killed by PID.
# PASS = 0 regressions AND 0 new-test failures AND 0 missing.   Result: /root/strat1/n1dw/gsuite_result_n1dw.json
# Sentinel (last line of logs/n1dw_gsuite.log): ZZ N1DW GSUITE DONE ZZ or ZZ N1DW GSUITE FAIL ZZ <why>
# LAUNCH: nohup setsid bash /root/strat1/n1dw/n1dw_gsuite.sh < /dev/null > /root/strat1/logs/n1dw_gsuite.driver.log 2>&1 &
R=/root/strat1
N=$R/n1dw
TREE=$N/tree
BT=$R/navrbd/tree
PY=/root/venv/bin/python
LOG=$R/logs/n1dw_gsuite.log
JB=$R/junit/navrbd
JN=$R/junit/n1dw
T=$R/tmp/n1dwsuite
COMMIT=134f5da1ee50a3c2fa242bb340a26dd6e4072ade
BASE=0c3b374fb1036b0ca4ca7620f98137dd6c59cca4
NFILES=10
fail() { echo "ZZ N1DW GSUITE FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
rssb() { grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{printf "%d", $2}'; }
mkdir -p $R/logs
echo "$(date -u +%FT%TZ) start" > "$LOG"
[ -d "$TREE" ] || fail "no $TREE"
(cd "$TREE" && sha256sum -c $N/n1dw_tree.sha256) >> "$LOG" 2>&1 || fail "the 10 N1DW files are not the 134f5da1 blobs"
D=$(diff -rq -x __pycache__ $BT/stack $TREE/stack; diff -rq -x __pycache__ $BT/taniteval $TREE/taniteval)
n_d=$(printf '%s\n' "$D" | grep -v '^$' | wc -l)
[ "$n_d" -eq "$NFILES" ] || fail "the tree differs from navrbd/tree (0c3b374f) in $n_d entries, expected exactly the $NFILES N1DW files"
echo "$(date -u +%T) tree == 0c3b374f + $NFILES N1DW files" >> "$LOG"
for k in 0 1 2 3 4 5 6 7; do [ -s "$JB/shard$k.xml" ] && [ -s "$JB/shard$k.files" ] || fail "base junit shard $k missing"; done
rm -rf "$JN" "$T"; mkdir -p "$JN" "$T"
for k in 0 1 2 3 4 5 6 7; do sed "s#^$BT/#$TREE/#" "$JB/shard$k.files" > "$JN/shard$k.files"; done
miss=0; for f in $(cat $JN/shard*.files); do [ -e "$f" ] || miss=$((miss+1)); done; [ $miss = 0 ] || fail "$miss shard files missing in the new tree"
for f in $(ls $TREE/stack/tests/test_*.py); do
  grep -qxF "$f" $JN/shard*.files && continue
  k=0; m=999999; for i in 0 1 2 3 4 5 6 7; do c=$(wc -l < $JN/shard$i.files); [ "$c" -lt "$m" ] && { m=$c; k=$i; }; done
  echo "$f" >> $JN/shard$k.files; echo "new-only -> $k $(basename $f)" >> "$LOG"
done
echo "$(cat $JN/shard*.files | wc -l) test files in 8 shards (junit/navrbd composition)" >> "$LOG"
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
$PY $TREE/stack/scripts/strat1_check.py suite --base-junit $JB --new-junit $JN --base-commit $BASE --new-commit $COMMIT --out $N/gsuite_result_n1dw.json > $R/logs/suite_compare_n1dw.log 2>&1 \
  || fail "G-SUITE regressions (see $N/gsuite_result_n1dw.json)"
$PY -c "
import json,sys
r=json.load(open('$N/gsuite_result_n1dw.json'))
print('n_regressions',r['n_regressions'],'new_test_failures',len(r['new_test_failures']),'n_new_only',r['n_new_only'],'missing',len(r['missing_in_new']),'base',r['base'],'new',r['new'])
sys.exit(1 if (r['n_regressions'] or r['new_test_failures'] or r['missing_in_new']) else 0)" >> "$LOG" 2>&1 || fail "regressions, new-test failures or missing tests (see $N/gsuite_result_n1dw.json)"
rm -rf "$T"
echo "ZZ N1DW GSUITE DONE ZZ" >> "$LOG"
