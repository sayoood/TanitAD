#!/bin/bash
# RBDPOD copy of navrbd_gsuite.sh for tree_fix (5d474c91): base = junit/navrbd (the 0c3b374f run, today), 5 files differ; DRY=1 stops before pytest.
# navrbd_gsuite.sh -- G-SUITE (full stack/tests) for 0c3b374f (e71782a3 + the 16 NAVRBD files) on the pod, CPU ONLY. Pattern: navrb2_fix2_gsuite.sh.
# Base = junit/navrb2_fix2: the e71782a3 run on THIS pod (same environment, 2026-11-18 06:29-06:52Z). Same shard files with the tree path swapped; the new test files go to the shard with the fewest files.
# Two waves of 4, OMP/MKL 2 threads, memory guard: container anon memory < 30 GB before a wave, watchdog fails the suite at >= 52 GB (shards killed by PID).
# PASS = 0 regressions AND 0 new-test failures AND 0 missing.   Result: /root/strat1/gsuite_result_navrbd.json
# Sentinel (last line of logs/navrbd_gsuite.log): ZZ NAVRBD GSUITE DONE ZZ or ZZ NAVRBD GSUITE FAIL ZZ <why>
# STAGED / LAUNCHED DETACHED:  nohup setsid bash /root/strat1/navrbd/navrbd_gsuite.sh < /dev/null > /root/strat1/logs/navrbd_gsuite.driver.log 2>&1 &
R=/root/strat1
N=$R/navrbd
SFX=${DRY:+_dry}
TREE=$N/tree_fix$SFX
BT=$N/tree
PY=/root/venv/bin/python
LOG=$R/logs/navrbd_gsuite_fix$SFX.log
JB=$R/junit/navrbd
JN=$R/junit/navrbd_fix$SFX
T=$R/tmp/navrbdfixsuite$SFX
COMMIT=5d474c9135298a40b3d50b56d611477d9fe97ca3
BASE=0c3b374fb1036b0ca4ca7620f98137dd6c59cca4
NFILES=5
fail() { echo "ZZ NAVRBD GSUITE FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
rss() { grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{print int($2/1073741824)}'; }
mkdir -p $R/logs
echo "$(date -u +%FT%TZ) start" > "$LOG"
[ -d "$TREE" ] || fail "no $TREE"
(cd "$TREE" && sha256sum -c $N/navrbd_tree_fix.sha256) >> "$LOG" 2>&1 || fail "the 16 NAVRBD files are not the 5d474c91 blobs"
D=$(diff -rq -x __pycache__ $BT/stack $TREE/stack; diff -rq -x __pycache__ $BT/taniteval $TREE/taniteval)
n_d=$(printf '%s\n' "$D" | grep -v '^$' | wc -l)
[ "$n_d" -eq "$NFILES" ] || fail "tree_fix differs from tree (0c3b374f) in $n_d entries, expected exactly the $NFILES NAVRBD files"
echo "$(date -u +%T) tree_fix == tree (0c3b374f) + $NFILES rbdfix files" >> "$LOG"
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
[ -n "$DRY" ] && { echo "ZZ DRY OK ZZ (shards composed, import resolves in the tree, no pytest run)" >> "$LOG"; exit 0; }
for wave in "0 1 2 3" "4 5 6 7"; do
  R0=$(rss); [ "${R0:-99}" -lt 30 ] || fail "container anon memory ${R0} GB >= 30 GB: wave $wave not started"
  echo "$(date -u +%T) wave $wave start (rss ${R0} GB)" >> "$LOG"
  pids=()
  for k in $wave; do
    ( mkdir -p "$T/s$k"
      TMPDIR="$T/s$k" TMP="$T/s$k" TEMP="$T/s$k" timeout 4h $PY -m pytest -q -p no:cacheprovider --continue-on-collection-errors \
        --basetemp="$T/s$k/bt" --junitxml="$JN/shard$k.xml" $(cat "$JN/shard$k.files" | tr '\n' ' ') > "$JN/shard$k.log" 2>&1
      echo "rc=$?" >> "$JN/shard$k.log" ) &
    pids+=($!)
  done
  alive() { for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
  peak=0
  while alive; do
    r=$(rss); [ "${r:-0}" -gt "$peak" ] && peak=$r
    if [ "${r:-0}" -ge 52 ]; then
      for p in "${pids[@]}"; do pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done
      fail "memory guard: container anon memory ${r} GB >= 52 GB during wave $wave; shards killed by PID"
    fi
    sleep 15
  done
  echo "$(date -u +%T) wave $wave done (peak container rss ${peak} GB)" >> "$LOG"
done
for k in 0 1 2 3 4 5 6 7; do [ -s "$JN/shard$k.xml" ] || fail "shard $k wrote no junit"; done
cd "$R" || fail "cd $R"
$PY $TREE/stack/scripts/strat1_check.py suite --base-junit $JB --new-junit $JN --base-commit $BASE --new-commit $COMMIT --out $R/gsuite_result_navrbd_fix$SFX.json > $R/logs/suite_compare_navrbd_fix$SFX.log 2>&1 \
  || fail "G-SUITE regressions (see $R/gsuite_result_navrbd_fix$SFX.json)"
$PY -c "
import json,sys
r=json.load(open('$R/gsuite_result_navrbd_fix$SFX.json'))
print('n_regressions',r['n_regressions'],'new_test_failures',len(r['new_test_failures']),'n_new_only',r['n_new_only'],'missing',len(r['missing_in_new']),'base',r['base'],'new',r['new'])
sys.exit(1 if (r['n_regressions'] or r['new_test_failures'] or r['missing_in_new']) else 0)" >> "$LOG" 2>&1 || fail "regressions, new-test failures or missing tests (see $R/gsuite_result_navrbd_fix$SFX.json)"
rm -rf "$T"
echo "ZZ NAVRBD GSUITE DONE ZZ" >> "$LOG"
