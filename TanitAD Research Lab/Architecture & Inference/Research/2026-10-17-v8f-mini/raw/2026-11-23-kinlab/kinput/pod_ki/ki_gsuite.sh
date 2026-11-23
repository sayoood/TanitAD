#!/bin/bash
# ki_gsuite.sh -- G-SUITE (full stack/tests, CPU ONLY) for the K-INPUT code of record (commit 4097a89e, `--dagger-base own|logged`) on the pod. A COPY of rk_gsuite.sh revision 2 (itself of ra_gsuite.sh / rh_gsuite.sh): only the env (ki_env.sh), the tree / blob list, the names and the sentinel differ,
# and the junit dir, the temp dir and the log live under the KIT folder (WRITE SCOPE of the KI prep: kit_ki/, tree_ki/ and the result file).
# tree = /root/rollhist/tree_ki (stack + taniteval of commit 4097a89e, blob-verified). BASE = the STAGEB tree's base: /root/strat1/junit/base, the cdbff971 run on /root/strat1/base (same pod).
# 8 shards in the base run's composition (/root/strat1/shards_like_base.py), waves of 3, 3, 2 (the pod is capped at ~15.3 cores: at most ~3 busy processes), pinned to cores 48-95, nice 19, OMP/MKL 2 threads,
# memory guard on the container's own added anon memory (< 20 GB; the watchdog kills the shards by PID); a wave is not started while the container's total_rss is >= 40 GB. The comparison tool is the overlay /root/strat1/overlay_strat1_check.py (read only).
# PASS = 0 regressions AND 0 new-test failures AND 0 missing. Result: /root/rollhist/gsuite_result_ki.json. Sentinel (last line of $HERE/prep_logs/ki_gsuite.log): ZZ KI GSUITE DONE ZZ or ZZ KI GSUITE FAIL ZZ <why>.
# Runs BEFORE the GPU lock (it holds no GPU). Wall MEASURED on the RK run of the same shards: 37 min 32 s (waves 10:15, 16:16, 11:01). Own temp ($T) is deleted at the end.
HERE=$(cd "$(dirname "$0")" && pwd)
. $HERE/ki_env.sh
BASE=cdbff971fa6dad4d20b937fe8abda6cb9f622adb
LOG=$HERE/prep_logs/ki_gsuite.log; JN=$HERE/junit_ki; T=$HERE/tmp_gsuite_ki; JB=$S1/junit/base
mkdir -p $HERE/prep_logs
fail() { echo "ZZ KI GSUITE FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
rssb() { grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{printf "%d", $2}'; }
echo "$(date -u +%FT%TZ) start" > "$LOG"
[ -d "$TREE" ] || fail "no $TREE"
$PY $R/blobcheck.py $TREE $BLOBS >> "$LOG" 2>&1 || fail "tree blob check"
[ "$(find $TREE -type f -not -path '*/__pycache__/*' | wc -l)" = "$(wc -l < $BLOBS)" ] || fail "extra or missing file in the tree"
for k in 0 1 2 3 4 5 6 7; do [ -s "$JB/shard$k.xml" ] && [ -s "$JB/shard$k.files" ] || fail "base junit $JB shard $k missing"; done
rm -rf "$JN" "$T"; mkdir -p "$JN" "$T"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
cd "$TREE" || fail "cd $TREE"
export PYTHONPATH="$TREE/stack:$TREE"
$PY -c "import tanitad, os; assert os.path.realpath(tanitad.__file__).startswith('$TREE/stack/'), tanitad.__file__" || fail "import resolves outside the tree"
$PY $S1/shards_like_base.py $JB $TREE $JN >> "$LOG" 2>&1 || fail "shard lists"
for wave in "0 1 2" "3 4 5" "6 7"; do
  R0=$(rssb); [ "${R0:-0}" -lt 42949672960 ] || fail "container anon memory $R0 B >= 40 GB: wave $wave not started"
  echo "$(date -u +%T) wave $wave start (container rss $((R0/1073741824)) GB baseline)" >> "$LOG"
  pids=()
  for k in $wave; do
    ( mkdir -p "$T/s$k"
      TMPDIR="$T/s$k" TMP="$T/s$k" TEMP="$T/s$k" taskset -c 48-95 nice -n 19 timeout 4h $PY -m pytest -q -p no:cacheprovider --continue-on-collection-errors \
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
$PY $S1/overlay_strat1_check.py suite --base-junit $JB --new-junit $JN --base-commit $BASE --new-commit $COMMIT --out $R/gsuite_result_ki.json > $HERE/prep_logs/suite_compare_ki.log 2>&1 \
  || fail "G-SUITE regressions (see $R/gsuite_result_ki.json)"
$PY -c "
import json,sys
r=json.load(open('$R/gsuite_result_ki.json'))
print('n_regressions',r['n_regressions'],'new_test_failures',len(r['new_test_failures']),'n_new_only',r['n_new_only'],'missing',len(r['missing_in_new']),'base',r['base'],'new',r['new'])
sys.exit(1 if (r['n_regressions'] or r['new_test_failures'] or r['missing_in_new']) else 0)" >> "$LOG" 2>&1 || fail "regressions, new-test failures or missing tests (see $R/gsuite_result_ki.json)"
echo "$(date -u +%FT%TZ) own temp $T deleted (pytest basetemps)" >> $HERE/prep_logs/deletions.log
rm -rf "${T:?}"
echo "ZZ KI GSUITE DONE ZZ" >> "$LOG"
