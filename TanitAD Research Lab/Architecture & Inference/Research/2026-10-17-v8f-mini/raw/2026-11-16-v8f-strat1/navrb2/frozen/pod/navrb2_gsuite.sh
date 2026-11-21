#!/bin/bash
# navrb2_gsuite.sh -- G-SUITE (full) for the NAVRB2 commit (4dedac45 + navrb2.patch, landed by the MM on agent/v8f-navrb2-20261117) on the pod, CPU ONLY.
# Pattern: /root/strat1/ckpt120_gsuite_fix3.sh. Tree = /root/strat1/navrb2/tree (the 4dedac45 tree new8 + the files of navrb2_tree.sha256), verified here; the BASE run is junit/new8 (4dedac45).
# Same 8 shards in the base run's composition (shards_like_base.py), two waves of 4 shards, OMP/MKL 2 threads each, memory guard: container anon memory (cgroup total_rss) < 30 GB before a wave starts,
# and a watchdog kills this script's shards by PID and FAILS the suite at >= 52 GB (never beside a training run's headroom).
# Result: /root/strat1/gsuite_result_navrb2.json (schema strat1-gsuite-result-v1, written by strat1_check.py suite).  Sentinel (last line of logs/navrb2_gsuite.log): ZZ NAVRB2 GSUITE DONE|FAIL ZZ.
# Edit COMMIT (the landed commit) before launch. STAGED, NOT LAUNCHED:
#   nohup setsid bash /root/strat1/navrb2/navrb2_gsuite.sh < /dev/null > /root/strat1/logs/navrb2_gsuite.driver.log 2>&1 &
R=/root/strat1
N=$R/navrb2
TREE=$N/tree
BASETREE=$R/new8
PY=/root/venv/bin/python
LOG=$R/logs/navrb2_gsuite.log
J=$R/junit/navrb2
T=$R/tmp/navrb2suite
COMMIT=__MM_FILLS_THE_NAVRB2_COMMIT__
BASE=4dedac45dc8d4f6922711ed9af63e0ab54e435be
NFILES=14
fail() { echo "ZZ NAVRB2 GSUITE FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
rss() { grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{print int($2/1073741824)}'; }
echo "$(date -u +%FT%TZ) start" > "$LOG"
case "$COMMIT" in __MM*) fail "COMMIT / NFILES not set";; esac
[ -d "$TREE" ] || fail "no $TREE"
(cd "$TREE" && sha256sum -c $N/navrb2_tree.sha256) >> "$LOG" 2>&1 || fail "the tree files are not the NAVRB2 blobs"
D=$(diff -rq -x __pycache__ $BASETREE/stack $TREE/stack; diff -rq -x __pycache__ $BASETREE/taniteval $TREE/taniteval)
n_d=$(printf '%s\n' "$D" | grep -v '^$' | wc -l)
[ "$n_d" -eq "$NFILES" ] || fail "the tree differs from new8 (4dedac45) in $n_d entries, expected exactly the $NFILES NAVRB2 files"
echo "$(date -u +%T) tree == 4dedac45 + $NFILES NAVRB2 files" >> "$LOG"
rm -rf "$J" "$T"; mkdir -p "$J" "$T"
cd "$TREE" || fail "cd"
export PYTHONPATH="$TREE/stack:$TREE" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
$PY -c "import tanitad, os; assert os.path.realpath(tanitad.__file__).startswith('$TREE/stack/'), tanitad.__file__" || fail "import resolves outside the tree"
$PY $R/shards_like_base.py $R/junit/base $TREE $J >> "$LOG" 2>&1 || fail "shard lists"
for wave in "0 1 2 3" "4 5 6 7"; do
  R0=$(rss); [ "${R0:-99}" -lt 30 ] || fail "container anon memory ${R0} GB >= 30 GB: wave $wave not started"
  echo "$(date -u +%T) wave $wave start (rss ${R0} GB)" >> "$LOG"
  pids=()
  for k in $wave; do
    ( mkdir -p "$T/s$k"
      TMPDIR="$T/s$k" TMP="$T/s$k" TEMP="$T/s$k" timeout 4h $PY -m pytest -q -p no:cacheprovider --continue-on-collection-errors \
        --basetemp="$T/s$k/bt" --junitxml="$J/shard$k.xml" $(cat "$J/shard$k.files" | tr '\n' ' ') > "$J/shard$k.log" 2>&1
      echo "rc=$?" >> "$J/shard$k.log" ) &
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
for k in 0 1 2 3 4 5 6 7; do [ -s "$J/shard$k.xml" ] || fail "shard $k wrote no junit"; done
$PY $TREE/stack/scripts/strat1_check.py suite --base-junit $R/junit/new8 --new-junit $J --base-commit $BASE \
    --new-commit $COMMIT --out $R/gsuite_result_navrb2.json > $R/logs/suite_compare_navrb2.log 2>&1 || fail "G-SUITE regressions (see $R/gsuite_result_navrb2.json)"
echo "suite: $(cat $R/logs/suite_compare_navrb2.log)" >> "$LOG"
rm -rf "$T"
echo "ZZ NAVRB2 GSUITE DONE ZZ" >> "$LOG"
