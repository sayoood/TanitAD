#!/bin/bash
# tfar_gsuite.sh -- G-SUITE (full) for the TFAR commit (aedf8716 + tfar.patch, landed by the MM) on the pod, CPU ONLY.
# Pattern: /root/strat1/navrb2/navrb2_gsuite.sh. Tree = /root/strat1/tfar/tree (= the aedf8716 tree /root/strat1/new11 + the files of tfar_tree.sha256), verified here; the BASE run is junit/new11 (aedf8716).
# Same 8 shards in the base run's composition (shards_like_base.py), two waves of 4 shards, OMP/MKL 2 threads each, memory guard: container anon memory (cgroup total_rss) < 30 GB before a wave starts,
# and a watchdog kills this script's shards by PID and FAILS the suite at >= 52 GB (never beside a training run's headroom).
# Result: /root/strat1/gsuite_result_tfar.json (schema tfar-gsuite-result-v1, written by tfar_check.py suite).  Sentinel (last line of logs/tfar_gsuite.log): ZZ TFAR GSUITE DONE|FAIL ZZ.
# The MM edits COMMIT (the landed commit) and NFILES (the line count of tfar_tree.sha256) before launch. STAGED, NOT LAUNCHED:
#   nohup setsid bash /root/strat1/tfar/tfar_gsuite.sh < /dev/null > /root/strat1/logs/tfar_gsuite.driver.log 2>&1 &
R=/root/strat1
N=$R/tfar
TREE=$N/tree
BASETREE=$R/new11
PY=/root/venv/bin/python
LOG=$R/logs/tfar_gsuite.log
J=$R/junit/tfar
T=$R/tmp/tfarsuite
COMMIT=__MM_FILLS_THE_TFAR_COMMIT__
BASE=aedf8716989494464277e50ec4decbbf1a2e65dc
NFILES=__MM_FILLS_NFILES__
fail() { echo "ZZ TFAR GSUITE FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
rss() { grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{print int($2/1073741824)}'; }
echo "$(date -u +%FT%TZ) start" > "$LOG"
case "$COMMIT$NFILES" in *__MM*) fail "COMMIT / NFILES not set";; esac
[ -d "$TREE" ] || fail "no $TREE"
(cd "$TREE" && sha256sum -c $N/tfar_tree.sha256) >> "$LOG" 2>&1 || fail "the tree files are not the TFAR blobs"
D=$(diff -rq -x __pycache__ $BASETREE/stack $TREE/stack; diff -rq -x __pycache__ $BASETREE/taniteval $TREE/taniteval)
n_d=$(printf '%s\n' "$D" | grep -v '^$' | wc -l)
[ "$n_d" -eq "$NFILES" ] || fail "the tree differs from new11 (aedf8716) in $n_d entries, expected exactly the $NFILES TFAR files"
echo "$(date -u +%T) tree == aedf8716 + $NFILES TFAR files" >> "$LOG"
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
$PY $TREE/stack/scripts/tfar_check.py suite --base-junit $R/junit/new11 --new-junit $J --base-commit $BASE \
    --new-commit $COMMIT --out $R/gsuite_result_tfar.json > $R/logs/suite_compare_tfar.log 2>&1 || fail "G-SUITE regressions (see $R/gsuite_result_tfar.json)"
echo "suite: $(cat $R/logs/suite_compare_tfar.log)" >> "$LOG"
rm -rf "$T"
echo "ZZ TFAR GSUITE DONE ZZ" >> "$LOG"
