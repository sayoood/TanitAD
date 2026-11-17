#!/bin/bash
# ckpt120_gsuite_fix3.sh -- G-SUITE (full) for the CKPT120 commit aedf8716989494464277e50ec4decbbf1a2e65dc (agent/v8f-ckpt120-20261117) on the pod, CPU ONLY.
# tree new11 = new4 (the a4357e79 tree) + the 20 commit files (sha256 list /root/strat1/ckpt120_new11.sha256), verified here.
# Same 8 shards in the base run's composition as the s1d suite (shards_like_base.py, base junit/new6 = the a4357e79 run), but run in two waves of 4 shards,
# OMP/MKL 2 threads each, with a memory guard: container anon memory (cgroup memory.stat total_rss) must be < 30 GB before a wave starts, and if it passes
# 52 GB (limit 70.9 GB less an 18 GB margin) the shards of this script are killed by PID and the suite FAILS (never beside a training run's headroom).
# Result: /root/strat1/gsuite_result_ckpt120_fix3.json (schema ckpt120-gsuite-result-v1, written by ckpt120_check.py suite).  Sentinel (last line of logs/ckpt120_gsuite_fix3.log): ZZ CKPT120 GSUITE FIX3 DONE|FAIL ZZ.
# Launch: nohup setsid bash /root/strat1/ckpt120_gsuite_fix3.sh < /dev/null > /root/strat1/logs/ckpt120_gsuite_fix3.driver.log 2>&1 &
R=/root/strat1
TREE=$R/new11
PY=/root/venv/bin/python
LOG=$R/logs/ckpt120_gsuite_fix3.log
J=$R/junit/new11
T=$R/tmp/new11suite
COMMIT=aedf8716989494464277e50ec4decbbf1a2e65dc
BASE=a4357e7986f3aa30dab2ed83fde2351fda8d5662
fail() { echo "ZZ CKPT120 GSUITE FIX3 FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
rss() { grep -E "^total_rss " /sys/fs/cgroup/memory/memory.stat | awk '{print int($2/1073741824)}'; }
echo "$(date -u +%FT%TZ) start" > "$LOG"
[ -d "$TREE" ] || fail "no $TREE"
(cd "$TREE" && sha256sum -c $R/ckpt120_new11.sha256) >> "$LOG" 2>&1 || fail "new11 files are not the aedf8716 blobs"
D=$(diff -rq -x __pycache__ $R/new4/stack $TREE/stack; diff -rq -x __pycache__ $R/new4/taniteval $TREE/taniteval)
n_d=$(printf '%s\n' "$D" | grep -v '^$' | wc -l)
[ "$n_d" -eq 20 ] || fail "new11 differs from new4 in $n_d entries, expected exactly the 20 commit files"
echo "$(date -u +%T) tree new11 == aedf8716 (20 files over new4)" >> "$LOG"
rm -rf "$J" "$T"; mkdir -p "$J" "$T"
cd "$TREE" || fail "cd"
export PYTHONPATH="$TREE/stack:$TREE" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1
$PY -c "import tanitad, os; assert os.path.realpath(tanitad.__file__).startswith('$TREE/stack/'), tanitad.__file__" || fail "import resolves outside new11"
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
$PY $TREE/stack/scripts/ckpt120_check.py suite --base-junit $R/junit/new6 --new-junit $J --base-commit $BASE \
    --new-commit $COMMIT --out $R/gsuite_result_ckpt120_fix3.json > $R/logs/suite_compare_ckpt120_fix3.log 2>&1 || fail "G-SUITE regressions (see $R/gsuite_result_ckpt120_fix3.json)"
echo "suite: $(cat $R/logs/suite_compare_ckpt120_fix3.log)" >> "$LOG"
rm -rf "$T"
echo "ZZ CKPT120 GSUITE FIX3 DONE ZZ" >> "$LOG"
