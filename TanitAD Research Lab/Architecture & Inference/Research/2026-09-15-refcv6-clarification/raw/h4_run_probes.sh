#!/usr/bin/env bash
# H4 (REFCV6_CLARIFICATION §6.4): does a REF-C trunk's BEV content grow with training exposure?
# Three probe arms, STRICTLY sequential (a second GPU job beside one is MEASURED to crawl):
#   1. main  on refcv4b @ 40,284   2. main on refcv4b @ 9,500   3. main_s1 @ 40,284 (probe-seed floor)
# Everything except the checkpoint is held constant: extractor, head, 6,000 steps, split, seed.
# Usage: h4_run_probes.sh [logfile]      (default: <cache>/h4_probes.log)
set -o pipefail
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CODE="/c/Users/Admin/tanitad-snap-20260915/TanitAD Research Lab/Architecture & Inference/Research/2026-09-13-bev-lidar-corpus-and-head/code"
W=/c/Users/Admin/tanitad-caches/bevhead-20260913
LOG=${1:-$W/h4_probes.log}
export PYTHONIOENCODING=utf-8
export PYTHONPATH="/c/Users/Admin/tanitad-snap-20260915/stack"
: > "$LOG"
for f in tokens_v4b40284 tokens_v4b9500; do
  [ -f "$W/$f/index.npz" ] || { echo "ZZABORT missing $f/index.npz" >> "$LOG"; exit 2; }
done
run () {  # $1 tok dir  $2 arm  $3 tag
  echo "=== $2 $3 start $(date '+%H:%M:%S')" >> "$LOG"
  BEVHEAD_TOK_DIR="$W/$1" "$PY" "$CODE/p4_bev_head.py" --arm "$2" --tag "$3" >> "$LOG" 2>&1
  echo "=== $2 $3 exit=$? $(date '+%H:%M:%S')" >> "$LOG"
}
run tokens_v4b40284 main    v4b40284
run tokens_v4b9500  main    v4b9500
run tokens_v4b40284 main_s1 v4b40284
echo "ZZH4-PROBES-DONE $(date '+%H:%M:%S')" >> "$LOG"
grep -E "^===|test_ap|AP |ZZ" "$LOG" | tail -30
