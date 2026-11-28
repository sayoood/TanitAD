#!/bin/bash
# Runs arrange_splits.py after EACH split finishes extracting.
#
# Why a separate watcher rather than a line added to pull_nuplan_splits.sh: that script is ALREADY
# RUNNING (detached, since 11:2x). Bash reads a script lazily by byte offset, so editing it in place
# can make the live shell execute garbage from the middle of a line -- a documented TanitAD trap.
# This is additive and touches nothing that is running.
#
# Waits for each `SPLIT_DONE <split>` marker, then does the same-volume rename that puts the DBs
# where THEIR runner reads them. Idempotent: arrange_splits.py refuses a non-empty target.
set -u
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
VPY="${REFE_DRIVERL_PY:-C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe}"
L="D:/Projects/TanitAD/data/nuplan/pull_splits.log"
seen=""
echo "$(date +%T) arrange_watcher armed; watching $L"
while true; do
  for s in test val; do
    if [[ "$seen" != *"|$s|"* ]] && grep -q "SPLIT_DONE $s" "$L" 2>/dev/null; then
      echo "$(date +%T) SPLIT_DONE $s seen -> arranging"
      "$VPY" "$PKG/code/arrange_splits.py"            # dry run first, for the record
      "$VPY" "$PKG/code/arrange_splits.py" --apply
      echo "$(date +%T) ARRANGED $s (exit $?)"
      seen="${seen}|$s|"
    fi
  done
  if grep -q "PULL_ALL_DONE" "$L" 2>/dev/null && [[ "$seen" == *"|test|"* && "$seen" == *"|val|"* ]]; then
    echo "$(date +%T) ARRANGE_WATCHER_DONE both splits arranged"; break
  fi
  if grep -qE "DOWNLOAD_FAIL" "$L" 2>/dev/null; then
    echo "$(date +%T) ARRANGE_WATCHER_ABORT download failed"; break
  fi
  sleep 120
done
