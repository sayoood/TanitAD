#!/bin/bash
# Arrange the TEST split the moment ITS extraction finishes, instead of ~40 h later.
#
# ⛔ THE PROBLEM THIS SOLVES, MEASURED 2026-09-20. `pull_nuplan_splits.sh` runs the test extraction
# in the BACKGROUND with its stdout redirected to `extract_test.log` (line 28), and only `cat`s that
# log into `pull_splits.log` at line 35 -- AFTER the val download returns. `arrange_watcher.sh`
# greps `pull_splits.log`, so it cannot see `SPLIT_DONE test` until val has finished downloading.
# Val is NETWORK-bound at ~0.7-1.6 MB/s (measured against 11.6 MB/s disk and a 1.62 MB/s
# memory-only S3 range GET, so it is the link, not the disk), i.e. ~40 h.
#
# ⭐ WHY THAT DELAY IS NOT ACADEMIC: the four **test14** tasks -- the rows DriveZero's headline table
# actually reports -- read `nuplan-v1.1/splits/test` and need the TEST split ONLY. The val14 tasks
# are the ones that need val. So a test-only arrange unblocks the headline reproduction now and
# leaves val on its own schedule.
#
# SAFETY. `arrange_splits.py` REFUSES a target directory that exists and is non-empty, so the
# existing watcher's later call is a harmless no-op and the two cannot fight. ⛔ It must NOT run
# while extraction is still writing into `data/cache/test/`, which is why this waits for the
# `SPLIT_DONE test` marker rather than for a file count or a timer.
#
# ⚠️ The wait exits on FAILURE as well as success. A loop that greps only the success marker polls
# forever through a crash and looks identical to "still running" (that has cost this programme 38 h
# once already).
set -u
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
VPY="${REFE_DRIVERL_PY:-C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe}"
EL="D:/Projects/TanitAD/data/nuplan/extract_test.log"
echo "$(date +%T) arrange_test_early armed; watching $EL for SPLIT_DONE/SPLIT_FAIL test"
while true; do
  if grep -q "SPLIT_DONE test" "$EL" 2>/dev/null; then
    echo "$(date +%T) SPLIT_DONE test seen -> arranging TEST only"
    "$VPY" "$PKG/code/arrange_splits.py"
    "$VPY" "$PKG/code/arrange_splits.py" --apply
    echo "$(date +%T) ARRANGE_TEST_EARLY_DONE (exit $?)"
    break
  fi
  if grep -qE "SPLIT_FAIL test" "$EL" 2>/dev/null; then
    echo "$(date +%T) ARRANGE_TEST_EARLY_ABORT: extraction reported SPLIT_FAIL"; break
  fi
  if ! grep -q "extracting to" "$EL" 2>/dev/null; then
    echo "$(date +%T) ARRANGE_TEST_EARLY_ABORT: extract log never started"; break
  fi
  sleep 120
done
