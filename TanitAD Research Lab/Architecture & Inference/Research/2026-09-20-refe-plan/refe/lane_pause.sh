#!/bin/bash
# Give the PDM training relabel the pod's CPU quota (7.65 CPUs, cpu.max; R32) while it runs, then resume the teacher-lane
# relabel. SFT-4's chain waits on ZZPDM_TRAIN_EXIT; the lane subset may be partial at launch (PREREG_SFT4 declares it).
# SIGSTOP / SIGCONT only: a stopped multiprocessing worker keeps its task and its memory, nothing is lost or re-queued.
#   bash lane_pause.sh <teacher-lane main pid> [<main pid> ...]
set -u
L=/workspace/data/refe_sft2/lane_pause.log
W=""
for m in "$@"; do
  for c in $(pgrep -P "$m"); do
    grep -q resource_tracker /proc/$c/cmdline 2>/dev/null || W="$W $c"   # the trackers are idle; leave them
  done
done
[ -n "$W" ] || { echo "$(date -u +%FT%T) no workers found for $*" >> $L; exit 1; }
kill -STOP $W && echo "$(date -u +%FT%T) STOPPED$W" >> $L
until grep -q "ZZPDM_TRAIN_EXIT" /workspace/data/refe_sft2/pdm_train.log 2>/dev/null; do sleep 120; done
kill -CONT $W && echo "$(date -u +%FT%T) RESUMED$W ($(grep ZZPDM_TRAIN_EXIT /workspace/data/refe_sft2/pdm_train.log))" >> $L
