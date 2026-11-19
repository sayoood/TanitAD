#!/bin/bash
# queue_reps1.sh -- takes the pod GPU queue lock and runs blockreps1.sh under it (memory pod-gpu-queue-flock). FORK of blockum/queue_blockum.sh. Waits for nothing unless QUEUE_LOG + QUEUE_PATTERN (an ERE for a line at the
# start of a line) are set; then QUEUE_MAXN x QUEUE_POLL s at most. Overrides for the dry run: QUEUE_LOCK, QUEUE_CHAIN, QUEUE_POLL, QUEUE_MAXN.
# launch (MM), as its OWN ssh command, never inside an && list:
#   nohup setsid bash /root/dscale10k/blockreps1/queue_reps1.sh < /dev/null > /root/dscale10k/blockreps1.log 2>&1 &
cd /root/dscale10k || { echo "ZZ DSCALE10K REPS1 FAIL ZZ queue: cd /root/dscale10k"; exit 1; }
LOCK=${QUEUE_LOCK:-/root/gpu_queue.lock}; CHAIN=${QUEUE_CHAIN:-/root/dscale10k/blockreps1/blockreps1.sh}
echo "$(date -u +%FT%TZ) REPS1 queued (lock $LOCK)"
if [ -n "${QUEUE_PATTERN:-}" ]; then
  n=0
  until grep -qE "$QUEUE_PATTERN" "${QUEUE_LOG:-/dev/null}" 2>/dev/null; do
    n=$((n+1)); [ $n -gt ${QUEUE_MAXN:-1440} ] && { echo "ZZ DSCALE10K REPS1 FAIL ZZ queue: no line matching $QUEUE_PATTERN after the wait"; exit 1; }
    sleep ${QUEUE_POLL:-120}
  done
  echo "$(date -u +%FT%TZ) wait ended: $(grep -E "$QUEUE_PATTERN" "$QUEUE_LOG" | tail -n 1 | cut -c1-200)"
fi
exec flock "$LOCK" bash "$CHAIN"
