#!/bin/bash
# FORK of block3/queue_block3.sh for the conditional arm X_s0 (the 10,000 data-scale clips alone; card 2.3 row X_s0, section 6 DATA-HURTS). Changes: blockx/FORK_NOTES.md. NOT launched: the MM asks the PI first.
# queue_block3.sh -- waits for the STRAT1 C0REP chain to end (its `ZZ STRAT1 C0REP DONE|FAIL ZZ` line at the start of a line in its log), then takes the pod GPU queue lock and runs block3.sh under it (memory pod-gpu-queue-flock).
# Same shape as /root/strat1/queue_c0rep.sh. The overrides are for the dry run only (QUEUE_LOG, QUEUE_LOCK, QUEUE_CHAIN, QUEUE_POLL, QUEUE_MAXN).
# launch (MM), as its OWN ssh command, never inside an && list:
#   nohup setsid bash /root/dscale10k/block3/queue_block3.sh < /dev/null > /root/dscale10k/block3.log 2>&1 &
L=${QUEUE_LOG:-/root/strat1/logs/navrb2_queue.log}; LOCK=${QUEUE_LOCK:-/root/gpu_queue.lock}; CHAIN=${QUEUE_CHAIN:-/root/dscale10k/blockx/blockx.sh}
echo "$(date -u +%FT%TZ) BLOCKX queued behind the NAVRB2 chain ($L)"
n=0
until grep -qE '^ZZ STRAT1 NAVRB2 (DONE|FAIL) ZZ' "$L" 2>/dev/null; do
  n=$((n+1)); [ $n -gt ${QUEUE_MAXN:-1440} ] && { echo "ZZ DSCALE10K BLOCKX FAIL ZZ queue: no NAVRB2 end line after 48 h"; exit 1; }
  sleep ${QUEUE_POLL:-120}
done
echo "$(date -u +%FT%TZ) NAVRB2 ended: $(grep -E '^ZZ STRAT1 NAVRB2 (DONE|FAIL) ZZ' "$L" | tail -n 1 | cut -c1-200)"
exec flock "$LOCK" bash "$CHAIN"
