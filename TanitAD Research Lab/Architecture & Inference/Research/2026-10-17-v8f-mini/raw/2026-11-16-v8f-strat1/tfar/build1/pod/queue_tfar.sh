#!/bin/bash
# queue_tfar.sh: waits for the END LINE of whatever runs before TFAR on the pod GPU, then queues the TFAR chain on the pod GPU lock. No model turn polls; sleep 300 between reads.
# The card's queue order (section 10): DSCALE block 3, then NAVRB2, then TFAR. The predecessor is a VARIABLE the MM sets (no default: an unset sentinel refuses):
#   PRED_LOG      the predecessor's log file on the pod (e.g. /root/strat1/logs/navrb2.log)
#   PRED_PATTERN  an extended regex that matches its end line, success OR failure (e.g. "ZZ STRAT1 NAVRB2 (DONE|FAIL) ZZ")
# Last line of this script's log: the chain's own sentinel, or ZZ TFAR QUEUE FAIL ZZ <why>.
# STAGED, NOT LAUNCHED:  PRED_LOG=... PRED_PATTERN=... nohup setsid bash /root/strat1/tfar/queue_tfar.sh < /dev/null > /root/strat1/logs/tfar_queue.log 2>&1 &
PRED_LOG=${PRED_LOG:-__MM_SETS_PRED_LOG__}
PRED_PATTERN=${PRED_PATTERN:-__MM_SETS_PRED_PATTERN__}
case "$PRED_LOG$PRED_PATTERN" in *__MM_SETS*) echo "ZZ TFAR QUEUE FAIL ZZ PRED_LOG / PRED_PATTERN not set by the MM"; exit 1;; esac
for i in $(seq 1 288); do
  if grep -qE "$PRED_PATTERN" "$PRED_LOG" 2>/dev/null; then break; fi
  sleep 300
done
grep -qE "$PRED_PATTERN" "$PRED_LOG" 2>/dev/null || { echo "ZZ TFAR QUEUE FAIL ZZ no end line matching /$PRED_PATTERN/ in $PRED_LOG after 24 h"; exit 1; }
echo "$(date -u +%FT%TZ) predecessor finished: $(grep -E "$PRED_PATTERN" "$PRED_LOG" | tail -n 1); queued on /root/gpu_queue.lock"
exec flock /root/gpu_queue.lock bash /root/strat1/tfar/tfar_chain.sh
