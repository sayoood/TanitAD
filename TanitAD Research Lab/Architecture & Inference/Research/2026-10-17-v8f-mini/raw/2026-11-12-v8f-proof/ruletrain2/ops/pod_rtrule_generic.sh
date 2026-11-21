#!/usr/bin/env bash
# pod_rtrule_generic.sh <TAG> <CHAIN> <NEW:BASE> [<NEW:BASE> ...] : runs ON THE POD (nohup setsid), started by the lead.
# The label-fed distance rule of card ADD26 (note N1, cap-only) at inference on the checkpoints runs/<CHAIN>/<BASE>, written as rollouts <NEW>.
# Evaluation only, it trains nothing. It waits for the chain <CHAIN> to end (its DONE or STOP marker), then runs one rollout at a time, never as a fourth GPU job.
set -u
TAG=${1:?tag}; CHAIN=${2:?chain}; shift 2
W=/workspace/v8proof; D=$W/distrule; R=$W/runs/distrule/t1; L=$D/rtrule_${TAG}_queue.log; CH=$W/chains/$CHAIN
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$L"; }
[ -e "$D/RTRULE_${TAG}_DONE.txt" ] && { echo "REFUSED: $D/RTRULE_${TAG}_DONE.txt exists"; exit 2; }
[ -f "$D/pod_distrule_run.sh" ] || { echo "REFUSED: no pod_distrule_run.sh"; exit 2; }
[ $# -ge 1 ] || { echo "REFUSED: no rollout given"; exit 2; }
log "queue $TAG start; chain $CHAIN; rollouts $*; run script sha256 $(sha256sum "$D/pod_distrule_run.sh" | cut -c1-16)"
for i in $(seq 1 1440); do
  [ -e "$CH/${CHAIN}_DONE.txt" ] || [ -e "$CH/${CHAIN}_STOP.txt" ] && break
  sleep 30
done
[ -e "$CH/${CHAIN}_DONE.txt" ] || [ -e "$CH/${CHAIN}_STOP.txt" ] || { log "ZZ RTRULE $TAG STOP: chain marker absent after 12 h"; echo WAIT > "$D/RTRULE_${TAG}_STOP.txt"; exit 4; }
log "the chain $CHAIN has ended ($(ls "$CH" | grep -E 'DONE|STOP' | tr '\n' ' ')); starting"
bad=0; tot=0; nd=0
for spec in "$@"; do
  NEW=${spec%%:*}; BASE=${spec##*:}; tot=$((tot+1))
  [ -e "$R/$NEW/T1_DONE" ] && { log "$NEW already done"; nd=$((nd+1)); continue; }
  [ -e "$R/$NEW" ] && { log "SKIP $NEW: output folder exists without T1_DONE"; bad=$((bad+1)); continue; }
  [ -f "$W/runs/$CHAIN/$BASE/ckpt_last.pt" ] || { log "SKIP $NEW: no checkpoint $CHAIN/$BASE"; bad=$((bad+1)); continue; }
  for i in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "${n:-0}" -lt 3 ] && break
    sleep 30
  done
  DR_FLOOR=cap-only bash "$D/pod_distrule_run.sh" "$NEW" "$CHAIN" "$BASE" label > "$D/run_$NEW.out" 2>&1; rc=$?
  log "$NEW (label-fed rule on $CHAIN/$BASE) rc $rc : $(tail -n 1 "$D/run_$NEW.out" | cut -c1-160)"
  if [ $rc = 0 ] && [ -e "$R/$NEW/T1_DONE" ]; then nd=$((nd+1)); else bad=$((bad+1)); fi
done
log "ZZ RTRULE $TAG QUEUE DONE: $nd of $tot rollouts with T1_DONE, $bad failed"
echo "$nd" > "$D/RTRULE_${TAG}_DONE.txt"
