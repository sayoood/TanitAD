#!/usr/bin/env bash
# pod_rtrule6_queue.sh : runs ON THE POD (nohup setsid), started by the lead. Note PREREG_V8F_PROOF_ADD32_N1, stage 1: the label-fed distance rule of card ADD26
# (note N1, cap-only) on the 6,000-step fine-tunes GFT6 and GRT6 (rollouts GFT6RL_s0, GRT6RL_s0, GFT6RL_s1, GRT6RL_s1). Evaluation only, it trains nothing.
# It waits for both chains ruledose_a and ruledose_b to end (DONE or STOP marker of each), then runs one rollout at a time, never as a fourth GPU job.
set -u
W=/workspace/v8proof; D=$W/distrule; R=$W/runs/distrule/t1; L=$D/rtrule6_queue.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$L"; }
[ -e "$D/RTRULE6_DONE.txt" ] && { echo "REFUSED: $D/RTRULE6_DONE.txt exists"; exit 2; }
[ -f "$D/pod_distrule_run.sh" ] || { echo "REFUSED: no pod_distrule_run.sh"; exit 2; }
log "queue start; run script sha256 $(sha256sum "$D/pod_distrule_run.sh" | cut -c1-16)"
ended() { [ -e "$W/chains/$1/${1}_DONE.txt" ] || [ -e "$W/chains/$1/${1}_STOP.txt" ]; }
for i in $(seq 1 1440); do
  ended ruledose_a && ended ruledose_b && break
  sleep 30
done
ended ruledose_a && ended ruledose_b || { log "ZZ RTRULE6 STOP: a chain marker is absent after 12 h"; echo WAIT > "$D/RTRULE6_STOP.txt"; exit 4; }
log "both chains have ended ($(ls "$W/chains/ruledose_a" "$W/chains/ruledose_b" | grep -E 'DONE|STOP' | tr '\n' ' ')); starting"
bad=0; tot=0
for spec in "GFT6RL_s0 GFT6_s0 ruledose_a" "GRT6RL_s0 GRT6_s0 ruledose_a" "GFT6RL_s1 GFT6_s1 ruledose_b" "GRT6RL_s1 GRT6_s1 ruledose_b"; do
  set -- $spec; NEW=$1; BASE=$2; Q=$3; tot=$((tot+1))
  [ -e "$R/$NEW/T1_DONE" ] && { log "$NEW already done"; continue; }
  [ -e "$R/$NEW" ] && { log "SKIP $NEW: output folder exists without T1_DONE"; bad=$((bad+1)); continue; }
  [ -f "$W/runs/$Q/$BASE/ckpt_last.pt" ] || { log "SKIP $NEW: no checkpoint $Q/$BASE"; bad=$((bad+1)); continue; }
  for i in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "${n:-0}" -lt 3 ] && break
    sleep 30
  done
  DR_FLOOR=cap-only bash "$D/pod_distrule_run.sh" "$NEW" "$Q" "$BASE" label > "$D/run_$NEW.out" 2>&1; rc=$?
  log "$NEW (label-fed rule on $Q/$BASE) rc $rc : $(tail -n 1 "$D/run_$NEW.out" | cut -c1-160)"
  [ $rc = 0 ] || bad=$((bad+1))
done
nd=$(ls -d $R/G[FR]T6RL_s[01]/T1_DONE 2>/dev/null | wc -l)
log "ZZ RTRULE6 QUEUE DONE: $nd of $tot rollouts with T1_DONE, $bad failed"
echo "$nd" > "$D/RTRULE6_DONE.txt"
