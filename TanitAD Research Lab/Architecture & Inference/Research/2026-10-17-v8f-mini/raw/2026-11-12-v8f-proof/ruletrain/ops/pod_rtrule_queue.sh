#!/usr/bin/env bash
# pod_rtrule_queue.sh : runs ON THE POD (nohup setsid), started by the lead. Card PREREG_V8F_PROOF_ADD32 (RULETRAIN), section 3: the label-fed distance rule of card
# ADD26 (note N1, cap-only) on the fine-tuned checkpoints GFT and GRT (rollouts GFTRL_s0, GFTRL_s1, GRTRL_s0, GRTRL_s1). Evaluation only, it trains nothing.
# It waits for the chain ruletrain_a to end (its DONE or STOP marker), then runs one rollout at a time, never as a fourth GPU job; the run script checks the memory.
set -u
W=/workspace/v8proof; D=$W/distrule; R=$W/runs/distrule/t1; L=$D/rtrule_queue.log; CH=$W/chains/ruletrain_a
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$L"; }
[ -e "$D/RTRULE_DONE.txt" ] && { echo "REFUSED: $D/RTRULE_DONE.txt exists"; exit 2; }
[ -f "$D/pod_distrule_run.sh" ] || { echo "REFUSED: no pod_distrule_run.sh"; exit 2; }
log "queue start; run script sha256 $(sha256sum "$D/pod_distrule_run.sh" | cut -c1-16)"
for i in $(seq 1 960); do
  [ -e "$CH/ruletrain_a_DONE.txt" ] || [ -e "$CH/ruletrain_a_STOP.txt" ] && break
  sleep 30
done
[ -e "$CH/ruletrain_a_DONE.txt" ] || [ -e "$CH/ruletrain_a_STOP.txt" ] || { log "ZZ RTRULE STOP: chain marker absent after 8 h"; echo WAIT > "$D/RTRULE_STOP.txt"; exit 4; }
log "the chain ruletrain_a has ended ($(ls "$CH" | grep -E 'DONE|STOP' | tr '\n' ' ')); starting"
bad=0; tot=0
for spec in "GFTRL_s0 GFT_s0" "GRTRL_s0 GRT_s0" "GFTRL_s1 GFT_s1" "GRTRL_s1 GRT_s1"; do
  set -- $spec; NEW=$1; BASE=$2; tot=$((tot+1))
  [ -e "$R/$NEW/T1_DONE" ] && { log "$NEW already done"; continue; }
  [ -e "$R/$NEW" ] && { log "SKIP $NEW: output folder exists without T1_DONE"; bad=$((bad+1)); continue; }
  [ -f "$W/runs/ruletrain_a/$BASE/ckpt_last.pt" ] || { log "SKIP $NEW: no checkpoint ruletrain_a/$BASE"; bad=$((bad+1)); continue; }
  for i in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "${n:-0}" -lt 3 ] && break
    sleep 30
  done
  DR_FLOOR=cap-only bash "$D/pod_distrule_run.sh" "$NEW" ruletrain_a "$BASE" label > "$D/run_$NEW.out" 2>&1; rc=$?
  log "$NEW (label-fed rule on ruletrain_a/$BASE) rc $rc : $(tail -n 1 "$D/run_$NEW.out" | cut -c1-160)"
  [ $rc = 0 ] || bad=$((bad+1))
done
nd=$(ls -d $R/G[FR]TRL_s[01]/T1_DONE 2>/dev/null | wc -l)
log "ZZ RTRULE QUEUE DONE: $nd of $tot rollouts with T1_DONE, $bad failed"
echo "$nd" > "$D/RTRULE_DONE.txt"
