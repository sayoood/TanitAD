#!/bin/bash
# pod_lcorl_rerun.sh : card PREREG_V8F_PROOF_ADD35. The two label-fed rule rollouts on the own-estimate lead-input arms (LCORL_s0, LCORL_s1) again,
# with the tree the arms were trained and rolled out in (tree_leadcond). The queue ran them in tree_leadsup, whose config does not know the
# lead-input field, and both ended rc 1 at the model build (no rollout, no number). Evaluation only; one rollout at a time; never a fourth GPU job.
set -u
W=/workspace/v8proof; D=$W/distrule; R=$W/runs/distrule/t1; L=$D/rtrule_LCB2_queue.log
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$L"; }
log "rerun start; tree tree_leadcond; run script sha256 $(sha256sum "$D/pod_distrule_run.sh" | cut -c1-16)"
nd=0
for s in 0 1; do
  NEW=LCORL_s$s; BASE=LCO_s$s
  [ -e "$R/$NEW/T1_DONE" ] && { log "$NEW already done"; nd=$((nd+1)); continue; }
  # the runner refuses an existing output folder: the folder of the first run (t1.log with the traceback, no dump) is kept under another name
  [ -e "$R/$NEW" ] && [ ! -e "$R/$NEW.first_run" ] && mv -n "$R/$NEW" "$R/$NEW.first_run"
  [ -e "$R/$NEW" ] && { log "SKIP $NEW: output folder still exists"; continue; }
  for i in $(seq 1 240); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ "${n:-0}" -lt 3 ] && break
    sleep 30
  done
  DR_TREE=$W/tree_leadcond DR_FLOOR=cap-only bash "$D/pod_distrule_run.sh" "$NEW" leadcond_b "$BASE" label > "$D/run_${NEW}_rerun.out" 2>&1; rc=$?
  log "$NEW (label-fed rule on leadcond_b/$BASE, tree_leadcond) rc $rc : $(tail -n 1 "$D/run_${NEW}_rerun.out" | cut -c1-160)"
  [ $rc = 0 ] && [ -e "$R/$NEW/T1_DONE" ] && nd=$((nd+1))
done
log "ZZ RTRULE LCB2 DONE: $nd of 2 rollouts with T1_DONE"
echo "$nd" > "$D/RTRULE_LCB2_DONE.txt"
