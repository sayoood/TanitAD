#!/usr/bin/env bash
# pod_distrule_queue.sh : runs ON THE POD (nohup setsid), started by the lead. Card PREREG_V8F_PROOF_ADD26 with note N1 (projection cap-only, control r2 by mode plumb).
# It waits until the three chains of card ADD24 are over (DONE or STOP marker each), then runs, evaluation only:
#   controls : HIERRO_s0 (rule off through the wrapper; r1: arrays equal to the banked dump), HIERRM_s0 (plumb; r2: arrays equal to HIERRO_s0), HIERRP_s0 (push; r3 is read on the dev box)
#   arms A   : label source on HIER_s0..s5 and G_s0..s5
#   arms B   : head source and label source on LS / LSD / LSH, seeds 0 and 1 (an arm without a checkpoint is skipped and logged)
# At most three rollouts at a time; every rollout goes through pod_distrule_run.sh (memory check, pinned tool check, no overwrite). It trains nothing.
set -u
W=/workspace/v8proof; D=$W/distrule; CH=$W/chains; R=$W/runs/distrule/t1; L=$D/queue.log
export W D R L
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$L"; }
export -f log
[ -e "$D/DISTRULE_DONE.txt" ] && { echo "REFUSED: $D/DISTRULE_DONE.txt exists"; exit 2; }
log "queue start; waits for the DONE or STOP marker of leadsup_a, leadsup_b, leadsup_c"
n=0
while :; do
  k=0; for c in leadsup_a leadsup_b leadsup_c; do { [ -e $CH/$c/${c}_DONE.txt ] || [ -e $CH/$c/${c}_STOP.txt ]; } && k=$((k+1)); done
  [ $k = 3 ] && break
  n=$((n+1)); [ $n -gt 480 ] && { log "ZZ DISTRULE STOP: the chains of card ADD24 are not over after 8 h"; echo WAIT > "$D/DISTRULE_STOP.txt"; exit 3; }
  sleep 60
done
log "chains over: $(ls $CH/leadsup_a $CH/leadsup_b $CH/leadsup_c | grep -E 'DONE|STOP' | tr '\n' ' ')"
run1() { # NEW Q BASE MODE
  local NEW=$1 Q=$2 BASE=$3 MODE=$4 rc
  [ -e "$R/$NEW/T1_DONE" ] && { log "$NEW already done"; return 0; }
  [ -e "$R/$NEW" ] && { log "SKIP $NEW: output folder exists without T1_DONE"; return 1; }
  [ -f "$W/runs/$Q/$BASE/ckpt_last.pt" ] || { log "SKIP $NEW: no checkpoint $Q/$BASE"; return 1; }
  sleep $(( RANDOM % 20 ))
  DR_FLOOR=cap-only bash "$D/pod_distrule_run.sh" "$NEW" "$Q" "$BASE" "$MODE" > "$D/run_$NEW.out" 2>&1; rc=$?
  log "$NEW ($MODE on $Q/$BASE) rc $rc : $(tail -n 1 "$D/run_$NEW.out" | cut -c1-160)"
  return $rc
}
export -f run1
# controls
run1 HIERRO_s0 hierrep HIER_s0 off   || { log "ZZ DISTRULE STOP: rule off did not run"; echo R1-RUN > "$D/DISTRULE_STOP.txt"; exit 4; }
c1=$(/workspace/venv/bin/python "$D/cmp_dumps.py" "$R/HIERRO_s0/dump_HIERRO_s0_m10_causal_heads" "$W/runs/hierrep/t1/HIER_s0/dump_HIER_s0_m10_causal_heads" 2>&1 | tail -n 1); rc1=$?
log "r1 off against the banked dump: $c1"
echo "$c1" | grep -q 'differing=0 ' || { log "ZZ DISTRULE STOP: control r1 failed"; echo R1 > "$D/DISTRULE_STOP.txt"; exit 4; }
run1 HIERRM_s0 hierrep HIER_s0 plumb || { log "ZZ DISTRULE STOP: plumb did not run"; echo R2-RUN > "$D/DISTRULE_STOP.txt"; exit 4; }
c2=$(/workspace/venv/bin/python "$D/cmp_dumps.py" "$R/HIERRM_s0/dump_HIERRM_s0_m10_causal_heads" "$R/HIERRO_s0/dump_HIERRO_s0_m10_causal_heads" 2>&1 | tail -n 1)
log "r2 plumb against off: $c2"
echo "$c2" | grep -q 'differing=0 ' || { log "ZZ DISTRULE STOP: control r2 failed"; echo R2 > "$D/DISTRULE_STOP.txt"; exit 4; }
J=$D/jobs.txt
{
  echo "HIERRP_s0 hierrep HIER_s0 push"
  for s in 0 1 2 3 4 5; do echo "HIERRL_s$s hierrep HIER_s$s label"; done
  echo "GRL_s0 spatial G_s0 label"; echo "GRL_s1 spatial G_s1 label"
  echo "GRL_s2 spatialrep_a G_s2 label"; echo "GRL_s3 spatialrep_a G_s3 label"
  echo "GRL_s4 spatialrep_b G_s4 label"; echo "GRL_s5 spatialrep_b G_s5 label"
  for s in 0 1; do
    echo "LSRH_s$s leadsup_a LS_s$s head";   echo "LSRL_s$s leadsup_a LS_s$s label"
    echo "LSDRH_s$s leadsup_b LSD_s$s head"; echo "LSDRL_s$s leadsup_b LSD_s$s label"
    echo "LSHRH_s$s leadsup_c LSH_s$s head"; echo "LSHRL_s$s leadsup_c LSH_s$s label"
  done
} > "$J"
log "controls r1 and r2 passed; $(wc -l < "$J") rollouts, three at a time"
xargs -P 3 -L 1 bash -c 'run1 "$@"' _ < "$J"
nd=$(ls -d $R/*/T1_DONE 2>/dev/null | wc -l)
log "ZZ DISTRULE QUEUE DONE: $nd rollouts with T1_DONE of $(( $(wc -l < "$J") + 2 ))"
echo "$nd" > "$D/DISTRULE_DONE.txt"
