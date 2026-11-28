#!/usr/bin/env bash
# pod_fracchain_v8proof.sh -- card PREREG_V8F_PROOF_ADD64 on the v8proof pod: QUARTER starts when the folder of card ADD60 (half1) has its end marker, EIGHTH when
# QUARTER has ended; one probe of this fork at a time beside the rollout queue. If card ADD60 or QUARTER ended with a stop marker, the run after it is not started.
# Start: cd /workspace/v8proof/dscale_lead_frac && nohup setsid bash pod_fracchain_v8proof.sh > fracchain_stdout.txt 2>&1 < /dev/null &
# End marker: /workspace/v8proof/dscale_lead_frac/FRACCHAIN_DONE.txt. Nothing is deleted.
set -u
F=/workspace/v8proof/dscale_lead_frac; H1=/workspace/v8proof/dscale_lead/half1
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] fracchain: $*" >> "$F/fracchain.log"; }
cd "$F" || exit 3
[ -e "$F/FRACCHAIN_STARTED.txt" ] && { echo "REFUSED: the chain was started before"; exit 3; }
ts > "$F/FRACCHAIN_STARTED.txt"
log "waiting for the end marker of card ADD60"
while [ ! -e "$H1/HALFSCALE_DONE.txt" ] && [ ! -e "$H1/HALFSCALE_STOP.txt" ]; do sleep 30; done
if [ -e "$H1/HALFSCALE_STOP.txt" ]; then
  log "card ADD60 ended with a stop marker: QUARTER and EIGHTH are not started"
else
  log "card ADD60 ended: QUARTER starts"
  FRAC_DIR=quarter1 FRAC_DEN=4 FRAC_N=1143 FRAC_SHA=0915ce4493c38e1efbe9fbb4e2620841735a6cc6e76880111cd3ad32b9952971 bash "$F/pod_fracscale_v8proof.sh" > "$F/quarter1_stdout.txt" 2>&1
  log "QUARTER launcher rc $?"
  if [ -e "$F/quarter1/FRACSCALE_DONE.txt" ]; then
    log "QUARTER ended: EIGHTH starts"
    FRAC_DIR=eighth1 FRAC_DEN=8 FRAC_N=571 FRAC_SHA=b1270c6698eeb427961e6358d8f77c7d500a4506a899f44c18126094eff6671c bash "$F/pod_fracscale_v8proof.sh" > "$F/eighth1_stdout.txt" 2>&1
    log "EIGHTH launcher rc $?"
  else
    log "QUARTER has no done marker: EIGHTH is not started"
  fi
fi
ts > "$F/FRACCHAIN_DONE.txt"
log "FRACCHAIN_DONE"
