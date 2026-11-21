#!/bin/bash
# POD-SIDE, launched DETACHED by post_merge_to_p6gate.sh.   Step (e): P6's launch-gate DRY RUN = the runbook's `P6_STOP_AFTER=launch_gate` run of refe/p6_pod.sh:
# every stage up to and including the launch gate (lanes -> lab -> [wait: GPU idle] -> mem -> egr -> launch gate), a PASS token minted and VERIFIED, then the chain STOPS
# BEFORE `train.py --preflight` and before training.  It does NOT launch P6.
#   P6_D7_FALLBACK=0   explicit: the PI's 2026-10-16 ruling is "faithful, not the fallback"; the pod's p6_start.env sets =1 and must not leak in (this script does not source it)
#   P6_CBANK=$OUT      the 3-C bank (needs the knob in p6_pod.sh; the orchestrator refuses to get here without it)
#   P6_OVERLAP=0       never run P6's CPU stages beside another trainer (the orchestrator has already required an idle GPU)
# The HF token is read from a file whose path is passed in the environment; it is never printed, never an argument, never written.
# Env in: P6PKG P6 OUT DD_GB HF_TOKEN_FILE P6N.
set -u
echo $$ > "$P6N/gate.PID"
rm -f "$P6N/gate_dryrun.rc"
cd "$P6PKG/refe" || { echo 90 > "$P6N/gate_dryrun.rc"; exit 90; }
[ -s "$HF_TOKEN_FILE" ] || { echo 91 > "$P6N/gate_dryrun.rc"; exit 91; }
HF_TOKEN="$(cat "$HF_TOKEN_FILE")" P6_STOP_AFTER=launch_gate P6_D7_FALLBACK=0 P6_CBANK="$OUT" P6_YAW_WRAP=off P6_FREE_NAVLOGS=1 P6_OVERLAP=0 P6_DD_GB="$DD_GB" \
  bash "$P6PKG/refe/p6_pod.sh" >> "$P6/p6_stdout.log" 2>&1
echo $? > "$P6N/gate_dryrun.rc"
