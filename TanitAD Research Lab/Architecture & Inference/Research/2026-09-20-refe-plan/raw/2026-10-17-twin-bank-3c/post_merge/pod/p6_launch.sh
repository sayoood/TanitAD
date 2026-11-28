#!/bin/bash
# POD-SIDE, launched DETACHED by post_merge_to_p6gate.sh step (f), and ONLY after the launch-gate dry run passed AND the pod was found free (no GPU compute process, no rls1_pod.sh).
# This is P6's REAL run, "exactly as P6_POD_RUNBOOK says": `bash refe/p6_pod.sh` under the SAME environment as the dry run (gate_dryrun.sh), MINUS P6_STOP_AFTER.
# Every finished stage is skipped (done markers); the chain re-verifies the launch-gate token against the exact argv, code tree and data before the preflight and before every
# train.py launch, and REFUSES (no training) if it does not verify.  `exec`: the process that stays alive IS p6_pod.sh (it holds chain.lock), not a wrapper.
# The HF token is read from a file whose path is passed in the environment; it is never printed, never an argument, never written.
# Env in: P6PKG P6 OUT DD_GB HF_TOKEN_FILE P6N.
set -u
echo $$ > "$P6N/launch.PID"
cd "$P6PKG/refe" || exit 90
[ -s "$HF_TOKEN_FILE" ] || exit 91
export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
exec env -u P6_STOP_AFTER P6_D7_FALLBACK=0 P6_CBANK="$OUT" P6_YAW_WRAP=off P6_FREE_NAVLOGS=1 P6_OVERLAP=0 P6_DD_GB="$DD_GB" \
  bash "$P6PKG/refe/p6_pod.sh" >> "$P6/p6_stdout.log" 2>&1 < /dev/null
