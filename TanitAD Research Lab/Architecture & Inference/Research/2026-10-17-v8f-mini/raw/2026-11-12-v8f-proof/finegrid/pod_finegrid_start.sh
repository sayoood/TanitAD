#!/bin/bash
# pod_finegrid_start.sh -- card PREREG_V8F_PROOF_ADD41 (FINEGRID). Runs ON THE POD, started once by the lead (nohup setsid; the start line holds no process test).
# Why a wrapper: the launcher of the chains actcons_e2 and actcons_e3 (runs/actcons_z/pod_actcons_g_auto.sh) takes the slot that each RL base frees, with the
# same two-probe slot test as the pilot runner. Two tests that pass at the same moment would start a fourth GPU job. This wrapper holds the pilot runner back
# until the launcher has started both chains (their chain.log exists), or has written its not-started file, or 6 h have passed. Then it waits 180 s and hands
# over to pod_finegrid_run.sh, which is not changed (it has its own slot wait with a 10 h bound and never starts a fourth GPU job).
# A second start is refused by a marker file, not by a look at the process list.
set -u
W=/workspace/v8proof
PB=$W/finegrid
OUT=$W/runs/finegrid
mkdir -p "$OUT"
L=$OUT/finegrid_start.log
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] finegrid-start: $*" | tee -a "$L"; }
if [ -f "$OUT/FINEGRID_START_TAKEN.txt" ]; then say "REFUSED: a start is already recorded in $OUT/FINEGRID_START_TAKEN.txt"; exit 3; fi
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$OUT/FINEGRID_START_TAKEN.txt"
say "start recorded; holds the pilot runner back until the launcher has started actcons_e2 and actcons_e3"
n=0
while true; do
  if [ -f "$W/chains/actcons_e2/chain.log" ] && [ -f "$W/chains/actcons_e3/chain.log" ]; then say "both chains actcons_e2 and actcons_e3 are started"; break; fi
  if [ -f "$W/runs/actcons_z/ACTCONS_G_NOT_STARTED.txt" ]; then say "the launcher wrote its not-started file; going on"; break; fi
  n=$((n+1))
  if [ "$n" -gt 360 ]; then say "6 h bound reached; going on"; break; fi
  if [ $((n % 30)) = 1 ]; then say "waits (minute $n of 360)"; fi
  sleep 60
done
sleep 180
say "hands over to pod_finegrid_run.sh"
exec bash "$PB/pod_finegrid_run.sh"
