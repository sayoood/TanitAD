#!/bin/bash
# DERIVED by k2dc/make_woff_kit.py from the PLAN-RESET script of the same role: WIRE-OFF = K2DC's tail with its 6 box_gates and the
# map_chan_proj weight/bias at 0, K0's tail unchanged (model_wireoff.pt, wire_off.py), pre-registered in
# e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM4.md (landed 81d409d9, before any WIRE-OFF number).
# wait_e2rawB_then_s6c.sh -- MM 4a3d05, 2026-11-16: waits for E2-WIRE-OFF stage B's Thor roll to stop at AWAIT-S6C (then runs the dev-box half,
# e2woff_s6c_pull_score.sh, and exits with its rc: 0 GO / 1 NO-GO / 2 REFUSED / 3 abort / 4 refused / 6 wrapper refused), or to abort
# (ROLL_ABORTED file -> exit 7), or 9 h pass (exit 8). Also exits 9 if the Thor roll process vanished without either marker.
export PATH="/usr/bin:/bin:$PATH"
OUT=/home/nvidia/r8ns/r9e2/runs/E2WISE_WOFF/stageB
S=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc
end=$(( $(date +%s) + 32400 )); gone=0
while [ "$(date +%s)" -lt "$end" ]; do
  ST=$(ssh -n -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "if [ -e $OUT/ROLL_ABORTED ]; then echo ABORTED; cat $OUT/ROLL_ABORTED; elif grep -q 'AWAIT-S6C' $OUT/ROLL_STATUS.txt; then echo AWAIT; else echo RUN \$(pgrep -fc 'r9e2_roll[.]sh'); tail -n 1 $OUT/ROLL_STATUS.txt; fi" 2>/dev/null | tr -d '\r')
  case "$ST" in
    AWAIT*) echo "STAGEB-AWAIT-S6C $(date -u +%H:%M:%SZ): running the dev-box S6c + verdict"; bash $S/e2woff_s6c_pull_score.sh < /dev/null; rc=$?; echo "STAGEB-S6C-END rc $rc"; exit $rc;;
    ABORTED*) echo "STAGEB-ROLL-ABORTED $(date -u +%H:%M:%SZ)"; printf '%s\n' "$ST"; exit 7;;
    "RUN 0"*) gone=$((gone + 1)); [ $gone -ge 2 ] && { echo "STAGEB-ROLL-GONE $(date -u +%H:%M:%SZ) (no r9e2_roll.sh process, no marker)"; printf '%s\n' "$ST"; exit 9; };;
    RUN*) gone=0;;
  esac
  sleep 300
done
echo "STAGEB-WAIT-TIMEOUT $(date -u +%H:%M:%SZ)"; printf '%s\n' "$ST"; exit 8
