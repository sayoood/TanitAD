#!/bin/bash
# DERIVED by k2dc/make_woff_kit.py from the PLAN-RESET script of the same role: WIRE-OFF = K2DC's tail with its 6 box_gates and the
# map_chan_proj weight/bias at 0, K0's tail unchanged (model_wireoff.pt, wire_off.py), pre-registered in
# e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM4.md (landed 81d409d9, before any WIRE-OFF number).
# auto_woff.sh -- MM 4a3d05, 2026-11-17 (dev box, run_in_background). WIRE-OFF end to end (ADDENDUM 4, 81d409d9) without a model turn:
#   ship 3 scripts (md5 both ends) -> e2woff_weights.sh (ZZWOFFW-DONEZZ) -> stage A (setsid) -> wait DONE (ABORT -> 7, 4 h -> 8)
#   -> LAUNCH_DRY stage B (rc 0, K2DC 6 + K0 4 sites on model_wireoff.pt, 10 total, model_ema 0; else 6) -> launch (else 5)
#   -> exec wait_e2woffB_then_s6c.sh (0 GO / 1 NO-GO / ...). Exits 4 on a ship or weights failure.
export PATH="/usr/bin:/bin:$PATH"
S=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc
R=/home/nvidia/r8ns/r9e2; ST=$R/runs/E2WISE_WOFF/stageA/STATUS.txt; L=$R/inputs/e2woff_stageB_launch.sh
th() { ssh -n -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "$1" 2>/dev/null | tr -d '\r'; }
for f in wire_off.py e2woff_weights.sh e2woff_stageA.sh e2woff_stageB_launch.sh; do
  th "test -e $R/inputs/$f && echo EXISTS" | grep -qx EXISTS && [ "$(th "md5sum < $R/inputs/$f | cut -c1-32")" != "$(md5sum < $S/$f | cut -c1-32)" ] && { echo "AUTOO-SHIP-CONFLICT $f differs on Thor"; exit 4; }
  scp -q -o BatchMode=yes $S/$f tanitad-thor:$R/inputs/$f < /dev/null
  [ "$(th "md5sum < $R/inputs/$f | cut -c1-32")" = "$(md5sum < $S/$f | cut -c1-32)" ] || { echo "AUTOO-SHIP-FAIL $f"; exit 4; }
done
echo "AUTOO-SHIPPED $(date -u +%H:%M:%SZ)"
Wt=$(th "bash $R/inputs/e2woff_weights.sh < /dev/null"); printf '%s\n' "$Wt" | sed 's/^/  weights: /'
printf '%s\n' "$Wt" | grep -q '^ZZWOFFW-DONEZZ' || { echo "AUTOO-WEIGHTS-FAIL"; exit 4; }
th "cd $R && setsid nohup bash inputs/e2woff_stageA.sh > $R/logs/e2woff_stageA.out 2>&1 < /dev/null & sleep 3; echo LAUNCHED" | grep -q LAUNCHED || { echo "AUTOO-STAGEA-LAUNCH-FAIL"; exit 4; }
echo "AUTOO-STAGEA-STARTED $(date -u +%H:%M:%SZ)"
end=$(( $(date +%s) + 14400 ))
while :; do
  sleep 240
  A=$(th "tail -n 40 $ST")
  if printf '%s\n' "$A" | grep -q -e ' ABORT-'; then echo "AUTOO-STAGEA-ABORT $(date -u +%H:%M:%SZ)"; printf '%s\n' "$A" | tail -n 5; exit 7; fi
  if printf '%s\n' "$A" | grep -q -e ' DONE stage A'; then break; fi
  [ "$(date +%s)" -ge "$end" ] && { echo "AUTOO-STAGEA-TIMEOUT $(date -u +%H:%M:%SZ)"; printf '%s\n' "$A" | tail -n 3; exit 8; }
done
echo "AUTOO-STAGEA-DONE $(date -u +%H:%M:%SZ)"; printf '%s\n' "$A" | grep -e 'EVAL-DONE' -e ' DONE '
D=$(th "LAUNCH_DRY=1 bash $L < /dev/null"); printf '%s\n' "$D" | sed 's/^/  dry: /'
DL=$(printf '%s\n' "$D" | grep '^ZZE2WOFFB-DRY ')
nK2=$(printf '%s\n' "$DL" | grep -o '[0-9]\+ --ckpt [^ ]*/r9e2-K2DC/model_wireoff[.]pt' | awk '{s+=$1} END {print s+0}')
nK0=$(printf '%s\n' "$DL" | grep -o '[0-9]\+ --ckpt [^ ]*/r9e2-K0/model_wireoff[.]pt' | awk '{s+=$1} END {print s+0}')
nAll=$(printf '%s\n' "$DL" | grep -o '[0-9]\+ --ckpt [^ ]*' | awk '{s+=$1} END {print s+0}')
nEma=$(printf '%s\n' "$DL" | sed -n 's/.* model_ema[.]pt \([0-9]*\).*/\1/p')
echo "AUTOO-DRY-CHECK K2DC $nK2 K0 $nK0 all $nAll model_ema $nEma"
if ! printf '%s\n' "$DL" | grep -q '^ZZE2WOFFB-DRY rc 0 ' || [ "$nK2" != 6 ] || [ "$nK0" != 4 ] || [ "$nAll" != 10 ] || [ "$nEma" != 0 ]; then
  echo "AUTOO-DRY-REFUSED $(date -u +%H:%M:%SZ): stage B NOT launched"; exit 6
fi
Lr=$(th "bash $L < /dev/null"); printf '%s\n' "$Lr" | sed 's/^/  launch: /'
printf '%s\n' "$Lr" | grep -q '^ZZE2WOFFB-STARTED ' || { echo "AUTOO-LAUNCH-FAILED $(date -u +%H:%M:%SZ)"; exit 5; }
echo "AUTOO-STAGEB-STARTED $(date -u +%H:%M:%SZ)"
exec bash $S/wait_e2woffB_then_s6c.sh < /dev/null
