#!/bin/bash
# auto_prst.sh -- MM 4a3d05, 2026-11-17 (dev box, run_in_background). PLAN-RESET end to end (ADDENDUM 3, 93688696) without a model turn:
#   ship 3 scripts (md5 both ends) -> e2prst_weights.sh (ZZPRSTW-DONEZZ) -> stage A (setsid) -> wait DONE (ABORT -> 7, 4 h -> 8)
#   -> LAUNCH_DRY stage B (rc 0, K2DC 6 + K0 4 sites on model_wise_plan_a000.pt, 10 total, model_ema 0; else 6) -> launch (else 5)
#   -> exec wait_e2prstB_then_s6c.sh (0 GO / 1 NO-GO / ...). Exits 4 on a ship or weights failure.
export PATH="/usr/bin:/bin:$PATH"
S=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc
R=/home/nvidia/r8ns/r9e2; ST=$R/runs/E2WISE_PLAN0/stageA/STATUS.txt; L=$R/inputs/e2prst_stageB_launch.sh
th() { ssh -n -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "$1" 2>/dev/null | tr -d '\r'; }
for f in e2prst_weights.sh e2prst_stageA.sh e2prst_stageB_launch.sh; do
  th "test -e $R/inputs/$f && echo EXISTS" | grep -qx EXISTS && [ "$(th "md5sum < $R/inputs/$f | cut -c1-32")" != "$(md5sum < $S/$f | cut -c1-32)" ] && { echo "AUTOP-SHIP-CONFLICT $f differs on Thor"; exit 4; }
  scp -q -o BatchMode=yes $S/$f tanitad-thor:$R/inputs/$f < /dev/null
  [ "$(th "md5sum < $R/inputs/$f | cut -c1-32")" = "$(md5sum < $S/$f | cut -c1-32)" ] || { echo "AUTOP-SHIP-FAIL $f"; exit 4; }
done
echo "AUTOP-SHIPPED $(date -u +%H:%M:%SZ)"
Wt=$(th "bash $R/inputs/e2prst_weights.sh < /dev/null"); printf '%s\n' "$Wt" | sed 's/^/  weights: /'
printf '%s\n' "$Wt" | grep -q '^ZZPRSTW-DONEZZ' || { echo "AUTOP-WEIGHTS-FAIL"; exit 4; }
th "cd $R && setsid nohup bash inputs/e2prst_stageA.sh > $R/logs/e2prst_stageA.out 2>&1 < /dev/null & sleep 3; echo LAUNCHED" | grep -q LAUNCHED || { echo "AUTOP-STAGEA-LAUNCH-FAIL"; exit 4; }
echo "AUTOP-STAGEA-STARTED $(date -u +%H:%M:%SZ)"
end=$(( $(date +%s) + 14400 ))
while :; do
  sleep 240
  A=$(th "tail -n 40 $ST")
  if printf '%s\n' "$A" | grep -q -e ' ABORT-'; then echo "AUTOP-STAGEA-ABORT $(date -u +%H:%M:%SZ)"; printf '%s\n' "$A" | tail -n 5; exit 7; fi
  if printf '%s\n' "$A" | grep -q -e ' DONE stage A'; then break; fi
  [ "$(date +%s)" -ge "$end" ] && { echo "AUTOP-STAGEA-TIMEOUT $(date -u +%H:%M:%SZ)"; printf '%s\n' "$A" | tail -n 3; exit 8; }
done
echo "AUTOP-STAGEA-DONE $(date -u +%H:%M:%SZ)"; printf '%s\n' "$A" | grep -e 'EVAL-DONE' -e ' DONE '
D=$(th "LAUNCH_DRY=1 bash $L < /dev/null"); printf '%s\n' "$D" | sed 's/^/  dry: /'
DL=$(printf '%s\n' "$D" | grep '^ZZE2PRSTB-DRY ')
nK2=$(printf '%s\n' "$DL" | grep -o '[0-9]\+ --ckpt [^ ]*/r9e2-K2DC/model_wise_plan_a000[.]pt' | awk '{s+=$1} END {print s+0}')
nK0=$(printf '%s\n' "$DL" | grep -o '[0-9]\+ --ckpt [^ ]*/r9e2-K0/model_wise_plan_a000[.]pt' | awk '{s+=$1} END {print s+0}')
nAll=$(printf '%s\n' "$DL" | grep -o '[0-9]\+ --ckpt [^ ]*' | awk '{s+=$1} END {print s+0}')
nEma=$(printf '%s\n' "$DL" | sed -n 's/.* model_ema[.]pt \([0-9]*\).*/\1/p')
echo "AUTOP-DRY-CHECK K2DC $nK2 K0 $nK0 all $nAll model_ema $nEma"
if ! printf '%s\n' "$DL" | grep -q '^ZZE2PRSTB-DRY rc 0 ' || [ "$nK2" != 6 ] || [ "$nK0" != 4 ] || [ "$nAll" != 10 ] || [ "$nEma" != 0 ]; then
  echo "AUTOP-DRY-REFUSED $(date -u +%H:%M:%SZ): stage B NOT launched"; exit 6
fi
Lr=$(th "bash $L < /dev/null"); printf '%s\n' "$Lr" | sed 's/^/  launch: /'
printf '%s\n' "$Lr" | grep -q '^ZZE2PRSTB-STARTED ' || { echo "AUTOP-LAUNCH-FAILED $(date -u +%H:%M:%SZ)"; exit 5; }
echo "AUTOP-STAGEB-STARTED $(date -u +%H:%M:%SZ)"
exec bash $S/wait_e2prstB_then_s6c.sh < /dev/null
