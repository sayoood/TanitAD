#!/usr/bin/env bash
# RPT (addendum 4): re-arm wake_rnext.sh after each new RUN_*.done until the chain ends, then pull. Script polling only.
# Box 2 held 8 RUN_*.done before RPT (HANDOFF_RPT.md step 2). Exit codes: 0 chain done and pull rc 0; 3 run failed; 4 safety stop;
# 5 waker timeout; 6 more dones than the chain can make; 7 pull failed.
cd /e/Projects/TanitAD-artifacts/dagland/recover_next || exit 9
s=8
while :; do
  out=$(RNEXT_SEEN=$s bash wake_rnext.sh); ec=$?
  echo "[$(date '+%m-%d %H:%M:%S')] seen=$s ec=$ec $out"
  case "$out" in *"CHAIN DONE"*) break ;; esac
  [ "$ec" -ne 0 ] && exit "$ec"
  s=$((s + 1)); [ "$s" -gt 10 ] && { echo "more RUN_*.done than the chain makes"; exit 6; }
done
echo "[$(date '+%m-%d %H:%M:%S')] pulling"
bash pull_rnext.sh > pull_rpt.out 2>&1; pec=$?
tail -5 pull_rpt.out
echo "[$(date '+%m-%d %H:%M:%S')] pull rc=$pec"
[ "$pec" -eq 0 ] || exit 7
