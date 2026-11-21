"""gen_auto_wboth.py -- derive auto_wboth.sh (WISE-BOTH end to end, PREREG_E2WISE ADDENDUM 2, 8153ee7d; ordered after PLAN-RESET by ADDENDUM 3)
from auto_prst.sh, every substitution count-asserted, plus after_prst_then_wboth.sh: waits until PLAN-RESET's Thor roll stops (AWAIT-S6C or
ROLL_ABORTED; Thor GPU then free, its S6c runs on the dev box), then runs auto_wboth.sh. usage: python -I gen_auto_wboth.py <k2dc dir>"""
import sys
from pathlib import Path

D = Path(sys.argv[1])
s = (D / "auto_prst.sh").read_bytes().decode("utf-8")
assert "\r" not in s and "wboth" not in s
PAIRS = [
    ("# auto_prst.sh -- MM 4a3d05, 2026-11-17 (dev box, run_in_background). PLAN-RESET end to end (ADDENDUM 3, 93688696) without a model turn:",
     "# auto_wboth.sh -- MM 4a3d05, 2026-11-17 (dev box). WISE-BOTH end to end (ADDENDUM 2, 8153ee7d; after PLAN-RESET per ADDENDUM 3). Derived from\n"
     "# auto_prst.sh by k2dc/gen_auto_wboth.py:", 1),
    ("e2prst_weights.sh", "e2wboth_weights.sh", None), ("e2prst_stageA.sh", "e2wboth_stageA.sh", None),
    ("e2prst_stageB_launch.sh", "e2wboth_stageB_launch.sh", None), ("wait_e2prstB_then_s6c.sh", "wait_e2wbothB_then_s6c.sh", None),
    ("e2prst_stageA.out", "e2wboth_stageA.out", None), ("E2WISE_PLAN0", "E2WISE_BOTH", None),
    ("model_wise_plan_a000", "model_wise_both_a050", None), ("ZZPRSTW-DONEZZ", "ZZWBOTHW-DONEZZ", None),
    ("ZZE2PRSTB-", "ZZE2WBOTHB-", None), ("AUTOP-", "AUTOW-", None),
]
for old, new, n in PAIRS:
    c = s.count(old)
    assert c >= 1 and (n is None or c == n), (old, c)
    s = s.replace(old, new)
for bad in ("prst", "PRST", "PLAN0", "a000", "AUTOP"):
    chk = s.replace("# auto_prst.sh by k2dc/gen_auto_wboth.py:", "")
    assert bad not in chk, (bad, [ln for ln in chk.split("\n") if bad in ln][:2])
(D / "auto_wboth.sh").write_bytes(s.encode("utf-8"))
g = r'''#!/bin/bash
# after_prst_then_wboth.sh -- MM 4a3d05, 2026-11-17 (dev box, run_in_background). Waits until PLAN-RESET's Thor roll has stopped (its stage B
# ROLL_STATUS shows AWAIT-S6C, or ROLL_ABORTED exists), so the Thor GPU is free, then execs auto_wboth.sh (its exit codes). 14 h -> exit 8.
export PATH="/usr/bin:/bin:$PATH"
S=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc
OUT=/home/nvidia/r8ns/r9e2/runs/E2WISE_PLAN0/stageB
end=$(( $(date +%s) + 50400 ))
while :; do
  st=$(ssh -n -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "if [ -e $OUT/ROLL_ABORTED ]; then echo ABORTED; elif grep -q AWAIT-S6C $OUT/ROLL_STATUS.txt 2>/dev/null; then echo AWAIT; else echo RUN; fi" 2>/dev/null | tr -d '\r')
  case "$st" in AWAIT|ABORTED) echo "AFTER-PRST $st $(date -u +%H:%M:%SZ): starting WISE-BOTH"; exec bash $S/auto_wboth.sh < /dev/null;; esac
  [ "$(date +%s)" -ge "$end" ] && { echo "AFTER-PRST-TIMEOUT $(date -u +%H:%M:%SZ)"; exit 8; }
  sleep 300
done
'''
(D / "after_prst_then_wboth.sh").write_bytes(g.encode("utf-8"))
print("WROTE auto_wboth.sh after_prst_then_wboth.sh")
