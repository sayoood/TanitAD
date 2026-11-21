"""make_prst_kit.py -- derive the PLAN-RESET kit (PREREG_E2WISE_2026-11-17_ADDENDUM3.md, landed 93688696) from the WISE-PLAN kit, LF out,
every substitution count-asserted; plus e2prst_weights.sh (Thor, CPU) and auto_prst.sh (dev box chain).
usage: python -I make_prst_kit.py <k2dc scratch dir>"""
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
NOTE = ("# DERIVED by k2dc/make_prst_kit.py from the WISE-PLAN script of the same role: PLAN-RESET = S_plan at alpha 0 (model_wise_plan_a000.pt),\n"
        "# pre-registered in e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM3.md (landed 93688696, before any PLAN-RESET number).\n")
PAIRS = [("wise_plan_a050", "wise_plan_a000"), ("WISE-PLAN", "PLAN-RESET"), ("ZZE2WISEA", "ZZE2PRSTA"), ("ZZE2WISEB", "ZZE2PRSTB"),
         ("e2wise_stageA", "e2prst_stageA"), ("e2wise_stageB", "e2prst_stageB"), ("e2wise_plan_stageB", "e2prst_stageB"),
         ("e2wise_s6c_pull_score", "e2prst_s6c_pull_score")]
for src, dst in (("e2wise_stageA.sh", "e2prst_stageA.sh"), ("e2wise_stageB_launch.sh", "e2prst_stageB_launch.sh"),
                 ("e2wise_s6c_pull_score.sh", "e2prst_s6c_pull_score.sh"), ("wait_e2wiseB_then_s6c.sh", "wait_e2prstB_then_s6c.sh")):
    s = (D / src).read_bytes().decode("utf-8")
    assert "\r" not in s and "prst" not in s and "PLAN0" not in s and "a000" not in s, src
    s = re.sub(r"E2WISE_PLAN(?!0)", "E2WISE_PLAN0", s)
    for old, new in PAIRS:
        s = s.replace(old, new)
    left = [ln for ln in s.split("\n") if re.search(r"a050|WISE-PLAN|ZZE2WISE|e2wise_", ln)]
    assert not left, (src, left[:3])
    lines = s.split("\n")
    assert lines[0].startswith("#!"), src
    (D / dst).write_bytes("\n".join([lines[0], NOTE.rstrip("\n")] + lines[1:]).encode("utf-8"))
    print("WROTE", dst)

W = r'''#!/bin/bash
# e2prst_weights.sh -- MM 4a3d05, 2026-11-17 (Thor; CPU). model_wise_plan_a000.pt for K2DC and K0: S_plan (the 20 prefixes recorded in WISE-PLAN's
# provenance) at alpha 0. PREREG_E2WISE_2026-11-17_ADDENDUM3.md (93688696). CONTROL: counts equal to the SAME arm's WISE-PLAN provenance
# (K2DC 324 interpolated / 6 copied; K0 324 / 0, it has no decoder box_gate), outside-S bit-equal, every interpolated tensor torch.equal to
# the init's. Idempotent: an existing output is re-verified, not re-made. Ends ZZPRSTW-DONEZZ or ZZPRSTW-FAIL-<why>ZZ.
set -u
R=/home/nvidia/r8ns/r9e2; PY=/home/nvidia/venvs/tanitad-train/bin/python; INIT=/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt
TOOL=$R/inputs/weight_interp.py_a4faef50
cd $R || { echo "ZZPRSTW-FAIL-CDZZ"; exit 2; }
for A in K2DC K0; do
  P=runs/r9e2-$A/model_wise_plan_a050.pt.provenance.json; O=runs/r9e2-$A/model_wise_plan_a000.pt
  [ -s $P ] || { echo "ZZPRSTW-FAIL-NOPLANPROVZZ $P"; exit 3; }
  PFX=$($PY -c 'import json,sys; p=json.load(open(sys.argv[1]))["prefixes"]; assert len(p)==20, p; print(",".join(p))' $P) || { echo "ZZPRSTW-FAIL-PREFIXZZ"; exit 4; }
  if [ -e $O ]; then echo "  $A: $O exists (made by an earlier run of this script): re-verifying, not re-making"
  else $PY $TOOL --run runs/r9e2-$A/model_ema_tail5k6k.pt --init $INIT --alpha 0 --prefixes "$PFX" --out $O > logs/e2prst_weights_$A.log 2>&1 || { echo "ZZPRSTW-FAIL-INTERP-$A""ZZ"; tail -3 logs/e2prst_weights_$A.log; exit 5; }; fi
  $PY -c '
import json, sys, torch
d = json.load(open(sys.argv[1] + ".provenance.json"))
ref = json.load(open(sys.argv[3]))   # WISE-PLAN provenance of the SAME arm: counts must match it (K2DC 324/6, K0 324/0 -- K0 has no box_gate)
ok = (d["n_interpolated"] == ref["n_interpolated"] == 324 and d["n_copied_in_S"] == ref["n_copied_in_S"] and d["n_untouched"] == ref["n_untouched"]
      and d["prefixes"] == ref["prefixes"] and d["perception_untouched_all_bit_equal"] is True and d["alpha"] == 0)
def sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    best = None
    for v in ([ck] + [x for x in ck.values() if isinstance(x, dict)]):
        n = sum(1 for t in v.values() if torch.is_tensor(t))
        if best is None or n > best[0]:
            best = (n, v)
    return best[1]
o, i = sd(sys.argv[1]), sd(sys.argv[2])
eq = sum(1 for k in d["interpolated"] if torch.equal(o[k], i[k]))
ok = ok and eq == len(d["interpolated"]) == 324
print("CONTROL-" + ("OK" if ok else "BAD"), d["n_interpolated"], d["n_copied_in_S"], d["perception_untouched_all_bit_equal"], "init-equal", eq, d["md5_out"])
sys.exit(0 if ok else 1)' $O $INIT $P || { echo "ZZPRSTW-FAIL-CONTROL-$A""ZZ"; exit 6; }
done
echo "ZZPRSTW-DONEZZ $(date -u +%H:%M:%SZ)"
'''
(D / "e2prst_weights.sh").write_bytes(W.encode("utf-8"))

C = r'''#!/bin/bash
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
'''
(D / "auto_prst.sh").write_bytes(C.encode("utf-8"))
print("WROTE e2prst_weights.sh auto_prst.sh")
