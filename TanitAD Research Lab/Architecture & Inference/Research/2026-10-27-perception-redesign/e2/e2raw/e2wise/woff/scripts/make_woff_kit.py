"""make_woff_kit.py -- derive the WIRE-OFF kit (PREREG_E2WISE_2026-11-17_ADDENDUM4.md, landed 81d409d9) from the PLAN-RESET kit, LF out, every
leftover PLAN-RESET token asserted absent; plus e2woff_weights.sh (Thor, CPU), auto_woff.sh (dev box chain) and after_wboth_then_woff.sh.
usage: python -I make_woff_kit.py <k2dc scratch dir>"""
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
NOTE = ("# DERIVED by k2dc/make_woff_kit.py from the PLAN-RESET script of the same role: WIRE-OFF = K2DC's tail with its 6 box_gates and the\n"
        "# map_chan_proj weight/bias at 0, K0's tail unchanged (model_wireoff.pt, wire_off.py), pre-registered in\n"
        "# e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM4.md (landed 81d409d9, before any WIRE-OFF number).\n")
PAIRS = [("model_wise_plan_a000", "model_wireoff"), ("wise_plan_a000", "wireoff"), ("E2WISE_PLAN0", "E2WISE_WOFF"), ("PLAN-RESET", "WIRE-OFF"),
         ("ZZE2PRSTA", "ZZE2WOFFA"), ("ZZE2PRSTB", "ZZE2WOFFB"), ("ZZPRSTW", "ZZWOFFW"), ("e2prst_", "e2woff_"), ("auto_prst", "auto_woff"),
         ("AUTOP-", "AUTOO-"), ("wait_e2prstB", "wait_e2woffB"), ("(alpha 0.5) weights", "weights"), ("(weight_interp.py a4faef50, 20ab9f54)", "(wire_off.py, ADDENDUM4 81d409d9)")]
LEFT = re.compile(r"a000|PLAN0|PLAN-RESET|prst|PRST|weight_interp|alpha")
for src, dst in (("e2prst_stageA.sh", "e2woff_stageA.sh"), ("e2prst_stageB_launch.sh", "e2woff_stageB_launch.sh"),
                 ("e2prst_s6c_pull_score.sh", "e2woff_s6c_pull_score.sh"), ("wait_e2prstB_then_s6c.sh", "wait_e2woffB_then_s6c.sh"),
                 ("auto_prst.sh", "auto_woff.sh")):
    s = (D / src).read_bytes().decode("utf-8")
    assert "\r" not in s and "woff" not in s and "WOFF" not in s, src
    lines = [ln for ln in s.split("\n") if not (ln.startswith("# DERIVED by k2dc/make_prst_kit.py")
                                                or ln.startswith("# pre-registered in e2/e2raw/PREREG_E2WISE_2026-11-17_ADDENDUM3.md"))]
    s = "\n".join(lines)
    for old, new in PAIRS:
        s = s.replace(old, new)
    if dst == "auto_woff.sh":
        old = "for f in e2woff_weights.sh e2woff_stageA.sh e2woff_stageB_launch.sh; do"
        assert s.count(old) == 1, "ship loop"
        s = s.replace(old, "for f in wire_off.py e2woff_weights.sh e2woff_stageA.sh e2woff_stageB_launch.sh; do")
    left = [ln for ln in s.split("\n") if LEFT.search(ln)]
    assert not left, (dst, left[:4])
    lines = s.split("\n")
    assert lines[0].startswith("#!"), src
    (D / dst).write_bytes("\n".join([lines[0], NOTE.rstrip("\n")] + lines[1:]).encode("utf-8"))
    print("WROTE", dst)

W = r'''#!/bin/bash
# e2woff_weights.sh -- MM 4a3d05, 2026-11-18 (Thor; CPU). model_wireoff.pt for K2DC and K0 (PREREG_E2WISE_2026-11-17_ADDENDUM4.md, 81d409d9):
# K2DC's tail with exactly 8 wire tensors at 0 (6 box_gates + map_chan_proj.proj.weight/.bias); K0 has none -> a byte copy of its tail.
# CONTROL (from the tool's re-load check + here): K2DC n_zeroed 8, K0 n_zeroed 0 and md5_out == md5 of its tail; every other tensor bit-equal.
# Idempotent: an existing output is re-verified, not re-made. Ends ZZWOFFW-DONEZZ or ZZWOFFW-FAIL-<why>ZZ.
set -u
R=/home/nvidia/r8ns/r9e2; PY=/home/nvidia/venvs/tanitad-train/bin/python; TOOL=$R/inputs/wire_off.py
cd $R || { echo "ZZWOFFW-FAIL-CDZZ"; exit 2; }
for A in K2DC K0; do
  case $A in K2DC) N=8;; K0) N=0;; esac
  T=runs/r9e2-$A/model_ema_tail5k6k.pt; O=runs/r9e2-$A/model_wireoff.pt
  [ -s $T ] || { echo "ZZWOFFW-FAIL-NOTAILZZ $T"; exit 3; }
  if [ -e $O ]; then echo "  $A: $O exists (made by an earlier run of this script): re-verifying, not re-making"
  else $PY -I $TOOL --run $T --out $O --expect $N > logs/e2woff_weights_$A.log 2>&1 || { echo "ZZWOFFW-FAIL-TOOL-$A""ZZ"; tail -3 logs/e2woff_weights_$A.log; exit 5; }; fi
  $PY -I -c '
import hashlib, json, sys
d = json.load(open(sys.argv[1] + ".provenance.json"))
m = hashlib.md5(open(sys.argv[1], "rb").read()).hexdigest()
t = hashlib.md5(open(sys.argv[2], "rb").read()).hexdigest()
n = int(sys.argv[3])
ok = d["n_zeroed"] == n and d["all_others_bit_equal"] is True and d["md5_out"] == m and d["md5_run"] == t and ((m == t) == (n == 0))
print("CONTROL-" + ("OK" if ok else "BAD"), "zeroed", d["n_zeroed"], "untouched", d["n_untouched_bit_equal"], "md5", m, "tail", t)
sys.exit(0 if ok else 1)' $O $T $N || { echo "ZZWOFFW-FAIL-CONTROL-$A""ZZ"; exit 6; }
done
echo "ZZWOFFW-DONEZZ $(date -u +%H:%M:%SZ)"
'''
(D / "e2woff_weights.sh").write_bytes(W.encode("utf-8"))

Q = r'''#!/bin/bash
# after_wboth_then_woff.sh -- MM 4a3d05, 2026-11-18 (dev box, run_in_background). Waits until WISE-BOTH's Thor roll has stopped (its stage B
# ROLL_STATUS shows AWAIT-S6C, or ROLL_ABORTED exists), so the Thor GPU is free, then execs auto_woff.sh (its exit codes). 16 h -> exit 8.
export PATH="/usr/bin:/bin:$PATH"
S=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc
OUT=/home/nvidia/r8ns/r9e2/runs/E2WISE_BOTH/stageB
end=$(( $(date +%s) + 57600 ))
while :; do
  st=$(ssh -n -o ConnectTimeout=20 -o BatchMode=yes tanitad-thor "if [ -e $OUT/ROLL_ABORTED ]; then echo ABORTED; elif grep -q AWAIT-S6C $OUT/ROLL_STATUS.txt 2>/dev/null; then echo AWAIT; else echo RUN; fi" 2>/dev/null | tr -d '\r')
  case "$st" in AWAIT|ABORTED) echo "AFTER-WBOTH $st $(date -u +%H:%M:%SZ): starting WIRE-OFF"; exec bash $S/auto_woff.sh < /dev/null;; esac
  [ "$(date +%s)" -ge "$end" ] && { echo "AFTER-WBOTH-TIMEOUT $(date -u +%H:%M:%SZ)"; exit 8; }
  sleep 300
done
'''
(D / "after_wboth_then_woff.sh").write_bytes(Q.encode("utf-8"))
print("WROTE e2woff_weights.sh after_wboth_then_woff.sh")
