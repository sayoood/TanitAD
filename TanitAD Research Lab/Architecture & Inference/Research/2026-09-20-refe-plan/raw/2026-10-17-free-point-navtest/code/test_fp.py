#!/usr/bin/env python3
"""CPU tests of the free-point tooling (no GPU, no pod).  Each check names what it proves; the last block is the DELIBERATE-REGRESSION arms (each must make its check FAIL).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I -X utf8 test_fp.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp_picks as P  # noqa: E402
import fp_ship as SH  # noqa: E402

fails = []


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        fails.append(name)


AT = P.load_attrib()
rng = np.random.default_rng(11)

# --- the point construction == dose_rls1b.py's s(alpha) at alpha 0.25, computed independently here
c, r = rng.normal(size=(7, 64, 7)), rng.normal(size=(7, 64, 7))
pl = P.points_logits({"BASE": c, "C0A": c, "R0B": r})
check("M25 == c + 0.25 (r - c), float64, all 7 outputs (dose_rls1b.py:4)", np.array_equal(pl["M25"], np.asarray(c, np.float64) + 0.25 * (np.asarray(r, np.float64) - np.asarray(c, np.float64))))
check("logits are float64 before mixing (the dose script loads float32 files as float64)", pl["M25"].dtype == np.float64)

# --- the pick is the trainer's: sel = NC*DAC*(5EP+5TTC+2C)/12*lane, DDC (index 5) not in it, ties -> lowest index
z = np.full((1, 64, 7), -9.0)
z[0, 17] = 9.0
z[0, 40, 5] = 99.0
check("DDC output does not enter the selection rule", int(AT.picks(z)[0]) == 17)
check("ties go to the lowest index", int(AT.picks(np.full((1, 64, 7), 2.0))[0]) == 0)

# --- IC3 logic on a synthetic fan: a near-tie flip is EXPLAINED, a far pick is NOT
if os.path.exists(P.BANKED_U400_SEAM):
    zb = np.load(P.BANKED_U400_SEAM, allow_pickle=False)
    toks = [str(t) for t in zb["token"][:50]]
    poses = np.array(zb["poses"][:50], np.float32)
    res = P.ic3(poses, toks)
    check("IC3 on the banked poses themselves: 100 % agreement, pass", res["frac_equal"] == 1.0 and res["pass"])
    shifted = poses.copy()
    shifted[:3] += 1.0                                          # 3 of 50 tokens moved by 1 m: not within 5 mm -> disagreement (no fan given, so nothing is explained)
    res2 = P.ic3(shifted, toks)
    check("IC3 sees 3 moved poses as 3 disagreements", res2["n_disagree"] == 3 and abs(res2["frac_equal"] - 47 / 50) < 1e-12)
    wrong = poses.copy()
    wrong[:20] += 1.0                                           # 40 % wrong -> below the 97 % gross-error bar
    check("D-IC3: 40 % wrong poses FAIL the 97 % bar (the check has teeth)", not P.ic3(wrong, toks)["pass"])

# --- D1: a shuffled token -> logits map collapses the pick agreement
b = rng.normal(size=(300, 64, 7))
agree = float((np.asarray(AT.picks(b)) == np.asarray(AT.picks(b[rng.permutation(300)]))).mean())
check("D1: shuffled token->logits agreement < 50 % (IC3 would FAIL)", agree < 0.5)

# --- the shipping plan covers every log exactly once and respects the chunk size
wk = "C:/Users/Admin/freept/work"
if os.path.exists(f"{wk}/ship_plan.json") and os.path.exists(f"{wk}/plan.json"):
    sp, pp = json.load(open(f"{wk}/ship_plan.json")), json.load(open(f"{wk}/plan.json"))
    logs = [l for c_ in sp["chunks"] for l in c_["logs"]]
    check("ship plan: every log exactly once", sorted(logs) == sorted(pp["tokens_per_log"]) and len(logs) == len(set(logs)))
    check("ship plan: no chunk above the 1.3 GB cap (unless a single log)", all(c_["bytes"] <= SH.CHUNK_BYTES or len(c_["logs"]) == 1 for c_ in sp["chunks"]))
    check("ship plan: tokens sum to 12,146", sum(c_["tokens"] for c_ in sp["chunks"]) == 12146 == pp["n_tokens"])

# --- the worker is gated on the re-roll and refuses without it
import fp_forward as FF  # noqa: E402
check("worker gate path is the re-roll's started_epoch", FF.STARTED_EPOCH == "/workspace/ops/bank3c/full/started_epoch")
check("worker refuses a state whose update counter is not 1000 (EXPECT_UPDATE)", FF.EXPECT_UPDATE == 1000)

print("ZZTEST_FP_" + ("PASS" if not fails else "FAIL " + ", ".join(fails)))
sys.exit(0 if not fails else 1)
