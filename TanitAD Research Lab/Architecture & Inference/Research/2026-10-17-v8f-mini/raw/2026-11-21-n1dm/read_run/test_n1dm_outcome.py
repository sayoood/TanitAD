"""test_n1dm_outcome.py -- pure-logic checks of n1dm_ho_read.outcome / zpost_block and n1dm_read141.guard_of (no data, no torch).  Run: python -I test_n1dm_outcome.py"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/eval141")
import n1dm_ho_read as R  # noqa: E402


def blk(g0, g1, c0=False, c1=False, w=(None, None)):
    row = lambda g, c: {"gain": g, "cost": c}                        # noqa: E731
    return {"D1_HO_deciding_N1DM-N1D": {"N1DM_s0": row(g0, c0), "N1DM_s1": row(g1, c1)}, "Z-POST": {"weak": {"s0": w[0], "s1": w[1]}}}


def fired(o, key):
    return [v for k, v in o["labels"].items() if k.startswith(key)][0]


ok = []
o = R.outcome(blk(True, True))
ok.append(("gain at both -> GAIN pending guard", fired(o, "N1DM-GAIN").startswith("PENDING-GUARD") and fired(o, "N1DM-ONE-SEED") == "not fired" and "not fired" in fired(o, "N1DM-POST-WEAK")))
o = R.outcome(blk(True, False))
ok.append(("gain at one -> ONE-SEED pending guard", fired(o, "N1DM-ONE-SEED").startswith("PENDING-GUARD") and fired(o, "N1DM-GAIN (").startswith("not fired")))
o = R.outcome(blk(False, False, True, True, (False, False)))
ok.append(("cost at both on D1-HO -> COST fired", fired(o, "N1DM-COST").startswith("FIRED")))
o = R.outcome(blk(False, False, False, False, (True, False)))
ok.append(("no gain, head < 0.685 at s0 -> POST-WEAK fired", fired(o, "N1DM-POST-WEAK").startswith("FIRED") and fired(o, "N1DM-NO-GAIN") == "not fired"))
o = R.outcome(blk(False, False, False, False, (False, False)))
ok.append(("no gain, head >= 0.685 both -> NO-GAIN pending guard", fired(o, "N1DM-NO-GAIN").startswith("PENDING-GUARD") and fired(o, "N1DM-POST-WEAK").startswith("not fired")))
o = R.outcome(blk(False, False))
ok.append(("no gain, Z-POST not joined -> PENDING-ZPOST", "PENDING-ZPOST" in fired(o, "N1DM-POST-WEAK") and "PENDING-ZPOST" in fired(o, "N1DM-NO-GAIN")))
# Z-POST join
with tempfile.TemporaryDirectory(dir=HERE + "/tmp") as d:
    p = d + "/z.json"
    json.dump({"status": "VALID", "arms": {"N1DM_s0": {"route_head_acc_T0": {"point": 0.70}}, "N1DM_s1": {"route_head_acc_T0": {"point": 0.66}}}}, open(p, "w"))
    zb = R.zpost_block(argparse.Namespace(zpost=p), {"N1DM_s0": {"point": 0.62}, "N1DM_s1": {"point": 0.63}})
    ok.append(("zpost join: s0 0.70 not weak, s1 0.66 weak", zb["weak"] == {"s0": False, "s1": True} and abs(zb["arms"]["N1DM_s1"]["vs_logistic_0.685"] + 0.025) < 1e-12))
import n1dm_read141 as N  # noqa: E402
G = N.guard_of
ok.append(("guard per seed", G({0: set(), 1: set()}, False) == "GUARD-HOLDS-BOTH" and G({0: {"a"}, 1: set()}, False) == "GUARD-COST-ONE-SEED" and G({0: {"a"}, 1: {"b"}}, False) == "GUARD-COST-BOTH-SEEDS"))
for name, good in ok:
    print(("ok   " if good else "FAIL ") + name)
sys.exit(0 if all(g for _, g in ok) else 1)
