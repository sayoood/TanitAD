#!/usr/bin/env python3
"""RL-1 step 3: the RL-val read of the head-wise composites.  Pre-registered in out/raw/PREREG_RL1_B_COMPOSITES.md (sha256 beside it).
    python composites.py --out out/raw/composites_rlval.json
"""
import argparse, json, os
import numpy as np
import rlh
from rlh import pick, compose

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--dl", default=rlh.DL)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
kv, ck, s, logs, st = rlh.load_labels()
N = len(kv)
ar = np.arange(N)
BT = rlh.Boot(logs)
Q = ("pdms", "nc", "dac", "ep", "ttc", "c", "ddc")
NAME = {0: "NC", 1: "DAC", 2: "EP", 3: "TTC", 4: "C", 5: "DDC", 6: "lane"}


def per_set(ip):
    return {q: 100.0 * s[q][ar, ip] for q in Q}


def delta(ia, ib, qs=Q):
    """picks ia minus picks ib -> {q: (mean, lo, hi)} x100"""
    return {q: list(BT.ci(100.0 * (s[q][ar, ia] - s[q][ar, ib]))) for q in qs}


def label(d):
    ok_gain = d["pdms"][1] > 0
    adv = [c for c in ("nc", "dac", "ttc") if d[c][2] < 0]
    if ok_gain and not adv:
        return "PASS-VAL", adv
    if (not ok_gain) and not adv:
        return "GAIN-LOST-VAL", adv
    if ok_gain and adv:
        return "SAFETY-LOST-VAL", adv
    return "BOTH-LOST-VAL", adv


def deficit(d):
    return max(0.0, -d["pdms"][1]) + sum(max(0.0, -d[c][2]) for c in ("nc", "dac", "ttc"))


def ld(arm, u, ab):
    root = DLD if arm == "r0d" else a.dl
    return rlh.load_dump(f"{root}/{arm}/final/scores_update{u}_{ab}.npz", kv, ck)


base0 = rlh.load_dump(f"{a.dl}/r0/final/scores_step0_base.npz", kv, ck)
ib0 = pick(base0)
res = {"tool": "composites/1", "prereg_B_sha256": open("C:/Users/Admin/par18/rl/out/raw/PREREG_RL1_B_COMPOSITES.sha256").read().split()[0],
       "prereg_A_sha256": open("C:/Users/Admin/par18/rl/out/raw/PREREG_RL1_A_MECHANISM.sha256").read().split()[0], "controls": {}, "states": {}}

STATES = [("r0", 1000), ("r0d", 250), ("r0d", 500), ("r0d", 750), ("r0d", 1000)]
DLD = "C:/Users/Admin/par18/rl/r0d_dl"
for arm, u in STATES:
    B, Aown, C0A = ld(arm, u, "B"), ld(arm, u, "A"), ld("c0", u, "A")
    iB, iAo, iC0 = pick(B), pick(Aown), pick(C0A)
    comp = {"V1": compose(B, base0, (2, 4)), "V2": compose(B, C0A, (2, 4))}
    blk = {"B_itself": {"vs_base0": delta(iB, ib0), "vs_ownA": delta(iB, iAo)},
           "A_own_itself": {"vs_base0": delta(iAo, ib0)}}
    for v, x in comp.items():
        ip = pick(x)
        d0 = delta(ip, ib0)
        lab, adv = label(d0)
        blk[v] = {"vs_base0": d0, "vs_ownA": delta(ip, iAo), "vs_B": delta(ip, iB), "label": lab, "adverse_components": adv, "deficit": deficit(d0),
                  "picks_changed_vs_base0": int((ip != ib0).sum()), "picks_changed_vs_B": int((ip != iB).sum()),
                  "barS_vs_ownA_ttc_nc_ge_-0.5": bool(delta(ip, iAo, ("ttc", "nc"))["ttc"][0] >= -0.5 and delta(ip, iAo, ("ttc", "nc"))["nc"][0] >= -0.5)}
        eB = blk["B_itself"]["vs_base0"]["ep"][0]
        blk[v]["EP_retention_vs_B"] = (d0["ep"][0] / eB) if eB != 0 else None
    # descriptive rows
    if (arm, u) in (("r0", 1000), ("r0d", 1000)):
        D = {"D1_H1shape_base0": compose(B, base0, (2, 4, 6)), "D2_lane_from_base0": compose(base0, B, (6,)), "D2_lane_from_A": compose(Aown, B, (6,)),
             "D4_EP_only_base0": compose(B, base0, (2,))}
        for h in range(7):
            D[f"D3_B_with_{NAME[h]}_from_base0"] = compose(base0, B, (h,))
            D[f"D3_base0_with_{NAME[h]}_from_B"] = compose(B, base0, (h,))
        blk["descriptive"] = {k: {"vs_base0": delta(pick(x), ib0), "vs_ownA": delta(pick(x), iAo, ("pdms", "ttc", "nc", "ep")), "picks_changed_vs_B": int((pick(x) != iB).sum())} for k, x in D.items()}
    res["states"][f"{arm}_u{u}"] = blk
    print("state", arm, u, {v: (blk[v]["label"], round(blk[v]["deficit"], 3)) for v in ("V1", "V2")}, flush=True)

# ---------------- controls (RL-2 extension: the git-show machinery check of the original is replaced by a REPRODUCTION of RL-1's banked composites_rlval.json)
C = res["controls"]
for arm in ("r0", "r0d"):
    B, Aown, C0A = ld(arm, 1000, "B"), ld(arm, 1000, "A"), ld("c0", 1000, "A")
    iB = pick(B)
    C[f"{arm}_u1000"] = {}
    k = C[f"{arm}_u1000"]
    k["all_RL_equals_B"] = {"V1_allheads_picks_equal": bool(np.array_equal(pick(compose(B, base0, range(7))), iB)),
                            "V2_allheads_picks_equal": bool(np.array_equal(pick(compose(B, C0A, range(7))), iB))}
    k["zero_RL_equals_SRC"] = {"base0": bool(np.array_equal(pick(compose(B, base0, ())), ib0)), "C0A": bool(np.array_equal(pick(compose(B, C0A, ())), pick(C0A)))}
    k["B_vs_base0"] = delta(iB, ib0)
    perm = np.random.default_rng(20261018).permutation(N)
    xr = compose(B, base0[perm], (2, 4))
    dr = delta(pick(xr), ib0)
    k["regression_arm_shuffled_SRC"] = {"vs_base0": dr, "pdms_lb_below_zero": bool(dr["pdms"][1] < 0)}
    k["boot_identity_max_abs"] = rlh.ctl_boot(kv, logs, 100 * (s["ttc"][ar, iB] - s["ttc"][ar, ib0]))[2]
# reproduction of RL-1's banked file (same code path; the R0 u1000 block must be identical to 1e-9)
banked = json.load(open("C:/Users/Admin/par18/rl/out/raw/composites_rlval.json"))["states"]["r0_u1000"]
mine = res["states"]["r0_u1000"]
mx, nk = 0.0, 0


def walk(x, y):
    global mx, nk
    if isinstance(x, dict):
        for kk in x:
            walk(x[kk], y[kk])
    elif isinstance(x, list):
        for p, q in zip(x, y):
            walk(p, q)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        mx = max(mx, abs(x - y)); nk += 1
    else:
        assert x == y, (x, y)


walk(mine, banked)
C["reproduce_RL1_banked_r0_u1000"] = {"max_abs_over_numbers": mx, "n_numbers": nk}
json.dump(res, open(a.out, "w"), indent=1)
print("controls:", json.dumps({k: v for k, v in C.items()})[:3000])
for st_ in ("r0d_u250", "r0d_u500", "r0d_u750", "r0d_u1000"):
    r = res["states"][st_]
    print(st_, {v: (r[v]["label"], [round(x, 3) for x in r[v]["vs_base0"]["pdms"]], round(r[v]["deficit"], 3)) for v in ("V1", "V2")})
