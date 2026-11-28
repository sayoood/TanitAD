#!/usr/bin/env python3
"""R8-WISEFT-GATE-PAI (registered ad0cbbf, 2026-10-21 16:25 Berlin, before any gate composite was computed on a PhysicalAI dump).

The arc gate on the refcv8 per-frame battery's S6 dumps (T1 self-action open loop on logged eval139 frames; waypoints 1..6 s, dt 1.0):
    GW_ARC      = ha0_ext where |ln(arc4(os_theta05) / max(arc4(ha0_ext), 0.05))| > 0.1368, else os_theta05
    G_ARC48852  = the same gate on the shipped plan os_48852
Rule: per component, X - os_48852 at the SAME seed (paired episode-cluster bootstrap B 2000, 139 episodes); X is PARITY-OR-BETTER
iff NO component is separated-WORSE (higher error / lower agreement) at BOTH seeds, else WORSE (listed).

Per-window components: taniteval/tools/refav1_arm.py::_components (four_families _seq_geometry + maneuver_kinematics +
factor_from_kinematics) at dt 1.0, plus pair-masked yaw rate and curvature MAE on four_families' own mask (as
families/code/paired_families_navsim.py).  Interval: refav1_arm._paired_families.  Controls: trees aligned (ws, clip_index, g, ha0_ext
identical per episode); self-pair exactly 0; g-as-prediction 0 / 1 literal; never-fires gate == plan and always-fires == ha0_ext.

    python pai_gate_families.py --tree-48852 <dir> --tree-theta <dir> --out <json>
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

D = "D:/Projects/TanitAD"
for p in (f"{D}/stack", f"{D}/taniteval", f"{D}/taniteval/tools"):
    if p not in sys.path:
        sys.path.append(p)
THR = 0.1368
DT = 1.0
IDX4 = 3            # S6 waypoints at 1, 2, 3, 4, 5, 6 s -> index 3 is 4.0 s
WORSE_IF_HIGHER = True


def arc4(p):
    seg = np.diff(np.concatenate([np.zeros((p.shape[0], 1, 2)), p[:, :, :2]], 1), axis=1)
    return np.cumsum(np.hypot(seg[..., 0], seg[..., 1]), axis=1)[:, IDX4]


def gate(plan, prior):
    lr = np.abs(np.log(np.maximum(arc4(plan) / np.maximum(arc4(prior), 0.05), 0.05)))
    fb = lr > THR
    return np.where(fb[:, None, None], prior, plan), fb


def load(tree, seed):
    fs = sorted(glob.glob(os.path.join(tree, f"s6_s{seed}", "ep*.npz")))
    if not fs:
        raise SystemExit(f"no S6 dumps in {tree}/s6_s{seed}")
    cat, eid, meta = {}, [], []
    for f in fs:
        with np.load(f) as z:
            n = int(z["ws"].shape[0])
            for k in ("os", "ha0_ext", "g", "ws", f"b_refcv7_50400_s{seed}"):
                cat.setdefault(k, []).append(np.asarray(z[k], np.float64))
            eid += [os.path.splitext(os.path.basename(f))[0]] * n
            meta.append((os.path.basename(f), int(z["clip_index"][0])))
    return {k: np.concatenate(v) for k, v in cat.items()}, eid, meta


def components(ra, ff, torch, P, G):
    c = ra._components(P, G, DT)
    Pg, Gg = ff._seq_geometry(torch.as_tensor(P).float(), DT), ff._seq_geometry(torch.as_tensor(G).float(), DT)
    bp = Pg["pair_valid"] & Gg["pair_valid"]
    nv = bp.sum(1)
    for nm, key in (("LAT_curvature_mae_1pm", "curvature"), ("LAT_yaw_rate_mae_masked_radps", "yaw_rate")):
        c[nm] = torch.where(nv > 0, ((Pg[key] - Gg[key]).abs() * bp).sum(1) / nv.clamp_min(1), torch.nan).numpy()
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-48852", required=True); ap.add_argument("--tree-theta", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1"); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    import refav1_arm as ra
    from taniteval import four_families as ff
    ra._FAMILY_OF = {**ra._FAMILY_OF, "LAT_curvature_mae_1pm": "lateral", "LAT_yaw_rate_mae_masked_radps": "lateral"}
    seeds = [int(s) for s in a.seeds.split(",")]
    res = {"id": "R8-WISEFT-GATE-PAI", "registered": "ad0cbbf (2026-10-21 16:25 Berlin)", "surface": "S6, dt 1.0 s, waypoints 1..6 s",
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (the SHIP-R8 tier); one training seed", "threshold": THR,
           "trees": {"48852": a.tree_48852, "theta05": a.tree_theta}, "seeds": {}, "controls": {}}
    worse_both = {"GW_ARC": None, "G_ARC48852": None}
    per_seed_worse = {}
    for s in seeds:
        A, eA, mA = load(a.tree_48852, s)
        Bt, eB, mB = load(a.tree_theta, s)
        c = res["controls"].setdefault(f"s{s}", {})
        c["aligned"] = bool(mA == mB and eA == eB and np.array_equal(A["ws"], Bt["ws"]) and np.array_equal(A["g"], Bt["g"])
                            and np.array_equal(A["ha0_ext"], Bt["ha0_ext"]))
        G, prior = A["g"], A["ha0_ext"]
        gw, fb_w = gate(Bt["os"], prior)
        g48, fb_48 = gate(A["os"], prior)
        never = np.where(np.zeros(len(gw), bool)[:, None, None], prior, Bt["os"])
        always = np.where(np.ones(len(gw), bool)[:, None, None], prior, Bt["os"])
        c["never_fires_is_plan"] = bool(np.array_equal(never, Bt["os"]))
        c["always_fires_is_prior"] = bool(np.array_equal(always, prior))
        arms = {"os_48852": A["os"], "os_theta05": Bt["os"], "GW_ARC": gw, "G_ARC48852": g48, "ha0_ext": prior,
                "refcv7_50400": Bt[f"b_refcv7_50400_s{s}"], "GT": G}
        comps = {k: components(ra, ff, torch, v, G) for k, v in arms.items()}
        tiers = {k: "T1" for k in comps}
        hm = {m: float(np.nanmean(v)) for m, v in comps["GT"].items()}
        c["g_as_prediction"] = all(abs(hm[m]) < 1e-6 for m in hm if not m.startswith("TAC_")) and all(
            hm[m] == 1.0 for m in hm if m.startswith("TAC_"))
        comps["os_48852_self"] = comps["os_48852"]
        sp = ra._paired_families(comps, "os_48852", "os_48852_self", eA, {**tiers, "os_48852_self": "T1"}, 200, 0)
        c["self_pair_zero"] = all(float(r["delta"]) == 0.0 for fam in sp["families"].values() for r in fam.values())
        pairs = {}
        for b, a_ in (("GW_ARC", "os_48852"), ("G_ARC48852", "os_48852"), ("os_theta05", "os_48852"), ("GW_ARC", "os_theta05"),
                      ("GW_ARC", "ha0_ext"), ("G_ARC48852", "ha0_ext"), ("GW_ARC", "refcv7_50400"), ("G_ARC48852", "refcv7_50400")):
            pairs[f"{b}__minus__{a_}"] = ra._paired_families(comps, a_, b, eA, tiers, a.n_boot, 0)
        res["seeds"][f"s{s}"] = {"n_windows": len(eA), "n_episodes": len(set(eA)),
                                 "fallback_frac": {"GW_ARC": round(float(fb_w.mean()), 4), "G_ARC48852": round(float(fb_48.mean()), 4)},
                                 "means": {k: {m: round(float(np.nanmean(v)), 6) for m, v in comps[k].items()} for k in arms},
                                 "pairs": pairs}
        for X in worse_both:
            w = []
            for fam, blk in pairs[f"{X}__minus__os_48852"]["families"].items():
                for m, r in blk.items():
                    if not r.get("separated"):
                        continue
                    agree = m.startswith("TAC_")
                    if (agree and r["delta"] < 0) or (not agree and r["delta"] > 0):
                        w.append(f"{fam}.{m}")
            per_seed_worse.setdefault(X, {})[f"s{s}"] = w
    res["controls"]["all_pass"] = all(all(v for v in cs.values()) for k, cs in res["controls"].items() if isinstance(cs, dict))
    res["verdicts"] = {}
    for X, ws in per_seed_worse.items():
        both = sorted(set.intersection(*[set(v) for v in ws.values()])) if len(ws) == len(seeds) else None
        v = ("PARITY-OR-BETTER" if both == [] else f"WORSE on {both}") if len(seeds) == 2 else "SINGLE-SEED (no verdict)"
        res["verdicts"][X] = {"verdict": v if res["controls"]["all_pass"] else "INCOMPLETE (a control failed): " + v,
                              "separated_worse_per_seed": ws}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print("QQPAIG " + json.dumps({"verdicts": {k: v["verdict"] for k, v in res["verdicts"].items()},
                                  "fallback": {k: v["fallback_frac"] for k, v in res["seeds"].items()},
                                  "controls": res["controls"]["all_pass"]}) + " QQ")


if __name__ == "__main__":
    main()
