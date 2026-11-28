#!/usr/bin/env python3
"""R8-WISEFT-GATEP stage P (registered af6803a, 2026-10-21 16:35 Berlin): the PATH-PRESERVING gate on PhysicalAI S6 dumps vs the SHIPPED
EMA-48,852 plan -- the R8-WISEFT-GATE-PAI rule unchanged (PARITY-OR-BETTER iff no family component is separated-worse at BOTH seeds).

    gate fires   : |ln(arc4(os_theta05) / max(arc4(ha0_ext), 0.05))| > 0.1368   (arc4 = chords through origin and the 1..4 s points)
    GP_PRIORPROG : theta_0.5's path (polyline through origin + its 1..6 s points) driven at the prior's arc length s' = M, where it fires
    GP_MINPROG   : the same path at s' = min(L, M), where it fires
Geometry = the shipped retime stage's polyline_geometry / eval_polyline / prior_arc.  Components, estimator and the alignment / self-pair /
GT controls are pai_gate_families.py's (imported).  Extra controls: never-firing == plan bit-exact; always-firing GP_PRIORPROG ==
eval_polyline at s' = M within 1e-12 m (two code paths of the stage; max recorded); GP_MINPROG never travels farther than the plan or the prior at any instant.
"""
import argparse
import importlib.util
import json
import sys

import numpy as np

GDIR = "C:/Users/Admin/refcv8_pf/navsim_ckpt/wiseft/gate/code"
STAGE = "C:/Users/Admin/refcv8_pf/navsim_ckpt/thor_bundle/stage/retime/retime_stage.py"


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def gp(RS, plan, prior, fire, mode):
    geo = RS.polyline_geometry(plan)
    L, M = geo["L"][:, 1:], RS.prior_arc(prior)
    if mode == "PRIORPROG":                      # the stage's OWN g = 0 call (as build_lever_seams.py's R8PATH_PRIORPROG)
        P, s = RS.retime_positions(geo, M, np.zeros(M.shape[1]))
    else:
        s = np.minimum(L, M)
        P = np.stack([RS.eval_polyline(geo, s[:, k]) for k in range(s.shape[1])], 1)
    return np.where(fire[:, None, None], P, plan), L, M, s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-48852", required=True); ap.add_argument("--tree-theta", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    PG = imp("pai_gate_families_m", f"{GDIR}/pai_gate_families.py")
    RS = imp("retime_stage_gp", STAGE)
    import torch
    import refav1_arm as ra
    from taniteval import four_families as ff
    ra._FAMILY_OF = {**ra._FAMILY_OF, "LAT_curvature_mae_1pm": "lateral", "LAT_yaw_rate_mae_masked_radps": "lateral"}
    res = {"id": "R8-WISEFT-GATEP stage P", "registered": "af6803a (2026-10-21 16:35 Berlin)", "surface": "S6, dt 1.0 s, 1..6 s",
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (the SHIP-R8 tier); one training seed", "seeds": {}, "controls": {}}
    worse = {}
    for s in (0, 1):
        A, eA, mA = PG.load(a.tree_48852, s)
        Bt, eB, mB = PG.load(a.tree_theta, s)
        c = res["controls"].setdefault(f"s{s}", {})
        c["aligned"] = bool(mA == mB and eA == eB and np.array_equal(A["g"], Bt["g"]) and np.array_equal(A["ha0_ext"], Bt["ha0_ext"]))
        G, prior, plan = A["g"], A["ha0_ext"], Bt["os"]
        lr = np.abs(np.log(np.maximum(PG.arc4(plan) / np.maximum(PG.arc4(prior), 0.05), 0.05)))
        fire = lr > PG.THR
        arms = {"os_48852": A["os"], "os_theta05": plan, "ha0_ext": prior, "GT": G}
        for mode in ("PRIORPROG", "MINPROG"):
            arms[f"GP_{mode}"], L, M, sp = gp(RS, plan, prior, fire, mode)
            never, _, _, _ = gp(RS, plan, prior, np.zeros(len(fire), bool), mode)
            c[f"never_fires_is_plan_{mode}"] = bool(np.array_equal(never, plan))
        always, _, _, _ = gp(RS, plan, prior, np.ones(len(fire), bool), "PRIORPROG")
        geo_p = RS.polyline_geometry(plan)
        alt = np.stack([RS.eval_polyline(geo_p, RS.prior_arc(prior)[:, k]) for k in range(plan.shape[1])], 1)
        dmax = float(np.abs(always - alt).max())    # the stage's g=0 path vs eval at s'=M directly: L + (M - L) != M in the last bit
        res.setdefault("control_values", {})[f"s{s}_PRIORPROG_two_paths_max_abs_m"] = dmax
        c["always_PRIORPROG_two_code_paths_agree_1e-12"] = dmax <= 1e-12
        geo_m = RS.polyline_geometry(arms["GP_MINPROG"])
        c["MINPROG_never_farther"] = bool((geo_m["L"][:, 1:] <= np.maximum(L, M) + 1e-9).all())
        comps = {k: PG.components(ra, ff, torch, v, G) for k, v in arms.items()}
        tiers = {k: "T1" for k in comps}
        hm = {m: float(np.nanmean(v)) for m, v in comps["GT"].items()}
        c["g_as_prediction"] = all(abs(hm[m]) < 1e-6 for m in hm if not m.startswith("TAC_")) and all(
            hm[m] == 1.0 for m in hm if m.startswith("TAC_"))
        comps["os_48852_self"] = comps["os_48852"]
        sp_ = ra._paired_families(comps, "os_48852", "os_48852_self", eA, {**tiers, "os_48852_self": "T1"}, 200, 0)
        c["self_pair_zero"] = all(float(r["delta"]) == 0.0 for fam in sp_["families"].values() for r in fam.values())
        pairs = {}
        for b, a_ in (("GP_PRIORPROG", "os_48852"), ("GP_MINPROG", "os_48852"), ("GP_PRIORPROG", "os_theta05"),
                      ("GP_MINPROG", "os_theta05"), ("GP_PRIORPROG", "ha0_ext"), ("GP_MINPROG", "ha0_ext")):
            pairs[f"{b}__minus__{a_}"] = ra._paired_families(comps, a_, b, eA, tiers, a.n_boot, 0)
        res["seeds"][f"s{s}"] = {"n_windows": len(eA), "fire_frac": round(float(fire.mean()), 4),
                                 "means": {k: {m: round(float(np.nanmean(v)), 6) for m, v in comps[k].items()} for k in arms},
                                 "pairs": pairs}
        for X in ("GP_PRIORPROG", "GP_MINPROG"):
            w = []
            for fam, blk in pairs[f"{X}__minus__os_48852"]["families"].items():
                for m, r in blk.items():
                    if r.get("separated") and ((m.startswith("TAC_") and r["delta"] < 0) or (not m.startswith("TAC_") and r["delta"] > 0)):
                        w.append(f"{fam}.{m}")
            worse.setdefault(X, {})[f"s{s}"] = w
    res["controls"]["all_pass"] = all(all(v for v in cs.values()) for cs in res["controls"].values() if isinstance(cs, dict))
    res["verdicts"] = {}
    for X, ws in worse.items():
        both = sorted(set(ws["s0"]) & set(ws["s1"]))
        v = "PARITY-OR-BETTER" if not both else f"WORSE on {both}"
        res["verdicts"][X] = {"verdict": v if res["controls"]["all_pass"] else "INCOMPLETE (a control failed): " + v, "separated_worse_per_seed": ws}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print("QQGPP " + json.dumps({"verdicts": {k: v["verdict"] for k, v in res["verdicts"].items()},
                                 "fire": {k: v["fire_frac"] for k, v in res["seeds"].items()}, "controls": res["controls"]["all_pass"]}) + " QQ")


if __name__ == "__main__":
    main()
