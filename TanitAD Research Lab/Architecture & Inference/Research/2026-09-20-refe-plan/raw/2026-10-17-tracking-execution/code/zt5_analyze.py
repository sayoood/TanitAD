#!/usr/bin/env python3
"""zt5_analyze.py -- PREREG_ZT5 sections 3-6: navtest mechanism (DAC level), dev mechanism replication, dev fix effects,
rho* choice, eligibility and the chosen fix.  numpy only (driverl-eval venv).

    python zt5_analyze.py --nav ../data/nav --dev ../data/dev --out ../RESULT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zt5_lib as L  # noqa: E402
import zero_tail as zt  # noqa: E402

CFS_NAV = ("H", "S", "V", "A", "P")
CFS_DEV = ("H", "S", "V", "A", "D", "P")
TM_PDMS_AT_STAKE = {"point": 1.07, "lo": 0.71, "hi": 1.55,
                    "src": "ZT3 RESULT.json report.lever_coverage_u400.proposal_side (Y-PDMS, = tracker-made)"}
SUBS = ("nc", "dac", "ep", "ttc", "c", "pdms")
FAM = ("speed_mae", "cross_mae", "heading_mae_deg", "delta")


def rnd(d, n=4):
    if isinstance(d, dict):
        return {k: rnd(v, n) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [rnd(v, n) for v in d]
    if isinstance(d, (float, np.floating)):
        return round(float(d), n)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def boot_for(logs):
    names = sorted(set(logs))
    li = np.array([names.index(x) for x in logs])
    return zt.Boot(li, len(names)), len(names)


def wquant(x, w, q=0.5):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x)
    x, w = x[ok], w[ok]
    if x.size == 0:
        return float("nan")
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, q * c[-1])])


def turn_bin(final_deg):
    a = abs(final_deg)
    return 0 if a < 15 else (1 if a < 45 else 2)


def v_bin(v):
    return 0 if v < 5 else (1 if v < 10 else 2)


def descr_block(recs, w=None):
    w = np.ones(len(recs)) if w is None else np.asarray(w, float)
    g = lambda k: np.array([r[k] for r in recs], float)  # noqa: E731
    nonstr = np.array([not r["straight"] for r in recs])
    ci = g("cross_inside")
    out = {"n": len(recs), "hinc_deg_med": wquant(g("hinc_deg"), w), "hinc_deg_p90": wquant(g("hinc_deg"), w, 0.9),
           "dv0_med": wquant(g("dv0"), w), "abs_dv0_med": wquant(np.abs(g("dv0")), w),
           "abs_kappa0_med": wquant(np.abs(g("kappa0")), w), "delta_med": wquant(g("delta"), w),
           "delta_p90": wquant(g("delta"), w, 0.9), "along_med": wquant(g("along"), w),
           "share_along_lag": float((w * (g("along") < 0)).sum() / w.sum()),
           "share_nonstraight_at_maxdev": float((w * nonstr).sum() / w.sum()),
           "share_inside_among_nonstraight": float((w * nonstr * (ci > 0)).sum() / max((w * nonstr).sum(), 1e-12)),
           "abs_cross_med": wquant(np.abs(g("cross")), w), "dev_t_med_s": wquant(g("dev_t") / 10.0, w)}
    return out


def rescue_table(recs, cfs, boot):
    out = {}
    for c in cfs:
        if c not in recs[0]["cf"]:
            continue
        x = np.array([r["cf"][c] for r in recs], float)
        out[c] = boot.ratio(x, np.ones_like(x), scale=100.0)
    return out


# ------------------------------------------------------------------------------------------------------------- navtest
def nav_part(nav_dir):
    recs = [json.loads(x) for f in sorted(glob.glob(f"{nav_dir}/nav_*.jsonl")) for x in open(f, encoding="utf-8")]
    tm = [r for r in recs if r["kind"] == "TM"]
    ctrl = [r for r in recs if r["kind"] == "CTRL" and r["stratum"] == "random"]
    gates = {"AG1_all": all(r["AG1"] for r in recs), "NG1_all": all(r["NG1"] for r in recs), "n_TM": len(tm),
             "n_CTRL_random_dac1": len(ctrl), "n_excluded_nonrandom_ctrl_units": sum(r["kind"] == "CTRL" and r["stratum"] != "random" for r in recs),
             "TM_raw_dac1_all": all(r["cf"]["raw"] == 1.0 for r in tm), "TM_base_dac0_all": all(r["cf"]["base"] == 0.0 for r in tm)}
    b_tm, nl = boot_for([r["log"] for r in tm])
    res = rescue_table(tm, CFS_NAV, b_tm)
    pd = np.array([r["cf"]["P_drawn"] for r in tm], float)
    res["P_drawn_compliant"] = b_tm.ratio(pd, np.ones_like(pd), scale=100.0)
    # controls, post-stratified to T_A's turn x v0 cells
    cell = lambda r: (turn_bin(r["final_deg"]), v_bin(r["v0"]))  # noqa: E731
    tm_c = {}
    for r in tm:
        tm_c[cell(r)] = tm_c.get(cell(r), 0) + 1
    ct_c = {}
    for r in ctrl:
        ct_c[cell(r)] = ct_c.get(cell(r), 0) + 1
    w = np.array([tm_c.get(cell(r), 0) / ct_c[cell(r)] for r in ctrl], float)
    covered = sum(n for c, n in tm_c.items() if c in ct_c) / len(tm)
    b_ct, _ = boot_for([r["log"] for r in ctrl])
    broken = {}
    for c in CFS_NAV:
        x = np.array([1.0 - r["cf"][c] for r in ctrl], float)
        broken[c] = b_ct.ratio(x * w, w, scale=100.0)
    dom = max(("H", "S", "V", "A"), key=lambda c: res[c]["point"])
    reading = ("DOMINANT " + dom) if res[dom]["point"] >= 25.0 else "DISTRIBUTED TRACKING LAG"
    return {"gates": gates, "n_logs_TM": nl, "rescue_pct_TM": res, "controls_broken_pct_weighted": broken,
            "control_weighting": {"cells_TM": {str(k): v for k, v in tm_c.items()}, "TM_share_in_covered_cells": covered},
            "descriptors": {"TM": descr_block(tm), "CTRL_weighted": descr_block(ctrl, w), "CTRL_unweighted": descr_block(ctrl)},
            "R_MECH": {"dominant_single_factor": dom, "reading": reading},
            "reach_estimate_pdms": {c: {"point": res[c]["point"] / 100 * TM_PDMS_AT_STAKE["point"],
                                        "evidence": "ESTIMATED = navtest rescue share x ZT3 tracker-made PDMS-at-stake; ignores harm and EP/C/TTC"}
                                    for c in ("H", "P")}}


# ------------------------------------------------------------------------------------------------------------- dev
def dev_load(dev_dir):
    parts = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(f"{dev_dir}/dev_*of2.npz"))]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files
         if k not in ("arms", "rho_grid", "subs", "n_skip")}
    D["arms"], D["rho_grid"] = parts[0]["arms"], parts[0]["rho_grid"]
    D["n_skip"] = int(sum(int(p["n_skip"]) for p in parts))
    units = [json.loads(x) for f in sorted(glob.glob(f"{dev_dir}/dev_*of2_units.jsonl")) for x in open(f, encoding="utf-8")]
    return D, units


def eff(boot, fixed, base, scale=100.0):
    d = np.asarray(fixed, float) - np.asarray(base, float)
    return boot.ratio(d, np.ones_like(d), scale=scale)


def fix_block(boot, base6, fix6, drawn_base, drawn_fix, fam_base, fam_fix, man_agree=None):
    out = {"dPDMS_x100": eff(boot, fix6[:, 5], base6[:, 5])}
    for i, s in enumerate(SUBS[:5]):
        out[f"d{s.upper()}_x100"] = eff(boot, fix6[:, i], base6[:, i])
    out["dZERO_pp"] = eff(boot, (fix6[:, 5] == 0).astype(float), (base6[:, 5] == 0).astype(float))
    out["dac_zero_rescued"] = int(((base6[:, 1] == 0) & (fix6[:, 1] == 1)).sum())
    out["dac_zero_broken"] = int(((base6[:, 1] == 1) & (fix6[:, 1] == 0)).sum())
    out["drawn_fail_pp"] = eff(boot, 1.0 - np.asarray(drawn_fix, float), 1.0 - np.asarray(drawn_base, float))
    out["families"] = {"LONGITUDINAL": {"dEP_x100": out["dEP_x100"], "d_speed_mae": eff(boot, fam_fix[:, 0], fam_base[:, 0], 1.0)},
                       "LATERAL": {"d_cross_mae": eff(boot, fam_fix[:, 1], fam_base[:, 1], 1.0),
                                   "d_heading_mae_deg": eff(boot, fam_fix[:, 2], fam_base[:, 2], 1.0),
                                   "d_delta": eff(boot, fam_fix[:, 3], fam_base[:, 3], 1.0)},
                       "TACTICAL": ({"manoeuvre_agreement_pct": boot.ratio(man_agree, np.ones_like(man_agree), 100.0)}
                                    if man_agree is not None else "1.0 by construction (same proposal)"),
                       "STRATEGIC": "PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE item 25)"}
    return out


def dev_part(dev_dir):
    D, units = dev_load(dev_dir)
    n = len(D["log"])
    gates = {"n_sets": n, "n_skipped": D["n_skip"], "AG1_all": bool(D["ag1"].all()),
             "DG1_max_abs": float(D["dg1"].max()), "DG1_pass": bool(D["dg1"].max() <= 1e-5),
             "DG2_agree_pct": float(D["dg2"].sum() / (64 * n) * 100), "DG2_pass": bool(D["dg2"].sum() / (64 * n) >= 0.999)}
    boot, nl = boot_for([str(x) for x in D["log"]])
    gates["n_logs"] = nl
    I = np.arange(n)
    arms = [str(a) for a in D["arms"]]
    rhos = [float(r) for r in D["rho_grid"]]
    out = {}
    for k, arm in enumerate(arms):
        pk = D["pick"][:, k]
        base6 = D["banked"][I, pk]                                       # == sc_base (DG1)
        A = {"base_PDMS_x100": boot.ratio(base6[:, 5], np.ones(n), 100.0),
             "base_zero_pct": boot.ratio((base6[:, 5] == 0).astype(float), np.ones(n), 100.0),
             "base_tracker_made_zero_picks": int(((base6[:, 5] == 0) & (base6[:, 1] == 0) & (D["dacR"][I, pk] == 1)).sum()),
             "base_dac_zero_picks": int((base6[:, 1] == 0).sum())}
        fb = D["fam_base"][:, k]
        A["F_c"] = fix_block(boot, D["sc_base"][:, k], D["sc_Fc"][:, k], D["dacR"][I, pk], D["dacRc"][I, pk], fb, D["fam_Fc"][:, k])
        A["F_a_ceiling"] = fix_block(boot, D["sc_base"][:, k], D["sc_Fa"][:, k], D["dacR"][I, pk], D["drawn_Fa"][:, k], fb, D["fam_Fa"][:, k])
        man = lambda j: np.array([L.manoeuvre(np.radians(x)) for x in D["final_deg"][I, j]])  # noqa: E731
        A["F_b"], A["F_bc"] = {}, {}
        for ri, rho in enumerate(rhos):
            j = D["fb"][:, k, ri]
            A["F_b"][str(rho)] = fix_block(boot, base6, D["banked"][I, j], D["dacR"][I, pk], D["dacR"][I, j], fb,
                                           D["fam_Fb"][:, k, ri], (man(j) == man(pk)).astype(float))
            A["F_b"][str(rho)]["switched_pct"] = float((j != pk).mean() * 100)
            jc = D["fbc"][:, k, ri]
            A["F_bc"][str(rho)] = fix_block(boot, D["sc_base"][:, k], D["sc_Fbc"][:, k, ri], D["dacR"][I, pk], D["dacRc"][I, jc], fb,
                                            D["fam_Fbc"][:, k, ri], (man(jc) == man(pk)).astype(float))
            A["F_bc"][str(rho)]["switched_pct"] = float((jc != pk).mean() * 100)
        out[arm] = A
    # rho* on N-s0 (primary): largest lower bound of dPDMS; ties -> larger rho
    prim = arms[0]
    rho_star = {}
    for fx in ("F_b", "F_bc"):
        best = max(rhos, key=lambda r: (round(out[prim][fx][str(r)]["dPDMS_x100"]["lo"], 10), r))
        rho_star[fx] = best
    # eligibility (PREREG section 6)
    def elig(fx, key=None):
        g = lambda arm: out[arm][fx] if key is None else out[arm][fx][key]  # noqa: E731
        p = g(prim)
        conds = {"admissible_real_driving": True,
                 "lo_gt_0_and_point_ge_0.05": p["dPDMS_x100"]["lo"] > 0 and p["dPDMS_x100"]["point"] >= 0.05,
                 "replicates_point_gt_0": all(g(a)["dPDMS_x100"]["point"] > 0 for a in arms[1:]),
                 "no_adverse_subscore": all(p[f"d{s.upper()}_x100"]["hi"] >= 0 for s in SUBS[:5]),
                 "drawn_plan_check": p["drawn_fail_pp"]["hi"] <= 0.5 / 100 * 100}
        return {"eligible": all(conds.values()), "conditions": conds, "dPDMS_x100": p["dPDMS_x100"]}
    E = {"F_c": elig("F_c"), "F_b": elig("F_b", str(rho_star["F_b"])), "F_a_ceiling": {"eligible": False,
         "reason": "pre-compensation draws a path off the intended one (real-driving rule; PI decision (a) open)",
         "dPDMS_x100": out[prim]["F_a_ceiling"]["dPDMS_x100"]}}
    e_bc = elig("F_bc", str(rho_star["F_bc"]))
    if not (E["F_b"]["eligible"] and E["F_c"]["eligible"]):
        e_bc["eligible"] = False
        e_bc["note"] = "competes only if F_b and F_c are each eligible (PREREG section 6)"
    E["F_bc"] = e_bc
    el = [f for f in ("F_c", "F_b", "F_bc") if E[f]["eligible"]]
    chosen = max(el, key=lambda f: E[f]["dPDMS_x100"]["point"]) if el else None
    # dev mechanism replication
    picks_u = {}
    for u in units:
        if u["kind"] == "pick":
            picks_u[(u["log"], u["token"], u["step"], u["rank"], u["j"])] = u
    pu = list(picks_u.values())
    fan = [u for u in units if u["kind"] == "fan"]
    ctrl = [u for u in units if u["kind"] == "ctrl"]
    mech = {}
    for name, rs in (("tracker_made_picks", pu), ("fan_tracker_made", fan)):
        if rs:
            b, nlu = boot_for([u["log"] for u in rs])
            mech[name] = {"n": len(rs), "n_logs": nlu, "rescue_pct": rescue_table(rs, CFS_DEV, b),
                          "descriptors": descr_block(rs)}
    if ctrl:
        b, _ = boot_for([u["log"] for u in ctrl])
        x = np.array([1.0 - u["cf"]["H"] for u in ctrl], float)
        mech["fan_controls"] = {"n": len(ctrl), "H_broken_pct": b.ratio(x, np.ones_like(x), 100.0), "descriptors": descr_block(ctrl)}
    return {"gates": gates, "arms": arms, "rho_grid": rhos, "rho_star_N-s0": rho_star, "effects": out,
            "eligibility": E, "chosen": chosen, "mechanism_replication": mech}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"package": "C:/Users/Admin/zt5/raw/2026-10-17-tracking-execution (ZT5)",
           "prereg": "PREREG_ZT5.md (sha256 in PREREG_ZT5.md.sha256)",
           "tier": "T1 NAVSIM v1.1 pseudo-simulation (LQR + kinematic bicycle); dev = navtrain held-out, PDM re-implementation (DG1-verified)",
           "estimator": "paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (zero_tail.Boot); answers another draw of LOGS only",
           "evidence": "MEASURED (ours) unless a field says ESTIMATED"}
    if glob.glob(f"{a.nav}/nav_*.jsonl"):
        res["navtest_mechanism"] = nav_part(a.nav)
    if glob.glob(f"{a.dev}/dev_*of2.npz"):
        res["dev"] = dev_part(a.dev)
    res = rnd(res)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("navtest_mechanism",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
