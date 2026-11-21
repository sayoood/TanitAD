#!/usr/bin/env python3
"""EP-1 step 3: the FIT-SET read of the three pre-registered candidates (PREREG_EP1.md section 2).  Fit table only; navtest is NOT read.
    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 python ep_fit.py --out ../raw/fit_read.json --frozen ../FROZEN.json
"""
import argparse
import json

import numpy as np

from ep_common import FIT, FIT_SHA, NC, DAC, EP, TTC, C, DDC, SEED, load, gates, at, level, paired, Boot
from ep_rules import (DELTA_GRID, TAU_GRID, heads, pick_base, pick_c1, pick_c1_identity, pick_c2, pick_c3)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--frozen", required=True)
a = ap.parse_args()
t = load(FIT, FIT_SHA[:16])
assert t["sha256"] == FIT_SHA
N = len(t["token"])
sub, pdms = t["sub"], t["pdms"]
p, g, w, L = heads(t)
boot = Boot(t["log_idx"], t["n_logs"])
base = pick_base(p, g, w)
assert (base == t["pick"]).all()
pd_b = at(pdms, base)
res = {"tier": "SELECTION tier on P2's navtrain held-out table (fit set); never an endpoint", "n_tokens": N, "n_logs": int(t["n_logs"]),
       "estimator": "paired log-cluster bootstrap, 10,000, seed 20260927, another draw of LOGS only (24 logs)", "gates": gates(t, "fit"),
       "prereg_sha256": open("../PREREG_EP1.sha256").read().split()[0]}

# fold assignment, PREREG section 2.4
perm = np.random.default_rng(SEED).permutation(t["n_logs"]).reshape(6, 4)
fold_of_log = np.empty(t["n_logs"], int)
for f in range(6):
    fold_of_log[perm[f]] = f
fold = fold_of_log[t["log_idx"]]


def comp(idx):
    return {"PDMS": at(pdms, idx), "NC": at(sub[..., NC], idx), "DAC": at(sub[..., DAC], idx), "TTC": at(sub[..., TTC], idx),
            "EP": at(sub[..., EP], idx), "C": at(sub[..., C], idx), "DDC": at(sub[..., DDC], idx)}


cb = comp(base)


def d_vec(idx):
    return at(pdms, idx) - pd_b


def choose(gains, tol=0.05):
    m = max(gains)
    for i, gv in enumerate(gains):
        if gv >= m - tol:
            return i


def read_candidate(name, idx_fun, grid):
    out = {"name": name}
    thetas = [None] if grid is None else grid
    D = np.stack([d_vec(idx_fun(th)) for th in thetas])
    gains = [float(D[k].mean() * 100) for k in range(len(thetas))]
    out["grid"] = list(thetas)
    out["in_sample_gain_x100_by_theta"] = gains
    ks = choose(gains)
    out["theta_star"] = thetas[ks]
    out["in_sample_at_theta_star"] = paired(boot, D[ks] + pd_b, pd_b)
    if grid is not None:
        oof = np.zeros(N)
        chosen = []
        for f in range(6):
            tr, te = fold != f, fold == f
            g_tr = [float(D[k][tr].mean() * 100) for k in range(len(thetas))]
            kf = choose(g_tr)
            chosen.append(thetas[kf])
            oof[te] = D[kf][te]
        fold_gain = [float(oof[fold == f].mean() * 100) for f in range(6)]
        out["nested_cv"] = {"chosen_theta_per_fold": chosen, "per_fold_oof_gain_x100": fold_gain,
                            "folds_positive": int(sum(x > 0 for x in fold_gain)), "pooled_oof": paired(boot, oof + pd_b, pd_b)}
        f1_val = out["nested_cv"]["pooled_oof"]["point"]
        f1_folds = out["nested_cv"]["folds_positive"]
    else:
        fold_gain = [float(D[0][fold == f].mean() * 100) for f in range(6)]
        out["per_fold_gain_x100"] = fold_gain
        f1_val = gains[0]
        f1_folds = int(sum(x > 0 for x in fold_gain))
    out["F1"] = {"value_x100": f1_val, "folds_positive_of_6": f1_folds, "pass": bool(f1_val > 0 and f1_folds >= 4)}
    ci = idx_fun(out["theta_star"])
    cc = comp(ci)
    comps = {}
    adverse = []
    for k in ["PDMS", "NC", "DAC", "TTC", "EP", "C", "DDC"]:
        d = paired(boot, cc[k], cb[k])
        comps[k] = d
        if k in ("NC", "DAC", "TTC") and d["hi"] < 0:
            adverse.append(k)
    out["components_paired_vs_baseline_x100"] = comps
    out["zero_pdms_picks"] = {"baseline": int((pd_b == 0).sum()), "candidate": int((cc["PDMS"] == 0).sum())}
    out["picks_changed"] = int((ci != base).sum())
    out["L_at_pick_m_mean"] = {"baseline": float(at(L, base).mean()), "candidate": float(at(L, ci).mean())}
    y4 = np.abs(t["proposals"][..., 7, 1].astype(float))
    out["abs_y4_at_pick_m_mean"] = {"baseline": float(at(y4, base).mean()), "candidate": float(at(y4, ci).mean())}
    out["F2"] = {"adverse_components": adverse, "pass": len(adverse) == 0}
    return out


res["identity_controls"] = {
    "C1_written_as_S0": int((pick_c1_identity(p, g, w) == base).sum()),
    "C2_delta0": int((pick_c2(p, g, w, L, 0.0) == base).sum()),
    "C3_tau1.0": int((pick_c3(p, g, w, L, 1.0) == base).sum()),
    "n": N}
assert all(v == N for k, v in res["identity_controls"].items() if k != "n")

rng = np.random.default_rng(SEED)
L_shuf = rng.permuted(L, axis=1)
pEP_shuf = rng.permuted(p[..., EP], axis=1)
p_shuf = p.copy()
p_shuf[..., EP] = pEP_shuf

cands = {}
cands["C1_XPDMS"] = read_candidate("C1_XPDMS", lambda th: pick_c1(p, g, w), None)
cands["C2_BAND"] = read_candidate("C2_BAND", lambda th: pick_c2(p, g, w, L, th), DELTA_GRID)
cands["C3_FLOOR"] = read_candidate("C3_FLOOR", lambda th: pick_c3(p, g, w, L, th), TAU_GRID)

ctrl = {}
ctrl["C1_pEP_shuffled"] = float(d_vec(pick_c1(p_shuf, g, w)).mean() * 100)
ctrl["C2_L_shuffled_best_over_grid"] = max(float(d_vec(pick_c2(p, g, w, L_shuf, th)).mean() * 100) for th in DELTA_GRID[1:])
ctrl["C3_L_shuffled_best_over_grid"] = max(float(d_vec(pick_c3(p, g, w, L_shuf, th)).mean() * 100) for th in TAU_GRID[1:])
res["controls_F3"] = ctrl
EPtrue = sub[..., EP]
res["informative_ceilings_not_gates"] = {
    "C2_true_EP_as_proxy_best_over_grid_x100": max(float(d_vec(pick_c2(p, g, w, EPtrue, th)).mean() * 100) for th in DELTA_GRID[1:]),
    "C3_true_EP_as_proxy_best_over_grid_x100": max(float(d_vec(pick_c3(p, g, w, EPtrue, th)).mean() * 100) for th in TAU_GRID[1:]),
    "C2_reversed_proxy_best_over_grid_x100": max(float(d_vec(pick_c2(p, g, w, -L, th)).mean() * 100) for th in DELTA_GRID[1:]),
    "C3_reversed_proxy_best_over_grid_x100": max(float(d_vec(pick_c3(p, g, w, -L, th)).mean() * 100) for th in TAU_GRID[1:])}
cands["C1_XPDMS"]["F3"] = {"control_x100": ctrl["C1_pEP_shuffled"], "pass": bool(ctrl["C1_pEP_shuffled"] <= 0.05)}
cands["C2_BAND"]["F3"] = {"control_x100": ctrl["C2_L_shuffled_best_over_grid"], "pass": bool(ctrl["C2_L_shuffled_best_over_grid"] <= 0.05)}
cands["C3_FLOOR"]["F3"] = {"control_x100": ctrl["C3_L_shuffled_best_over_grid"], "pass": bool(ctrl["C3_L_shuffled_best_over_grid"] <= 0.05)}
for c in cands.values():
    c["FIT_GATE_PASS"] = bool(c["F1"]["pass"] and c["F2"]["pass"] and c["F3"]["pass"])
res["baseline"] = {"PDMS_x100": level(boot, pd_b), **{k: level(boot, v) for k, v in cb.items() if k != "PDMS"}}
res["candidates"] = cands
passers = [k for k, c in cands.items() if c["FIT_GATE_PASS"]]
res["passers"] = passers
json.dump(res, open(a.out, "w"), indent=1)
frozen = {"note": "FROZEN before navtest is opened (PREREG_EP1.md section 2.8)", "prereg_sha256": res["prereg_sha256"], "m": len(passers),
          "candidates": {k: {"theta_star": cands[k]["theta_star"]} for k in passers}}
json.dump(frozen, open(a.frozen, "w"), indent=1)

print("baseline PDMS x100", round(res["baseline"]["PDMS_x100"]["point"], 3), "identity controls", res["identity_controls"])
print("controls F3", {k: round(v, 4) for k, v in ctrl.items()})
print("ceilings", {k: round(v, 3) for k, v in res["informative_ceilings_not_gates"].items()})
for k, c in cands.items():
    print("\n==", k, "theta*", c["theta_star"], "F1", c["F1"], "F2", c["F2"], "F3", c["F3"], "PASS", c["FIT_GATE_PASS"])
    print("   in-sample gain by theta:", [round(x, 3) for x in c["in_sample_gain_x100_by_theta"]])
    print("   in-sample at theta*:", {x: round(v, 3) for x, v in c["in_sample_at_theta_star"].items() if x in ("point", "lo", "hi")})
    if "nested_cv" in c:
        print("   nested:", c["nested_cv"]["chosen_theta_per_fold"], [round(x, 3) for x in c["nested_cv"]["per_fold_oof_gain_x100"]],
              {x: round(v, 3) for x, v in c["nested_cv"]["pooled_oof"].items() if x in ("point", "lo", "hi")})
    else:
        print("   folds:", [round(x, 3) for x in c["per_fold_gain_x100"]])
    print("   comps:", {kk: (round(v["point"], 3), round(v["lo"], 3), round(v["hi"], 3)) for kk, v in c["components_paired_vs_baseline_x100"].items()})
    print("   zero picks", c["zero_pdms_picks"], "changed", c["picks_changed"], "L", c["L_at_pick_m_mean"], "|y4|", c["abs_y4_at_pick_m_mean"])
