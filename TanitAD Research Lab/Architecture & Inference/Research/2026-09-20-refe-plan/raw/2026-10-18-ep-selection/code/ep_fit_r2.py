#!/usr/bin/env python3
"""EP-1 round 2, step 3: fit-set read of C4 BANDGUARD (PREREG_EP1_R2.md).  Fit table only; navtest NOT read.
    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 python ep_fit_r2.py --out ../raw/fit_read_r2.json --frozen ../FROZEN_R2.json
"""
import argparse
import json

import numpy as np

from ep_common import FIT, FIT_SHA, NC, DAC, EP, TTC, C, DDC, SEED, load, gates, at, level, paired, Boot
from ep_rules import heads, pick_base

DELTAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
EPSS = [0.0, 0.005, 0.01, 0.02, 0.05]
GRID = [(0.0, 0.0)] + [(d, e) for d in DELTAS for e in EPSS]          # ordered (delta asc, eps asc); first = identity


def pick_c4(p, g, w, L, delta, eps):
    base = pick_base(p, g, w)
    if delta == 0.0:
        return base
    ls = np.log(np.maximum(g * w, 1e-300))
    band = ls >= ls.max(1, keepdims=True) - delta
    ar = np.arange(len(base))
    g_b = g[ar, base][:, None]
    t_b = p[ar, base, TTC][:, None]
    ok = band & (g >= g_b - eps) & (p[..., TTC] >= t_b - eps)
    ok[ar, base] = True
    return np.argmax(np.where(ok, L, -np.inf), axis=1)


if __name__ == "__main__":
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
    perm = np.random.default_rng(SEED).permutation(t["n_logs"]).reshape(6, 4)
    fold_of_log = np.empty(t["n_logs"], int)
    for f in range(6):
        fold_of_log[perm[f]] = f
    fold = fold_of_log[t["log_idx"]]
    res = {"tier": "SELECTION tier on P2's navtrain held-out table (fit set); never an endpoint", "gates": gates(t, "fit"),
           "prereg_sha256": open("../PREREG_EP1_R2.sha256").read().split()[0], "identity_control": int((pick_c4(p, g, w, L, 0.0, 0.0) == base).sum()), "n": N}
    assert res["identity_control"] == N

    def comp(idx):
        return {"PDMS": at(pdms, idx), "NC": at(sub[..., NC], idx), "DAC": at(sub[..., DAC], idx), "TTC": at(sub[..., TTC], idx),
                "EP": at(sub[..., EP], idx), "C": at(sub[..., C], idx), "DDC": at(sub[..., DDC], idx)}

    def choose(gains, tol=0.05):
        m = max(gains)
        for i, gv in enumerate(gains):
            if gv >= m - tol:
                return i

    D = np.stack([at(pdms, pick_c4(p, g, w, L, d, e)) - pd_b for d, e in GRID])
    gains = [float(D[k].mean() * 100) for k in range(len(GRID))]
    res["grid"] = GRID
    res["in_sample_gain_x100_by_cell"] = gains
    ks = choose(gains)
    th = GRID[ks]
    res["theta_star"] = th
    oof = np.zeros(N)
    chosen = []
    for f in range(6):
        tr, te = fold != f, fold == f
        g_tr = [float(D[k][tr].mean() * 100) for k in range(len(GRID))]
        kf = choose(g_tr)
        chosen.append(GRID[kf])
        oof[te] = D[kf][te]
    fold_gain = [float(oof[fold == f].mean() * 100) for f in range(6)]
    res["nested_cv"] = {"chosen_theta_per_fold": chosen, "per_fold_oof_gain_x100": fold_gain, "folds_positive": int(sum(x > 0 for x in fold_gain)),
                        "pooled_oof": paired(boot, oof + pd_b, pd_b)}
    res["F1"] = {"value_x100": res["nested_cv"]["pooled_oof"]["point"], "folds_positive_of_6": res["nested_cv"]["folds_positive"],
                 "pass": bool(res["nested_cv"]["pooled_oof"]["point"] > 0 and res["nested_cv"]["folds_positive"] >= 4)}
    ci = pick_c4(p, g, w, L, *th)
    cc, cb = comp(ci), comp(base)
    comps, adverse = {}, []
    for k in ["PDMS", "NC", "DAC", "TTC", "EP", "C", "DDC"]:
        d = paired(boot, cc[k], cb[k])
        comps[k] = d
        if k in ("NC", "DAC", "TTC") and d["hi"] < 0:
            adverse.append(k)
    res["in_sample_at_theta_star"] = paired(boot, cc["PDMS"], cb["PDMS"])
    res["components_paired_vs_baseline_x100"] = comps
    res["zero_pdms_picks"] = {"baseline": int((pd_b == 0).sum()), "candidate": int((cc["PDMS"] == 0).sum())}
    res["picks_changed"] = int((ci != base).sum())
    res["L_at_pick_m_mean"] = {"baseline": float(at(L, base).mean()), "candidate": float(at(L, ci).mean())}
    res["F2"] = {"adverse_components": adverse, "pass": len(adverse) == 0}
    L_shuf = np.random.default_rng(SEED).permuted(L, axis=1)
    ctrl = max(float((at(pdms, pick_c4(p, g, w, L_shuf, d, e)) - pd_b).mean() * 100) for d, e in GRID[1:])
    res["F3"] = {"control_best_over_grid_x100": ctrl, "pass": bool(ctrl <= 0.05)}
    res["FIT_GATE_PASS"] = bool(res["F1"]["pass"] and res["F2"]["pass"] and res["F3"]["pass"])
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump({"note": "FROZEN before navtest is opened (PREREG_EP1_R2.md section 2.7)", "prereg_sha256": res["prereg_sha256"],
               "m": int(res["FIT_GATE_PASS"]), "candidates": {"C4_BANDGUARD": {"theta_star": th}} if res["FIT_GATE_PASS"] else {}},
              open(a.frozen, "w"), indent=1)
    print("grid gains (delta,eps):gain")
    for (d, e), gv in zip(GRID, gains):
        print(f"  ({d},{e}): {gv:.3f}", end="")
    print("\ntheta*", th, "in-sample", {k: round(v, 3) for k, v in res["in_sample_at_theta_star"].items() if k in ("point", "lo", "hi")})
    print("nested", chosen, [round(x, 3) for x in fold_gain], {k: round(v, 3) for k, v in res["nested_cv"]["pooled_oof"].items() if k in ("point", "lo", "hi")})
    print("F1", res["F1"], "F2", res["F2"], "F3", res["F3"], "PASS", res["FIT_GATE_PASS"])
    print("comps", {k: (round(v["point"], 3), round(v["lo"], 3), round(v["hi"], 3)) for k, v in comps.items()})
    print("zero", res["zero_pdms_picks"], "changed", res["picks_changed"], res["L_at_pick_m_mean"])
