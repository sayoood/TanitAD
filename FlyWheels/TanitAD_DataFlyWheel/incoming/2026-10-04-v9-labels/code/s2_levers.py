"""WP-A Stage 2 — Rule-Zero follow-up: E4 found NO eligible RC variant (E2 failed under the tree class on trainS).
SPEC §9.E E4 pre-registered the next levers IN THIS ORDER: (L1) sigma = 15 m; (L2) drop the heading field; (L3) the
noised input (sigma_along 2.0 m, sigma_lat 0.75 m) as the training default — each re-measured. This script runs all
three, plus:
  * an INSTRUMENT REPAIR (post-hoc, stated): the registered trees (HGB v1) failed their own deranged-input control
    (rho -0.064 trainS, -0.181 eval139: clip-specific memorisation, ~171 autocorrelated windows per clip).
    HGB v2 = the same trees with min_samples_leaf = 200 (a leaf must span more than one clip's windows) and
    l2_regularization = 1.0. v2 is CERTIFIED only if its own controls pass (|rho_deranged| <= 0.01, target >= 0.99).
  * diagnostics (no bar): the heavy route (sigma 25 m) as the input; L1 + L2 combined.

usage: python s2_levers.py extract <eval139|trainS> <s2_windows.npz> <out_sigma15.npz>
       python s2_levers.py e2 <ols|hgb1|hgb2> <s2_windows_X.npz> <sigma15_X.npz> <out.json>
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\r8_wpa\stack\scripts")
import build_v9_labels as B  # noqa: E402

VARS = ("A30", "A50", "A80", "B")
SEED = 20261004


def extract(split, win_p, out_p):
    import torch
    sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
    import s2_extract as X
    cfg = X.SRC[split]
    Z = np.load(win_p, allow_pickle=True)
    sha = np.array([str(s) for s in Z["clip_sha12"]])
    now = Z["t_now_s"]
    m = torch.load(cfg["manifest"], map_location="cpu", weights_only=False)
    cid_by_sha = {B.sha12(c): c for c in m["clip_id"]}
    import gzip
    sup = {}
    with gzip.open(cfg["labels"], "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                sup[B.sha12(r["clip_id"])] = r.get("turn_suppression")
    out = {f"rc15_{v}_{k}": np.full(len(now), np.nan) for v in VARS for k in ("x", "y", "psi", "valid")}
    t0 = time.time()
    for ci, s12 in enumerate(sorted(set(sha.tolist()))):
        log = B.load_log(X.EGO.format(cid_by_sha[s12]))
        tr = B.track10(log)
        turns = B.turn_table(log, tr, sup.get(s12))
        paths = {15.0: B.smooth_path(tr, 15.0), 25.0: B.smooth_path(tr, 25.0)}
        s_nat = B.s_at(tr, log.ts)
        for i in np.nonzero(sha == s12)[0]:
            rc = B.rc_fields(tr, log, paths, turns, float(now[i]), s_nat, sigma=15.0, sigma_h=25.0)
            for v in VARS:
                for k in ("x", "y", "psi", "valid"):
                    out[f"rc15_{v}_{k}"][i] = rc[f"rc_{v}_{k}"]
        if ci % 100 == 0:
            print(f"[levers extract {split}] {ci + 1} clips {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out_p, **out)
    print(f"[levers extract {split}] done {time.time() - t0:.0f}s")


def e2(instr, win_p, s15_p, out_p):
    sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
    import s2_analyze as A
    Z = dict(np.load(win_p, allow_pickle=True))
    Z.update(dict(np.load(s15_p, allow_pickle=True)))
    Z["clip_sha12"] = np.array([str(s) for s in Z["clip_sha12"]])
    shuf = A.deranged_by_clip(Z, [f"rc_A50_{k}" for k in ("x", "y", "psi")], SEED)
    sel = np.isfinite(Z["vfut"][:, 0]) & np.isfinite(Z["a0"]) & np.isfinite(Z["o1_ms"])
    for v in VARS:
        sel &= (Z[f"rc_{v}_valid"] == 1) & (Z[f"rc15_{v}_valid"] == 1) & (Z[f"rcH_{v}_valid"] == 1)
    for k in ("x", "y", "psi"):
        sel &= np.isfinite(shuf[f"rc_A50_{k}"])
    Y = Z["vfut"][sel]
    g = Z["clip_sha12"][sel]
    v0 = Z["v0"][sel][:, None]
    NAV = A.nav_block(Z, sel)
    rng = np.random.default_rng(SEED)

    def rc(prefix, v, cols=("x", "y", "psi")):
        return np.c_[tuple(Z[f"{prefix}_{v}_{c}"][sel] for c in cols)]

    def noised(v):
        x, y, psi = Z[f"rc_{v}_x"][sel], Z[f"rc_{v}_y"][sel], np.radians(Z[f"rc_{v}_psi"][sel])
        ea, el = rng.normal(0, 2.0, len(x)), rng.normal(0, 0.75, len(x))
        return np.c_[x + ea * np.cos(psi) - el * np.sin(psi), y + ea * np.sin(psi) + el * np.cos(psi), Z[f"rc_{v}_psi"][sel]]

    sets = {"F0": v0, "NAV": np.c_[v0, NAV],
            "CTRL_RC_A50_deranged": np.c_[v0, shuf["rc_A50_x"][sel], shuf["rc_A50_y"][sel], shuf["rc_A50_psi"][sel]],
            "CTRL_O1_oracle": np.c_[v0, Z["o1_ms"][sel]]}
    for v in VARS:
        arms = {"REG_s8": rc("rc", v), "L1_s15": rc("rc15", v), "L2_s8_noheading": rc("rc", v, ("x", "y")),
                "L3_s8_noised": noised(v), "DIAG_L1L2_s15_noheading": rc("rc15", v, ("x", "y"))}
        if v == "A50":
            arms["DIAG_heavy_s25"] = rc("rcH", v)
        for an, X_ in arms.items():
            sets[f"{an}_{v}"] = np.c_[v0, X_]
            sets[f"NAV_{an}_{v}"] = np.c_[v0, NAV, X_]
    out = {"instrument": instr, "n_rows": int(sel.sum()), "n_clips": int(len(np.unique(g))), "r2": {}, "rho": {}}
    t0 = time.time()

    def fit(X_, Y_):
        if instr == "ols":
            return A.r2_oof(X_, Y_, g, "ols")
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import GroupKFold
        kw = dict(max_iter=200, max_depth=6, learning_rate=0.05, random_state=0)
        if instr == "hgb2":
            kw.update(min_samples_leaf=200, l2_regularization=1.0)
        pred = np.zeros_like(Y_)
        for tr_i, te_i in GroupKFold(n_splits=5).split(X_, Y_[:, 0], g):
            for j in range(Y_.shape[1]):
                pred[te_i, j] = HistGradientBoostingRegressor(**kw).fit(X_[tr_i], Y_[tr_i, j]).predict(X_[te_i])
        return 1.0 - ((Y_ - pred) ** 2).sum(0) / ((Y_ - Y_.mean(0)) ** 2).sum(0)

    for k, X_ in sets.items():
        out["r2"][k] = [round(float(a), 5) for a in fit(X_, Y)]
    out["r2"]["CTRL_target_itself"] = [round(float(fit(np.c_[v0, Y[:, j]], Y[:, [j]])[0]), 5) for j in range(6)]
    base = np.array(out["r2"]["F0"])
    for k, r in out["r2"].items():
        out["rho"][k] = round(float(np.mean((np.array(r) - base) / (1.0 - base))), 5)
    rho = out["rho"]
    out["controls"] = {"deranged_abs_le_0.01": abs(rho["CTRL_RC_A50_deranged"]) <= 0.01,
                       "target_ge_0.99": rho["CTRL_target_itself"] >= 0.99, "o1_ge_0.40": rho["CTRL_O1_oracle"] >= 0.40}
    out["certified"] = bool(out["controls"]["deranged_abs_le_0.01"] and out["controls"]["target_ge_0.99"])
    out["bar"] = {}
    for k in rho:
        if k.startswith(("REG_", "L1_", "L2_", "L3_", "DIAG_")):
            nk = "NAV_" + k
            out["bar"][k] = {"rho": rho[k], "rho_nav": rho["NAV"], "nav_plus_minus_nav": round(rho[nk] - rho["NAV"], 5),
                             "pass": bool(rho[k] <= rho["NAV"] + 0.03 and rho[nk] - rho["NAV"] <= 0.03)}
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(out_p, "w"), indent=1)
    print(f"[levers e2 {instr}] {out['n_rows']} rows certified={out['certified']} controls={out['controls']} {out['wall_s']}s")
    for k, b in out["bar"].items():
        print(f"  {k:32s} rho {b['rho']:+.3f} nav {b['rho_nav']:+.3f} navRC-nav {b['nav_plus_minus_nav']:+.3f} {'PASS' if b['pass'] else 'FAIL'}")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(*sys.argv[2:5])
    else:
        e2(*sys.argv[2:6])
