#!/usr/bin/env python3
"""nav_arm2.py -- ARM 1 (PREREG_FROZEN.md 4): what the nav token encodes.  Stages (resumable, time-boxed):
    python -I nav_arm2.py --stage desc                      descriptive tables on the FULL populations (GO rate by nav state)
    python -I nav_arm2.py --stage lin --pop pstop|pmove     linear arms N5, NL, L (alone, +E), decomposition NV/NT/ND, controls C1/C3 on N5+E
    python -I nav_arm2.py --stage mlp --pop pstop|pmove     MLP arms N5, NL, +E
    python -I nav_arm2.py --stage pair --pop pstop|pmove    paired deltas vs the banked E / V+E / Z+E  -> raw/results_nav2_<pop>.json
nav5 = (FOLLOW, TURN_L, TURN_R one-hot, d_next/100 clipped [0,3], args_valid) at NOW = J["nav"][off[clip] + t] of the rebuilt TRAIN join; limit = limit4 one-hot ++ limit_ok.
CUDA_VISIBLE_DEVICES=-1."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
warnings.filterwarnings("ignore")
import numpy as np

import common2 as C  # noqa: E402
L = C.L


def nav_feats(d, A, J):
    clips = [str(c) for c in A["clips"]]
    nav, lim, _, _ = C.join_nav(J, clips, d["clip"], d["t"])
    return nav, lim


def arm_sets(d, nav, lim):
    E, y = d["E"], d["y"]
    N5, NL, Lm = nav, np.concatenate([nav, lim], 1), lim
    NV, NT, ND = nav[:, 4:5], nav[:, :3], nav[:, 3:5]
    cat = lambda *a: (lambda f: np.concatenate(a, 1))
    lin = {"N5": (cat(N5), None), "NL": (cat(NL), None), "L": (cat(Lm), None),
           "N5+E": (cat(N5, E), None), "NL+E": (cat(NL, E), None), "L+E": (cat(Lm, E), None),
           "NV": (cat(NV), None), "NT": (cat(NT), None), "ND": (cat(ND), None),
           "NV+E": (cat(NV, E), None), "NT+E": (cat(NT, E), None), "ND+E": (cat(ND, E), None)}
    ys = L.shuffle_labels(y, 1)
    lin["C1:shuffle:N5+E"] = (cat(N5, E), ys)
    lin["C3:leak:N5+E"] = (lambda f: np.concatenate([N5, E, y[:, None].astype(np.float64)], 1), None)
    mlp = {"N5": lin["N5"], "N5+E": lin["N5+E"], "NL": lin["NL"], "NL+E": lin["NL+E"]}
    return lin, mlp


def rate_ci(go, clip, n_boot=2000, seed=0):
    u, inv = np.unique(clip, return_inverse=True)
    s = np.bincount(inv, weights=go, minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        b.append((c @ s) / max(c @ n, 1.0))
    return {"rate": float(s.sum() / max(n.sum(), 1.0)), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "n": int(n.sum())}


def stage_desc():
    A = C.load_windows()
    J = C.load_join_npz(f"{C.OLDSCR}/join_train_rebuilt.npz")
    clips = [str(c) for c in A["clips"]]
    v0, go, ci_, tt = A["v0"], A["go"].astype(float), A["clip_i"], A["t"]
    nav, lim, row, _ = C.join_nav(J, clips, ci_, tt)
    has = J["has"][row]
    out = {"note": "GO rate by nav state on the FULL populations (all windows, not the sample); clip-cluster bootstrap B=2000 seed 0",
           "all_windows": {"n": int(len(v0)), "share_has_v9_row": float(has.mean()), "share_nav_all_zero": float((nav.sum(1) == 0).mean()),
                           "share_args_valid": float(nav[:, 4].mean()), "share_token_TURN": float((nav[:, 1:3].sum(1) > 0).mean())},
           "pops": {}}
    for pop, m in (("pstop", v0 < 0.5), ("pmove", (v0 >= 0.5) & (v0 < 6.0))):
        n_, g_, c_ = nav[m], go[m], ci_[m]
        valid = n_[:, 4] > 0
        tok = np.where(n_[:, :3].sum(1) == 0, -1, n_[:, :3].argmax(1))
        dn = np.where(valid, n_[:, 3] * 100.0, np.nan)
        P = {"n": int(m.sum()), "go_rate": rate_ci(g_, c_), "by_args_valid": {}, "by_token": {}, "by_d_next_bin": {}}
        for nm, mm in (("valid", valid), ("not_valid", ~valid)):
            P["by_args_valid"][nm] = {**rate_ci(g_[mm], c_[mm]), "share_of_population": float(mm.mean())}
        for nm, k in (("FOLLOW", 0), ("TURN_L", 1), ("TURN_R", 2), ("no_row(all-zero)", -1)):
            mm = tok == k
            if mm.sum() >= 30:
                P["by_token"][nm] = {**rate_ci(g_[mm], c_[mm]), "share_of_population": float(mm.mean())}
        bins = (("<=5m", valid & (dn <= 5)), ("5-15m", valid & (dn > 5) & (dn <= 15)), ("15-30m", valid & (dn > 15) & (dn <= 30)), (">30m", valid & (dn > 30)), ("none", ~valid))
        for nm, mm in bins:
            if mm.sum() >= 30:
                P["by_d_next_bin"][nm] = {**rate_ci(g_[mm], c_[mm]), "share_of_population": float(mm.mean())}
        gom = g_ > 0
        P["share_of_GO_windows_with_args_valid"] = float(valid[gom].mean())
        P["share_of_notGO_windows_with_args_valid"] = float(valid[~gom].mean())
        P["share_of_GO_windows_with_TURN_token"] = float((tok[gom] > 0).mean())
        P["share_of_notGO_windows_with_TURN_token"] = float((tok[~gom] > 0).mean())
        P["mean_d_next_m_valid_GO"] = float(np.nanmean(dn[gom])) if np.isfinite(dn[gom]).any() else None
        P["mean_d_next_m_valid_notGO"] = float(np.nanmean(dn[~gom])) if np.isfinite(dn[~gom]).any() else None
        P["median_d_next_m_valid_GO"] = float(np.nanmedian(dn[gom])) if np.isfinite(dn[gom]).any() else None
        P["median_d_next_m_valid_notGO"] = float(np.nanmedian(dn[~gom])) if np.isfinite(dn[~gom]).any() else None
        out["pops"][pop] = P
        print(pop, "n", P["n"], "go", round(P["go_rate"]["rate"], 3), "| GO|valid", round(P["by_args_valid"]["valid"]["rate"], 3), "GO|not valid", round(P["by_args_valid"]["not_valid"]["rate"], 3),
              "| valid share GO/notGO", round(P["share_of_GO_windows_with_args_valid"], 3), round(P["share_of_notGO_windows_with_args_valid"], 3), flush=True)
    json.dump(out, open(f"{C.RAW}/results_nav2_desc.json", "w"), indent=1, default=float)


def stage_fit(pop, fam):
    A = C.load_windows()
    J = C.load_join_npz(f"{C.OLDSCR}/join_train_rebuilt.npz")
    d = C.load_pop(pop)
    nav, lim = nav_feats(d, A, J)
    lin, mlp = arm_sets(d, nav, lim)
    return C.run_arms(f"nav_{pop}", d, fam, lin if fam == "linear" else mlp)


def stage_pair(pop):
    d = C.load_pop(pop)
    y, clip, outer = d["y"], d["clip"], d["outer"]
    counts = L.boot_counts(clip)
    out = {"pop": pop, "n": int(len(y)), "note": "paired on the SAME 12,000 windows / folds / bootstrap as the banked arms; 'fold' = mean of per-fold AUCs (primary), 'pooled' = pooled OOF AUC", "arms": {}, "paired": {}}
    for fam in ("linear", "mlp"):
        bk = C.banked(pop, fam)
        if fam == "linear":
            assert np.array_equal(bk["y"], y) and np.array_equal(bk["clip"], clip), "row order differs from the banked arms"
        oof = dict(np.load(f"{C.RAW}/oof2_nav_{pop}_{fam}.npz"))
        res = json.load(open(f"{C.RAW}/res2_nav_{pop}_{fam}.json"))
        out["arms"][fam] = res["arms"]
        out["arms"][fam].update({f"banked:{k}": C.rec(bk[k], y, clip, outer, counts) for k in ("E", "V+E", "Z+E", "Z") if k in bk})
        P = {}
        for a in ("N5", "NL", "L", "NV", "NT", "ND"):
            if f"{a}+E" in oof and a != "L" or (a == "L" and "L+E" in oof):
                P[f"{a}+E_minus_E"] = C.paired(oof[f"{a}+E"], bk["E"], y, clip, outer, counts)
        for a in ("N5", "NL"):
            if a in oof:
                P[f"{a}_minus_E"] = C.paired(oof[a], bk["E"], y, clip, outer, counts)
            if f"{a}+E" in oof:
                P[f"Z+E_minus_{a}+E"] = C.paired(bk["Z+E"], oof[f"{a}+E"], y, clip, outer, counts)
                P[f"{a}+E_minus_V+E"] = C.paired(oof[f"{a}+E"], bk["V+E"], y, clip, outer, counts)
        out["paired"][fam] = P
    json.dump(out, open(f"{C.RAW}/results_nav2_{pop}.json", "w"), indent=1, default=float)
    for fam in ("linear", "mlp"):
        for k, v in out["paired"][fam].items():
            print(pop, fam, k, f"fold {v['fold']['delta']:+.4f} [{v['fold']['lo']:+.4f},{v['fold']['hi']:+.4f}]  pooled {v['pooled']['delta']:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("desc", "lin", "mlp", "pair"))
    ap.add_argument("--pop", choices=C.POPS)
    a = ap.parse_args()
    print("free MB", C.require_ram(3000, "nav_arm2"), flush=True)
    if a.stage == "desc":
        stage_desc()
    elif a.stage == "pair":
        stage_pair(a.pop)
    else:
        ok = stage_fit(a.pop, "linear" if a.stage == "lin" else "mlp")
        print("complete" if ok else "INCOMPLETE")
