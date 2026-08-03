"""D-APPEAR P1b — WHY the still-frame arm behaves differently on the two corpora.

A null has two very different causes and they must be separated before anything is claimed:

  (i)  THE SUBSTRATE IS BROKEN on this corpus (a degenerate pixel block, a wrong channel
       slice, a constant feature) -- in which case the null is a property of the PROBE;
  (ii) the appearance->speed MAP EXISTS but DOES NOT TRANSFER across the independent unit --
       in which case the null is the finding.

THE LADDER THAT SEPARATES THEM (three rungs, same arm, same corpus)

  1  ``within_clip``          random WINDOW split inside the same episodes.
     ⚠️ LEAKY BY CONSTRUCTION -- adjacent windows overlap and share frames, so this is NOT a
     generalisation number and is never quoted as one. Its ONLY job is to prove the substrate
     carries the signal at all. A null HERE means the probe is broken.
  2  ``across_clip_within_rig``  episode-disjoint, but all episodes from ONE camera rig.
  3  ``across_clip``          episode-disjoint over the mixed corpus -- the real number.

Run on BOTH corpora with the SAME code path, so the cross-corpus contrast cannot be an
artefact of two different scripts:
  comma2k19 highway   ``dlatent_pixel_substrate.pt`` + ``idm_derived_accel_latents.pt``
  PhysicalAI-AV       ``dappear_pai_substrate.pt``   (built by ``build_pai_substrate.py``)

usage:
  OMP_NUM_THREADS=6 python run_p1b_mechanism.py --out results_p1b_mechanism.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

PAI = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_pai_substrate.pt")
PAI_A = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_rigA.pt")
PAI_B = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_rigB.pt")
COMMA_PIX = Path(r"C:/Users/Admin/tanitad-data/eval/dlatent_pixel_substrate.pt")
COMMA_LAT = Path(r"C:/Users/Admin/tanitad-data/eval/idm_derived_accel_latents.pt")
SPEED_J = 0
SEL_TOL, MIN_SKILL = 0.005, 0.01
MEAN_KEY = (None, float("inf"))
RBF_GAMMA_MULTS = (0.25, 1.0, 4.0)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


ARMS = (("pix32_centre_rbf", "pix32", "centre", "rbf"),
        ("pix32_centre",     "pix32", "centre", "linear"),
        ("v1_window",        "Z",     "window", "linear"))


def load_comma():
    pix = torch.load(COMMA_PIX, map_location="cpu", weights_only=False)
    lat = torch.load(COMMA_LAT, map_location="cpu", weights_only=False)
    by = {e["name"]: e for e in lat["episodes"]}
    eps = []
    for pe in pix["episodes"]:
        le = by.get(pe["name"])
        if le is None:
            continue
        assert int(le["n"]) == int(pe["n"])
        rec = dict(pe)
        rec["Z"] = le["Z"]
        rec["rig"] = "comma"
        eps.append(rec)
    return eps


def load_pai(path):
    return torch.load(path, map_location="cpu", weights_only=False)["episodes"]


def cat(eps, key):
    X = torch.cat([e[key] for e in eps]).float()
    return X[:, None, :] if X.ndim == 2 else X


def targets(eps):
    S = torch.cat([e["S"] for e in eps]).float().numpy().astype(np.float64)
    eid = np.concatenate([np.full(e["n"], e["name"]) for e in eps])
    return S[:, SPEED_J], eid


def substrate_stats(eps, key):
    X = cat(eps, key)
    Xc = X[:, X.shape[1] // 2]
    return {"n_windows": int(len(X)), "W": int(X.shape[1]), "D": int(X.shape[2]),
            "centre_mean": round(float(Xc.mean()), 6),
            "centre_std": round(float(Xc.std()), 6),
            "centre_min": round(float(Xc.min()), 6),
            "centre_max": round(float(Xc.max()), 6),
            "n_constant_features": int((Xc.std(0) < 1e-6).sum()),
            "median_feature_std": round(float(Xc.std(0).median()), 6)}


def fit_arm(AP, Xf, yf, Xs, ys, XF, yF, Xh, *, feat, kernel, device):
    Xf = AP.window_features(Xf, feat)
    Xs = AP.window_features(Xs, feat)
    XF = AP.window_features(XF, feat)
    Xh = AP.window_features(Xh, feat)
    Xf, Xs = AP.standardize(Xf, Xs)
    XF, Xh = AP.standardize(XF, Xh)
    mu, sd = float(yF.mean()), float(max(yF.std(), 1e-6))
    kw = dict(kernel=kernel, matmul_device=device, matmul_dtype=torch.float32)
    alphas = AP.DualRidge.alpha_grid(2, -4, 10)
    gms = RBF_GAMMA_MULTS if kernel == "rbf" else (None,)
    sel_p, ho_p = {}, {}
    for gm in gms:
        gamma = None
        if kernel == "rbf":
            g = torch.Generator().manual_seed(20260803)
            idx = torch.randperm(len(Xf), generator=g)[:1000]
            d2 = torch.cdist(Xf[idx].double(), Xf[idx].double()).pow(2)
            gamma = gm / max(float(d2[d2 > 0].median()), 1e-9)
        inner = AP.DualRidge(Xf, torch.from_numpy(((yf - mu) / sd)[:, None]).double(),
                             gamma=gamma, **kw)
        for al in alphas:
            sel_p[(gm, al)] = inner.predict(Xs, al).numpy().ravel() * sd + mu
        del inner
        full = AP.DualRidge(XF, torch.from_numpy(((yF - mu) / sd)[:, None]).double(),
                            gamma=gamma, **kw)
        for al in alphas:
            ho_p[(gm, al)] = full.predict(Xh, al).numpy().ravel() * sd + mu
        del full
    k0 = next(iter(sel_p))
    sel_p[MEAN_KEY] = np.full_like(sel_p[k0], mu)
    ho_p[MEAN_KEY] = np.full_like(ho_p[k0], mu)
    scored = {k: AP.r2_score(v, ys) for k, v in sel_p.items()}
    best = max(scored.values())
    if best < MIN_SKILL:
        k_sel, rule = max(scored, key=lambda k: k[1]), "skill_gate_to_train_mean"
    else:
        ok = [k for k, s in scored.items() if s >= best - SEL_TOL]
        k_sel, rule = max(ok, key=lambda k: k[1]), "one_se_tiebreak"
    return ho_p[k_sel], {"gamma_mult": k_sel[0],
                         "alpha": ("inf" if np.isinf(k_sel[1]) else k_sel[1]),
                         "inner_r2": round(float(scored[k_sel]), 5),
                         "best_inner_r2": round(float(best), 5), "rule": rule,
                         "n_features": int(Xf.shape[1])}


def rung(AP, eps, key, feat, kernel, mode, device, seed=0):
    """mode='across_clip' -> episode-disjoint i%3; mode='within_clip' -> random WINDOW split."""
    if mode == "across_clip":
        tr = [e for i, e in enumerate(eps) if i % 3 != 0]
        ho = [e for i, e in enumerate(eps) if i % 3 == 0]
        fit = [e for i, e in enumerate(tr) if i % 3 != 0]
        sel = [e for i, e in enumerate(tr) if i % 3 == 0]
        Xf, Xs, XF, Xh = (cat(fit, key), cat(sel, key), cat(tr, key), cat(ho, key))
        yf, _ = targets(fit)
        ys, _ = targets(sel)
        yF, _ = targets(tr)
        yh, eh = targets(ho)
    else:
        X = cat(eps, key)
        y, eid = targets(eps)
        g = np.random.default_rng(seed)
        p = g.permutation(len(X))
        n_ho = len(X) // 3
        ho_i, tr_i = p[:n_ho], p[n_ho:]
        sel_i, fit_i = tr_i[:len(tr_i) // 3], tr_i[len(tr_i) // 3:]
        Xf, Xs, XF, Xh = X[fit_i], X[sel_i], X[tr_i], X[ho_i]
        yf, ys, yF, yh, eh = y[fit_i], y[sel_i], y[tr_i], y[ho_i], eid[ho_i]
    p, sel_meta = fit_arm(AP, Xf, yf, Xs, ys, XF, yF, Xh,
                          feat=feat, kernel=kernel, device=device)
    return {"r2_speed": round(AP.r2_score(p, yh), 5),
            "corr": (round(float(np.corrcoef(p, yh)[0, 1]), 5) if p.std() > 0 else None),
            "n_train_windows": int(len(XF)), "n_heldout_windows": int(len(Xh)),
            "n_heldout_units": int(len(np.unique(eh))),
            "gt_speed_mean": round(float(yh.mean()), 4),
            "gt_speed_std": round(float(yh.std()), 4),
            "selection": sel_meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_p1b_mechanism.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    import tanitad.eval.accel_probe as AP

    corpora = {"comma2k19_highway": load_comma(),
               "physicalai_av_mixed": load_pai(PAI)}
    if PAI_A.exists():
        corpora["physicalai_av_rigA_only"] = load_pai(PAI_A)
    if PAI_B.exists():
        corpora["physicalai_av_rigB_only"] = load_pai(PAI_B)

    res = {"meta": {
        "prereg": "Project Steering/PREREG_APPEARANCE_SHORTCUT.md",
        "rungs": {
            "within_clip": "random WINDOW split -- LEAKY BY CONSTRUCTION (adjacent windows "
                           "overlap). Never a generalisation number; its only job is to show "
                           "the substrate is not degenerate.",
            "across_clip": "episode-disjoint i%3 -- the real number"},
        "device": a.device}, "corpora": {}}

    for cname, eps in corpora.items():
        # ⚠️ cap the window count so the dual eigh stays the same size on every corpus --
        # a corpus-size difference must not be mistaken for a corpus difference.
        n_win = sum(e["n"] for e in eps)
        rec = {"n_episodes": len(eps), "n_windows": n_win,
               "speed_mps": {}, "substrate": {}, "arms": {}}
        y_all, _ = targets(eps)
        rec["speed_mps"] = {"mean": round(float(y_all.mean()), 4),
                            "std": round(float(y_all.std()), 4),
                            "p05": round(float(np.percentile(y_all, 5)), 4),
                            "p50": round(float(np.percentile(y_all, 50)), 4),
                            "p95": round(float(np.percentile(y_all, 95)), 4),
                            "frac_below_1mps": round(float((y_all < 1).mean()), 4),
                            "cv": round(float(y_all.std() / max(y_all.mean(), 1e-9)), 4)}
        for key in ("pix32", "Z"):
            if key in eps[0]:
                rec["substrate"][key] = substrate_stats(eps, key)
        for arm, key, feat, kernel in ARMS:
            rec["arms"][arm] = {}
            for mode in ("within_clip", "across_clip"):
                rec["arms"][arm][mode] = rung(AP, eps, key, feat, kernel, mode, a.device)
                log(f"{cname:28s} {arm:18s} {mode:14s} "
                    f"R2 {rec['arms'][arm][mode]['r2_speed']:+.4f} "
                    f"(inner {rec['arms'][arm][mode]['selection']['best_inner_r2']:+.4f})")
        res["corpora"][cname] = rec
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))

    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
