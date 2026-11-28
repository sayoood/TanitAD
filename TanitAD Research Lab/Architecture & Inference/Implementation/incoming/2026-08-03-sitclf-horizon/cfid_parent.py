"""C-FID-PARENT — does THIS pipeline reproduce the sibling stream's banked horizon row?

Pre-registration Sec 4 registers this control: one cell re-run under the parent `run_horizon.py`
row rule and lambda rule, asserted against `../2026-08-03-sitclf-temporal/results_horizon.json` at
`lead 3.0`. A mismatch means this pipeline is not the one that produced the banked table and
NOTHING from the per-situation study may be quoted.

⚠️ It is run BEFORE the long study on purpose. A pipeline check that runs afterwards is a
post-mortem, not a control.

The parent's conventions, replicated verbatim rather than approximated:
  * PCA fitted on `splits[f][0]` = (train AND NOT sel) rows, NO hist_ok restriction;
  * rows scored = `valid(lead) AND hist_ok(WIN=8)`;
  * ONE lambda shared across the three situations, chosen by the MEAN of per-situation AP
    (raw AP, not AP-lift) on the SEL rows;
  * ridge standardisation over ALL fit rows of the fold, validity applied inside `ridge_scores`.

usage:
  python cfid_parent.py --substrate <npz>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.data.situations import (anticipation_target,            # noqa: E402
                                     detect_intersection,
                                     detect_lane_change,
                                     detect_roundabout, kinematics)
from tanitad.eval.ap_ci import (ap_episode_cluster_bootstrap,        # noqa: E402
                                average_precision)
from tanitad.eval.sitclf import (causal_window, clip_runs,           # noqa: E402
                                 cluster_folds, ridge_scores)

SITS = ("lane_change", "roundabout", "intersection")
CACHES = (r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74",
          r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836")
RANK, WIN, LEAD = 16, 8, 3.0
SEL_FRAC = 0.20
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--substrate",
                     default=r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz")
    ap_.add_argument("--parent",
                     default=str(Path(__file__).resolve().parents[1] /
                                 "2026-08-03-sitclf-temporal" / "results_horizon.json"))
    ap_.add_argument("--out", default="cfid_parent.json")
    a = ap_.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(a.substrate)
    F = z["F"]
    cc = z["clip_cluster"]
    st, en = clip_runs(cc)
    N = F.shape[0]
    folds = cluster_folds(cc, 2, seed=0)

    files = []
    for root in CACHES:
        files += sorted(glob.glob(os.path.join(root, "ep_*.pt")))
    per_clip = []
    for f in files:
        P = np.asarray(torch.load(f, map_location="cpu", weights_only=True,
                                  mmap=True)["poses"]).astype(np.float64)
        K = kinematics(P)
        per_clip.append((int(K["T"]), {
            "lane_change": detect_lane_change(K),
            "roundabout": detect_roundabout(K, bracket=True),
            "intersection": detect_intersection(K, cross=None)[0]}))
    if sum(t for t, _ in per_clip) != N:
        raise SystemExit("C-FID: frame count mismatch")

    Yb, Vb = [], []
    for T, ev in per_clip:
        y = np.zeros((T, 3), bool)
        v = np.zeros((T, 3), bool)
        for i, s in enumerate(SITS):
            y[:, i], v[:, i] = anticipation_target(T, ev[s], lead_s=LEAD)
        Yb.append(y)
        Vb.append(v)
    Yb, Vb = np.concatenate(Yb), np.concatenate(Vb)
    _, hist_ok = causal_window(np.zeros((N, 1), np.float32), st, en, WIN)
    V = Vb & hist_ok[:, None]

    splits = []
    for f in (0, 1):
        te = folds == f
        tr = ~te
        tr_cl = np.unique(cc[tr])
        perm = np.random.default_rng(0 + f).permutation(tr_cl)
        sel_cl = set(int(x) for x in perm[:max(1, int(round(SEL_FRAC * len(tr_cl))))])
        is_sel = np.array([int(x) in sel_cl for x in cc])
        splits.append((tr & ~is_sel, tr & is_sel, te))

    def fit_pca(src, rows, r):
        A = torch.from_numpy(np.asarray(src[rows], dtype=np.float32)).to(device)
        mu = A.mean(0, keepdim=True)
        A = A - mu
        _u, _s, v = torch.svd_lowrank(A, q=r + 16, niter=4)
        del A
        if device != "cpu":
            torch.cuda.empty_cache()
        return mu.cpu().numpy(), v[:, :r].cpu().numpy().astype(np.float32)

    def project(src, mu, W):
        out = np.empty((src.shape[0], W.shape[1]), np.float32)
        for b in range(0, src.shape[0], 20000):
            out[b:b + 20000] = (np.asarray(src[b:b + 20000], np.float32) - mu) @ W
        out /= max(float(np.abs(out).mean()), 1e-6)
        return out

    def mean_ap(y, s, v):
        vals = []
        for i in range(y.shape[1]):
            m = v[:, i]
            if m.sum() < 50 or y[m, i].sum() < 5:
                continue
            vals.append(average_precision(y[m, i], s[m, i]))
        return float(np.mean(vals)) if vals else float("nan")

    Y = Yb.astype(np.float32)
    Yi = Yb.astype(np.int64)
    Vf = V.astype(np.float32)
    out = np.zeros_like(Y)
    for f in (0, 1):
        mu, Wp = fit_pca(F, np.flatnonzero(splits[f][0]), RANK)
        X, _ = causal_window(project(F, mu, Wp), st, en, WIN)
        fit, sel, te = splits[f]
        best = (-np.inf, 1.0)
        for lam in LAMBDAS:
            m_ = mean_ap(Yi[sel], ridge_scores(X[fit], Y[fit], Vf[fit], X[sel], lam=lam), V[sel])
            if np.isfinite(m_) and m_ > best[0]:
                best = (m_, lam)
        out[te] = ridge_scores(X[fit], Y[fit], Vf[fit], X[te], lam=best[1])
        log(f"  fold {f}: lambda {best[1]} (mean SEL AP {best[0]:.5f})")

    parent = json.loads(Path(a.parent).read_text(encoding="utf-8"))
    rep = {"_what": "C-FID-PARENT — reproduce the sibling stream's banked lead-3.0 horizon row",
           "parent": str(a.parent), "lead_s": LEAD, "per_situation": {}}
    bad = []
    for i, s in enumerate(SITS):
        m = V[:, i]
        yv = Yi[m, i]
        got_ap = round(average_precision(yv, out[m, i].astype(np.float64)), 5)
        got_lift = ap_episode_cluster_bootstrap(yv, out[m, i].astype(np.float64), cc[m],
                                                n_boot=2000, lift=True)
        want = parent["per_situation"][s][f"lead_{LEAD}"]
        row = {"n_pos": int(yv.sum()), "want_n_pos": want["n_pos"],
               "ap": got_ap, "want_ap": want["ap"],
               "lift": [got_lift["point"], got_lift["lo"], got_lift["hi"]],
               "want_lift": [want["ap_lift"]["point"], want["ap_lift"]["lo"],
                             want["ap_lift"]["hi"]]}
        row["MATCH"] = bool(row["n_pos"] == row["want_n_pos"]
                            and abs(row["ap"] - row["want_ap"]) <= 2e-5
                            and all(abs(x - y) <= 2e-5
                                    for x, y in zip(row["lift"], row["want_lift"])))
        if not row["MATCH"]:
            bad.append(s)
        rep["per_situation"][s] = row
        log(f"  {s:>13}: AP {got_ap} vs {want['ap']} | lift {row['lift']} vs "
            f"{row['want_lift']} -> {'MATCH' if row['MATCH'] else 'MISMATCH'}")
    rep["C_FID_PARENT"] = "PASS" if not bad else f"FAIL on {bad}"
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log(rep["C_FID_PARENT"])
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
