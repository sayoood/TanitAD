"""P1 lead-gap FOLLOW-UP (WM_PHYSICS_PROOF, pre-registered 2026-08-11): which of
three hypotheses explains R2(enc) <= 0 on METRIC lead gap with a LINEAR probe?

  H-param  : the latent codes NEARNESS, not metres  -> log1p / inverse / TTC-proxy
             transforms become linearly readable (R2(enc) >= 0.30 at k=10).
  H-nonlin : the information is present but non-linear -> a 2-layer MLP ceiling
             reads it (MLP R2(enc) >= 0.30) while every linear transform fails.
  H-absent : nothing reads it -> MODEL verdict "missing state variable"
             (consequences pre-registered in the battery doc).

Runs CPU-only off the banked ``probe_arrays.pt`` (no new GPU rolls; the same
windows, episode-disjoint folds, and NaN-exclusion discipline as the parent
run). TTC here is the PROXY gap / max(v_ego, 0.5) — stationary-lead assumption,
stated; a closing-speed TTC needs lead velocity, which the dump does not carry.

Usage:
  python3 scripts/p1_lead_transforms.py \
      --arrays /workspace/experiments/p1-rerun-clsfilter/probe_arrays.pt \
      --out /workspace/experiments/p1-lead-transforms
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

GATE_R2 = 0.30          # pre-registered: a transform/probe "reads" the variable
GATE_K = 10             # the P1 gate horizon
RETENTION = 0.85        # P1's retention rule, applied on the WINNING transform
MLP_HIDDEN = 64
MLP_EPOCHS = 300
MLP_LR = 1e-2
MLP_WD = 1e-4
RIDGE_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def out_of_fold_r2(preds: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def ridge_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    """Out-of-fold ridge R2, lambda chosen per fold on the TRAIN split."""
    preds = np.zeros_like(y)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        Xtr, ytr = X[tr], y[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_n, Xte_n = (Xtr - mu) / sd, (X[te] - mu) / sd
        best, best_pred = -np.inf, None
        n_val = max(1, int(0.2 * len(ytr)))
        for lam in RIDGE_LAMBDAS:
            A = Xtr_n[:-n_val].T @ Xtr_n[:-n_val] + lam * np.eye(X.shape[1])
            w = np.linalg.solve(A, Xtr_n[:-n_val].T @ ytr[:-n_val])
            r2v = out_of_fold_r2(Xtr_n[-n_val:] @ w, ytr[-n_val:])
            if r2v > best:
                best, best_pred = r2v, Xte_n @ w
        preds[te] = best_pred
    return out_of_fold_r2(preds, y)


def mlp_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
            seed: int = 0) -> float:
    """Out-of-fold 2-layer MLP R2 — the capability CEILING, never the headline."""
    preds = np.zeros_like(y)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        torch.manual_seed(seed)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xt = torch.from_numpy((X[tr] - mu) / sd).float()
        yt = torch.from_numpy(y[tr]).float()[:, None]
        ym, ys = float(yt.mean()), float(yt.std()) + 1e-8
        yt = (yt - ym) / ys
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], MLP_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(MLP_HIDDEN, 1))
        opt = torch.optim.Adam(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
        for _ in range(MLP_EPOCHS):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            Xe = torch.from_numpy((X[te] - mu) / sd).float()
            preds[te] = (net(Xe)[:, 0] * ys + ym).numpy()
    return out_of_fold_r2(preds, y)


def transforms_of(gap: np.ndarray, v_ego: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "gap_m": gap,
        "log1p_gap": np.log1p(gap),
        "inv_gap": 1.0 / (1.0 + gap),
        "ttc_proxy": gap / np.maximum(v_ego, 0.5),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("p1_lead_transforms", description=__doc__)
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    d = torch.load(a.arrays, map_location="cpu", weights_only=False)
    folds_all = d["fold_ids_episode_disjoint"].numpy()
    ks = d["meta"]["ks"]
    res: dict = {
        "item": "P1 lead-gap transform + capability-ceiling probes",
        "gates": {"reads_r2": GATE_R2, "gate_k": GATE_K,
                  "retention": RETENTION},
        "estimator": "pooled out-of-fold R2, episode-disjoint folds (parent "
                     "run's fold ids); registry claims need the episode-"
                     "cluster bootstrap on top",
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
        "per_k": {},
    }
    for k in ks:
        gap = d["targets"]["lead_gap"][k].numpy().astype(np.float64)
        v_ego = d["targets"]["speed"][k].numpy().astype(np.float64)
        m = np.isfinite(gap) & np.isfinite(v_ego)
        row: dict = {"n": int(m.sum())}
        if m.sum() < 60:
            row["reason"] = "n too small — reported, not judged"
            res["per_k"][k] = row
            continue
        folds = folds_all[m]
        for arm, key in (("enc", "z_enc"), ("pred", "z_hat")):
            X = d[key][k].numpy().astype(np.float64)[m]
            for tname, ty in transforms_of(gap[m], v_ego[m]).items():
                row[f"r2_linear_{tname}_{arm}"] = round(
                    ridge_oof(X, ty, folds), 4)
            row[f"r2_mlp_gap_{arm}"] = round(
                mlp_oof(X, gap[m], folds, seed=a.seed), 4)
        res["per_k"][k] = row

    g = res["per_k"].get(GATE_K, {})
    lin_enc = {t: g.get(f"r2_linear_{t}_enc", float("-inf"))
               for t in ("gap_m", "log1p_gap", "inv_gap", "ttc_proxy")}
    best_t = max(lin_enc, key=lin_enc.get)
    mlp_enc = g.get("r2_mlp_gap_enc", float("-inf"))
    if lin_enc[best_t] >= GATE_R2:
        rp = g.get(f"r2_linear_{best_t}_pred", float("-inf"))
        verdict = {"hypothesis": "H-param CONFIRMED",
                   "winning_transform": best_t,
                   "r2_enc": lin_enc[best_t], "r2_pred": rp,
                   "retention_pass": bool(rp >= RETENTION * lin_enc[best_t]),
                   "read": "the latent codes lead NEARNESS in this form; the "
                           "P1 lead target is measurable — retention gate "
                           "applies on this transform"}
    elif mlp_enc >= GATE_R2:
        verdict = {"hypothesis": "H-nonlin CONFIRMED", "r2_mlp_enc": mlp_enc,
                   "linear_best": {best_t: lin_enc[best_t]},
                   "read": "lead distance is carried NON-linearly — P1 lead "
                           "row gets an MLP-tier annotation; linear-probe "
                           "interpretability claim does not hold for it"}
    else:
        verdict = {"hypothesis": "H-absent SUPPORTED",
                   "linear_best": {best_t: lin_enc[best_t]},
                   "r2_mlp_enc": mlp_enc,
                   "read": "MODEL verdict: the latent lacks a readable lead-"
                           "distance variable. Pre-registered consequences: "
                           "headway/TTC stay GT-join instruments; auxiliary "
                           "lead-readout loss is a v6/stage-B lever; P8's "
                           "decoded-BEV lead read-off is the convergent test"}
    res["verdict_at_gate_k"] = verdict
    with open(os.path.join(a.out, "p1_lead_transforms.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(verdict, indent=1))
    print("P1LT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
