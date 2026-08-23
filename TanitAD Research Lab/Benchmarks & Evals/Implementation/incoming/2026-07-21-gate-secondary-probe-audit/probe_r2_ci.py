"""Split-robust interval for the gate secondary `encoder_speed_probe_r2`.

WHY (gate-audit role, Mission Plan independent-test duty)
--------------------------------------------------------
`Project Steering/Gates/flagship-v3enc-gate-10k-2026-07-21.json` returns
**RESTART** for `flagship-v3enc`. Its primary (held-out ADE@2s 1.9654 <= 2.5)
PASSED; the verdict rests entirely on ONE secondary,
`encoder_speed_probe_r2 = 0.393 < 0.55`.

That number is produced by `taniteval/diag_v2mech.py:ridge_r2`, which

  1. uses ONE deterministic held-out split -- `uniq[::len(uniq)//8][:8]`, i.e.
     every 5th of the 40 val episodes -- so it is a **point estimate with no
     interval** (CLAUDE.md: "never quote an interval without its estimator");
  2. selects the ridge lambda **on the held-out set itself** (optimistic bias);
  3. searches a lambda grid whose **maximum, 10.0, is the value v3enc selected**
     -- a boundary hit, so the reported R^2 may be a lower bound.

This module re-measures the same quantity with the estimator the decision needs:
a repeated random episode-held-out split distribution (an episode-cluster
resampling of the probe), plus nested lambda selection and an extended grid, on
the SAME cached tensors the gate used (`diagv2_<key>.pt` -> `z_t`, `speed`,
`eid`). It computes nothing new about the models; it re-estimates the gate's own
number with its uncertainty.

Fidelity requirement: `orig_split_r2()` reproduces `ridge_r2` EXACTLY (the dual
form is algebraically identical to the primal `A = X^T X + lam*n*I` solve), so
any disagreement with the gate artifact is a real finding, not a re-implementation
artifact.

Pre-registered decision rule (both outcomes committed before running):
  * v3enc P97.5 < 0.55 AND v1 P2.5 > 0.55  -> the secondary is discriminative;
    the RESTART verdict stands on a sound instrument.
  * v3enc P97.5 >= 0.55, OR the split-to-split SD is of the order of the
    0.157 margin (0.55 - 0.393), OR the extended grid lifts v3enc over 0.55
    -> the secondary is a single-split artifact; the verdict is not
    decision-grade AS ADJUDICATED and must be re-run with an interval.

Usage (eval pod, CPU or GPU):
    python3 probe_r2_ci.py --results-dir /root/taniteval/results \
        --keys flagship-30k flagship-speed flagship-v2-6k flagship-v3enc-10k \
        --out probe_r2_ci.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

BASE_GRID = (1e-2, 1e-1, 1.0, 10.0)          # the grid the gate used
EXT_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
N_HELD = 8                                    # held-out episodes per split
GATE_THRESHOLD = 0.55


# --------------------------------------------------------------------------
# core estimator
# --------------------------------------------------------------------------
def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _gram(Xtr, Xte):
    """Precompute the two Gram products a whole lambda sweep reuses."""
    return Xtr @ Xtr.T, Xte @ Xtr.T


def _ridge_predict_gram(K, Kte, ytr, lam):
    n = K.shape[0]
    ym = ytr.mean()
    A = K + lam * n * torch.eye(n, dtype=K.dtype, device=K.device)
    return Kte @ torch.linalg.solve(A, (ytr - ym)) + ym


def _ridge_fit_predict(Xtr, ytr, Xte, lam):
    """Dual-form ridge, algebraically identical to
    `solve(X^T X + lam*n*I, X^T (y - ybar))` used by diag_v2mech.ridge_r2.

    n < p here (n~700 windows, p=2048 latent dims) so the dual n x n solve is
    both faster and better conditioned; it is the SAME estimator.
    """
    K, Kte = _gram(Xtr, Xte)
    return _ridge_predict_gram(K, Kte, ytr, lam)


def _standardize(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def probe_r2(z, y, held_mask, lam_grid=BASE_GRID, nested=False, seed=0):
    """Held-out R^2 of a ridge probe z -> y.

    nested=False reproduces the gate's estimator (lambda picked by best
    held-out R^2 -- selection on the test set).
    nested=True picks lambda on an inner split of the TRAIN episodes only.
    Returns (r2, lam).
    """
    Xtr, Xte = z[~held_mask].double(), z[held_mask].double()
    ytr, yte = y[~held_mask].double(), y[held_mask].double()
    Xtr, Xte = _standardize(Xtr, Xte)
    K, Kte = _gram(Xtr, Xte)
    if not nested:
        best = None
        for lam in lam_grid:
            r2 = _r2(yte, _ridge_predict_gram(K, Kte, ytr, lam))
            if best is None or r2 > best[0]:
                best = (r2, lam)
        return best
    # nested: inner random 25% of TRAIN rows as a validation fold
    g = torch.Generator().manual_seed(seed)
    n = Xtr.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = max(1, n // 4)
    vi, ti = perm[:n_val], perm[n_val:]
    Xi, Xv = _standardize(Xtr[ti], Xtr[vi])
    Ki, Kv = _gram(Xi, Xv)
    best_lam, best_inner = None, None
    for lam in lam_grid:
        r2 = _r2(ytr[vi], _ridge_predict_gram(Ki, Kv, ytr[ti], lam))
        if best_inner is None or r2 > best_inner:
            best_inner, best_lam = r2, lam
    return _r2(yte, _ridge_predict_gram(K, Kte, ytr, best_lam)), best_lam


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def gate_held_episodes(eids):
    """The deterministic split hard-coded in diag_v2mech.ridge_r2."""
    uniq = sorted(set(eids))
    return set(uniq[:: max(1, len(uniq) // N_HELD)][:N_HELD])


def mask_for(eids, held):
    return torch.tensor([e in held for e in eids])


def random_splits(eids, n_splits, n_held=N_HELD, seed=0):
    """n_splits random choices of n_held distinct episodes (episode-cluster
    resampling: windows move with their episode, never across)."""
    uniq = sorted(set(eids))
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_splits):
        idx = torch.randperm(len(uniq), generator=g)[:n_held]
        out.append({uniq[int(i)] for i in idx})
    return out


def summarize(vals):
    t = torch.tensor([v for v in vals if not math.isnan(v)], dtype=torch.double)
    if t.numel() == 0:
        return {"n": 0}
    q = torch.tensor([0.025, 0.25, 0.5, 0.75, 0.975], dtype=torch.double)
    p = torch.quantile(t, q)
    return {
        "n": int(t.numel()),
        "mean": round(float(t.mean()), 4),
        "sd": round(float(t.std()), 4),
        "min": round(float(t.min()), 4),
        "p2.5": round(float(p[0]), 4),
        "p25": round(float(p[1]), 4),
        "median": round(float(p[2]), 4),
        "p75": round(float(p[3]), 4),
        "p97.5": round(float(p[4]), 4),
        "max": round(float(t.max()), 4),
        "frac_ge_gate": round(float((t >= GATE_THRESHOLD).double().mean()), 4),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def audit_model(z, y, eids, n_splits=200, seed=0, device="cpu"):
    z = z.to(device).double()
    y = y.to(device).double()
    res = {"n_windows": int(z.shape[0]), "n_episodes": len(set(eids)),
           "latent_dim": int(z.shape[1])}

    held0 = gate_held_episodes(eids)
    m0 = mask_for(eids, held0).to(device)
    r2_0, lam_0 = probe_r2(z, y, m0, BASE_GRID)
    res["gate_split"] = {"r2": round(r2_0, 4), "lam": lam_0,
                         "n_held_eps": len(held0),
                         "lam_at_grid_max": lam_0 == max(BASE_GRID)}
    r2_ext, lam_ext = probe_r2(z, y, m0, EXT_GRID)
    res["gate_split_extended_grid"] = {"r2": round(r2_ext, 4), "lam": lam_ext}
    r2_nest, lam_nest = probe_r2(z, y, m0, EXT_GRID, nested=True, seed=seed)
    res["gate_split_nested_lambda"] = {"r2": round(r2_nest, 4), "lam": lam_nest}

    splits = random_splits(eids, n_splits, seed=seed)
    base, ext, nest, lams = [], [], [], []
    for i, held in enumerate(splits):
        m = mask_for(eids, held).to(device)
        if int(m.sum()) < 2 or int((~m).sum()) < 2:
            continue
        # one standardize + one Gram per split; the base grid is a subset of
        # the extended one, so both sweeps read the same factorisation.
        Xtr, Xte = _standardize(z[~m].double(), z[m].double())
        ytr, yte = y[~m].double(), y[m].double()
        K, Kte = _gram(Xtr, Xte)
        per_lam = {lam: _r2(yte, _ridge_predict_gram(K, Kte, ytr, lam))
                   for lam in EXT_GRID}
        r2b, lb = max(((per_lam[l], l) for l in BASE_GRID), key=lambda t: t[0])
        base.append(r2b)
        lams.append(lb)
        ext.append(max(per_lam.values()))
        nest.append(probe_r2(z, y, m, EXT_GRID, nested=True, seed=seed + i)[0])
    res["random_splits_base_grid"] = summarize(base)
    res["random_splits_extended_grid"] = summarize(ext)
    res["random_splits_nested_lambda"] = summarize(nest)
    res["lambda_hist_base_grid"] = {
        str(l): int(sum(1 for x in lams if x == l)) for l in BASE_GRID}

    # permutation null: break the z<->v0 correspondence at EPISODE level, so
    # any within-episode leakage that survives is what "chance" really is here.
    uniq = sorted(set(eids))
    g = torch.Generator().manual_seed(seed + 999)
    perm = torch.randperm(len(uniq), generator=g)
    remap = {uniq[i]: uniq[int(perm[i])] for i in range(len(uniq))}
    by_ep = {e: (torch.tensor([j for j, x in enumerate(eids) if x == e]))
             for e in uniq}
    y_null = y.clone()
    for e in uniq:
        src, dst = by_ep[remap[e]], by_ep[e]
        k = min(len(src), len(dst))
        y_null[dst[:k]] = y[src[:k]]
    null_vals = []
    for held in splits[:50]:
        m = mask_for(eids, held).to(device)
        null_vals.append(probe_r2(z, y_null, m, BASE_GRID)[0])
    res["permutation_null_base_grid"] = summarize(null_vals)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/root/taniteval/results")
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="probe_r2_ci.json")
    args = ap.parse_args()

    out = {"estimator": "repeated random 8-of-40 EPISODE held-out split "
                        "(episode-cluster resampling of the probe)",
           "n_splits": args.n_splits, "seed": args.seed,
           "gate_threshold": GATE_THRESHOLD,
           "base_grid": list(BASE_GRID), "extended_grid": list(EXT_GRID),
           "source": "diagv2_<key>.pt cached z_t/speed/eid (the gate's own tensors)",
           "models": {}}
    for key in args.keys:
        path = os.path.join(args.results_dir, f"diagv2_{key}.pt")
        c = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[audit] {key}: z_t={tuple(c['z_t'].shape)}", flush=True)
        out["models"][key] = audit_model(
            c["z_t"].float(), c["speed"].float(), list(c["eid"]),
            n_splits=args.n_splits, seed=args.seed, device=args.device)
        print(json.dumps(out["models"][key], indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[audit] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
