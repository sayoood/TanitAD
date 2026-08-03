"""H-T2's MECHANISM, measured directly: does appearance PCA discard the motion subspace?

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
`run_temporal.py` computes this inline as its STAGE 0, but the FIRST version of that stage was
WRONG in a way that flattered the hypothesis's opposite: it measured the temporal difference's
variance about the APPEARANCE mean instead of about the difference's own mean. The difference has
mean ~0, so subtracting a large appearance mean makes the total dominated by ||mu_appearance||^2 --
a constant offset the appearance basis reproduces almost perfectly by construction. It reported the
appearance basis retaining **0.9520** of the delta variance at rank 16, which is an artefact of the
centring and not a measurement of anything.

This script recomputes the diagnostic with every fraction taken about the OWN MEAN of the matrix
being measured, on the SAME fold-0 fit rows (same seeds, same splits), so the corrected numbers can
replace the run's inline block without re-running the hour-long fit.

⭐ WHAT IT DECIDES. H-T2 says the rank-16 appearance PCA truncates away the frame-to-frame change,
which would mean no head reading that basis can see motion however large it is. That is a claim
about SUBSPACES and it is checkable without any label, any classifier and any AP:

  * if the appearance basis retains most of the DELTA variance, there is no discarded motion
    subspace and H-T2 is refuted MECHANISTICALLY -- a motion-basis arm has nothing to recover;
  * if it retains little, H-T2 has a real mechanism and the motion arms are worth their fit.

usage:
  python subspace_diag.py --substrate <npz> --out results_subspace.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.eval.sitclf import (clip_runs, cluster_folds,          # noqa: E402
                                 temporal_difference)

SEL_FRAC = 0.20               # identical to run_temporal.py, so the rows match


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit_pca(X, rows, r, seed=0, device="cpu"):
    A = torch.from_numpy(np.asarray(X[rows], dtype=np.float32)).to(device)
    A = A - A.mean(0, keepdim=True)
    _u, _s, v = torch.svd_lowrank(A, q=min(r + 16, A.shape[1] - 1), niter=8)
    del A
    if device != "cpu":
        torch.cuda.empty_cache()
    return v[:, :r].cpu().numpy().astype(np.float32)


def mean_of(X, rows, chunk=20000):
    acc = np.zeros(X.shape[1], np.float64)
    for b in range(0, rows.size, chunk):
        acc += np.asarray(X[rows[b:b + chunk]], dtype=np.float64).sum(0)
    return (acc / max(rows.size, 1)).astype(np.float32)[None, :]


def var_explained(X, rows, mu, W, chunk=20000):
    """||(X - mu) W||^2 / ||X - mu||^2 with ``mu`` the mean of X ON THESE ROWS."""
    tot = cap = 0.0
    for b in range(0, rows.size, chunk):
        A = np.asarray(X[rows[b:b + chunk]], dtype=np.float32) - mu
        tot += float(np.einsum("ij,ij->", A, A))
        P = A @ W
        cap += float(np.einsum("ij,ij->", P, P))
    return float(cap / max(tot, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", required=True)
    ap.add_argument("--out", default="results_subspace.json")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(a.substrate)
    F = z["F"]
    cc = z["clip_cluster"]
    st, en = clip_runs(cc)
    folds = cluster_folds(cc, 2, seed=0)

    # fold 0's FIT rows, reproduced exactly from run_temporal.fold_splits(seed=0, f=0)
    te = folds == 0
    tr = ~te
    tr_cl = np.unique(cc[tr])
    perm = np.random.default_rng(0).permutation(tr_cl)
    sel_cl = set(int(x) for x in perm[:max(1, int(round(SEL_FRAC * len(tr_cl))))])
    is_sel = np.array([int(x) in sel_cl for x in cc])
    fit0 = np.flatnonzero(tr & ~is_sel)
    log(f"substrate {F.shape[0]:,} x {F.shape[1]}  fold-0 fit rows {fit0.size:,}  device={device}")

    out = {"_what": "H-T2 mechanism: does appearance PCA discard the motion subspace?",
           "_supersedes": ("results_temporal.json -> controls.H_T2_SUBSPACE_DIAGNOSTIC, whose "
                           "DELTA_var_in_APPEARANCE_basis was computed about the APPEARANCE mean "
                           "and is an artefact of that centring (0.9520 at rank 16)"),
           "substrate": str(a.substrate), "n_fit_rows": int(fit0.size),
           "centring": "every fraction is variance about the OWN MEAN of the matrix measured",
           "ranks": {}}

    for k in (1, 3):
        d, ok = temporal_difference(F, st, en, k=k)
        fit_d = fit0[ok[fit0]]
        mu_F, mu_D = mean_of(F, fit0), mean_of(d, fit_d)
        for r in (16, 64, 256):
            Wa = fit_pca(F, fit0, r, device=device)
            Wm = fit_pca(d, fit_d, r, device=device)
            cos = np.linalg.svd(Wa.T @ Wm, compute_uv=False)
            row = {
                "delta_lag_frames": k,
                "delta_lag_s": round(k / 10.0, 2),
                "appearance_var_in_APPEARANCE_basis": round(var_explained(F, fit0, mu_F, Wa), 5),
                "appearance_var_in_MOTION_basis": round(var_explained(F, fit0, mu_F, Wm), 5),
                "DELTA_var_in_APPEARANCE_basis": round(var_explained(d, fit_d, mu_D, Wa), 5),
                "DELTA_var_in_MOTION_basis": round(var_explained(d, fit_d, mu_D, Wm), 5),
                "mean_principal_cosine": round(float(cos.mean()), 5),
                "min_principal_cosine": round(float(cos.min()), 5),
                "n_rows_with_delta": int(fit_d.size)}
            row["delta_variance_LOST_by_using_the_appearance_basis"] = round(
                row["DELTA_var_in_MOTION_basis"] - row["DELTA_var_in_APPEARANCE_basis"], 5)
            out["ranks"][f"k{k}_rank{r}"] = row
            log(f"  k={k} rank={r}: DELTA variance kept -- appearance "
                f"{row['DELTA_var_in_APPEARANCE_basis']:.4f} vs motion "
                f"{row['DELTA_var_in_MOTION_basis']:.4f} "
                f"(motion advantage {row['delta_variance_LOST_by_using_the_appearance_basis']:+.4f}); "
                f"appearance variance kept -- appearance "
                f"{row['appearance_var_in_APPEARANCE_basis']:.4f} vs motion "
                f"{row['appearance_var_in_MOTION_basis']:.4f}; "
                f"mean/min principal cos {row['mean_principal_cosine']:.4f}/"
                f"{row['min_principal_cosine']:.4f}")
        del d

    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
