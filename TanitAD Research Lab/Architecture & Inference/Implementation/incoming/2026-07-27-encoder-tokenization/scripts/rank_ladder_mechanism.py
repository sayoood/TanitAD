"""R2 — is the rank-16 "dose-response" a property of VISION, or of the READER?

R1 (`rank16_reanalysis.py`) established two facts from the banked held-out scores:
  * `ego + img_pca16` is NOT separated from `ego` alone (+0.00085 [-0.02204,
    +0.02299]) -- the "peak at k=16" is indistinguishable from vision adding zero;
  * the IMAGE-ONLY ladder is FLAT (k16/k64/k256/raw2048 identical to 5 d.p.),
    so rank is not an information property of the visual state;
  * yet the CONCATENATED ladder's decline IS separated.

That combination has one natural explanation: the decline is produced by the
READER -- a single shared ridge penalty over a concatenated [ego | image] design
matrix, fit on 198 train clip-clusters -- and not by the image content.

THIS SCRIPT TESTS THAT EXPLANATION DIRECTLY, because the real 2048-d features
live on a pod (`/workspace/h2probe/feats`) and every pod is busy. It is a
SYNTHETIC REPRODUCTION at the true shapes, and it is declared as such: it cannot
tell us anything about PhysicalAI, only about the estimator applied to it.

  ARM `noise`   the image block is PURE NOISE -- zero mutual information with y.
                >>> THE DISCRIMINATOR. If the monotone collapse reproduces here,
                the ladder's SHAPE cannot be evidence about visual content,
                because here there IS no visual content.
  ARM `signal`  the image block carries a weak real signal, calibrated to land
                near the measured image-alone 1.89x base. The both-directions
                control: the harness must be able to show a block that helps.
  ARM `perblock` identical to `noise`/`signal` but with a SEPARATE ridge penalty
                for the ego block and the image block. If the collapse vanishes
                under per-block lambda, the mechanism is pinned exactly: it is
                one shared penalty, which is a fixable reader defect and NOT an
                architectural fact about the world model.

Shapes are taken from the real experiment (t1_probe.json):
    train 31,032 rows / 198 clip clusters / base rate 0.02694
    heldout 50,119 rows / 322 clip clusters / base rate 0.03276
Lambda grid, AP definition, and grouped-CV selection are the source stream's.

Run: python rank_ladder_mechanism.py --out ../artifacts/rank_ladder_mechanism.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))

N_TR, C_TR, BASE_TR = 31032, 198, 0.02694
N_HO, C_HO, BASE_HO = 50119, 322, 0.03276
D_EGO = 16
KS = (16, 64, 256, 2048)
LAMS = 10.0 ** np.arange(-1, 7.5, 0.5)
RATIOS = 10.0 ** np.arange(-2, 3, 1.0)        # lam_image / lam_ego, 4 decades
FOLDS = 5
# The per-block arm needs ONE eigendecomposition PER RATIO per fold, so it is
# len(RATIOS)x the shared-lambda cost. It is a MECHANISM refinement, not the
# discriminator (that is `noise_sharedlam`), so it is capped at k <= 256 where
# the shared-lambda collapse is already fully developed and separated in the
# REAL data (+k16 - +k256 = +0.05142 [+0.01701, +0.08792]).
K_MAX_PERBLOCK = 256


def average_precision(y, s):
    """Verbatim from 2026-07-26-situation-semantics/scripts/t1_probe.py."""
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yt = y[o]
    tp = np.cumsum(yt)
    fp = np.cumsum(1.0 - yt)
    P = tp / np.maximum(tp + fp, 1e-12)
    R = tp / yt.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], R])) * P))


def make_side(rng, n, n_clusters, base, ego_w, img_w, k_max):
    """Clustered design: a per-cluster random effect makes rows within a clip
    dependent, exactly as real frames are. `ego` carries `ego_w` of the logit,
    `img` carries `img_w`."""
    cl = np.sort(rng.integers(0, n_clusters, size=n))
    ego = rng.standard_normal((n, D_EGO)).astype(np.float32)
    img = rng.standard_normal((n, k_max)).astype(np.float32)
    # cluster random effect injected into BOTH blocks (shared nuisance structure)
    ce = rng.standard_normal((n_clusters, D_EGO)).astype(np.float32)
    ego += 0.7 * ce[cl]
    # the true logit: a fixed direction in ego, and (optionally) one in img
    w_e = rng.standard_normal(D_EGO).astype(np.float32)
    w_e /= np.linalg.norm(w_e)
    w_i = rng.standard_normal(k_max).astype(np.float32)
    w_i /= np.linalg.norm(w_i)
    lin = ego_w * (ego @ w_e) + img_w * (img @ w_i)
    lin = (lin - lin.mean()) / (lin.std() + 1e-9)
    thr = np.quantile(lin, 1.0 - base)
    y = (lin + 0.35 * rng.standard_normal(n) > thr).astype(np.float32)
    return ego, img, y, cl


class RidgePath:
    """One Gram + one eigendecomposition, then EVERY lambda is O(d^2).

    Same estimator as a per-lambda `solve`, just not recomputed: for a diagonal
    penalty P, A = S + P.  With P = s * diag(p) (p the per-block RATIO pattern,
    s the overall scale) we can whiten by p once --
        A = Q^T (Sw + s I) Q,   Sw = Q^-T S Q^-1,  Q = diag(sqrt(p))
    -- eigendecompose Sw ONCE, and then sweep s for free. The per-block arm
    therefore costs one eigh per RATIO, not one solve per (lam_a, lam_b) pair.
    The bias column is left unpenalised by giving it a tiny p and is handled by
    centring y, exactly as an unpenalised intercept would be.
    """

    def __init__(self, X, y, blocks=None, ratio=1.0):
        Xc = np.asarray(X, np.float64)
        self.mu_y = float((2.0 * y - 1.0).mean())
        yy = (2.0 * y - 1.0).astype(np.float64) - self.mu_y
        S = Xc.T @ Xc
        b = Xc.T @ yy
        d = S.shape[0]
        p = np.ones(d)
        if blocks is not None:
            da, db = blocks
            p[da:da + db] = ratio            # image block penalised by lam*ratio
        self.p = p
        q = np.sqrt(p)
        Sw = S / np.outer(q, q)
        self.evals, self.evecs = np.linalg.eigh(Sw)
        self.bw = self.evecs.T @ (b / q)
        self.q = q

    def w(self, lam):
        z = self.bw / (self.evals + lam)
        return (self.evecs @ z) / self.q


def score(X, w, mu_y=0.0):
    return np.asarray(X, np.float64) @ w + mu_y


def zfit(X):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    return mu, sd


def select_lam(Xtr, ytr, cl, blocks=None, per_block=False):
    """Grouped 5-fold CV by cluster, the source stream's selection rule.

    Shared-lambda: one ratio (1.0). Per-block: sweep RATIOS, and for each ratio
    sweep the scale along the eigen-path -- so the per-block arm searches
    lam_image / lam_ego over 4 decades at the cost of 9 eigendecompositions."""
    uc = np.unique(cl)
    fold = {c: i % FOLDS for i, c in enumerate(uc)}
    fid = np.array([fold[c] for c in cl])
    ratios = RATIOS if per_block else (1.0,)
    best, best_ap = None, -1.0
    paths = {}
    for r in ratios:
        for f in range(FOLDS):
            tr = fid != f
            if ytr[tr].sum() == 0 or ytr[fid == f].sum() == 0:
                continue
            mu, sd = zfit(Xtr[tr])
            paths[(r, f)] = (RidgePath((Xtr[tr] - mu) / sd, ytr[tr],
                                       blocks if per_block else None, r), mu, sd)
    for r in ratios:
        for lam in LAMS:
            oof = np.zeros(len(ytr))
            hit = False
            for f in range(FOLDS):
                if (r, f) not in paths:
                    continue
                pth, mu, sd = paths[(r, f)]
                va = fid == f
                oof[va] = score((Xtr[va] - mu) / sd, pth.w(lam), pth.mu_y)
                hit = True
            if not hit:
                continue
            ap = average_precision(ytr, oof)
            if ap > best_ap:
                best_ap, best = ap, (float(lam), float(lam * r))
    return best, best_ap


def run_arm(rng, img_w, per_block, tag):
    ks = [k for k in KS if not per_block or k <= K_MAX_PERBLOCK]
    kmax = max(KS)
    egoT, imgT, yT, clT = make_side(rng, N_TR, C_TR, BASE_TR, 1.0, img_w, kmax)
    egoH, imgH, yH, _clH = make_side(rng, N_HO, C_HO, BASE_HO, 1.0, img_w, kmax)
    base = float(yH.mean())
    rows = {}

    def eval_design(Xtr, Xho, blocks):
        (la, lb), _cv = select_lam(Xtr, yT, clT, blocks, per_block)
        mu, sd = zfit(Xtr)
        r = (lb / la) if per_block else 1.0
        pth = RidgePath((Xtr - mu) / sd, yT, blocks if per_block else None, r)
        s = score((Xho - mu) / sd, pth.w(la), pth.mu_y)
        return average_precision(yH, s) / base, (la, lb)

    r, lam = eval_design(egoT, egoH, None)
    rows["ego_alone"] = {"AP_over_base": round(r, 4), "dim": D_EGO, "lam": lam}
    for k in ks:
        Xtr = np.concatenate([egoT, imgT[:, :k]], 1)
        Xho = np.concatenate([egoH, imgH[:, :k]], 1)
        r, lam = eval_design(Xtr, Xho, (D_EGO, k))
        rows[f"ego+img_k{k}"] = {"AP_over_base": round(r, 4),
                                 "dim": D_EGO + k, "lam": lam}
    r, lam = eval_design(imgT[:, :16], imgH[:, :16], None)
    rows["img_k16_alone"] = {"AP_over_base": round(r, 4), "dim": 16, "lam": lam}
    if not per_block:
        r, lam = eval_design(imgT, imgH, None)
        rows["img_k2048_alone"] = {"AP_over_base": round(r, 4), "dim": kmax,
                                   "lam": lam}
    seq = [rows["ego_alone"]["AP_over_base"]] + \
          [rows[f"ego+img_k{k}"]["AP_over_base"] for k in ks]
    return {"arm": tag, "img_signal_weight": img_w, "per_block_lambda": per_block,
            "ks_evaluated": ks,
            "heldout_base_rate": round(base, 5), "rows": rows,
            "ladder_ego_then_ks": seq,
            "monotone_decline_after_peak": bool(
                all(seq[i] >= seq[i + 1] - 1e-9 for i in range(1, len(seq) - 1))),
            "collapse_ratio_k16_over_kmax": round(seq[1] / max(seq[-1], 1e-9), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "artifacts",
                                                  "rank_ladder_mechanism.json"))
    args = ap.parse_args()
    res = {
        "what": ("Is the rank-16 dose-response a property of VISION or of the "
                 "READER? Synthetic reproduction at the real shapes."),
        "DECLARED_LIMITATION": ("SYNTHETIC. The real 2048-d features are on a busy "
                                "pod. This says nothing about PhysicalAI content; "
                                "it characterises the ESTIMATOR at these shapes."),
        "shapes": {"train_rows": N_TR, "train_clusters": C_TR,
                   "heldout_rows": N_HO, "heldout_clusters": C_HO,
                   "d_ego": D_EGO, "ks": list(KS)},
        "arms": {},
    }
    for tag, img_w, per_block in (
            ("noise_sharedlam", 0.0, False),
            ("signal_sharedlam", 0.45, False),
            ("noise_perblocklam", 0.0, True),
            ("signal_perblocklam", 0.45, True)):
        rng = np.random.default_rng(SEED)
        res["arms"][tag] = run_arm(rng, img_w, per_block, tag)
        a = res["arms"][tag]
        print(f"{tag:22s} ladder(ego,+k...) = "
              f"{[f'{v:.3f}' for v in a['ladder_ego_then_ks']]}  "
              f"collapse={a['collapse_ratio_k16_over_kmax']:.2f}x  "
              f"monotone={a['monotone_decline_after_peak']}")

    n = res["arms"]["noise_sharedlam"]
    res["VERDICT"] = {
        "noise_block_reproduces_the_collapse": n["monotone_decline_after_peak"],
        "noise_collapse_ratio": n["collapse_ratio_k16_over_kmax"],
        "published_collapse_ratio_k16_over_k256": round(3.685 / 2.116, 3),
        "reading": ("If a ZERO-INFORMATION image block reproduces the monotone "
                    "collapse, the ladder's shape is not evidence about visual "
                    "content or about the predictor's dimensional need."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
