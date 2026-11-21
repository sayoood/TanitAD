#!/usr/bin/env python3
"""GS-9 — the TRANSITION-LEVEL rung of the probe ladder: predict Δx from Δz.

⭐ THE QUESTION (H-GS9-1). Our L1–L3 ladder probes STATES: ``z_t → attributes``
(``V7_LAUNCH_GATE.md`` P5; ``MODEL_REGISTRY.md`` §13.0d). Delta-JEPA (arXiv 2606.31232,
§"Physical and State-Delta Probing", Table 5) probes TRANSITIONS: ``Δz_t = z_{t+1} − z_t``
→ ``Δx_t = x_{t+1} − x_t``, train/test split BY TRAJECTORY, ~20k consecutive pairs, linear
and MLP probes. That is where P5/L3 — *does the PREDICTOR add anything over z_t?* —
actually lives, because a predictor that restates the scene has a Δẑ that carries no
transition. This tool adds that rung with the programme's own probe-panel rules
(CLAUDE.md: constant-only control reading the no-information value EXACTLY, a raw-input
floor, a time-shuffled control, n and d printed, every hyper-parameter fitted on the FIT
split only).

WHAT IS PROBED (inputs → the same 4 targets)
--------------------------------------------
Targets, ALL derived from ``poses`` [x, y, yaw, v] (the v2ep cache) over ONE tick
(k = 1, 0.1 s — the predictor's only trained horizon):
    dx_fwd   forward displacement in the ego frame at t   (m)
    dy_left  lateral displacement in the ego frame at t   (m)
    dyaw     wrapped heading change                       (rad)
    dv       speed change, ``v_{t+1} − v_t``, from the pose speed column (m/s)
⛔ ``v`` is REALISED MOTION (r 0.9988 with the pose change). It is NEVER a probe INPUT
here; it appears only as the TARGET ``dv`` — derived from poses — and, unavoidably, as the
third channel of the action the PREDICTOR receives (the model's own design, ``_lift3``).
That is why the panel carries ``dzhat_zero`` and ``dzhat_anch``: they separate what the
predictor's transition carries WITHOUT its action from what the action adds.

Inputs (per row t; W = the predictor's window; all encoder/predictor quantities from the
checkpoint under test):
    const        ones — the no-information control; reads skill = 0.0000 EXACTLY
    pixdelta     raw-input FLOOR: ``pix_{t+1} − pix_t`` of the newest frame at 8×20 gray
    act2         ``[steer_t, accel_t]`` — the fed action WITHOUT v (the echo reference:
                 a latent that does not beat the action it was fed carries nothing beyond it)
    z_t          the encoder state (a STATE rung, here for the marginal)
    dz_enc       ``z_{t+1} − z_t``                  — the paper's Table-5 object (encoder only)
    dzhat_true   ``ẑ_{t+1}(a_true) − z_t``          — the predictor's transition, true actions
    dzhat_zero   ``ẑ_{t+1}(a=0) − z_t``             — the predictor's transition, zero action
    dzhat_anch   ``ẑ_{t+1}(a_true) − ẑ_{t+1}(0)``   — the anchored action response (GS-8's d)
    z_t+dz_enc / z_t+dzhat_true — concatenations, so the MARGINAL of the transition over the
                 state can be read paired (clip-resampled).

THE ESTIMATOR
-------------
Ridge with an UNPENALISED intercept (centre on the FIT split; C92/C97 family), columns
standardised on FIT statistics. λ selected by clip-grouped inner K-fold on the FIT clips
only (never a random-row split — RETRACTION_LOG 2026-08-22 panel). Outer protocol:
K-fold BY CLIP; every clip is scored exactly once out-of-fold, so the score set is
episode-disjoint from every fit that scored it. PCA (optional, ``--pca``) is fitted on the
FIT split only. Reported per cell: n_fit / n_score rows, d, λ per fold, and per target
    skill = 1 − SSE(pred) / SSE(fit-split-mean predictor)   (pooled over OOF rows)
    r     = Pearson(pred, y) over OOF rows
with a CLIP-CLUSTER BOOTSTRAP CI on skill (resample clips with replacement, recompute the
pooled ratio; the construction of ``taniteval/ci.py``, applied to a ratio-of-sums
statistic) and PAIRED clip-resampled deltas for the marginals.

CONTROLS
--------
    constant        must read 0.0000 EXACTLY (by construction of the skill score)
    global shuffle  targets permuted across ALL rows: skill must sit in the null band (~0)
    within-clip shuffle  targets permuted WITHIN each clip: what survives is the CLIP-LEVEL
                    marginal (a scene-class cue), what dies is the TRANSITION-specific part;
                    the transition-specific skill is reported as real − within-shuffled
    raw floor       pixdelta; a learned input that does not beat it has added nothing

⛔ A NEGATIVE HERE IS A NEGATIVE ABOUT THE LINEAR MAP, not about learnability (CLAUDE.md,
linear-probe rule). ``--mlp`` adds a small ridge-regularised random-feature (RFF) probe as
the non-linear comparison, with the same controls; the paper's MLP column is that.

TIER ``T0-DIAGNOSTIC``. Not a driving claim; four families REFUSED with the reason stamped.
PROVENANCE (MM-C12): stack preflighted, ``tanitad.__file__`` and ckpt md5 stamped.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import pathlib
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE_PARENT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_TE_PARENT)
for _pth in (os.path.join(_REPO, "stack"),
             os.path.join(_REPO, "stack", "scripts"), _TE_PARENT):
    if os.path.isdir(_pth) and _pth not in sys.path:
        sys.path.insert(0, _pth)

EVAL_TIER = "T0-DIAGNOSTIC"
HYPOTHESIS = "H-GS9-1"
PAPER = "2606.31232 (Delta-JEPA), Table 5 / §Physical and State-Delta Probing"

N_STACK = 3
F_FRAMES = 100            # frames per clip (the E-DEC-59 / latentmotion instrument's F)
DT = 0.1                  # s per tick (10 Hz)
PIX_HW = (8, 20)          # raw floor geometry (gray) -> 160 dims
TARGETS = ("dx_fwd", "dy_left", "dyaw", "dv")
LAMBDA_GRID = tuple(float(x) for x in np.logspace(-4, 4, 17))   # x mean-diag of XᵀX
K_OUTER = 5
K_INNER = 5
N_BOOT = 1000
FEATURE_ORDER = ("const", "pixdelta", "act2", "z_t", "dz_enc", "dzhat_true", "dzhat_zero",
                 "dzhat_anch", "z_t+dz_enc", "z_t+dzhat_true")
MARGINALS = (("z_t+dz_enc", "z_t"), ("z_t+dzhat_true", "z_t"),
             ("dzhat_true", "dz_enc"), ("dzhat_true", "dzhat_zero"),
             ("dz_enc", "pixdelta"), ("dzhat_true", "pixdelta"), ("dzhat_true", "act2"))


def _p(*a):
    print(*a, flush=True)


def md5_of(path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# =====================================================================================
# targets from poses — numpy only
# =====================================================================================
def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def targets_from_poses(poses, k=1) -> np.ndarray:
    """``poses`` [T, 4] = (x, y, yaw, v) → [T-k, 4] = (dx_fwd, dy_left, dyaw, dv), the
    world displacement rotated into the ego frame at t. ``dv`` comes from the pose speed
    column — the ONLY place v enters this probe, and only as a target."""
    P = np.asarray(poses, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < 4:
        raise ValueError(f"poses must be [T, >=4] (x, y, yaw, v), got {P.shape}")
    d = P[k:, :2] - P[:-k, :2]
    yaw = P[:-k, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    dx_fwd = c * d[:, 0] + s * d[:, 1]
    dy_left = -s * d[:, 0] + c * d[:, 1]
    dyaw = wrap(P[k:, 2] - P[:-k, 2])
    dv = P[k:, 3] - P[:-k, 3]
    return np.stack([dx_fwd, dy_left, dyaw, dv], axis=1)


# =====================================================================================
# ridge with unpenalised intercept, FIT-split standardisation, grouped λ selection
# =====================================================================================
class _RidgeBasis:
    """One eigendecomposition of XᵀX (centred, standardised on the fit split) serves
    every λ and every target set — the shuffled controls change only Y."""

    def __init__(self, X_fit):
        X = np.asarray(X_fit, dtype=np.float64)
        self.mu = X.mean(0)
        sd = X.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        Xc = (X - self.mu) / self.sd
        self.n, self.d = Xc.shape
        G = Xc.T @ Xc
        # mean diag of XᵀX: λ is relative to the data scale. ⛔ An all-constant X (the
        # `const` control) has trace 0; the guard keeps λ > 0 so β is exactly 0 and the
        # prediction is exactly the fit mean — which is what makes the control read 0.0000
        # exactly instead of NaN.
        tr = float(np.trace(G) / max(self.d, 1))
        self.scale = tr if tr > 0 else 1.0
        w, V = np.linalg.eigh(G)
        self.w = np.clip(w, 0.0, None)
        self.V = V
        self.XcT = Xc.T
        self.Xc = Xc

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sd

    def coef(self, Y_fit, lam_rel):
        """β for every λ in ``lam_rel`` (relative to mean diag). Y_fit is centred here
        (the intercept is the fit mean — unpenalised by construction)."""
        Y = np.asarray(Y_fit, dtype=np.float64)
        ymu = Y.mean(0)
        R = self.V.T @ (self.XcT @ (Y - ymu))                  # [d, T]
        lams = np.atleast_1d(np.asarray(lam_rel, dtype=np.float64)) * self.scale
        betas = np.stack([self.V @ (R / (self.w + L)[:, None]) for L in lams])  # [L, d, T]
        return betas, ymu

    def predict(self, X, beta, ymu):
        return self.transform(X) @ beta + ymu


def grouped_folds(groups, k, seed=0):
    """K folds over UNIQUE group ids (clips), shuffled with a seed. Returns a list of
    arrays of group ids per fold."""
    uniq = np.unique(np.asarray(groups))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    k = max(2, min(k, len(uniq)))
    return [uniq[order[i::k]] for i in range(k)]


def select_lambda(X_fit, Y_fit, groups_fit, lambdas=LAMBDA_GRID, k_inner=K_INNER, seed=0):
    """λ by clip-grouped inner CV on the FIT split ONLY. Pooled SSE over targets
    standardised with FIT statistics. Returns (λ_rel, inner-CV curve)."""
    X = np.asarray(X_fit, dtype=np.float64)
    Y = np.asarray(Y_fit, dtype=np.float64)
    g = np.asarray(groups_fit)
    ysd = Y.std(0)
    ysd = np.where(ysd > 1e-12, ysd, 1.0)
    lams = np.asarray(lambdas, dtype=np.float64)
    sse = np.zeros(len(lams))
    for hold in grouped_folds(g, k_inner, seed=seed + 17):
        te = np.isin(g, hold)
        tr = ~te
        if tr.sum() < 2 or te.sum() < 1:
            continue
        basis = _RidgeBasis(X[tr])
        betas, ymu = basis.coef(Y[tr], lams)
        Xte = basis.transform(X[te])
        for j in range(len(lams)):
            pred = Xte @ betas[j] + ymu
            sse[j] += float((((pred - Y[te]) / ysd) ** 2).sum())
    best = int(np.argmin(sse))
    return float(lams[best]), {"lambdas_rel": lams.tolist(), "inner_sse": sse.tolist(),
                               "argmin_at_grid_edge": bool(best in (0, len(lams) - 1))}


def pca_fit(X_fit, n_comp):
    X = np.asarray(X_fit, dtype=np.float64)
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:n_comp]


def crossfit(X_clips, Y_clips, *, k_outer=K_OUTER, lambdas=LAMBDA_GRID, k_inner=K_INNER,
             seed=0, pca=0, y_variants=None):
    """Out-of-fold predictions for every clip. ``y_variants`` maps name → list of
    per-clip Y arrays (the shuffled controls) scored through the IDENTICAL fits —
    λ is selected on the REAL fit targets and reused, PCA/standardisation likewise.

    Returns ``{"pred": {variant: [N, T]}, "const": {variant: [N, T]}, "y": {variant: [N, T]},
    "clip": [N], "d": int, "lambda_per_fold": [...], "n_fit_per_fold": [...]}``."""
    clips = list(range(len(X_clips)))
    groups_rows = np.concatenate([np.full(len(X_clips[c]), c) for c in clips])
    X_all = np.concatenate([np.asarray(x, dtype=np.float64) for x in X_clips])
    variants = {"real": Y_clips}
    if y_variants:
        variants.update(y_variants)
    Y_all = {v: np.concatenate([np.asarray(y, dtype=np.float64) for y in ys])
             for v, ys in variants.items()}
    N = X_all.shape[0]
    T = Y_all["real"].shape[1]
    pred = {v: np.full((N, T), np.nan) for v in variants}
    const = {v: np.full((N, T), np.nan) for v in variants}
    lam_folds, nfit_folds, d_used = [], [], None
    for hold in grouped_folds(groups_rows, k_outer, seed=seed):
        te = np.isin(groups_rows, hold)
        tr = ~te
        Xtr, Xte = X_all[tr], X_all[te]
        if pca and pca < Xtr.shape[1]:
            mu, Vt = pca_fit(Xtr, pca)
            Xtr, Xte = (Xtr - mu) @ Vt.T, (Xte - mu) @ Vt.T
        d_used = int(Xtr.shape[1])
        lam, _ = select_lambda(Xtr, Y_all["real"][tr], groups_rows[tr], lambdas, k_inner, seed)
        basis = _RidgeBasis(Xtr)
        Xte_s = basis.transform(Xte)
        for v in variants:
            betas, ymu = basis.coef(Y_all[v][tr], [lam])
            pred[v][te] = Xte_s @ betas[0] + ymu
            const[v][te] = ymu[None, :]
        lam_folds.append(lam)
        nfit_folds.append(int(tr.sum()))
    return {"pred": pred, "const": const, "y": Y_all, "clip": groups_rows, "d": d_used,
            "lambda_per_fold": lam_folds, "n_fit_per_fold": nfit_folds, "n_score": int(N)}


# =====================================================================================
# scoring — skill (exact 0 for the constant), r, clip-cluster bootstrap
# =====================================================================================
def skill_score(pred, const, y):
    """1 − SSE(pred)/SSE(const) per target, pooled over rows. The constant predictor
    reads EXACTLY 0.0 because pred == const."""
    pred, const, y = (np.asarray(a, dtype=np.float64) for a in (pred, const, y))
    sse_p = ((pred - y) ** 2).sum(0)
    sse_c = ((const - y) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1.0 - sse_p / sse_c
    return np.where(sse_c > 0, s, 0.0)


def pearson_r(pred, y):
    pred, y = np.asarray(pred, dtype=np.float64), np.asarray(y, dtype=np.float64)
    out = []
    for j in range(y.shape[1]):
        a, b = pred[:, j] - pred[:, j].mean(), y[:, j] - y[:, j].mean()
        den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
        out.append(float((a * b).sum() / den) if den > 0 else 0.0)
    return np.asarray(out)


def _clip_draws(clip_ids, n_boot, seed):
    uniq = np.unique(clip_ids)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.choice(uniq, size=len(uniq), replace=True)


def clip_bootstrap_skill(pred, const, y, clip_ids, n_boot=N_BOOT, seed=0, pred_b=None,
                         const_b=None):
    """Clip-cluster bootstrap of the pooled skill (per target). With ``pred_b``, the
    PAIRED delta skill(a) − skill(b) on the same clip draws. Per-clip SSE components
    are precomputed so each draw is a weighted sum."""
    clip_ids = np.asarray(clip_ids)
    uniq = np.unique(clip_ids)
    idx = {c: np.where(clip_ids == c)[0] for c in uniq}
    y = np.asarray(y, dtype=np.float64)

    def comps(p, c0):
        p, c0 = np.asarray(p, dtype=np.float64), np.asarray(c0, dtype=np.float64)
        sp = np.stack([((p[idx[c]] - y[idx[c]]) ** 2).sum(0) for c in uniq])   # [C, T]
        sc = np.stack([((c0[idx[c]] - y[idx[c]]) ** 2).sum(0) for c in uniq])
        return sp, sc

    sp_a, sc_a = comps(pred, const)
    paired = pred_b is not None
    if paired:
        sp_b, sc_b = comps(pred_b, const_b)
    pos = {c: i for i, c in enumerate(uniq)}
    boots = []
    for draw in _clip_draws(clip_ids, n_boot, seed):
        w = np.bincount([pos[c] for c in draw], minlength=len(uniq)).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sa = 1.0 - (w @ sp_a) / (w @ sc_a)
            if paired:
                sb = 1.0 - (w @ sp_b) / (w @ sc_b)
                boots.append(sa - sb)
            else:
                boots.append(sa)
    boots = np.asarray(boots)
    point = skill_score(pred, const, y)
    if paired:
        point = point - skill_score(pred_b, const_b, y)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return {"point": point.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "separated": [bool(l > 0 or h < 0) for l, h in zip(lo, hi)],
            "n_clips": int(len(uniq)), "n_boot": int(n_boot),
            "estimator": ("paired_" if paired else "") + "clip_cluster_bootstrap(skill)"}


def shuffle_targets(Y_clips, mode, seed=0):
    rng = np.random.default_rng(seed)
    if mode == "within":
        return [np.asarray(y)[rng.permutation(len(y))] for y in Y_clips]
    if mode == "global":
        cat = np.concatenate([np.asarray(y) for y in Y_clips])
        perm = cat[rng.permutation(len(cat))]
        out, o = [], 0
        for y in Y_clips:
            out.append(perm[o:o + len(y)])
            o += len(y)
        return out
    raise ValueError(mode)


def rff_features(X, n_feat, seed=0, gamma=None):
    """Random Fourier features for the non-linear comparison (fixed seed, fitted scale
    on whatever is passed — callers pass FIT statistics through ``gamma``)."""
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    if gamma is None:
        gamma = 1.0 / max(d, 1)
    Wm = rng.standard_normal((d, n_feat)) * np.sqrt(2.0 * gamma)
    b = rng.uniform(0, 2 * np.pi, n_feat)
    return np.sqrt(2.0 / n_feat) * np.cos(X @ Wm + b)


# =====================================================================================
# the panel
# =====================================================================================
def run_panel(feats, targets, clip_names, *, k_outer=K_OUTER, k_inner=K_INNER, seed=0,
              n_boot=N_BOOT, pca=0, mlp=False, mlp_feats=1024, lambdas=LAMBDA_GRID) -> dict:
    """``feats``: {name: [per-clip arrays [n_i, d]]}; ``targets``: [per-clip [n_i, 4]].
    Every cell gets the real, within-clip-shuffled and globally-shuffled targets
    through the identical fits."""
    n_rows = int(sum(len(t) for t in targets))
    y_within = shuffle_targets(targets, "within", seed + 1)
    y_global = shuffle_targets(targets, "global", seed + 2)
    variants = {"within_shuffle": y_within, "global_shuffle": y_global}
    cells = {}
    fits = {}
    for name in FEATURE_ORDER:
        if name not in feats:
            continue
        X = feats[name]
        t0 = time.time()
        # ⛔ the standardiser refuses a constant column silently? No: sd -> 1, so the
        # constant cell's column is centred to exactly 0 and every β is 0: pred == mean.
        cf = crossfit(X, targets, k_outer=k_outer, lambdas=lambdas, k_inner=k_inner,
                      seed=seed, pca=pca, y_variants=variants)
        fits[name] = cf
        cell = {"n_score": cf["n_score"], "n_fit_per_fold": cf["n_fit_per_fold"],
                "d": cf["d"], "lambda_rel_per_fold": cf["lambda_per_fold"],
                "seconds": round(time.time() - t0, 1)}
        for v in ("real", "within_shuffle", "global_shuffle"):
            sk = skill_score(cf["pred"][v], cf["const"][v], cf["y"][v])
            r = pearson_r(cf["pred"][v], cf["y"][v])
            cell[v] = {"skill": sk.tolist(), "r": r.tolist()}
        cell["real"]["skill_ci"] = clip_bootstrap_skill(cf["pred"]["real"], cf["const"]["real"],
                                                        cf["y"]["real"], cf["clip"], n_boot, seed)
        cell["transition_specific_skill"] = (np.asarray(cell["real"]["skill"])
                                             - np.asarray(cell["within_shuffle"]["skill"])).tolist()
        cells[name] = cell
        _p(f"    {name:<16} d={cf['d']:<5} n_score={cf['n_score']}  skill "
           + " ".join(f"{t}:{s:+.4f}" for t, s in zip(TARGETS, cell["real"]["skill"]))
           + "  | within-shuf " + " ".join(f"{s:+.3f}" for s in cell["within_shuffle"]["skill"])
           + "  | global-shuf " + " ".join(f"{s:+.3f}" for s in cell["global_shuffle"]["skill"]))
    marg = {}
    for a, b in MARGINALS:
        if a in fits and b in fits:
            fa, fb = fits[a], fits[b]
            marg[f"{a} - {b}"] = clip_bootstrap_skill(
                fa["pred"]["real"], fa["const"]["real"], fa["y"]["real"], fa["clip"], n_boot, seed,
                pred_b=fb["pred"]["real"], const_b=fb["const"]["real"])
    out = {"n_rows": n_rows, "n_clips": int(len(targets)), "targets": list(TARGETS),
           "k_outer": k_outer, "k_inner": k_inner, "lambda_grid_rel": list(lambdas),
           "pca": int(pca), "cells": cells, "paired_marginals": marg,
           "controls": {"constant_reads_exactly_zero": bool(
               "const" in cells and all(s == 0.0 for s in cells["const"]["real"]["skill"]))}}
    if mlp:
        out["mlp_rff"] = {}
        for name in ("dz_enc", "dzhat_true", "dzhat_zero", "pixdelta"):
            if name not in feats:
                continue
            # scale from the pooled data (a fixed-seed feature map; the ridge inside
            # crossfit still selects λ on the fit split only)
            allX = np.concatenate([np.asarray(x, dtype=np.float64) for x in feats[name]])
            gamma = 1.0 / max(float(allX.var(0).sum()), 1e-12)
            Xr = [rff_features(x, mlp_feats, seed=seed + 5, gamma=gamma) for x in feats[name]]
            cf = crossfit(Xr, targets, k_outer=k_outer, lambdas=lambdas, k_inner=k_inner,
                          seed=seed, pca=0, y_variants=variants)
            cell = {"d": cf["d"], "n_score": cf["n_score"], "lambda_rel_per_fold": cf["lambda_per_fold"]}
            for v in ("real", "within_shuffle", "global_shuffle"):
                cell[v] = {"skill": skill_score(cf["pred"][v], cf["const"][v], cf["y"][v]).tolist(),
                           "r": pearson_r(cf["pred"][v], cf["y"][v]).tolist()}
            out["mlp_rff"][name] = cell
            _p(f"    RFF {name:<12} d={cf['d']}  skill "
               + " ".join(f"{t}:{s:+.4f}" for t, s in zip(TARGETS, cell["real"]["skill"]))
               + "  | global-shuf " + " ".join(f"{s:+.3f}" for s in cell["global_shuffle"]["skill"]))
    return out


# =====================================================================================
# torch glue — features from a checkpoint and a corpus
# =====================================================================================
def _decode_frames(path, max_frames):
    import torch
    from PIL import Image
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    lens = d["jpeg_len"].numpy().tolist()
    off = [0]
    for L in lens:
        off.append(off[-1] + int(L))
    n = min(len(lens), max_frames)
    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(np.asarray(im).copy())
    if not imgs or float(np.abs(imgs[0]).mean()) == 0.0:
        raise SystemExit(f"[FATAL] {path} decoded to all-zero frames")
    return d, imgs


def gray_pix(img, hw=PIX_HW):
    """Newest frame → gray → block-mean downsample to ``hw`` → flat [h*w] in [0, 1]."""
    g = np.asarray(img, dtype=np.float64).mean(-1) / 255.0
    H, W = g.shape
    h, w = hw
    return g[:H - H % h, :W - W % w].reshape(h, H // h, w, W // w).mean((1, 3)).ravel()


def clip_features(world, path, dev, cond_param, *, max_frames=F_FRAMES, replace="last",
                  batch=64) -> dict | None:
    """All per-row inputs and targets for one clip. Rows are DATASET-consistent stacked
    frames (no padded stacks: stacked row j = raw frames j..j+2, paired with pose j+2),
    with a full W-window ending at the row and a pose at row+1."""
    import torch
    from train_v6_staged import _lift3  # noqa: PLC0415
    d, imgs = _decode_frames(path, max_frames)
    n = len(imgs)
    k = N_STACK - 1
    if n < k + int(world.window) + 2:
        return None
    frames = [torch.from_numpy(im).permute(2, 0, 1).float() / 255.0 for im in imgs]
    stacks = [torch.cat([frames[j], frames[j + 1], frames[j + 2]], 0) for j in range(n - k)]
    Z = []
    with torch.no_grad():
        for s in range(0, len(stacks), 16):
            x = torch.stack(stacks[s:s + 16])[:, None].to(dev)
            Z.append(world.encode_window(x)[:, 0].float().cpu())
    z = torch.cat(Z)                                         # [n-k, S]; row j <-> pose j+k
    poses = np.asarray(d["poses"], dtype=np.float64)[k:k + len(z)]
    act = d["actions"].float()[k:k + len(z)]                 # [n-k, 2]
    pix = np.stack([gray_pix(im) for im in imgs[k:k + len(z)]])
    W = int(world.window)
    rows = list(range(W - 1, len(z) - 1))                    # window ends at t, pose t+1 exists
    zs = torch.stack([z[t - W + 1:t + 1] for t in rows])
    a2 = torch.stack([act[t - W + 1:t + 1] for t in rows])
    v0 = torch.tensor(poses[rows, 3], dtype=torch.float32)  # the trainer's pose_last[:, 3]
    preds_true, preds_zero = [], []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            b = slice(s, s + batch)
            a3 = _lift3(a2[b].to(dev), v0[b].to(dev), cond_param)
            a3z = a3.clone()
            if replace == "last":
                a3z[:, -1, :2] = 0.0
            else:
                a3z[:, :, :2] = 0.0
            preds_true.append(world.predictor(zs[b].to(dev), a3)[1].float().cpu())
            preds_zero.append(world.predictor(zs[b].to(dev), a3z)[1].float().cpu())
    zt = z[rows].numpy().astype(np.float64)
    zn = z[[t + 1 for t in rows]].numpy().astype(np.float64)
    zh_t = torch.cat(preds_true).numpy().astype(np.float64)
    zh_0 = torch.cat(preds_zero).numpy().astype(np.float64)
    tg = targets_from_poses(poses, 1)[rows]
    feats = {"const": np.ones((len(rows), 1)),
             "pixdelta": pix[[t + 1 for t in rows]] - pix[rows],
             "act2": act[rows].numpy().astype(np.float64),
             "z_t": zt, "dz_enc": zn - zt, "dzhat_true": zh_t - zt, "dzhat_zero": zh_0 - zt,
             "dzhat_anch": zh_t - zh_0}
    feats["z_t+dz_enc"] = np.concatenate([zt, feats["dz_enc"]], 1)
    feats["z_t+dzhat_true"] = np.concatenate([zt, feats["dzhat_true"]], 1)
    return {"feats": feats, "targets": tg, "n_rows": len(rows),
            "clip_id": str(d.get("clip_id", os.path.basename(path)))}


def build_corpus_features(world, clips, dev, cond_param, **kw):
    F = {}
    T, names = [], []
    for i, c in enumerate(clips, 1):
        r = clip_features(world, c, dev, cond_param, **kw)
        if r is None:
            continue
        for k, v in r["feats"].items():
            F.setdefault(k, []).append(v)
        T.append(r["targets"])
        names.append(r["clip_id"])
        if i % 10 == 0 or i == len(clips):
            _p(f"    encoded {i}/{len(clips)} clips, {sum(len(t) for t in T)} rows")
    return F, T, names


# =====================================================================================
# CLI
# =====================================================================================
def resolve_stack(requested):
    cand = pathlib.Path(requested) if requested else pathlib.Path(_REPO) / "stack"
    if not (cand / "tanitad" / "__init__.py").is_file():
        _p(f"[REFUSED] stack {cand} does not hold tanitad/__init__.py")
        raise SystemExit(2)
    sp = str(cand.resolve())
    for p in (sp, os.path.join(sp, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return cand.resolve()


def preflight(stack):
    import tanitad
    got = pathlib.Path(tanitad.__file__).resolve()
    if stack not in got.parents:
        _p(f"[REFUSED] tanitad imported from the WRONG TREE: {got} (requested {stack}) — MM-C12")
        raise SystemExit(2)
    import tanitad.eval.v6_probe_trunk  # noqa: F401
    import train_v6_staged  # noqa: F401
    _p(f"  [stack] tanitad imported from: {got}")
    return str(got)


def parse_arms(a):
    arms = []
    for spec in (a.arm or []):
        n, p = spec.split("=", 1)
        arms.append((n, pathlib.Path(p)))
    if a.assets and a.arms:
        for n in str(a.arms).split(","):
            if n:
                arms.append((n, pathlib.Path(a.assets) / f"v7tiny_{n}" / "ckpt.pt"))
    if not arms:
        raise SystemExit("no arms: pass --arm NAME=PATH (repeatable) or --assets DIR --arms a,b")
    return arms


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="GS-9 transition-level probe: Δx from Δz")
    ap.add_argument("--arm", action="append")
    ap.add_argument("--assets", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--corpus", default="", help="dir of *.v2ep.pt HELD-OUT clips")
    ap.add_argument("--nclips", type=int, default=100)
    ap.add_argument("--max-frames", type=int, default=F_FRAMES)
    ap.add_argument("--stack", default="")
    ap.add_argument("--device", default="", choices=("", "cuda", "cpu"))
    ap.add_argument("--out", default="")
    ap.add_argument("--replace", default="last", choices=("last", "all"))
    ap.add_argument("--k-outer", type=int, default=K_OUTER)
    ap.add_argument("--k-inner", type=int, default=K_INNER)
    ap.add_argument("--pca", type=int, default=0, help="0 = none; fitted on the FIT split only")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp", action="store_true", help="add the RFF non-linear comparison")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    import torch
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    stack = resolve_stack(a.stack or None)
    tanitad_file = preflight(stack)
    from tanitad.eval.v6_probe_trunk import load_trunk_auto  # noqa: PLC0415
    from tanitad.models.flagship_v15 import SPEED_SCALE  # noqa: PLC0415
    from train_v6_staged import COND_INCUMBENT  # noqa: PLC0415
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not a.corpus or not a.out:
        raise SystemExit("--corpus and --out are required (unless --selftest)")
    clips = sorted(glob.glob(os.path.join(a.corpus, "*.v2ep.pt")))[:a.nclips]
    if not clips:
        raise SystemExit(f"[REFUSED] no *.v2ep.pt under {a.corpus}")
    res = {"_evidence_class": f"MEASURED (ours; dev-box {dev})", "eval_tier": EVAL_TIER,
           "hypothesis": HYPOTHESIS, "paper": PAPER, "tanitad_imported_from": tanitad_file,
           "metric_families": {"longitudinal": "REFUSED", "lateral": "REFUSED",
                               "tactical": "REFUSED", "strategic": "REFUSED",
                               "reason": "T0-DIAGNOSTIC probe panel — no driving metric "
                                         "exists here to family-ise (EVAL_DOCTRINE)"},
           "corpus": a.corpus, "n_clips_requested": len(clips), "max_frames": a.max_frames,
           "dt_s": DT, "k_ticks": 1, "targets": list(TARGETS),
           "v_policy": ("v is NEVER a probe input; it is the TARGET dv (derived from poses) "
                        "and the predictor's own third action channel (_lift3, v_last/"
                        f"{SPEED_SCALE:g})"),
           "replace": a.replace, "seed": a.seed, "k_outer": a.k_outer, "k_inner": a.k_inner,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arms": {}}
    for arm, p in parse_arms(a):
        if not p.is_file():
            _p(f"  {arm}: NO CKPT at {p} — SKIPPED (never approximated)")
            res["arms"][arm] = {"skipped": f"no checkpoint at {p}"}
            continue
        t0 = time.time()
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        world, _g, step = load_trunk_auto(ck, dev, ckpt_path=str(p))
        world.eval()
        cond_param = str(((ck.get("config") or {}).get("args") or {}).get("cond_param", COND_INCUMBENT))
        _p(f"[{arm}] step {step}  W={int(world.window)}  S={world.state_dim}  cond_param={cond_param}")
        F, T, names = build_corpus_features(world, clips, dev, cond_param,
                                            max_frames=a.max_frames, replace=a.replace)
        panel = run_panel(F, T, names, k_outer=a.k_outer, k_inner=a.k_inner, seed=a.seed,
                          n_boot=a.n_boot, pca=a.pca, mlp=a.mlp)
        res["arms"][arm] = {"ckpt_path": str(p), "ckpt_md5": md5_of(p), "step": int(step),
                            "state_dim": int(world.state_dim), "window": int(world.window),
                            "cond_param": cond_param, "clip_ids": names,
                            "seconds": round(time.time() - t0, 1), "panel": panel}
        del world
        if dev == "cuda":
            torch.cuda.empty_cache()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=_json_default)
    _p("WROTE", a.out)
    return 0


# =====================================================================================
# self-test — planted structure must be recovered; the controls must read known values
# =====================================================================================
def synthetic_clips(n_clips=20, rows=60, d=24, seed=0, clip_offset=0.0, noise=0.1):
    """Δx = A·Δz + noise with Δz ~ N(0, I); ``clip_offset`` adds a clip-level mean to
    the targets (a scene-class cue the within-clip shuffle must PRESERVE)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, 4))
    X, Y, P = [], [], []
    for c in range(n_clips):
        dz = rng.standard_normal((rows, d))
        off = clip_offset * rng.standard_normal(4)
        y = dz @ A + noise * rng.standard_normal((rows, 4)) + off
        X.append(dz)
        Y.append(y)
        P.append(rng.standard_normal((rows, 160)))          # a structureless "pixel" floor
    return X, Y, P


def selftest() -> int:
    ok = True
    X, Y, P = synthetic_clips()
    feats = {"const": [np.ones((len(y), 1)) for y in Y], "pixdelta": P, "dz_enc": X}
    out = run_panel(feats, Y, [f"c{i}" for i in range(len(Y))], n_boot=200)
    c = out["cells"]
    ok &= out["controls"]["constant_reads_exactly_zero"]
    ok &= all(s > 0.9 for s in c["dz_enc"]["real"]["skill"])
    ok &= all(abs(s) < 0.05 for s in c["dz_enc"]["global_shuffle"]["skill"])
    ok &= all(abs(s) < 0.05 for s in c["pixdelta"]["real"]["skill"])
    _p("  constant exactly 0:", out["controls"]["constant_reads_exactly_zero"],
       " planted skill:", [round(s, 3) for s in c["dz_enc"]["real"]["skill"]],
       " global-shuffle:", [round(s, 3) for s in c["dz_enc"]["global_shuffle"]["skill"]],
       " floor:", [round(s, 3) for s in c["pixdelta"]["real"]["skill"]])
    _p("  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
