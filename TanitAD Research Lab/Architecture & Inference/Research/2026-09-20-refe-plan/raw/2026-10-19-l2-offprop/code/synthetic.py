#!/usr/bin/env python3
"""Synthetic stand-in for the L2 data: the SAME shapes, folds, pair rules and Step A structure as the real run, with a PLANTED within-slot EP direction of known strength.

Used for (a) every unit test, (b) the design choices (`design_sweep.py`: the BCE weight and the optimiser recipe are justified HERE and never on a real label), (c) the cost measurement,
(d) the C-LEAK behavioural half and the C-REACH dry run.  Nothing here reads a real label.

GENERATIVE MODEL (all parameters explicit; defaults mimic what the real features show LABEL-FREE: token norm ~30, per-dim std ~1, within-slot token move ~3.3, EP-head weight norm ~0.55, EP logit std
across slots ~0.9, within-slot logit std ~0.36):
  in-fan:  EP label t ~ clip(N(0.8, 0.15)); token x = c_token + noise + kappa * z * v   (z = standardised t; v = the "slot prior" direction P2's head reads)
  P2 head: w0 = c * (v + rho * u), b0 calibrated to the in-fan mean.  rho is how much of the within-slot direction u P2 already reads (rho = 0: none).
  pert:    true EP shift d = shift(kind, param) + eta;  t_p = clip(t_a + d);   token x_p = x_a + gamma * standardised(t_p - t_a) * u + xi,   xi ~ N(0, sigma_xi^2 I).
  gamma is the STRENGTH of the planted within-slot signal (gamma = 0: the token carries no within-slot ordering at all -> the null world).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2lib  # noqa: E402
from l2lib import stepA  # noqa: E402


def _traj(rng, T=20, dt=0.2):
    v0 = rng.uniform(2.0, 16.0)
    a = rng.uniform(-0.12, 0.12)
    kappa = rng.normal(0.0, 0.012)
    t = dt * np.arange(1, T + 1)
    s = v0 * t * (1 + 0.5 * a * t)
    s = np.maximum.accumulate(np.maximum(s, 0.0))
    yaw = kappa * s
    if abs(kappa) > 1e-6:
        x, y = np.sin(kappa * s) / kappa, (1 - np.cos(kappa * s)) / kappa
    else:
        x, y = s, np.zeros_like(s)
    out = np.stack([x, y, yaw], -1)
    out[-1, 2] += rng.normal(0, 0.3)                      # the raw last heading is corrupted (A7): the perturbation code is built around it
    return out.astype(np.float32)


def make_structure(n_tokens: int, n_logs: int, seed: int) -> tuple[dict, dict]:
    """label-free Step-A structure: tokens, logs, 64 proposals, banked-like logits and the REAL perturbation code (`stepA.make_perts`)."""
    rng = np.random.default_rng(seed)
    N = n_tokens
    log_of = np.concatenate([np.arange(n_logs), rng.integers(0, n_logs, size=max(N - n_logs, 0))])[:N]
    traj = np.stack([np.stack([_traj(rng) for _ in range(64)]) for _ in range(N)])
    logits = rng.normal(1.0, 1.0, size=(N, 64, 6)).astype(np.float32)
    bundle = {"token": np.array([f"tok{n:05d}" for n in range(N)]), "log": np.array([f"log{l:02d}" for l in log_of]), "traj_raw": traj, "logits": logits,
              "pick": np.argmax(stepA.aggregate(logits.astype(np.float64)), 1).astype(np.int64), "valid": np.ones((N, 64), bool)}
    perts = stepA.make_perts(bundle)
    return bundle, perts


def real_structure(run_dir: str = "D:/Projects/TanitAD-artifacts/l1_stepA_run") -> tuple[dict, dict]:
    """the REAL label-free structure of Step A (tokens, logs, trajectories, perturbation geometry).  No label or score is read from it."""
    b = stepA.load_npz(f"{run_dir}/bundle/bundle.npz")
    p = stepA.load_npz(f"{run_dir}/perts.npz")
    bundle = {k: b[k] for k in ("token", "log", "traj_raw", "logits", "pick", "valid")}
    return bundle, p


def _unit(rng, D, orth_to=None):
    v = rng.normal(size=D)
    if orth_to is not None:
        v = v - (v @ orth_to) * orth_to
    return v / np.linalg.norm(v)


def make_data(n_tokens: int = 96, n_logs: int = 24, seed: int = 0, gamma: float = 0.2, rho: float = 0.5, D: int = 256, sigma_e: float = 0.8, kappa_in: float = 1.6,
              sigma_xi: float = 0.2, eta_sd: float = 0.06, shift_scale: float = 0.6, structure=None) -> l2lib.Data:
    """-> an `l2lib.Data` whose labels and features are synthetic.  `structure` = (bundle, perts) from `real_structure()` to reuse the real geometry (n_tokens/n_logs then follow it)."""
    bundle0, perts = structure if structure is not None else make_structure(n_tokens, n_logs, seed)
    rng = np.random.default_rng(seed + 991)
    N, Q = perts["anchor"].shape
    v = _unit(rng, D)
    u = _unit(rng, D, orth_to=v)
    # ---- in-fan
    t_in = np.clip(rng.normal(0.8, 0.15, size=(N, 64)), 0.0, 1.0)
    z = (t_in - 0.8) / 0.15
    c_tok = rng.normal(0, 0.5, size=(N, 1, D))
    x_in = (c_tok + rng.normal(0, sigma_e, size=(N, 64, D)) + kappa_in * sigma_e * z[..., None] * v).astype(np.float32)
    w0 = v + rho * u
    ep0 = x_in @ w0
    c = 0.9 / ep0.std()
    w0 = (c * w0).astype(np.float32)
    b0 = float(np.float32(np.log(0.8 / 0.2) - (x_in @ w0).mean()))     # float32-exact, so the P2 arm and the synthetic 'banked' logits are the same numbers
    # ---- perturbations
    anchor, kind, param = perts["anchor"], perts["kind"], perts["param"]
    t_a = np.take_along_axis(t_in, anchor, 1)
    shift = np.where(kind == 0, shift_scale * (param - 1.0), -0.03) + rng.normal(0, eta_sd, size=(N, Q))
    t_p = np.clip(t_a + shift, 0.0, 1.0)
    d = t_p - t_a
    ds = d / max(d.std(), 1e-9)
    x_a = np.take_along_axis(x_in, anchor[..., None], 1)
    x_p = (x_a + gamma * ds[..., None] * u + rng.normal(0, sigma_xi, size=(N, Q, D))).astype(np.float32)
    # ---- labels (nc, dac, ep, ttc, c, ddc); a few NC/DAC flips on the perturbed plans so A4/A5 have a denominator
    sub_in = np.ones((N, 64, 6))
    sub_in[..., 2] = t_in
    sub_p = np.ones((N, Q, 6))
    sub_p[..., 2] = t_p
    flip = rng.random((N, Q)) < 0.02
    sub_p[..., 0] = np.where(flip, 0.0, 1.0)

    def pdms(s):
        return s[..., 0] * s[..., 1] * (5 * s[..., 2] + 5 * s[..., 3] + 2 * s[..., 4]) / 12.0

    bundle = dict(bundle0)
    bundle["sub"], bundle["pdms"] = sub_in, pdms(sub_in)
    labels = {"sub": sub_p, "pdms": pdms(sub_p)}
    # ---- banked-like logits: every head but EP is a fixed noisy constant; EP is P2's head on the token
    a_log = np.zeros((N, 64, 6))
    a_log[..., 0], a_log[..., 1] = 4.0 + rng.normal(0, 0.3, (N, 64)), 4.0 + rng.normal(0, 0.3, (N, 64))
    a_log[..., 3], a_log[..., 4], a_log[..., 5] = 3.0 + rng.normal(0, 0.3, (N, 64)), 3.0 + rng.normal(0, 0.3, (N, 64)), 3.0
    a_log[..., 2] = l2lib.ep_apply(x_in, np.vstack([np.zeros((2, D), np.float32), w0[None]]), np.array([0, 0, b0], np.float64))   # exactly what the P2 arm computes
    p_log = np.take_along_axis(a_log, anchor[..., None], 1).copy()
    p_log[..., :2] += rng.normal(0, 0.3, (N, Q, 2))
    p_log[..., 2] = l2lib.ep_apply(x_p, np.vstack([np.zeros((2, D), np.float32), w0[None]]), np.array([0, 0, b0], np.float64))
    scores = {"a_logit": a_log, "p_logit": p_log}
    bundle["logits"] = a_log.astype(np.float32)
    bundle["pick"] = np.argmax(stepA.aggregate(a_log), 1).astype(np.int64)
    W6 = np.zeros((6, D), np.float32)
    W6[:] = rng.normal(0, 0.03, size=(6, D))
    W6[2] = w0
    b6 = np.zeros(6, np.float32)
    b6[2] = np.float32(b0)
    return l2lib.Data(bundle, perts, labels, scores, x_in, x_p, W6, b6, meta={"kind": "synthetic", "gamma": gamma, "rho": rho, "seed": seed, "u": u.astype(np.float32), "v": v.astype(np.float32)})


# ====================================================================================================================================================================================
# SEMI-SYNTHETIC: REAL tokens, SYNTHETIC labels.  Used to fix the recipe where the optimiser sees the REAL feature covariance (conditioning, scale) but no real label is read.
# ====================================================================================================================================================================================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _infan_conc(top8, t_in, ep_in, thr=0.05):
    N = t_in.shape[0]
    y = np.take_along_axis(t_in, top8, 1)                 # [N,8]
    x = np.take_along_axis(ep_in, top8, 1)
    dy = y[:, :, None] - y[:, None, :]
    m = dy > thr
    dx = x[:, :, None] - x[:, None, :]
    return float(((m & (dx > 0)).sum() + 0.5 * (m & (dx == 0)).sum()) / max(m.sum(), 1))


def _a1(t_p, t_a, s_p, s_a, thr=0.05):
    m = np.abs(t_p - t_a) >= thr - 1e-12
    d = np.sign(s_p - s_a)
    c = np.where(d == 0, 0.5, (d == np.sign(t_p - t_a)).astype(float))
    return float((c * m).sum() / max(m.sum(), 1))


def semi_synthetic(real: l2lib.Data, beta: float = 0.5, alpha: float = 0.3, seed: int = 0, shift_scale: float = 0.6, target_infan: float = 0.889, target_a1: float = 0.685, target_share: float = 0.2575,
                   signal_amp: float = 0.08, n_pc: int = 20, p2_follow: float = 0.15) -> l2lib.Data:
    """REAL tokens and structure; labels generated from the tokens by a PLANTED model.
      in-fan   t = clip(sigmoid(P2's EP logit) + N(0, tau)), tau calibrated so P2's in-fan top-8 concordance = `target_infan` (0.889, the registered SELRANK value; a published/banked number, not our labels)
      pert     t_p = clip(t_a + shift(kind, param) + beta * eta * g + eta * N(0,1)) (beta = the planted signal's sd in units of the label-noise sd eta),  g = standardised projection of the REAL within-slot token move (x_p - x_anchor) on u
               u = alpha * (P2's EP direction) + (1 - alpha) * (a random direction inside the top-`n_pc` principal directions of the real moves), normalised.
               eta calibrated so that at beta = 0 P2's A1 equals `target_a1` (0.685, Step A's banked value) and a global scale c so that the share of pairs with |dEP| >= 0.05 equals `target_share` (0.2575 = 9,888 / 38,400, Step A's banked A1 denominator).  beta > 0 adds a learnable within-slot ordering that P2 reads only through alpha.
    Every non-EP label is the constant 1 (the non-EP heads carry no information here); every non-EP logit is the real banked one (label-free)."""
    rng = np.random.default_rng(seed)
    N, Q, D = real.N, real.Q, real.D
    W0, b0 = real.W6, real.b6
    s_in = l2lib.ep_apply(real.tok_in, W0, b0)
    s_p = l2lib.ep_apply(real.tok_p, W0, b0)
    z_in = rng.normal(size=(N, 64))
    p_in = _sigmoid(s_in)
    lo, hi = 1e-3, 1.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if _infan_conc(real.top8, np.clip(p_in + mid * z_in, 0, 1), s_in) > target_infan:
            lo = mid
        else:
            hi = mid
    tau = math.sqrt(lo * hi)
    t_in = np.clip(p_in + tau * z_in, 0.0, 1.0)
    anchor, kind, param = real.anchor, real.kind, real.param
    t_a = np.take_along_axis(t_in, anchor, 1)
    s_a = np.take_along_axis(s_in, anchor, 1)
    x_a = np.take_along_axis(real.tok_in, anchor[..., None], 1)
    delta = (real.tok_p - x_a).reshape(-1, D).astype(np.float64)
    sub = delta[rng.choice(len(delta), size=min(6000, len(delta)), replace=False)]
    sub = sub - sub.mean(0)
    _, _, vt = np.linalg.svd(sub, full_matrices=False)
    r = vt[:n_pc].T @ rng.normal(size=n_pc)
    r /= np.linalg.norm(r)
    w = np.asarray(W0[l2lib.EP], np.float64)
    w = w / np.linalg.norm(w)
    u = alpha * w + (1 - alpha) * r
    u /= np.linalg.norm(u)
    proj = (delta @ u).reshape(N, Q)
    g = (proj - proj.mean()) / proj.std()
    shift = np.where(kind == 0, shift_scale * (param - 1.0), -0.03) + p2_follow * (s_p - s_a)      # the part of the truth that P2's own within-slot response already follows (dEP ~ 0.15 * dLogit)
    noise = rng.normal(size=(N, Q))

    def t_p_of(beta_, eta_, c_):
        return np.clip(t_a + c_ * (shift + beta_ * eta_ * g + eta_ * noise), 0.0, 1.0)

    def share_of(t_):
        return float((np.abs(t_ - t_a) >= 0.05 - 1e-12).mean())

    eta, c = 0.2, 1.0
    for _ in range(4):                                    # two coupled calibrations: eta -> A1(P2) at beta = 0 (falls as eta grows); c -> the share of pairs with |dEP| >= 0.05 (rises with c)
        lo, hi = 1e-3, 3.0
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            if _a1(t_p_of(0.0, mid, c), t_a, s_p, s_a) > target_a1:
                lo = mid
            else:
                hi = mid
        eta = math.sqrt(lo * hi)
        lo, hi = 1e-3, 3.0
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            if share_of(t_p_of(0.0, eta, mid)) < target_share:
                lo = mid
            else:
                hi = mid
        c = math.sqrt(lo * hi)
    a1_at, share_at = _a1(t_p_of(0.0, eta, c), t_a, s_p, s_a), share_of(t_p_of(0.0, eta, c))
    if abs(a1_at - target_a1) > 0.01 or abs(share_at - target_share) > 0.02:
        raise ValueError(f"semi_synthetic: cannot calibrate P2's A1 to {target_a1} and the pair share to {target_share} (reached {a1_at:.3f}, {share_at:.3f}; eta={eta:.4f}, c={c:.4f})")
    t_p = t_p_of(beta, eta, c)

    def pdms(s):
        return s[..., 0] * s[..., 1] * (5 * s[..., 2] + 5 * s[..., 3] + 2 * s[..., 4]) / 12.0

    sub_in = np.ones((N, 64, 6))
    sub_in[..., 2] = t_in
    sub_p = np.ones((N, Q, 6))
    sub_p[..., 2] = t_p
    bundle = {k: v for k, v in real.bundle.items()}
    bundle["sub"], bundle["pdms"] = sub_in, pdms(sub_in)
    labels = {"sub": sub_p, "pdms": pdms(sub_p)}
    a_log, p_log = np.array(real.scores["a_logit"], np.float64, copy=True), np.array(real.scores["p_logit"], np.float64, copy=True)
    a_log[..., l2lib.EP], p_log[..., l2lib.EP] = s_in, s_p
    return l2lib.Data(bundle, real.perts, labels, {"a_logit": a_log, "p_logit": p_log}, real.tok_in, real.tok_p, real.W6, real.b6,
                      meta={"kind": "semi-synthetic", "beta": beta, "alpha": alpha, "seed": seed, "tau_in": tau, "eta": eta, "P2_infan_target": target_infan, "P2_A1_target_at_beta0": target_a1, "P2_A1_reached_at_beta0": a1_at, "pair_share_target": target_share, "pair_share_reached": share_at, "scale_c": c})
