#!/usr/bin/env python3
"""RL-S1 reward (part 1): the per-proposal reward over one scene's 64 hypotheses.  PURE: numpy only, no model / torch / navsim import.

Design: Research/2026-10-11-refe-data-and-rl/RL_POSTTRAIN_REVIEW.md section 3 (reward) -- implemented, not redesigned.

    R_i = PDMS_i - lam_d * D_i + w_prog * A_prog_i                       (the RL-core arm: R0 / R1)
          [ - lam_k * K_i ]  only with cfg.use_k      \\  the R2 screening arm; PI decision D3 is OPEN, so BOTH default OFF
          [ + lam_l * L_i ]  only with cfg.use_l      /
    lam_d = 0.15, w_prog = 0.5, lam_k = 0.05, lam_l = 0.05          (all [H], review 3.1 / 3.4; no weight-sensitivity study exists)

INPUT = the on-policy PDM label format (onpolicy_relabel_pdm.py side files / train.OnPolicyBank.PDM_ORDER): per proposal the sub-scores
NC, DAC, EP, TTC, C, DDC in `train.ScorerBank.COMPONENTS` order, shape [..., M, >=5] (a leading batch axis is fine).  PDMS is RECOMPUTED from
them with NAVSIM v1.1's weights NC*DAC*(5EP+5TTC+2C)/12 (== the side file's `pdms` column; `check_pdms` asserts it, labels are rounded to 5 dp).
D, K and the lane-change inputs are NOT in those labels: they come from the part-2 extended labeller (geometry on the drivable-area map).  The
reductions that turn its per-time-step traces into per-proposal scalars ARE here (`drivable_excess`, `lane_consistency`) so the maths is tested.

TWO INTERPRETATIONS the review leaves open (flagged, not silently chosen):
  1. "safe" (review 3.4 g2 never defines it): here NC = DAC = TTC = 1 (the 'full safety gate' of review 3.3), thresholds in `safe_mask`.
  2. For a SAFE proposal ep_i == min(p_i, p_ref) / p_ref EXACTLY (navsim_pdm_targets.score_plans: ep = raw / max(raw_ref, raw), forced 1 when
     max <= 5 m), so A_prog_i = clip(ep_i - max_safe(ep), -1, 0): no extra label is needed for the progress guard.
FINDING -> RULING.  Uncapped, the guard at w_prog = 0.5 put a TTC-violating full-speed plan (PDMS 0.583) ABOVE a safe half-speed plan
(PDMS 0.792 - 0.5*0.5 = 0.542): it rewarded crossing the safety boundary, which review principle P4 forbids ("may reorder near-ties, never
override PDMS").  No fixed w_prog can guarantee P4 (any w_prog > 0 flips some pair), so the Master Mind ruled (2026-10-12) a STRUCTURAL fix:
`guard_penalty` caps each safe proposal's penalty at half its pre-guard margin over the best UNSAFE proposal below it.  The guard still reorders
proposals INSIDE the safe set (progress vs comfort -- the slow-pick hack it exists for), but can never push a safe plan to or below an unsafe
one it beat.  w_prog stays 0.5 [H].  Pinned by test_rl_reward.py: guard_never_crosses_safety_boundary, the P4 property over random fans,
guard_reorders_inside_safe, and three mutations (cap removed / keep = 1 / cap counting safe rivals) that must go RED.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

V1_W = (5.0, 5.0, 2.0)                       # NAVSIM v1.1: EP, TTC, C weights (sum 12); DDC enters with weight 0
COMPONENTS = ("nc", "dac", "ep", "ttc", "c", "ddc")        # train.ScorerBank.COMPONENTS / OnPolicyBank.PDM_ORDER
SAFE_TOL = 1e-6


@dataclass(frozen=True)
class RewardCfg:
    lam_d: float = 0.15            # drivable-area excess weight (review 3.1)
    w_prog: float = 0.5            # progress-guard weight (review 3.4 g2)
    guard: bool = True             # g2 on: it is part of the R-core reward (review 6.1 arms R0 / R1)
    lam_k: float = 0.05
    use_k: bool = False            # R2 screening arm only; D3 open
    lam_l: float = 0.05
    use_l: bool = False            # R2 screening arm only; D3 open


def _f(x, name):
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite values")
    return a


def split_components(tg) -> dict:
    """[..., M, >=5] sub-score array in COMPONENTS order -> {'nc','dac','ep','ttc','c'[,'ddc']} each [..., M], all validated in [0, 1]."""
    a = _f(tg, "sub-scores")
    if a.ndim < 2 or a.shape[-1] < 5:
        raise ValueError(f"sub-scores need shape [..., M, >=5], got {a.shape}")
    if (a[..., :6] < -1e-9).any() or (a[..., :6] > 1 + 1e-9).any():
        raise ValueError("sub-scores outside [0, 1]")
    return {n: a[..., i] for i, n in enumerate(COMPONENTS[:min(a.shape[-1], 6)])}


def tg_from_pdm_record(rec: dict) -> np.ndarray:
    """One onpolicy_relabel_pdm.py side-file record -> [M, 6] in COMPONENTS order (the very stacking train.OnPolicyBank._apply_pdm_labels does)."""
    return np.clip(np.stack([np.asarray(rec[c], np.float64) for c in COMPONENTS], -1), 0.0, 1.0)


def pdms_v1(nc, dac, ep, ttc, c):
    """NAVSIM v1.1 PDMS of each proposal: NC * DAC * (5 EP + 5 TTC + 2 C) / 12."""
    return nc * dac * (V1_W[0] * ep + V1_W[1] * ttc + V1_W[2] * c) / sum(V1_W)


def check_pdms(tg, pdms_col, atol: float = 2e-4) -> float:
    """max |recomputed PDMS - the label file's own `pdms` column|; raises when above `atol` (5-decimal rounding of five inputs ~ 1e-4)."""
    s = split_components(tg)
    err = float(np.abs(pdms_v1(s["nc"], s["dac"], s["ep"], s["ttc"], s["c"]) - np.asarray(pdms_col, np.float64)).max())
    if err > atol:
        raise ValueError(f"recomputed PDMS differs from the label file's pdms column by {err:.3g} > {atol}")
    return err


def safe_mask(nc, dac, ttc, tol: float = SAFE_TOL):
    """INTERPRETATION 1: safe = NC = DAC = TTC = 1 (review 3.3's 'full safety gate'); review 3.4 does not define it."""
    return (nc >= 1.0 - tol) & (dac >= 1.0 - tol) & (ttc >= 1.0 - tol)


def progress_shortfall(ep, safe):
    """(g2) A_prog_i = clip(ep_i - ep_best_safe, -1, 0) for SAFE proposals, 0 for the rest and for a scene with no safe proposal.
    == clip((min(p_i,p_ref) - min(p_best_safe,p_ref)) / p_ref, -1, 0) because ep_i = min(p_i, p_ref) / p_ref on a safe proposal."""
    ep = np.asarray(ep, np.float64)
    safe = np.asarray(safe, bool)
    ref = np.where(safe, ep, -np.inf).max(axis=-1, keepdims=True)
    has = np.isfinite(ref)
    a = np.clip(ep - np.where(has, ref, 0.0), -1.0, 0.0)
    return np.where(safe & has, a, 0.0)


def guard_penalty(base, raw, safe, keep: float = 0.5):
    """P4 enforced (Master Mind ruling 2026-10-12).  base [..., M] = the pre-guard reward; raw = w_prog * A_prog (<= 0, exactly 0 off the safe
    set).  For a SAFE proposal i the penalty -raw_i is capped at keep * (base_i - max{base_j : j UNSAFE, base_j < base_i}); with no unsafe
    proposal below i the cap is infinite.  keep < 1 keeps every safe-over-unsafe order STRICT.  Unsafe proposals get exactly 0."""
    base, raw, safe = np.asarray(base, np.float64), np.asarray(raw, np.float64), np.asarray(safe, bool)
    bi, bj = base[..., :, None], base[..., None, :]
    below = np.where((~safe)[..., None, :] & (bj < bi), bj, -np.inf).max(axis=-1)
    cap = np.where(np.isfinite(below), keep * (base - np.where(np.isfinite(below), below, 0.0)), np.inf)
    return np.where(safe, -np.minimum(-raw, cap), 0.0)


def drivable_excess(gap, scale: float = 0.5):
    """D_i = mean_t min(g_i(t) / 0.5 m, 1) on the executed states; `gap` [..., M, T] = max over the 4 bbox corners of the distance to the union
    of drivable polygons (0 when inside).  Zero iff the ego stays inside (the G-REWARD-ID identity: g > 0 iff DAC = 0 is checked on labelled data in part 2)."""
    g = _f(gap, "gap")
    if (g < 0).any():
        raise ValueError("gap distances must be >= 0")
    return np.minimum(g / scale, 1.0).mean(axis=-1)


def lane_consistency(offset, dpsi_deg, in_intersection, straddle, *, k1_dead=0.25, k1_span=0.75, k2_dead_deg=10.0, k2_span_deg=20.0, near=1.5,
                     w=(0.6, 0.4, 0.2)):
    """K_i (review 3.2 c).  offset/dpsi_deg/in_intersection [..., M, T]: distance of the ego centre to the PDM-Closed centreline (m), heading
    error to it (deg), the scorer's own INTERSECTION switch; straddle [..., M] = fraction of time the bbox crosses a lane boundary with no
    lane-change label.  k1 = mean_t clip((o - 0.25) / 0.75, 0, 1) outside intersections; k2 = mean_t clip((|dpsi| - 10) / 20, 0, 1) outside
    intersections and only where o < 1.5 m (a deliberate lane change is not a heading violation).  Masked steps contribute 0 to a mean over ALL T.
    K = 0.6 k1 + 0.4 k2 + 0.2 k3  (the review's 'k3 added at weight 0.2' read literally: K <= 1.2, no clip)."""
    o, dp = _f(offset, "offset"), np.abs(_f(dpsi_deg, "dpsi_deg"))
    out = ~np.asarray(in_intersection, bool)
    k1 = (np.clip((o - k1_dead) / k1_span, 0.0, 1.0) * out).mean(axis=-1)
    k2 = (np.clip((dp - k2_dead_deg) / k2_span_deg, 0.0, 1.0) * (out & (o < near))).mean(axis=-1)
    k3 = _f(straddle, "straddle")
    return w[0] * k1 + w[1] * k2 + w[2] * k3


def lane_change_term(lc_executed, label_fires, dither, ep, safe, *, gain_span=0.2):
    """L_i (review 3.3), clipped to [-1, 1].  lc_executed [..., M] bool; label_fires [...] bool, scene level (label (i) OR (iii)); dither [..., M] in {0,1};
    +1 * progress_gain only for a SAFE executed lane change under a firing label; -1 for an executed change with no label; minus the dither flag.
    progress_gain_i = clip((EP_i - EP_stay) / 0.2, 0, 1), EP_stay = EP of the best SAFE non-lane-changing proposal (0 gain if there is none)."""
    lc, safe = np.asarray(lc_executed, bool), np.asarray(safe, bool)
    lab = np.asarray(label_fires, bool)[..., None]
    ep = np.asarray(ep, np.float64)
    stay = np.where(safe & ~lc, ep, -np.inf).max(axis=-1, keepdims=True)
    gain = np.where(np.isfinite(stay), np.clip((ep - np.where(np.isfinite(stay), stay, 0.0)) / gain_span, 0.0, 1.0), 0.0)
    pos = (lc & safe & lab) * gain
    neg = (lc & ~lab) * 1.0
    return np.clip(pos - neg - np.asarray(dither, np.float64), -1.0, 1.0)


def reward(tg, cfg: RewardCfg = RewardCfg(), *, d=None, k=None, lane=None, pdms=None) -> dict:
    """R [..., M] and its parts.  A term the cfg turns ON but whose input is missing RAISES (a silently-zero term is the 'built, tested,
    unreachable' failure): lam_d > 0 needs `d`, use_k needs `k`, use_l needs `lane` = {lc_executed, label_fires, dither}."""
    s = split_components(tg)
    p = pdms_v1(s["nc"], s["dac"], s["ep"], s["ttc"], s["c"])
    if pdms is not None:
        check_pdms(tg, pdms)
    safe = safe_mask(s["nc"], s["dac"], s["ttc"])
    r = p.copy()
    out = {"pdms": p, "safe": safe}
    if cfg.lam_d != 0.0:
        if d is None:
            raise ValueError("cfg.lam_d != 0 but no drivable-excess input `d` (pass d=, or lam_d=0 EXPLICITLY to run without it)")
        dd = _f(d, "d")
        if dd.shape != p.shape or (dd < 0).any() or (dd > 1 + 1e-9).any():
            raise ValueError("d must have the shape of the proposals and lie in [0, 1]")
        r = r - cfg.lam_d * dd
        out["D"] = dd
    if cfg.guard:
        a = progress_shortfall(s["ep"], safe)
        g = guard_penalty(r, cfg.w_prog * a, safe)
        r = r + g
        out["A_prog"] = a
        out["guard"] = g
    if cfg.use_k:
        if k is None:
            raise ValueError("cfg.use_k but no `k` input")
        kk = _f(k, "k")
        r = r - cfg.lam_k * kk
        out["K"] = kk
    if cfg.use_l:
        if lane is None:
            raise ValueError("cfg.use_l but no `lane` input")
        ll = lane_change_term(lane["lc_executed"], lane["label_fires"], lane["dither"], s["ep"], safe)
        r = r + cfg.lam_l * ll
        out["L"] = ll
    out["R"] = r
    return out
