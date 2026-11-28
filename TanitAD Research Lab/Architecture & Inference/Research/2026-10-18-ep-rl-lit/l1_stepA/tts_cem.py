#!/usr/bin/env python3
"""Step B machinery of PREREG_L1_SEARCH_ADMISSION.md (arm launched ONLY by outcome O1): a cross-entropy search on the scorer, K=5, M=64 per iteration scored as ONE complete
set, elites M/8, warm start at the scorer's pick, 3 search seeds; the final pick re-scores the elites inside the ORIGINAL 64-fan (prereg section 5).

Built and tested NOW so that Step B is a launch, not a build.  Nothing here has run on a real scorer.  numpy-only, CPU, `score_fn` = a set [K,20,3] -> logits [K,6] (stepA.py).

THE SEARCH SPACE IS THE ADMITTED ONE.  theta = (u, d): speed scale exp(u) and lateral offset d, composed on the pick's executed plan -- exactly the two perturbation families Step A
measured (speed 0.85-1.15, offset +-0.3 m), so the search cannot leave the region the admission test covered.  The initial widths and the clips are PLACEHOLDERS that the prereg did not
fix (amendment C8); O4's "widen the sampling covariance once" is `sigma0 * widen`.

IDENTITY: n_iters = 0 returns the pick untouched, bit for bit.  INERT (O4): the output differs from the pick by > 0.5 m ADE on < 5 % of sets.
"""
from __future__ import annotations

import math

import numpy as np

import stepA as A

K_ITERS, M_POP, N_ELITE = 5, 64, 8
SEARCH_SEEDS = (A.SEED, A.SEED + 1, A.SEED + 2)             # three search seeds (the prereg fixes the count, not the values)
SIGMA0 = (0.10, 0.20)                                       # (u, d) initial widths: PLACEHOLDER (C8), covers the Step A grid at ~1.5 sigma
U_CLIP = (math.log(0.70), math.log(1.30))
D_CLIP = (-0.60, 0.60)
INERT_ADE_M, INERT_FRAC = 0.5, 0.05


def composite(raw_pick, u: float, d: float, profile: str = A.LAT_PROFILE):
    """speed scale exp(u) then lateral offset d on the pick's executed plan -> (exec [T,3], scorer_in [T,3]); (u, d) = (0, 0) returns the raw pick exactly (C4)."""
    raw = np.asarray(raw_pick, dtype=np.float32)
    ex = A.repair_last_heading_np(raw).astype(np.float64)
    ex = A.scale_speed(ex, float(math.exp(u)))
    ex = A.lateral_offset(ex, float(d), profile)
    ex = A.repair_last_heading_np(ex)
    si = np.array(ex, copy=True)
    si[-1, 2] = float(raw[-1, 2])
    return ex.astype(np.float32), si.astype(np.float32)


def cem_search(score_fn, fan_raw, pick: int, n_iters: int = K_ITERS, m: int = M_POP, n_elite: int = N_ELITE, seed: int = SEARCH_SEEDS[0], sigma0=SIGMA0, widen: float = 1.0,
               profile: str = A.LAT_PROFILE) -> dict:
    """one set, one seed.  Returns {'raw','exec','theta','moved','history'}; 'moved' False means the output IS the pick."""
    fan = np.asarray(fan_raw, dtype=np.float32)
    pick_raw = np.array(fan[pick], copy=True)
    pick_exec = A.repair_last_heading_np(pick_raw)
    mu = np.zeros(2)
    sig = np.asarray(sigma0, dtype=np.float64) * widen
    rng = np.random.default_rng(seed)
    elites = np.zeros((0, 2))
    hist = []
    for _ in range(int(n_iters)):
        th = mu + sig * rng.standard_normal((m, 2))
        th[0] = mu                                                              # warm start: the current mean is always a member
        th[:, 0] = np.clip(th[:, 0], *U_CLIP)
        th[:, 1] = np.clip(th[:, 1], *D_CLIP)
        pop = np.stack([composite(pick_raw, u, d, profile)[1] for u, d in th])
        logits = np.asarray(score_fn(pop), dtype=np.float64)                    # the population is scored as ONE complete m-set
        agg = A.aggregate(logits)
        el = np.argsort(-agg, kind="stable")[:n_elite]
        elites = th[el]
        mu, sig = elites.mean(0), np.maximum(elites.std(0), 1e-3)
        hist.append({"best_agg": float(agg[el[0]]), "mu": mu.tolist()})
    # final pick: the pick and the last elites, each re-scored INSIDE the original fan (replacing the pick's slot)
    base = np.asarray(score_fn(np.array(fan, copy=True)), dtype=np.float64)
    best_v, best = float(A.aggregate(base[pick])), None
    for u, d in elites:
        ex, si = composite(pick_raw, u, d, profile)
        v = float(A.aggregate(A.score_replaced(score_fn, fan, pick, si)[pick]))
        if v > best_v:
            best_v, best = v, (ex, si, (float(u), float(d)))
    if best is None:
        return {"raw": pick_raw, "exec": pick_exec, "theta": (0.0, 0.0), "moved": False, "history": hist}
    return {"raw": best[1], "exec": best[0], "theta": best[2], "moved": True, "history": hist}


def ade_xy(a, b) -> float:
    """mean Euclidean distance over the poses of two executed plans [T,3] (metres)"""
    return float(np.mean(np.linalg.norm(np.asarray(a, np.float64)[:, :2] - np.asarray(b, np.float64)[:, :2], axis=1)))


def inert_verdict(out_execs, pick_execs, thr_m: float = INERT_ADE_M, min_frac: float = INERT_FRAC) -> dict:
    """O4: the search output differs from the pick by > thr_m ADE on fewer than min_frac of the sets -> INERT (reported as INERT, not refuted: the precedent is P4 F1, 1 of 120)."""
    ade = np.array([ade_xy(o, p) for o, p in zip(out_execs, pick_execs)])
    frac = float(np.mean(ade > thr_m)) if len(ade) else float("nan")
    return {"inert": bool(frac < min_frac), "frac_moved": frac, "n": int(len(ade)), "ade_median_m": float(np.median(ade)) if len(ade) else float("nan"), "thr_m": thr_m, "min_frac": min_frac}


def delta_ci(delta_by_token, log_idx, n_logs: int, B: int = A.N_BOOT, seed: int = A.SEED) -> dict:
    """Step B endpoint: mean of (true PDMS of the search output - true PDMS of the pick) with the paired log-cluster bootstrap (ratio of sums, whole logs, the SAME draws every call)."""
    d = np.asarray(delta_by_token, dtype=np.float64)
    li = np.asarray(log_idx)
    num = np.bincount(li, weights=d, minlength=n_logs)
    den = np.bincount(li, minlength=n_logs).astype(np.float64)
    return A.ratio_ci(num, den, A.boot_indices(n_logs, B, seed))


def seed_spread(per_seed_points) -> dict:
    """prereg section 5: the SPREAD across the 3 search seeds is reported beside the interval; an effect below it is not an effect."""
    p = np.asarray(per_seed_points, dtype=np.float64)
    return {"points": p.tolist(), "range": float(p.max() - p.min()), "std": float(p.std(ddof=1)) if len(p) > 1 else float("nan")}
