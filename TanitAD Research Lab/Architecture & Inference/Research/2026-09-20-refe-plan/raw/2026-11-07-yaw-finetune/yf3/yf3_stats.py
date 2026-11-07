"""YF3 statistics and the COMMITTED DECISION RULE (PREREG_YAWFT.md section 6, text quoted next to each condition). numpy only; no I/O.

ESTIMATOR (named): paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927 -- the SFT-4 / M6 estimator. The resample indices are drawn EXACTLY as
`eval/proxy_eval.py::boot` draws them (one `rng.integers(0, G, G)` per resample from `default_rng(seed)`), so every arm and every statistic sees the SAME log resamples, and a mean of
per-token values reproduces `proxy_eval.boot` (tested against the banked result.json). It answers ONE question: would another draw of LOGS say this. It is blind to another TRAINING run
(`H-ESTIM-SEED-1`; hence two seeds and the floor F) and, for this deterministic planner, there is no inference-seed variance (ego probe C1: replicate bit-identical on 265/265 tokens).
"""
from __future__ import annotations

import numpy as np

N_BOOT, SEED = 10000, 20260927
REL_BAR, ABS_BAR_AT_A0, PDMS_NI_MARGIN = 0.05, 0.00094, -1.0          # PREREG section 6 (YF proposals, FINETUNE_PLAN section 6)


class Boot:
    """Group tokens by log; draw the resample matrix once; evaluate any ratio-of-sums or mean statistic on it."""

    def __init__(self, logs, n=N_BOOT, seed=SEED):
        keys = sorted(set(map(str, logs)))
        pos = {k: i for i, k in enumerate(keys)}
        self.gi = np.asarray([pos[str(l)] for l in logs])
        self.G, self.n = len(keys), n
        rng = np.random.default_rng(seed)
        picks = np.empty((n, self.G), dtype=np.int64)
        for b in range(n):
            picks[b] = rng.integers(0, self.G, self.G)
        self.counts = np.zeros((n, self.G), dtype=np.float64)
        for b in range(n):
            self.counts[b] = np.bincount(picks[b], minlength=self.G)

    def gsum(self, x):
        return np.bincount(self.gi, weights=np.asarray(x, np.float64), minlength=self.G)

    def stat(self, num, den=None):
        """per-resample pooled statistic: sum(num)/sum(den) over the resampled tokens (den None -> token mean)"""
        gn = self.gsum(num)
        gd = self.gsum(np.ones(len(self.gi)) if den is None else den)
        return (self.counts @ gn) / (self.counts @ gd)

    def point(self, num, den=None):
        num = np.asarray(num, np.float64)
        return float(num.sum() / (len(num) if den is None else np.asarray(den, np.float64).sum()))


def ci(samples, lo=2.5, hi=97.5):
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def contrast(bt: Boot, arms_b, arms_ref):
    """arms_b / arms_ref: lists (one entry per SEED) of (num, den). Seed-averaged paired contrast mean_s [stat(B_s) - stat(Ref_s)]; also each seed alone and the floor F = |D_s0 - D_s1|."""
    per_seed, pts = [], []
    for (nb, db), (nr, dr) in zip(arms_b, arms_ref):
        s = bt.stat(nb, db) - bt.stat(nr, dr)
        per_seed.append(s)
        pts.append(bt.point(nb, db) - bt.point(nr, dr))
    pooled = np.mean(per_seed, axis=0)
    out = {"point": float(np.mean(pts)), "ci95": ci(pooled), "per_seed_point": [float(p) for p in pts], "per_seed_ci95": [ci(s) for s in per_seed], "n_seeds": len(pts)}
    out["F_seed_floor"] = float(abs(pts[0] - pts[1])) if len(pts) == 2 else None
    return out


def decide(*, gates_ok: bool, gates_red: list, prim: dict, ref_mae: float, pdms: dict, adverse_by_seed: list, sign_nonzero: bool = True) -> dict:
    """The COMMITTED rule, mechanically. prim = contrast() of curvature MAE (1/m), B - Ref (negative = B better); pdms = contrast() of PDMS (x100); adverse_by_seed = list of
    `fam_adverse(...)['adverse']` lists, one per seed. Returns {verdict, conditions, text}."""
    lo, hi = prim["ci95"]
    pt = prim["point"]
    d = prim["per_seed_point"]
    F = prim.get("F_seed_floor")
    cond = {
        "all_gates_green": bool(gates_ok),
        "primary_CI_upper_bound_lt_0": bool(hi < 0),
        "relative_improvement_ge_5pct": bool(ref_mae > 0 and (-pt / ref_mae) >= REL_BAR),
        "abs_improvement_ge_0.00094_(restatement_at_A0_0.0188)": bool(-pt >= ABS_BAR_AT_A0),
        "both_seeds_agree_in_sign": bool(len(d) == 2 and (d[0] < 0) == (d[1] < 0) and d[0] != 0 and d[1] != 0),
        "F_lt_half_abs_mean_delta": bool(F is not None and F < abs(pt) / 2),
        "pdms_delta_CI_lower_bound_ge_-1.0": bool(pdms["ci95"][0] >= PDMS_NI_MARGIN),
        "families_adverse_empty_every_seed": bool(all(len(a) == 0 for a in adverse_by_seed)),
    }
    rel = (-pt / ref_mae) if ref_mae > 0 else float("nan")
    refuted_why = []
    if pdms["ci95"][1] < PDMS_NI_MARGIN:
        refuted_why.append("PDMS upper bound < -1.0")
    if lo > 0:
        refuted_why.append("primary CI lies entirely above 0 (worse)")
    if any(len(a) for a in adverse_by_seed):
        refuted_why.append("a lateral / longitudinal component separates adversely: " + "; ".join(sorted({x['metric'] for a in adverse_by_seed for x in a})))
    gating = ["primary_CI_upper_bound_lt_0", "relative_improvement_ge_5pct", "both_seeds_agree_in_sign", "F_lt_half_abs_mean_delta", "pdms_delta_CI_lower_bound_ge_-1.0", "families_adverse_empty_every_seed"]
    if not gates_ok:
        v, text = "VOID", "a red gate voids the read (PREREG section 4): " + ", ".join(gates_red)
    elif refuted_why:
        v, text = "REFUTED", "; ".join(refuted_why)
    elif all(cond[k] for k in gating):
        v, text = "ADOPT", "all ADOPT conditions hold (B's rung only; claim = on NAVSIM open-loop the yaw input is usable and lowers lateral error, nothing about closed loop)"
    else:
        why = [k for k in gating if not cond[k]]
        v, text = "NOT PROVEN", "unmet: " + ", ".join(why) + " -> advance the rung (Rule Zero)"
    return {"verdict": v, "text": text, "conditions": cond, "relative_improvement": rel, "refuted_reasons": refuted_why,
            "rule": ("ADOPT iff ALL: G1-G8 green; primary Delta(B - C) paired-CI upper bound < 0 AND relative improvement >= 5 % (>= 0.00094 1/m on A0's 0.0188); both seeds agree in sign AND "
                     "F < |mean Delta| / 2; PDMS Delta CI lower bound >= -1.0; families_adverse empty on every longitudinal and lateral component. NOT PROVEN if the primary CI straddles 0, "
                     "or the seeds disagree, or F >= |mean Delta| / 2. REFUTED if the PDMS upper bound < -1.0, or the primary CI lies entirely above 0 (worse), or a lateral / "
                     "longitudinal component separates adversely.")}
