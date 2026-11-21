#!/usr/bin/env python3
"""Scorer validation battery, selection tier (SCORER_GYM_DESIGN section 4.1; PREREG_SCORER1 section 3/4/8). CPU only.

Input: a per-token table (the C3 `table.npz` joined to the ego probe by `sel_common.load()`): 64 proposals per token with
true per-proposal PDMS / sub-scores and a score per proposal (logits [N,64,7], or any [N,64] surrogate score).

Every interval is the paired LOG-CLUSTER bootstrap (10,000 draws, seed 20260927, 95 % percentile).
  QUESTION IT ANSWERS: "would another draw of LOGS say this?"  It does NOT answer "would another TRAINING run say
  this?" (H-ESTIM-SEED-1: needs a replicate arm) and the planner is deterministic, so there is no inference variance.
  Tier: selection tier (T0/T1 lookup on true per-proposal scores), never a closed-loop number.

Estimator rules baked in:
  * within-token Spearman uses AVERAGE ranks (scipy.stats.rankdata).  NEVER argsort-of-argsort (RETRACTION_LOG 2026-11-07 ~20:50);
    the `legacy` method exists only to reproduce the retracted P1 numbers and is labelled as such.
  * a token enters a head only if its TRUE labels vary (and >= 5 valid proposals); n is reported per head.
  * pick = first index of the max score (torch.argmax semantics), the same rule the programme's banked picks use.
"""
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy.stats import rankdata

SEL_DIR = "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107"
OUT_DIR = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/battery"
B_DRAWS, SEED = 10000, 20260927
W_ZERO, W_RANDOM = 529 / 12146, 11617 / 12146      # navtest stratum weights (selc prereg section 3)
SLOW, OVER = 0.85, 1.15
HI, LO = 0.95, 0.50
KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
HEADS = ("NC", "DAC", "EP", "TTC", "C", "DDC", "lane")      # logit order; true labels exist for the first 6 (table `sub`)
DT = 0.2
NAN3 = (float("nan"),) * 3


# --------------------------------------------------------------------------------------------- bootstrap
class Boot:
    """One shared [B, n_logs] resample of log ids: every arm and every pair reads the SAME draws."""

    def __init__(self, log, stratum_zero, B=B_DRAWS, seed=SEED, w_zero=W_ZERO, w_random=W_RANDOM):
        self.ul, self.li = np.unique(log, return_inverse=True)
        self.nl = len(self.ul)
        self.draws = np.random.default_rng(seed).choice(self.nl, size=(B, self.nl), replace=True)
        self.zero = np.asarray(stratum_zero, bool)
        self.w_zero, self.w_random = w_zero, w_random

    def parts(self, vals, mask):
        vals = np.asarray(vals, np.float64)
        m = np.asarray(mask, bool) & np.isfinite(vals)
        s = np.bincount(self.li[m], weights=vals[m], minlength=self.nl)
        c = np.bincount(self.li[m], minlength=self.nl).astype(np.float64)
        return s, c

    def mean(self, vals, mask):
        """Unweighted ratio estimator sum/count over resampled logs. -> (point, lo, hi)."""
        s, c = self.parts(vals, mask)
        S, C = s[self.draws].sum(1), c[self.draws].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = S / C
        r = r[np.isfinite(r)]
        if len(r) == 0:
            return NAN3
        pt = s.sum() / max(c.sum(), 1e-12)
        return float(pt), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

    def wmean(self, vals, mask):
        """Stratum-weighted mean (W_zero * zeroed + W_random * random); draws that empty a stratum are dropped.
        -> (point, lo, hi, n_draws_used, share_zero, share_random)."""
        mask = np.asarray(mask, bool)
        sz, cz = self.parts(vals, mask & self.zero)
        sr, cr = self.parts(vals, mask & ~self.zero)
        if cz.sum() == 0 or cr.sum() == 0:
            return NAN3 + (0, float("nan"), float("nan"))
        pt = self.w_zero * sz.sum() / cz.sum() + self.w_random * sr.sum() / cr.sum()
        Z = sz[self.draws].sum(1) / np.maximum(cz[self.draws].sum(1), 1e-12)
        R = sr[self.draws].sum(1) / np.maximum(cr[self.draws].sum(1), 1e-12)
        ok = (cz[self.draws].sum(1) > 0) & (cr[self.draws].sum(1) > 0)
        b = (self.w_zero * Z + self.w_random * R)[ok]
        return (float(pt), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(ok.sum()),
                float(sz.sum() / cz.sum()), float(sr.sum() / cr.sum()))

    def summarize(self, vals, mask):
        """The standard block for one per-token statistic: weighted, per stratum, unweighted, with n."""
        vals = np.asarray(vals, np.float64)
        mask = np.asarray(mask, bool) & np.isfinite(vals)
        w = self.wmean(vals, mask)
        z = self.mean(vals, mask & self.zero) if (mask & self.zero).any() else NAN3
        r = self.mean(vals, mask & ~self.zero) if (mask & ~self.zero).any() else NAN3
        u = self.mean(vals, mask)
        return {"weighted": list(w[:3]), "weighted_draws_used": w[3], "zeroed": list(z), "random": list(r),
                "unweighted": list(u), "n": int(mask.sum()), "n_zeroed": int((mask & self.zero).sum()),
                "n_random": int((mask & ~self.zero).sum())}

    def delta(self, va, vb, mask):
        """PAIRED difference a - b on the same tokens (shared draws; a token counts only if defined in BOTH)."""
        va, vb = np.asarray(va, np.float64), np.asarray(vb, np.float64)
        m = np.asarray(mask, bool) & np.isfinite(va) & np.isfinite(vb)
        return self.summarize(np.where(m, va - vb, np.nan), m)


# --------------------------------------------------------------------------------------------- arms
def pick_of(score):
    """First index of the max, as torch.argmax / np.argmax."""
    return np.argmax(np.asarray(score), axis=-1)


class Arm:
    """name, score [N,64] (pick = first argmax). Optional probs [N,64,7] for the per-head statistics."""

    def __init__(self, name, score, probs=None, pick=None):
        self.name = name
        self.score = np.asarray(score)
        self.probs = None if probs is None else np.asarray(probs, np.float64)
        self.pick = pick_of(self.score) if pick is None else np.asarray(pick, np.int64)


def aggregate(logits, lane=True):
    """navsim_v1_lane (planner.py:551-556) in float32 torch, bit-identical to sel_common.agg_torch."""
    import torch
    p = torch.as_tensor(np.asarray(logits, np.float32)).sigmoid()
    w = torch.tensor((5.0, 5.0, 2.0), dtype=p.dtype)
    a = p[..., 0] * p[..., 1] * (p[..., 2] * w[0] + p[..., 3] * w[1] + p[..., 4] * w[2]) / w.sum()
    return (a * p[..., 6] if lane else a).numpy()


def arm_from_logits(name, logits, lane=True):
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits).astype(np.float64)))
    return Arm(name, aggregate(logits, lane), probs=probs)


def regression_arms(T, seed=0):
    """Deliberate-regression arms (must read as known): always-slowest, always-fastest, random, oracle."""
    rng = np.random.default_rng(seed)
    return [Arm("always_slowest", -T["v4_all"]), Arm("always_fastest", T["v4_all"]),
            Arm("random_pick", rng.random(T["pdms"].shape)), Arm("oracle", T["pdms"])]


# --------------------------------------------------------------------------------------------- per-token statistics
def regret_tok(pdms, valid, pick):
    r = np.arange(len(pick))
    oracle = np.where(valid, pdms, -np.inf).max(1)
    return oracle - pdms[r, pick]


def pairwise_acc_tok(pdms, valid, score):
    """Per token: fraction of (>=0.95, <=0.50) true-PDMS proposal pairs the score orders correctly (ties 0.5).
    NaN where either group is empty. Also returns the pair count."""
    hi = valid & (pdms >= HI)
    lo = valid & (pdms <= LO)
    s = np.asarray(score, np.float64)
    gt = (s[:, :, None] > s[:, None, :]).astype(np.float64) + 0.5 * (s[:, :, None] == s[:, None, :])
    pm = hi[:, :, None] & lo[:, None, :]
    npair = pm.sum((1, 2)).astype(np.float64)
    corr = (gt * pm).sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(npair > 0, corr / npair, np.nan)
    return acc, npair


def topk_oracle_tok(pdms, valid, score, ks=KS):
    """Per token: best true PDMS among the score's top-K (stable, ties by index). Returns [N, len(ks)]."""
    order = np.argsort(-np.asarray(score, np.float64), axis=1, kind="stable")
    sp = np.take_along_axis(np.where(valid, pdms, -np.inf), order, 1)
    cm = np.maximum.accumulate(sp, axis=1)
    return cm[:, [k - 1 for k in ks]]


def spearman_tok(X, Y, M, method="tie"):
    """Within-token Spearman of X vs Y over the proposals in M. `tie` = AVERAGE ranks (the pre-registered estimator).
    `legacy` = argsort-of-argsort (P1; NOT tie-safe, kept ONLY to reproduce retracted numbers). NaN if < 5 proposals
    or X or Y is constant."""
    N = X.shape[0]
    out = np.full(N, np.nan)
    for i in np.where(M.any(1))[0]:
        m = M[i]
        if m.sum() < 5:
            continue
        if method == "legacy":
            rx, ry = np.argsort(np.argsort(X[i, m])), np.argsort(np.argsort(Y[i, m]))
        else:
            rx, ry = rankdata(X[i, m]), rankdata(Y[i, m])
        if rx.std() == 0 or ry.std() == 0:
            continue
        out[i] = np.corrcoef(rx, ry)[0, 1]
    return out


def speed_ratio(v4_all, v0):
    return v4_all / np.maximum(v0, 1e-6)[:, None]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lateral_tok(plan, gt, speed_floor=0.5):
    """Lateral family of a picked plan [N,20,3] vs the human future [N,20,3] (ego frame, 0.2 s grid, columns x, y, c2).
    WARNING: column 2 of the REFe PROPOSALS is NOT heading (median |c2 - chord heading| = 1.14 rad at t=4 s on the u400 picks;
    the GT column IS, 0.008 rad), so heading is the CHORD direction for both sides, defined only where the chord > speed_floor.
    heading_err_t4: |wrap(h_plan - h_gt)| at the t=3.8-4.0 s chord (rad).  cross_track: mean over steps of the distance from
    each plan point to the GT polyline (origin + 20 points) (m).  yaw_rate_err / curvature_err: mean |plan - gt| over the steps
    where BOTH consecutive chords exceed speed_floor (a tangent-less standstill step has no yaw rate: RETR-2026-09-26-YAWMASK);
    NaN if no step qualifies. Curvature = yaw rate / chord speed."""
    plan, gt = np.asarray(plan, np.float64), np.asarray(gt, np.float64)
    path = np.concatenate([np.zeros_like(gt[:, :1, :2]), gt[:, :, :2]], axis=1)          # [N,21,2]
    a, b = path[:, :-1][:, None], path[:, 1:][:, None]                                    # [N,1,20,2]
    p = plan[:, :, None, :2]                                                              # [N,20,1,2]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    d = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)                              # [N,20,20]
    out = {"cross_track": d.min(2).mean(1)}

    def chords(P):
        q = np.concatenate([np.zeros_like(P[:, :1, :2]), P[:, :, :2]], 1)
        dq = np.diff(q, axis=1)                                                           # [N,20,2]
        return np.hypot(dq[..., 0], dq[..., 1]) / DT, np.arctan2(dq[..., 1], dq[..., 0])

    vp, hp = chords(plan)
    vg, hg = chords(gt)
    okh = (vp[:, 19] > speed_floor) & (vg[:, 19] > speed_floor)
    out["heading_err_t4"] = np.where(okh, np.abs(wrap(hp[:, 19] - hg[:, 19])), np.nan)
    wp, wg = wrap(np.diff(hp, axis=1)) / DT, wrap(np.diff(hg, axis=1)) / DT               # [N,19]
    ok = (vp[:, 1:] > speed_floor) & (vp[:, :-1] > speed_floor) & (vg[:, 1:] > speed_floor) & (vg[:, :-1] > speed_floor)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["yaw_rate_err"] = np.where(n > 0, (np.abs(wp - wg) * ok).sum(1) / n, np.nan)
        kp, kg = wp / np.maximum(vp[:, 1:], 1e-6), wg / np.maximum(vg[:, 1:], 1e-6)
        out["curvature_err"] = np.where(n > 0, (np.abs(kp - kg) * ok).sum(1) / n, np.nan)
    return out


# --------------------------------------------------------------------------------------------- the battery
def run_arm(T, boot, arm):
    """All selection-tier statistics for one arm. T is the dict from load_table()."""
    N = len(T["token"])
    r = np.arange(N)
    pdms, valid, cru = T["pdms"], T["valid"], T["cruise"]
    allm = np.ones(N, bool)
    res = {"arm": arm.name, "estimator": "paired log-cluster bootstrap, B=%d, seed=%d, 95%% percentile" % (B_DRAWS, SEED),
           "question_answered": "would another draw of LOGS say this? (not: another training run; planner deterministic)",
           "tier": "selection tier (lookup on true per-proposal scores)"}
    rg = regret_tok(pdms, valid, arm.pick)
    res["regret"] = boot.summarize(rg, allm)
    res["pdms_at_pick"] = boot.summarize(pdms[r, arm.pick], allm)
    acc, npair = pairwise_acc_tok(pdms, valid, arm.score)
    res["pairwise_acc"] = boot.summarize(acc, allm)
    res["pairwise_acc"]["definition"] = "per token, share of (true>=0.95, true<=0.50) pairs ordered correctly, ties 0.5; tokens with both groups only"
    res["pairwise_acc_pairs"] = {"total_pairs": int(npair[np.isfinite(acc)].sum()), "tokens_with_both_groups": int(np.isfinite(acc).sum())}
    tk = topk_oracle_tok(pdms, valid, arm.score)
    oracle = np.where(valid, pdms, -np.inf).max(1)
    res["topk_oracle_curve"] = {str(k): {"best_pdms_in_topk": boot.summarize(tk[:, j], allm)["weighted"],
                                         "regret": boot.summarize(oracle - tk[:, j], allm)["weighted"]} for j, k in enumerate(KS)}
    ratio = speed_ratio(T["v4_all"], T["v0"])
    pr = ratio[r, arm.pick]
    slow = (pr < SLOW).astype(float)
    over = (pr > OVER).astype(float)
    bias = T["v4_all"][r, arm.pick] - T["v4_gt"]
    res["longitudinal"] = {
        "cruise_slow_share": boot.summarize(slow, cru), "overspeed_share": boot.summarize(over, cru),
        "v4_bias_plan_minus_gt_mps": boot.summarize(bias, cru & T["gt_valid"]), "mean_plan_v4_over_v0": boot.summarize(pr, cru),
        "all_tokens_slow_share": boot.summarize(slow, allm),
        "distance_keeping": "UNAVAILABLE at this tier (no lead track on navtest)"}
    lat = lateral_tok(T["traj"][r, arm.pick], T["gt_pose20"], 0.5)
    res["lateral"] = {k: boot.summarize(v, T["gt_valid"]) for k, v in lat.items()}
    res["tactical"] = "n/a at this tier: the C3 table carries no tactical labels or tactical head (computed where available in the closed-loop read)"
    res["strategic"] = "PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25); not a pass, not omitted"
    if arm.probs is not None:
        res["spearman_tie_vs_v4_over_v0_cruise"] = spearman_block(T, boot, arm.probs, ratio)
    return res, dict(regret=rg, slow=slow, acc=acc, bias=bias, over=over, pr=pr, pdms_pick=pdms[r, arm.pick])


def spearman_block(T, boot, probs, ratio, method="tie"):
    """Per head: within-token Spearman of the predicted prob and of the TRUE sub-score vs v4/v0 on CRUISE, and pred minus true.
    A token enters a head only if BOTH are defined (true labels vary); n reported per head."""
    cru, valid = T["cruise"], T["valid"]
    cm = cru[:, None] & valid
    out = {}
    for j, h in enumerate(HEADS[:6]):
        rp = spearman_tok(probs[..., j], ratio, cm, method)
        rt = spearman_tok(T["sub"][..., j], ratio, cm, method)
        ok = np.isfinite(rp) & np.isfinite(rt)
        gap = np.where(ok, rp - rt, np.nan)
        out[h] = {"pred": boot.mean(np.where(ok, rp, np.nan), cru), "true": boot.mean(np.where(ok, rt, np.nan), cru),
                  "gap_pred_minus_true": boot.mean(gap, cru), "n_tokens_paired": int(ok.sum()),
                  "n_pred": int(np.isfinite(rp[cru]).sum()), "n_true": int(np.isfinite(rt[cru]).sum()),
                  "pred_unpaired_mean": float(np.nanmean(rp[cru])),
                  "true_unpaired_mean": float(np.nanmean(rt[cru])) if np.isfinite(rt[cru]).any() else None}
    rl = spearman_tok(probs[..., 6], ratio, cm, method)
    out["lane"] = {"true": "NO TRUTH in the C3 table (see LANE_TRUTH.md)", "pred_vs_speed": boot.mean(rl, cru),
                   "n_pred": int(np.isfinite(rl[cru]).sum())}
    return out


def pair_vs(boot, T, base_tok, arm_tok):
    """Paired a - b deltas (a = arm, b = base) on the tokens both define: every number answers the log-draw question."""
    N = len(T["token"])
    allm = np.ones(N, bool)
    cru = T["cruise"]
    return {"regret_delta": boot.delta(arm_tok["regret"], base_tok["regret"], allm),
            "pdms_at_pick_delta": boot.delta(arm_tok["pdms_pick"], base_tok["pdms_pick"], allm),
            "cruise_slow_share_delta": boot.delta(arm_tok["slow"], base_tok["slow"], cru),
            "pairwise_acc_delta": boot.delta(arm_tok["acc"], base_tok["acc"], allm),
            "v4_bias_delta_mps": boot.delta(arm_tok["bias"], base_tok["bias"], cru & T["gt_valid"])}


def load_table():
    sys.path.insert(0, SEL_DIR)
    import sel_common as sc
    T = sc.load()
    Z = np.load(sc.EGO + "/full/results.npz", allow_pickle=False)       # same alignment as sc.load(): by token string
    zt = {str(t): i for i, t in enumerate(Z["token"])}
    zi = np.array([zt[t] for t in T["token"]])
    T["gt_pose20"] = Z["gt_pose20"][zi].astype(np.float64)
    return T, sc


def make_boot(T):
    return Boot(T["log"], T["stratum"] == "zero")


def run_all(T, boot):
    arms = [arm_from_logits("u400", T["logits"], lane=True), arm_from_logits("u400_no_lane_factor", T["logits"], lane=False)]
    arms += regression_arms(T)
    out, toks = {}, {}
    for arm in arms:
        out[arm.name], toks[arm.name] = run_arm(T, boot, arm)
    out["paired_vs_u400"] = {n: pair_vs(boot, T, toks["u400"], toks[n]) for n in toks if n != "u400"}
    out["meta"] = {"n_tokens": int(len(T["token"])), "n_cruise": int(T["cruise"].sum()), "n_logs": int(boot.nl),
                   "weights": {"W_zero": W_ZERO, "W_random": W_RANDOM}, "B": B_DRAWS, "seed": SEED,
                   "banked_pick_reproduced": int((arms[0].pick == T["pick"]).sum())}
    return out, arms, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR + "/battery_u400.json")
    a = ap.parse_args()
    T, _ = load_table()
    out, _, _ = run_all(T, make_boot(T))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(a.out, hashlib.sha256(open(a.out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
