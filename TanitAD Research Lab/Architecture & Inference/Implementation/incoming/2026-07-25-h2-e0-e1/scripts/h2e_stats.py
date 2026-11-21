"""Interval estimators for H2 E1/E0 — episode-cluster only, never overlapping_holdout_se.

The resampling machinery (`episode_index`, `_draws`) is imported from the program's own
`taniteval/taniteval/ci.py`, so these intervals use literally the same construction as every
other decision-grade number in the program. What is added here is a PAIRED reducer for a
RATIO of two conditional rates (the `L1_gate` lift), which ci.py does not ship: its
`paired_episode_cluster_bootstrap` pairs two *arms on the same windows*, whereas the lift
pairs two *disjoint window sets inside the same episodes*. Both arms are recomputed inside the
same episode draw, so episode-level difficulty cancels exactly as in the ci.py paired form.
"""
import sys

import numpy as np

sys.path.insert(0, r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\taniteval")
from taniteval.ci import DEFAULT_N_BOOT, _draws, episode_index   # noqa: E402

ESTIMATOR = "paired_episode_cluster_bootstrap (ratio form; taniteval.ci._draws, B=2000)"
ESTIMATOR_RATE = "episode_cluster_bootstrap (taniteval.ci._draws, B=2000)"


def paired_cluster_lift(resp, gate, eid, n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05, min_n=5):
    """P(resp|gate=1) / P(resp|gate=0), paired episode-cluster bootstrap.

    Point estimate is the FULL-SET value; the bootstrap supplies the interval only.
    Degenerate draws (fewer than `min_n` frames on a side, or a zero denominator) are
    counted and reported rather than silently dropped.
    """
    r = np.asarray(resp, bool)
    m = np.asarray(gate, bool)
    uniq, idx_by_ep = episode_index(eid)
    p1 = float(r[m].mean()) if m.sum() else float("nan")
    p0 = float(r[~m].mean()) if (~m).sum() else float("nan")
    point = p1 / p0 if p0 > 0 else float("nan")
    boots, skipped = [], 0
    for sel in _draws(uniq, idx_by_ep, n_boot, seed):
        mm, rr = m[sel], r[sel]
        if mm.sum() < min_n or (~mm).sum() < min_n:
            skipped += 1
            continue
        b0 = rr[~mm].mean()
        if b0 <= 0:
            skipped += 1
            continue
        boots.append(rr[mm].mean() / b0)
    boots = np.asarray(boots)
    lo, hi = (np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if boots.size > 50 else (float("nan"), float("nan")))
    return {"lift": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "p_resp_given_pos": round(p1, 5), "p_resp_given_neg": round(p0, 5),
            "n_pos": int(m.sum()), "n_neg": int((~m).sum()),
            "n_episodes": int(len(uniq)), "n_boot": int(n_boot),
            "n_draws_used": int(boots.size), "n_draws_skipped": int(skipped),
            "excludes_1_above": bool(lo > 1.0), "excludes_1_below": bool(hi < 1.0),
            "estimator": ESTIMATOR}


def paired_cluster_diff(resp, gate, eid, n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05):
    """P(resp|gate=1) - P(resp|gate=0). Robustness companion to the ratio (no zero-denominator
    degeneracy), same draws, same estimator family."""
    r = np.asarray(resp, float)
    m = np.asarray(gate, bool)
    uniq, idx_by_ep = episode_index(eid)
    point = float(np.mean(r[m]) - np.mean(r[~m]))
    d, skipped = [], 0
    for sel in _draws(uniq, idx_by_ep, n_boot, seed):
        mm, rr = m[sel], r[sel]
        if mm.sum() < 1 or (~mm).sum() < 1:
            skipped += 1
            continue
        d.append(rr[mm].mean() - rr[~mm].mean())
    d = np.asarray(d)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta": round(point, 5), "lo": round(float(lo), 5), "hi": round(float(hi), 5),
            "separated": bool(lo > 0 or hi < 0), "n_episodes": int(len(uniq)),
            "n_boot": int(n_boot), "n_draws_skipped": int(skipped),
            "estimator": "paired_episode_cluster_bootstrap (difference form; taniteval.ci._draws)"}


def cluster_rate(x, eid, n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05):
    """Episode-cluster bootstrap CI on a simple rate mean(x)."""
    v = np.asarray(x, float)
    uniq, idx_by_ep = episode_index(eid)
    point = float(np.nanmean(v))
    b = np.array([float(np.nanmean(v[sel])) for sel in _draws(uniq, idx_by_ep, n_boot, seed)])
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"rate": round(point, 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "n": int(v.size), "n_episodes": int(len(uniq)), "n_boot": int(n_boot),
            "estimator": ESTIMATOR_RATE}


def cluster_share(num, den, eid, n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05):
    """Episode-cluster bootstrap CI on a SHARE num/den where both are per-row 0/1 and the
    denominator is itself a subset (e.g. 'of gate-positive frames, what fraction are
    recoverable-by-crop'). Resamples episodes, recomputes both sums inside the draw."""
    n = np.asarray(num, float)
    d = np.asarray(den, float)
    uniq, idx_by_ep = episode_index(eid)
    point = float(n.sum() / d.sum()) if d.sum() else float("nan")
    b, skipped = [], 0
    for sel in _draws(uniq, idx_by_ep, n_boot, seed):
        dd = d[sel].sum()
        if dd <= 0:
            skipped += 1
            continue
        b.append(n[sel].sum() / dd)
    b = np.asarray(b)
    lo, hi = (np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if b.size > 50 else (float("nan"), float("nan")))
    return {"share": round(point, 5), "lo": round(float(lo), 5), "hi": round(float(hi), 5),
            "n_num": int(n.sum()), "n_den": int(d.sum()), "n_episodes": int(len(uniq)),
            "n_boot": int(n_boot), "n_draws_skipped": int(skipped),
            "estimator": ESTIMATOR_RATE + " [share form]"}
