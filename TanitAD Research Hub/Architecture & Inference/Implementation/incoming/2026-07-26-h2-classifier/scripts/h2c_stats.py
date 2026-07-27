"""Interval estimators for the H2 classifier — episode-cluster only, never overlapping_holdout_se.

The resampling machinery (`episode_index`, `_draws`) is imported from the program's own
`taniteval/taniteval/ci.py`, so these intervals use literally the same construction as every other
decision-grade number in the program — and as the H2 label work they extend.

What is added here that neither `ci.py` nor `h2e_stats.py` ships: a **paired bootstrap over an
arbitrary reducer**. AP, precision, recall and firing rate are not means of per-row quantities — AP
in particular is a rank statistic over the whole set — so they must be RECOMPUTED inside each
episode draw, for both arms, on the same draw. That is what `paired_stat` does.

⚠️ The unit of resampling is the **episode cluster (= one 20 s clip)**, never the frame. This label
is frame-level and frames inside one clip are strongly dependent, so frame-level resampling would
understate every interval here by more than anywhere else in the program.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", "..", "..", "..", "..", "taniteval"))
if os.path.isdir(_REPO):
    sys.path.insert(0, _REPO)
else:                                      # pod-side copy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taniteval.ci import DEFAULT_N_BOOT, _draws, episode_index   # noqa: E402
from taniteval.rank_metrics import (assert_chance_comparator,  # noqa: E402,F401
                                    chance_ap, comparator_audit)
from taniteval.rank_metrics import average_precision as _ap_pkg   # noqa: E402

ESTIMATOR = "paired_episode_cluster_bootstrap (reducer form; taniteval.ci._draws, B=2000)"
ESTIMATOR_1 = "episode_cluster_bootstrap (taniteval.ci._draws, B=2000)"

#: ⛔ FIXED 2026-07-27. The original implementation broke ties with a STABLE
#: argsort, i.e. by ROW ORDER, while its own docstring claimed to compute what
#: `sklearn.average_precision_score` computes. It does not — sklearn COLLAPSES
#: ties. On the all-tied "chance" comparator (`h2c_eval.py:138`,
#: `np.zeros_like(y)`) a stable sort returns the identity permutation, so the
#: comparator silently became the ranker *"fire the left camera everywhere"* and
#: scored **AP 0.005269 against a base rate of 0.0030527 = 1.7259x chance**.
#: Every AP-vs-chance null in `H2_CLASSIFIER.md` was adjudicated against it.
#: `"row_order"` reproduces the pre-fix numbers bit-for-bit and is kept ONLY for
#: that; no NEW number may be measured with it.
TIE_POLICY = "collapse"


def average_precision(y, s, ties=None):
    """AP — delegated to `taniteval.rank_metrics`, never re-implemented here.

    Ties COLLAPSE into a single precision/recall point (the sklearn definition),
    so the metric cannot read information out of a row order that carries none,
    and a constant score scores exactly the base rate — which is what every
    "is this above chance?" test in this stream assumed and did not have.

    Pass `ties="row_order"` ONLY to reproduce a pre-2026-07-27 committed number.
    """
    return _ap_pkg(y, s, ties=TIE_POLICY if ties is None else ties)


def _legacy_average_precision(y, s):
    """⛔ THE PRE-2026-07-27 IMPLEMENTATION. Reproduction only — never a new number.

    Kept verbatim so every committed H2 number stays bit-reproducible, and so the
    retraction has an executable exhibit. Equivalent to
    `average_precision(y, s, ties="row_order")`.

    Its own docstring (retained below) is the retraction: it claimed to be what
    `sklearn.average_precision_score` computes. sklearn COLLAPSES ties; this does
    not. The paragraph correctly identifies that row-order tie-breaking flatters
    heavily-tied scores and correctly reasons that this is "the safe direction"
    — but it never asked what happens to the score that is tied EVERYWHERE, the
    chance comparator, which is the one number the whole above-chance test rests
    on. C13 class: the failure mode was named and then not applied to the guard.

    -- original docstring --
    AP = sum_k (R_k - R_{k-1}) * P_k over the realised PR points.

    Named exactly (C1: a metric NAME is not a metric DEFINITION): this is the step-interpolated
    AP that `sklearn.average_precision_score` computes, NOT the trapezoidal area under an
    interpolated PR curve and NOT AUROC.

    ⚠️ **Tie handling, stated because it is not neutral.** Ties are broken by a STABLE sort, i.e.
    by row order. In the (camera, frame) layout the left-camera rows come first, and the left
    camera carries the larger share of positives — so a heavily-tied score (the non-learned ego
    baselines, where the two camera rows of a frame share one value) gets a mildly OPTIMISTIC AP.
    That biases in favour of the BASELINES, i.e. against the primary arm, which is the safe
    direction for the verdict this metric decides.
    """
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


def roc_auc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    n1, n0 = float(y.sum()), float((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_stat(fn, eid, n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05, **arrays):
    """Episode-cluster bootstrap CI for ANY reducer `fn(**arrays_subset) -> float`."""
    uniq, idx_by_ep = episode_index(eid)
    point = float(fn(**arrays))
    b = []
    for sel in _draws(uniq, idx_by_ep, n_boot, seed):
        v = fn(**{k: a[sel] for k, a in arrays.items()})
        if np.isfinite(v):
            b.append(v)
    b = np.asarray(b, float)
    lo, hi = (np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if b.size > 50 else (float("nan"), float("nan")))
    return {"point": round(point, 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "n_episodes": int(len(uniq)), "n_boot": int(n_boot), "n_draws_used": int(b.size),
            "estimator": ESTIMATOR_1}


def paired_stat(fn_a, fn_b, eid, n_boot=DEFAULT_N_BOOT, seed=0, alpha=0.05, **arrays):
    """PAIRED episode-cluster bootstrap on `fn_a - fn_b`, both recomputed inside the SAME draw
    so episode difficulty cancels exactly as in `ci.py`'s paired form."""
    uniq, idx_by_ep = episode_index(eid)
    pa, pb = float(fn_a(**arrays)), float(fn_b(**arrays))
    d = []
    for sel in _draws(uniq, idx_by_ep, n_boot, seed):
        sub = {k: a[sel] for k, a in arrays.items()}
        va, vb = fn_a(**sub), fn_b(**sub)
        if np.isfinite(va) and np.isfinite(vb):
            d.append(va - vb)
    d = np.asarray(d, float)
    lo, hi = (np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if d.size > 50 else (float("nan"), float("nan")))
    return {"a": round(pa, 6), "b": round(pb, 6), "delta": round(pa - pb, 6),
            "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "separated": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
            "favours_a": bool(np.isfinite(lo) and lo > 0),
            "n_episodes": int(len(uniq)), "n_boot": int(n_boot), "n_draws_used": int(d.size),
            "estimator": ESTIMATOR}
