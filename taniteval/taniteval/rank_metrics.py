"""TanitEval — rank metrics with HONEST TIE HANDLING, and a guard on the
"chance" comparator.

WHY THIS FILE EXISTS (a post-mortem, not a hypothetical)
--------------------------------------------------------
The H2 sensor-need classifier reported **"⛔ NO ARM IS ABOVE CHANCE"** and that
sentence gated a ~52 GB gated-camera re-download. It is FALSE for the ego arm,
and the cause is four characters of tie handling:

* ``h2c_eval.py:138`` builds the chance comparator as ``chance = np.zeros_like(y)``
  — a constant score — on the documented belief that *"a constant score has AP
  exactly equal to the base rate WITHIN EACH DRAW"*.
* ``h2c_stats.average_precision`` ranks with ``np.argsort(-s, kind="mergesort")``.
  A **stable** sort on an all-tied score returns the IDENTITY permutation, i.e.
  **row order** — and the row order is ``h2c_eval.py:85``'s
  ``[every left-camera row, then every right-camera row]``.

So the "constant score" was really the ranker *"fire the left camera
everywhere"*, and the left camera carries the larger share of positives.
MEASURED (``…/2026-07-27-confirmed-fixes/raw/fix3_chance_comparator.json``, and
reproduced from the committed ``scores_heldout.npz``): the comparator scores
**AP 0.005269** against a base rate of **0.0030527** — **1.7259× chance**. A
TRUE random ranking scores **0.003172** (24 seeds).

Direction of the damage: the comparator is HARDER than chance, so every
``AP − chance`` delta was **understated** and every above-chance null was biased
toward *"not separated"*. For a control, that is bias toward the desired verdict
— the worst possible direction.

THE RULE THIS FILE ENFORCES
---------------------------
**A comparator that claims to be chance must score chance.** :func:`chance_ap`
says what chance is (the base rate); :func:`assert_chance_comparator` refuses a
comparator that does not hit it. A "chance" baseline nobody ever checked against
chance is the same class as C13 — *a guard that cannot fail*.

``ties="collapse"`` vs ``ties="row_order"``
-------------------------------------------
``collapse`` is the correct default and is what
``sklearn.metrics.average_precision_score`` computes: tied scores form ONE
precision/recall point, so the metric cannot read information out of the row
order, which carries none. ``row_order`` reproduces the pre-fix behaviour **and
exists only so committed numbers stay reproducible** — never for a new number.
"""
from __future__ import annotations

import numpy as np

BLOCK = "taniteval.rank_metrics"
VERSION = "1.0.0"

#: how close a putative chance comparator must sit to the base rate. AP is a
#: rank statistic on a finite sample, so an exactly-tied comparator hits the base
#: rate to floating point; a random-ranking comparator scatters around it.
CHANCE_RTOL = 1e-9

TIE_POLICIES = ("collapse", "row_order")


class ComparatorNotChance(AssertionError):
    """A baseline sold as "chance" that does not, in fact, score chance."""


def average_precision(y, s, *, ties: str = "collapse") -> float:
    """Step-interpolated average precision.

    ``ties="collapse"`` (default) — tied scores form a SINGLE precision/recall
    point, identical to ``sklearn.metrics.average_precision_score``. This is the
    only policy under which a constant score scores the base rate.

    ``ties="row_order"`` — the legacy stable-argsort behaviour, in which a tie is
    broken by the order the rows happen to sit in. **Reproduction only.** Row
    order is not evidence; when the rows are laid out ``[all left, all right]``
    and one side carries more positives, this reads the layout as skill.
    """
    if ties not in TIE_POLICIES:
        raise ValueError(f"ties must be one of {TIE_POLICIES}, got {ties!r}")
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    if y.shape != s.shape:
        raise ValueError(f"y {y.shape} and s {s.shape} must have the same shape")
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yt = y[o]
    tp = np.cumsum(yt)
    fp = np.cumsum(1.0 - yt)
    if ties == "collapse":
        # keep only the LAST index of each run of equal scores: a threshold can
        # never split a tie group, so the group contributes one PR point.
        st = s[o]
        keep = np.r_[np.diff(st) != 0, True]
        tp, fp = tp[keep], fp[keep]
    P = tp / np.maximum(tp + fp, 1e-12)
    R = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, R]) * P))


def chance_ap(y) -> float:
    """The AP of a chance ranking on ``y`` — the base rate, and nothing else.

    Under a uniformly random ranking every prefix has expected precision equal
    to the base rate, so E[AP] = base rate. Under a *tied* score with collapsed
    ties there is exactly one PR point, at recall 1 and precision = base rate, so
    AP = base rate identically. Both roads lead here; row order does not.
    """
    y = np.asarray(y, float)
    return float(y.mean()) if y.size else float("nan")


def random_ranking_ap(y, *, n_seeds: int = 24, seed0: int = 10_000) -> dict:
    """AP of a genuinely random ranking over ``n_seeds`` draws.

    The empirical companion to :func:`chance_ap`: it shows the scatter a real
    chance comparator has, which the constant-score comparator does not (it is
    deterministic, and — before this fix — deterministically wrong).
    """
    y = np.asarray(y, float)
    aps = [average_precision(y, np.random.default_rng(seed0 + i).random(y.size))
           for i in range(int(n_seeds))]
    return {"n_seeds": int(n_seeds), "mean": float(np.mean(aps)),
            "p2.5": float(np.percentile(aps, 2.5)),
            "p97.5": float(np.percentile(aps, 97.5)),
            "seed0": int(seed0)}


def comparator_audit(y, s, *, name: str = "comparator",
                     ties: str = "collapse") -> dict:
    """Measure whether ``s`` actually is a chance comparator. Never raises."""
    base = chance_ap(y)
    ap = average_precision(y, s, ties=ties)
    ap_row = average_precision(y, s, ties="row_order")
    s = np.asarray(s, float)
    n_distinct = int(np.unique(s).size)
    return {
        "block": BLOCK, "version": VERSION, "name": name,
        "base_rate": base, "AP": ap, "AP_row_order": ap_row,
        "inflation_vs_chance": (float("nan") if base <= 0 else ap / base),
        "inflation_vs_chance_row_order": (float("nan") if base <= 0
                                          else ap_row / base),
        "n_distinct_scores": n_distinct,
        "is_constant": bool(n_distinct <= 1),
        "is_chance": bool(abs(ap - base) <= CHANCE_RTOL * max(base, 1e-12)),
        "row_order_leaks": bool(abs(ap_row - base) > CHANCE_RTOL * max(base, 1e-12)),
        "_read": ("a constant score is a valid chance comparator ONLY under "
                  "ties='collapse'; under row-order tie-breaking it ranks by "
                  "the array layout, which is what "
                  "h2c_eval.py:138 unknowingly did"),
    }


def assert_chance_comparator(y, s, *, name: str = "comparator",
                             ties: str = "collapse",
                             rtol: float = CHANCE_RTOL) -> dict:
    """Refuse a "chance" baseline whose AP is not the base rate.

    This is the guard that would have caught the H2 defect on day one. It is
    deliberately unwaivable: there is no ``force`` argument, because the person
    who wants to waive it is the person whose null depends on waiving it.
    """
    rec = comparator_audit(y, s, name=name, ties=ties)
    base, ap = rec["base_rate"], rec["AP"]
    if not (abs(ap - base) <= rtol * max(base, 1e-12)):
        raise ComparatorNotChance(
            f"{name!r} is sold as CHANCE but scores AP {ap:.6f} against a base "
            f"rate of {base:.7f} ({rec['inflation_vs_chance']:.4f}x chance)"
            + (" — the score is CONSTANT, so this is a TIE-HANDLING defect: a "
               "stable argsort on an all-tied score ranks by ROW ORDER. Use "
               "ties='collapse'." if rec["is_constant"] else
               " — this comparator carries information and cannot stand in for "
               "chance.")
            + " Every AP-vs-chance delta measured against it is biased, and for "
              "a control the bias runs toward the desired verdict.")
    return rec
