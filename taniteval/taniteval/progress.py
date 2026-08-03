"""EGO PROGRESS for the LONGITUDINAL family — and the convention that decides what it means.

WHY THIS MODULE EXISTS
----------------------
PUBLISHED, [arXiv 2605.00066](https://arxiv.org/html/2605.00066v1) (Wang et al., 2026-04-30):
across **8 methods with complete paired data** on NAVSIM x Bench2Drive, traditional L2 (ADE/FDE)
correlates with the closed-loop Driving Score at **rho = -0.36, p = 0.43 — NOT significant**, while
the PDMS aggregate reaches 0.90 and **Ego Progress ALONE reaches 0.83**, ahead of No-Collision's
0.45. ⚠️ **Scepticism that must travel with those numbers: n = 8, p-values, NO confidence interval,
and two benchmarks differing in sensor suite and scenario mix.** The DIRECTION is established; the
magnitudes are indicative and are **not a baseline anyone may claim to beat.**

Ego progress is a LONGITUDINAL quantity, and 88.7 % of our own measured oracle gap is longitudinal.

⚠️ **WHAT ALREADY EXISTED — probed before writing a line of this (CLAUDE.md rule 2).**

* ``taniteval.pseudosim`` already has an audited, versioned ``ego_progress`` **scoring term**
  (``progress_ratios_per_step``, ``progress_from_ratio``, ``PROGRESS_TERMS``) and already cites the
  same paper. It is a bounded PDM-style **sub-score**, not a reportable metric.
* ``taniteval.driving`` already has ``progress_abs_err_m`` = ``|arclength(pred) - arclength(gt)|``
  — a **metres** path-length difference, unprojected and unnormalised.
* ``taniteval.four_families.longitudinal`` had **neither**.

So this module is NOT a new idea; it is the missing **reporting** form of one the programme already
owns, and it deliberately reuses that constant rather than minting a second one.

⛔ THE CONVENTION, AND WHY IT IS NOT A DETAIL
---------------------------------------------
"Along-track distance covered / the human's" has two inequivalent readings, and they disagree
exactly on curves:

``human_dir`` (**the default here**)
    project BOTH endpoints on the **human's own final-displacement direction**. GT then scores
    **exactly 1.0 by construction**, so the metric is a true *error* with a known optimum, which is
    what makes a GT-vs-CV discrimination control meaningful at all.

``t0_axis`` (**pseudosim's published reading**, kept here so the difference is measurable)
    numerator = the plan's **x-coordinate in the t0 ego frame**; denominator = the human's **chord
    length**. Then ``ratio_GT = cos(theta)``, where ``theta`` is the angle between the human's chord
    and the ego's t0 heading. ⇒ **on a curve the human scores below 1 and is charged for
    under-progress it did not commit**, and an arm that drives straighter than the human scores
    *higher*. That is a curvature confound in a term with weight 5.0.

Both are implemented; :func:`progress` reports the default and carries the other as
``t0_axis_gt_self_ratio`` so the size of the confound is visible in the record instead of argued
about. ⛔ Do not compare a ``human_dir`` number to a published ``t0_axis`` PSS number.

⛔ THE DEGENERATE CASE, AND WHY IT IS NOT CLAMPED
------------------------------------------------
When the human is stopped or crawling the denominator goes to 0 and the ratio explodes. Clamping
would silently turn "the human did not move" into "the arm scored 1.0", manufacturing agreement in
exactly the standing-still windows where longitudinal behaviour matters most — a red light is not a
free pass. Windows below :data:`PROGRESS_HUMAN_MIN_M` are **EXCLUDED and COUNTED**, and the count is
part of the result: a progress number over a silently-shrinking subset is not comparable across arms.

FRAME CONVENTION
----------------
Waypoints are ego-frame metres at the window's last observed pose, ``x`` forward, ``y`` left,
matching ``four_families``. The metric is **dt-invariant** (a ratio of two distances on one grid),
so unlike ``speed_*`` it cannot be corrupted by the sparse-grid dt defect that inflated every
published speed by 5x.
"""
from __future__ import annotations

import numpy as np

__all__ = ["PROGRESS_HUMAN_MIN_M", "CONVENTIONS", "progress", "progress_per_window"]

#: The human chord below which the ratio is undefined. ⛔ This is **pseudosim's published
#: constant, lifted verbatim** (``taniteval.pseudosim.PROGRESS_HUMAN_MIN_M = 0.5``) rather than a
#: second threshold — two thresholds for one concept in one programme is a retraction waiting to
#: happen. ``test_progress.py`` pins the two to be equal, so a future change to one FAILS LOUDLY
#: instead of silently forking the definition.
PROGRESS_HUMAN_MIN_M = 0.5

#: The two readings of "along-track". See the module docstring — they disagree on curves.
CONVENTIONS = ("human_dir", "t0_axis")


def _as2d(a) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError(f"expected [n,H,2] ego-frame metres, got {x.shape}")
    return x


def progress_per_window(pred, gt, convention: str = "human_dir") -> dict:
    """-> per-window arrays. ``pred``/``gt`` are ``[n,H,2]`` ego-frame metres.

    Returns ``ratio``, ``error`` (``|1-ratio|``), ``gt_progress_m``, ``pred_progress_m`` and a
    boolean ``valid`` mask. Invalid entries are ``nan``, never a filled value — a caller that
    forgets the mask gets a nan, not a plausible wrong number.

    ``human_dir``::

        u             = gt[:, -1] / |gt[:, -1]|
        gt_progress   = |gt[:, -1]|
        pred_progress = pred[:, -1] . u

    ``t0_axis`` (pseudosim's published reading)::

        gt_progress   = |gt[:, -1]|          (the human's CHORD)
        pred_progress = pred[:, -1, 0]       (the t0 forward axis)

    ⚠️ ``pred_progress`` may be **negative** — an arm predicting motion opposed to the human's
    direction. It is left signed; a clamp at 0 would hide the pathology.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; known {CONVENTIONS}. Refusing to "
                         f"fall back to a default — a typo must not silently produce a number "
                         f"under the wrong definition.")
    p, g = _as2d(pred), _as2d(gt)
    if p.shape != g.shape:
        raise ValueError(f"pred {p.shape} != gt {g.shape}")
    gt_final = g[:, -1, :]                                   # [n,2]
    gt_prog = np.linalg.norm(gt_final, axis=-1)              # [n] the human's chord
    valid = gt_prog >= PROGRESS_HUMAN_MIN_M
    if convention == "human_dir":
        u = np.zeros_like(gt_final)
        np.divide(gt_final, gt_prog[:, None], out=u, where=valid[:, None])
        pred_prog = np.einsum("nd,nd->n", p[:, -1, :], u)    # signed projection
    else:
        pred_prog = p[:, -1, 0]                              # t0 forward axis
    ratio = np.full_like(gt_prog, np.nan)
    np.divide(pred_prog, gt_prog, out=ratio, where=valid)
    return {"ratio": ratio, "error": np.abs(ratio - 1.0),
            "gt_progress_m": gt_prog,
            "pred_progress_m": np.where(valid, pred_prog, np.nan),
            "valid": valid, "convention": convention}


def progress(pred, gt) -> dict:
    """The reportable EGO PROGRESS block for the LONGITUDINAL family.

    ⛔ ``n_excluded_low_gt_progress`` is part of the result, not a footnote: two arms whose progress
    numbers were computed over different window subsets are not comparable, and this is the only
    field that lets a reader tell.

    ⭐ ``t0_axis_gt_self_ratio`` is the **built-in convention audit**: it is what the GROUND TRUTH
    itself scores under pseudosim's published reading. It is 1.0 only on perfectly straight windows;
    the shortfall IS the curvature confound, measured on the caller's own data rather than asserted.
    """
    w = progress_per_window(pred, gt, "human_dir")
    v = w["valid"]
    n_ok = int(v.sum())
    if n_ok == 0:
        return {
            "status": "UNAVAILABLE",
            "reason": (f"every window's GT chord is below PROGRESS_HUMAN_MIN_M="
                       f"{PROGRESS_HUMAN_MIN_M} m — the human did not move, so a progress RATIO "
                       f"has no denominator. Reported unavailable rather than clamped to 1.0, "
                       f"which would score a stopped arm as perfect."),
            "n": 0, "n_windows": int(len(v)),
        }
    r, e = w["ratio"][v], w["error"][v]
    # the convention audit: GT scored against ITSELF under the t0-axis reading
    gt_self = progress_per_window(gt, gt, "t0_axis")
    gsr = gt_self["ratio"][gt_self["valid"]]
    return {
        "status": "OK",
        "convention": "human_dir",
        "progress_ratio_mean": round(float(r.mean()), 4),
        "progress_ratio_median": round(float(np.median(r)), 4),
        "progress_error_mean": round(float(e.mean()), 4),          # the scoring scalar
        "under_progress_rate": round(float((r < 1.0).mean()), 4),  # timid vs eager, not |.|
        "gt_progress_mean_m": round(float(w["gt_progress_m"][v].mean()), 4),
        "n": n_ok,
        "n_windows": int(len(v)),
        "n_excluded_low_gt_progress": int((~v).sum()),
        "min_progress_m": PROGRESS_HUMAN_MIN_M,
        "t0_axis_gt_self_ratio": round(float(gsr.mean()), 4),
        "t0_axis_gt_self_ratio_note": (
            "what the GROUND TRUTH scores under pseudosim's published t0-axis reading. 1.0 means "
            "the windows are straight; below 1.0 is the CURVATURE CONFOUND — the human charged "
            "for under-progress it did not commit, and an arm that drives straighter than the "
            "human rewarded for it."),
        "definition": ("along-track distance covered / the human's, both projected on the human's "
                       "own final-displacement direction; dt-invariant; <1 = under-drives, "
                       ">1 = over-drives; GT scores exactly 1.0 by construction"),
        "published_motivation": ("arXiv 2605.00066 — Ego Progress alone rho=0.83 vs closed-loop "
                                 "Driving Score while traditional L2 gives rho=-0.36 (p=0.43, "
                                 "n=8, NO CI). Direction only; those magnitudes are NOT a target."),
    }
