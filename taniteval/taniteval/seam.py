"""TanitEval — X2: the BAND-SEAM instrument. **Verify, never repair.**

WHY THIS MODULE EXISTS (F-16, DIAGRAM_CONFORMANCE.md:131)
---------------------------------------------------------
The v6 architecture claims the 0–2 s (operative) and 2–6 s (tactical) bands are
**seam-free BY CONSTRUCTION**: they are SLICES of ONE 60-step ``(a, κ)`` @ 10 Hz
rollout (``V6Config.split_bands`` returns *views* — "materialising it as a copy
is how a 'seam' gets invented"), and no seam-repair term exists anywhere in the
loss. The conformance audit found that claim ✅ in the code and ⬜ **UNVERIFIED
BY ANY INSTRUMENT**: the trainer logs ``plan_ade_0_2s`` / ``plan_ade_2_6s``, but
two band ADEs cannot see a discontinuity — an arm can have identical band ADEs
and still jump at the boundary.

⛔ **THIS MODULE EXISTS TO FALSIFY THE CLAIM, NOT TO CONFIRM IT.** It is an
INSTRUMENT: it adds no loss term, it returns no gradient, and nothing in it may
ever become a seam-repair mechanism. If it finds a seam, the finding is
REPORTED — the architecture is what gets re-examined, not the number.

THE NULL, STATED SO IT CAN FAIL
-------------------------------
    H0  the band boundary is EXCHANGEABLE with every other step boundary of the
        same rollout: the discontinuity at the 2 s edge is drawn from the same
        distribution as the WITHIN-BAND step-to-step discontinuities.

⚠️ **"non-zero" is not "seam".** Every real control sequence has non-zero
step-to-step differences everywhere; a boundary that merely differs from zero
proves nothing. A seam is a boundary whose discontinuity is an **OUTLIER**
against the within-band null. Everything here is therefore a CONTRAST between
the seam boundary and the within-band boundaries **of the same window** — which
is also why the estimator is the PAIRED episode-cluster bootstrap.

THE STATISTIC
-------------
For a per-step scalar channel ``x[0..T-1]`` the order-``m`` discontinuity at
boundary index ``i`` (the boundary sitting between step ``i-1`` and step ``i``)
is the m-th finite difference anchored at ``i-1``::

    D1(i) = |x[i] - x[i-1]|                              level  jump
    D2(i) = |x[i+1] - 2x[i] + x[i-1]|                    slope  jump
    D3(i) = |x[i+2] - 3x[i+1] + 3x[i] - x[i-1]|          curvature jump

valid for ``i in [1, T-m]``; ``D_m(i) == abs(np.diff(x, n=m))[i-1]``, which is
how it is computed (one line, no hand-rolled stencil). The **seam boundary is
i = 20** — between plan step 19 (last operative step, t = 1.9 s) and plan step 20
(first tactical step, t = 2.0 s), because ``V6Config.band_slice`` is
``op = slice(0, 20)`` / ``tac = slice(20, 60)``.

Per window we then form

    ``d_seam``    = D_m(20)
    ``null_ref``  = **MEAN** of D_m(i) over the null boundaries i ≠ 20
    ``excess``    = d_seam − null_ref                      (the paired contrast)
    ``rank``      = mid-rank of d_seam among the null, normalised to [0, 1]
    ``top1``      = 1 if d_seam strictly exceeds every null boundary

⚠️ **THE NULL REFERENCE IS THE MEAN, NOT THE MEDIAN — and that is a MEASURED
correction, not a taste.** The first build of this module used the per-window
MEDIAN of the null, which looks like the robust choice. It is not admissible
here: for the right-skewed ``|Δ^m x|`` distribution ``E[median of n draws] <
E[draw]``, so ``E[d_seam − median(null)] > 0`` **under H0**. MEASURED
2026-08-16 on the self-test's genuine single-rollout arm (144 windows / 12
episodes, OU controls, no seam present): the median reference produced
``a/d1 excess +0.0387 [+0.0174, +0.0587]`` and ``kappa/d1 +0.0028 [+0.0018,
+0.0039]`` — **intervals separated from zero on a trajectory with no seam in
it.** A statistic that is positive under its own null is a seam detector that
finds seams in seamless rollouts. The MEAN reference is exactly unbiased under
exchangeability (``E[d_seam] = E[d_i]`` for every i ⇒ ``E[excess] = 0``) and is
what ships. The median survives only as an informational ``null_median`` column.

and report the decision-grade readouts, each with its estimator attached:

* ``excess_ci``   :func:`taniteval.ci.paired_episode_cluster_bootstrap` on
                  (d_seam, null_ref) — the SAME resampled episodes for both
                  arms, because they are the same windows. **Physical units.**
* ``excess_rel_ci``  the identical paired bootstrap on the SCALE-NORMALISED
                  arms (both divided by the global within-band scale). Because
                  the normalisation is by a constant the bootstrap is
                  equivariant — this is the same interval in units of "a
                  typical within-band step", and it is the one the VERDICT
                  reads, so a channel whose physical units are ~1e-2 (κ, in
                  1/m) is not adjudicated at the 4-dp display resolution the
                  programme publishes physical numbers at.
* ``rank_ci``     :func:`taniteval.ci.episode_cluster_bootstrap` on the rank;
                  under H0 the expectation is **0.5** exactly, whatever the
                  signal's scale. Distribution-free, and unbiased where the
                  magnitude contrast needs care.
* ``top1_ci``     the same on the top-1 indicator; under H0 the rate is
                  ``1/(n_null+1)``.
* ``power``       the bootstrap SE of the excess, the **MDE at 80 % power**, and
                  whether that MDE is below the materiality floor.

TWO NULLS, BECAUSE EXCHANGEABILITY IS AN ASSUMPTION
---------------------------------------------------
``global`` uses every other boundary; ``local`` uses only ``|i - 20| <=
halfwidth``. If the emitted control profile has a smooth index trend (early
steps systematically more active than late ones), the GLOBAL null is not
exchangeable and can manufacture a rank effect with no seam present. The LOCAL
null removes any such trend to first order. **A real seam fires on both.** A
finding that appears only in the global null is an index trend, and the record
says so rather than leaving the reader to discover it.

⛔ THE ESTIMATOR (CLAUDE.md, binding)
-------------------------------------
Every interval is delegated **verbatim** to ``taniteval/ci.py`` — the
episode-cluster bootstrap over the val episodes, paired where two quantities
live on the same windows. ``overlapping_holdout_se`` is **never** used and is
not importable from here: it narrows the interval 1.107–3.100× (median 1.499×)
*and* biases the point estimate bidirectionally, up to a SIGN FLIP on paired
deltas. The seam contrast IS a paired delta, i.e. exactly the shape that lesson
was measured on (``ctx→tactical`` +0.0439 → true +0.0148).

⛔ AN INSTRUMENT THAT CANNOT FAIL IS WORTHLESS (the C13 family)
---------------------------------------------------------------
Three mechanisms keep this one falsifiable, and all three are MEASURED, not
asserted:

1. **A three-valued verdict.** ``SEAM`` / ``NO_MATERIAL_SEAM`` / ``INCONCLUSIVE``
   — a null result is only ``NO_MATERIAL_SEAM`` when the MDE at 80 % power is
   at or below the materiality floor, i.e. when the test COULD have seen a
   material seam and did not. An under-powered null returns ``INCONCLUSIVE``.
   (The ``o6_rank_verdict`` precedent: a gate that can say "I do not know".)
2. **The boundary scan** (:func:`boundary_scan`) re-runs the identical rule with
   every OTHER boundary standing in for the seam. That measures the rule's own
   false-positive rate ON THE SAME DATA, and it names the boundary with the
   largest excess — so "the hotspot is not at 20" is a positive statement.
3. **A seam-injection validation** — the instrument is exercised against two
   independently-rolled bands concatenated (``stack/tests/test_v6_seam_probe.py``
   and ``seam_probe.py --self-test``). An instrument never shown to detect the
   defect it hunts is not validated.

MATERIALITY
-----------
``scale`` is the full-set mean of the per-window ``null_ref`` — "a typical
within-band step" — and the floor is ``k * scale`` with ``k`` defaulting to
:data:`MATERIALITY_K` = 1.0. Excess > floor therefore means "the boundary step
is more than **twice** a typical within-band step". In the normalised units the
verdict actually reads, the floor is exactly ``k``. The floor is scale-free (it
works for ``a`` in m/s², for ``κ`` in 1/m and for waypoint metres without three
separate hand-tuned thresholds) and it is stamped into every record, so a reader
always sees the bar the verdict was read against.

⚠️ **EXCHANGEABILITY IS AN ASSUMPTION AND THE INSTRUMENT SAYS SO.** MEASURED on
the self-test's genuine arm: ``wp_x`` at order 1 (the step displacement, ≈ v·dt)
carries a strong index TREND, and against the global null its excess reads
``−0.1563 [−0.2257, −0.0990]`` — separated, with no seam present. Against the
local null it shrinks to ``−0.0367``, and at order ≥ 2 (which annihilates a
linear trend) the local-null interval straddles zero. That is the whole reason
both nulls are reported and why a row counts as CONFIRMED only when both fire.

TIER
----
The continuity blocks consume the emitted 60-step plan and **nothing else** — no
recorded future actions, no future frames, no ground truth. They are therefore
**tier-invariant**: there is no teacher-forcing channel for them to be
contaminated by, and the same numbers come out at T0 and T1. They are stamped
``T1`` (the primary tier) with ``tier_invariant: true`` rather than left
unstamped. The **band-error** block compares the plan against the GT future and
IS a capability-adjacent number, so it inherits the DUMP's declared tier and an
undeclared tier is a hard error at the CLI — an un-tiered number is exactly what
the doctrine forbids.

Only numpy. No torch import, no pod paths — so the statistics are unit-testable
anywhere. Torch tensors are accepted and converted (the ``selgap`` convention).
"""
from __future__ import annotations

import math

import numpy as np

from . import ci as _ci

__all__ = [
    "BLOCK", "VERSION",
    "PLAN_STEPS", "DT", "SEAM_BOUNDARY", "OP_BAND_S", "TAC_BAND_S",
    "ORDERS", "ORDER_NAMES", "MATERIALITY_K", "POWER_TARGET", "MDE_Z",
    "LOCAL_HALFWIDTH", "SCAN_N_BOOT", "VERDICTS",
    "MIN_EPISODES_FOR_CLEAN_BILL", "seam_boundary_of", "band_errors_paired",
    "boundary_diffs", "seam_and_null", "continuity", "continuity_panel",
    "boundary_scan", "band_errors", "control_channels", "seam_report",
]

BLOCK = "taniteval.seam"
VERSION = "1.0.0"

# --- the §4b horizon contract, mirrored from tanitad.models.v6 -------------- #
#: 60 steps @ 10 Hz = 6.0 s — ONE rollout, ONE integrator.
PLAN_STEPS = 60
DT = 0.1
OP_BAND_S = (0.0, 2.0)
TAC_BAND_S = (2.0, 6.0)
#: ⭐ THE BOUNDARY UNDER TEST. ``V6Config.band_slice`` is ``op = slice(0, 20)``
#: and ``tac = slice(20, 60)``, so the band edge sits BETWEEN step 19 and step
#: 20 — i.e. boundary index 20 in this module's convention. Derived, not
#: guessed: ``seam_boundary_of(op_band_s, tac_band_s, dt)`` recomputes it and
#: the CLI refuses a dump whose declared geometry disagrees.
SEAM_BOUNDARY = 20

#: finite-difference orders reported by default. 1 = level jump, 2 = slope jump,
#: 3 = curvature jump. For an INTEGRATED channel (waypoints) the correspondence
#: shifts by one: a control-level jump shows as order 2 in position space.
ORDERS: tuple[int, ...] = (1, 2, 3)
ORDER_NAMES = {1: "level", 2: "slope", 3: "curvature"}

#: excess > MATERIALITY_K * within-band scale == "the boundary step is more than
#: (1 + k)x a typical within-band step". Declarable at the CLI; always stamped.
MATERIALITY_K = 1.0
POWER_TARGET = 0.80
_Z_ALPHA = 1.9599639845400545       # two-sided 95 %
_Z_POWER = 0.8416212335729143       # 80 % power
#: MDE multiplier: an effect of MDE_Z * SE is detected with POWER_TARGET power
#: at the two-sided 5 % level, under a normal approximation to the bootstrap
#: distribution (stated as an approximation; validated by the injection test).
MDE_Z = _Z_ALPHA + _Z_POWER         # 2.8015852181129688

#: ⛔ THE FLOOR ON A CLEAN BILL. Below this many episode CLUSTERS the verdict may
#: not be ``NO_MATERIAL_SEAM``; it becomes ``INCONCLUSIVE`` and says why.
#:
#: The reason is combinatorial, not a taste: a cluster bootstrap that resamples
#: ``n`` episodes with replacement has only ``C(2n-1, n)`` DISTINCT resamples —
#: **3** at n=2, 10 at n=3, 35 at n=4 — so its 2.5th percentile is supported on a
#: handful of values and cannot approximate a 95 % interval at all. At n=8 there
#: are 6 435, which is comfortably past the point where the percentile is
#: carrying information rather than arithmetic.
#:
#: ⚠️ MEASURED here, because the failure is silent rather than loud: on i.i.d.
#: exchangeable data with 2 episodes × 2 windows the bootstrap SE collapsed to
#: **0.019× the within-band scale**, i.e. an 80 %-power MDE of **0.054×** — a
#: claim of "well-powered null" off FOUR windows. An under-covering interval
#: does not look broken; it looks like a very precise result. (Same family as
#: the ``overlapping_holdout_se`` lesson one level up.)
MIN_EPISODES_FOR_CLEAN_BILL = 8

#: the LOCAL null's half-width in boundaries: i in [20-h, 20+h] \\ {20}.
LOCAL_HALFWIDTH = 5
#: the boundary scan is a CALIBRATION block, not a decision block, so it runs a
#: cheaper bootstrap and says so in its own record.
SCAN_N_BOOT = 400

VERDICTS = ("SEAM", "NO_MATERIAL_SEAM", "INCONCLUSIVE", "DEGENERATE")

_READ = ("X2 seam instrument — VERIFY, NEVER REPAIR. The seam is an OUTLIER "
         "test against the within-band null, not a non-zero test. Intervals "
         "are taniteval/ci.py's episode-cluster bootstrap (paired where the "
         "two quantities share windows), NEVER overlapping_holdout_se.")


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _to_numpy(x, dtype=np.float64):
    """numpy view of ``x``; torch tensors are detached + moved to CPU first."""
    if hasattr(x, "detach"):                    # torch.Tensor, no torch import
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _phi(z: float) -> float:
    """Standard normal CDF via ``math.erf`` — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _power_at(delta: float, se: float) -> float:
    """Two-sided power to reject ``mean excess == 0`` at an effect of ``delta``.

    Normal approximation to the episode-cluster bootstrap distribution. Stated
    as an approximation on purpose: the bootstrap distribution of a mean over
    40 episode clusters is close to normal but not exactly so, and the honest
    validation of this number is the injection test, not the formula.
    """
    if not np.isfinite(se) or se <= 0.0:
        return float("nan")
    t = abs(delta) / se
    return float(_phi(t - _Z_ALPHA) + _phi(-t - _Z_ALPHA))


def seam_boundary_of(op_band_s=OP_BAND_S, tac_band_s=TAC_BAND_S, dt=DT) -> int:
    """The boundary index the two bands meet at — DERIVED from the geometry.

    Mirrors ``V6Config.band_slice``: the operative band is ``slice(0, hi/dt)``
    and the tactical band starts there, so the seam boundary index is
    ``round(op_hi / dt)``. Refuses a gap or an overlap, exactly as
    ``V6Config.__post_init__`` does ("a gap or an overlap here IS the stitched-
    trajectory defect") — an instrument that silently probed the wrong index
    would report "no seam" for the best possible reason and the worst.
    """
    op_lo, op_hi = float(op_band_s[0]), float(op_band_s[1])
    tac_lo, tac_hi = float(tac_band_s[0]), float(tac_band_s[1])
    if op_lo != 0.0:
        raise ValueError(f"operative band must start at 0.0 s, got {op_lo}")
    if abs(tac_lo - op_hi) > 1e-9:
        raise ValueError(
            f"band gap/overlap: operative ends at {op_hi} s, tactical starts "
            f"at {tac_lo} s. A gap or an overlap IS the stitched-trajectory "
            f"defect this instrument exists to detect — fix the geometry, do "
            f"not probe past it.")
    if tac_hi <= tac_lo:
        raise ValueError(f"tactical band is empty: {tac_band_s}")
    return int(round(op_hi / float(dt)))


def boundary_diffs(x, order: int = 1):
    """``x`` [..., T] -> ``(d [..., T-order], idx [T-order])``.

    ``d[..., j]`` is ``|Δ^order x|`` anchored at boundary ``idx[j] = j + 1``,
    i.e. the order-th finite difference whose leftmost sample is step
    ``idx[j] - 1``. ``np.diff`` IS the stencil — writing one by hand is how a
    convention drifts.
    """
    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    a = _to_numpy(x)
    if a.ndim < 1:
        raise ValueError("x must have a step axis")
    if a.shape[-1] <= order:
        raise ValueError(f"series of length {a.shape[-1]} cannot carry an "
                         f"order-{order} difference (needs > {order})")
    d = np.abs(np.diff(a, n=order, axis=-1))
    return d, 1 + np.arange(d.shape[-1], dtype=np.int64)


def seam_and_null(d, idx, seam: int = SEAM_BOUNDARY, halfwidth=None):
    """Split an order-m difference array into (seam column, null columns).

    ``halfwidth`` None -> the GLOBAL null (every boundary except the seam);
    an int h -> the LOCAL null ``|i - seam| <= h``, seam excluded. The local
    null is what makes an index TREND separable from a seam.
    """
    d = _to_numpy(d)
    idx = np.asarray(idx, dtype=np.int64)
    if d.shape[-1] != idx.size:
        raise ValueError(f"d has {d.shape[-1]} boundaries but idx has "
                         f"{idx.size}")
    where = np.flatnonzero(idx == int(seam))
    if where.size != 1:
        raise ValueError(
            f"seam boundary {seam} is not among the {idx.size} valid "
            f"boundaries [{idx[0]}, {idx[-1]}] for this order — the order is "
            f"too high for the series length, or the seam index is wrong. "
            f"Refusing rather than probing a different boundary.")
    pos = int(where[0])
    keep = np.ones(idx.size, dtype=bool)
    keep[pos] = False
    if halfwidth is not None:
        keep &= np.abs(idx - int(seam)) <= int(halfwidth)
    if not keep.any():
        raise ValueError(f"the null is empty (seam={seam}, "
                         f"halfwidth={halfwidth}) — nothing to compare against")
    return d[..., pos], d[..., keep], idx[keep]


def _rank_and_top1(d_seam, d_null):
    """Mid-rank of the seam among its null, normalised to [0, 1], + top-1 flag.

    Mid-rank (``strictly_less + 0.5 * ties``) is what makes a DEGENERATE
    all-equal signal read 0.5 — the H0 value — instead of 0.0 or 1.0. A tie
    convention that pushed ties to one end would manufacture a verdict on a
    zero-init emission head, whose controls are exactly 0 at every step.
    """
    s = np.asarray(d_seam, dtype=np.float64)[..., None]
    n = np.asarray(d_null, dtype=np.float64)
    less = (n < s).sum(axis=-1).astype(np.float64)
    ties = (n == s).sum(axis=-1).astype(np.float64)
    rank = (less + 0.5 * ties) / float(n.shape[-1])
    top1 = (n < s).all(axis=-1).astype(np.float64)
    return rank, top1


# --------------------------------------------------------------------------- #
# the headline block                                                           #
# --------------------------------------------------------------------------- #
def continuity(x, eid, *, seam: int = SEAM_BOUNDARY, order: int = 1,
               halfwidth=None, channel: str | None = None,
               materiality_k: float = MATERIALITY_K,
               n_boot: int = _ci.DEFAULT_N_BOOT, seed: int = 0,
               alpha: float = 0.05, units: str | None = None) -> dict:
    """Seam-vs-within-band contrast for ONE channel at ONE difference order.

    Parameters
    ----------
    x : [N, T] array/tensor
        The per-window, per-step scalar channel (``a``, ``κ``, ``wp_x`` …).
    eid : [N]
        Episode id per window — the RESAMPLING UNIT of every interval here.
    seam : int
        Boundary index under test (default :data:`SEAM_BOUNDARY` = 20).
    order : int
        Finite-difference order — 1 level, 2 slope, 3 curvature.
    halfwidth : int | None
        None = the GLOBAL null; h = the LOCAL null ``|i - seam| <= h``.
    materiality_k : float
        The floor multiplier (see the module docstring).

    Returns
    -------
    dict — every number with its estimator, its n and its verdict:
      ``seam_mean`` / ``null_scale``  the two arms' full-set means
      ``excess`` / ``excess_ci``      the PAIRED episode-cluster bootstrap
      ``ratio``                       seam_mean / null_scale (interpretation)
      ``rank_ci`` / ``top1_ci``       distribution-free companions (H0 0.5 and
                                      ``1/(n_null+1)``)
      ``power``                       se, MDE at 80 %, floor, power_at_floor
      ``verdict``                     one of :data:`VERDICTS`
      ``notes``                       every caveat that applies, in words

    ⛔ This function MEASURES. It returns no gradient and touches no loss.
    """
    a = _to_numpy(x)
    if a.ndim != 2:
        raise ValueError(f"x must be [N, T] per-window per-step values, got "
                         f"{a.shape}")
    n = a.shape[0]
    if len(eid) != n:
        raise ValueError(f"eid/x length mismatch: {len(eid)} vs {n}")
    if n == 0:
        raise ValueError("continuity needs at least one window")

    d, idx = boundary_diffs(a, order)
    d_seam, d_null, null_idx = seam_and_null(d, idx, seam, halfwidth)
    # ⚠️ MEAN, not median — the median reference is BIASED UPWARD under H0.
    # See the module docstring: MEASURED on a seam-free rollout it produced
    # intervals separated from zero. E[d_seam - mean_i(d_i)] == 0 exactly under
    # exchangeability, whatever the shape of the |Δ^m x| distribution.
    null_ref = d_null.mean(axis=-1)                             # [N]
    null_med = np.median(d_null, axis=-1)                       # [N] info only
    excess_pw = d_seam - null_ref                               # [N] paired
    rank_pw, top1_pw = _rank_and_top1(d_seam, d_null)

    seam_mean = float(np.mean(d_seam))
    null_scale = float(np.mean(null_ref))

    # THE record interval, in PHYSICAL units — paired, because both arms are the
    # SAME windows (the shared per-window difficulty cancels inside each draw,
    # and a quadrature combination of two single-arm CIs would be invalid).
    excess_ci = _ci.paired_episode_cluster_bootstrap(
        d_seam, null_ref, eid, n_boot=n_boot, seed=seed, alpha=alpha)

    # THE VERDICT interval, in units of a typical within-band step. Dividing
    # BOTH arms by the same CONSTANT makes the bootstrap exactly equivariant —
    # this is the identical interval, re-expressed — and it moves the
    # adjudication off the programme's 4-dp physical display resolution, which
    # for kappa (~1e-2 1/m) is coarse enough to decide a verdict by rounding.
    if null_scale > 0.0:
        rel_ci = _ci.paired_episode_cluster_bootstrap(
            d_seam / null_scale, null_ref / null_scale, eid,
            n_boot=n_boot, seed=seed, alpha=alpha)
        se_src = _ci.episode_cluster_bootstrap(
            excess_pw / null_scale, eid, reduce="mean", n_boot=n_boot,
            seed=seed, alpha=alpha)
        rel_floor = float(materiality_k)
        rel_lo = float(rel_ci["lo"])
        se_rel = float(se_src["se"])
        se_phys = se_rel * null_scale
    else:
        # a perfectly flat within-band null: there is no scale to normalise by,
        # so the physical interval IS the verdict interval and the floor is 0.
        rel_ci, se_src = None, _ci.episode_cluster_bootstrap(
            excess_pw, eid, reduce="mean", n_boot=n_boot, seed=seed,
            alpha=alpha)
        rel_floor = float("nan")
        rel_lo = float(excess_ci["lo"])
        se_rel = float("nan")
        se_phys = float(se_src["se"])

    rank_ci = _ci.episode_cluster_bootstrap(
        rank_pw, eid, reduce="mean", n_boot=n_boot, seed=seed, alpha=alpha)
    top1_ci = _ci.episode_cluster_bootstrap(
        top1_pw, eid, reduce="mean", n_boot=n_boot, seed=seed, alpha=alpha)

    n_null = int(d_null.shape[-1])
    top1_null_rate = 1.0 / float(n_null + 1)
    floor = float(materiality_k) * null_scale
    mde80_phys = MDE_Z * se_phys if np.isfinite(se_phys) else float("nan")
    mde80_rel = (MDE_Z * se_rel if np.isfinite(se_rel) else float("nan"))
    powered = bool(np.isfinite(mde80_phys) and mde80_phys <= floor) \
        if null_scale > 0.0 else False

    notes: list[str] = []
    flat = (seam_mean == 0.0 and null_scale == 0.0)
    ci_degenerate = bool(excess_ci.get("degenerate", False))
    lo = rel_lo
    mde80 = mde80_rel if null_scale > 0.0 else mde80_phys
    floor_cmp = rel_floor if null_scale > 0.0 else 0.0

    if flat:
        verdict = "DEGENERATE"
        notes.append(
            "the channel is IDENTICALLY ZERO at every boundary (seam and "
            "null): no seam question is being answered. This is the expected "
            "reading for a zero-init emission head, whose controls are exactly "
            "(0, 0) at every step — a CV straight rollout has no "
            "discontinuities to rank. Not a pass.")
    elif ci_degenerate:
        verdict = "INCONCLUSIVE"
        notes.append(
            "the paired interval is at float64 resolution (ci.py flagged it "
            "degenerate): 'separated' here is arithmetic, not evidence.")
    elif lo > floor_cmp:
        verdict = "SEAM"
        notes.append(
            f"the paired CI's LOWER bound ({lo:.6g} in units of a typical "
            f"within-band step; {float(excess_ci['lo']):.6g} physical) exceeds "
            f"the materiality floor ({floor_cmp:.6g} == {materiality_k}x the "
            f"within-band scale {null_scale:.6g}) — the boundary discontinuity "
            f"is an OUTLIER against the within-band null. ⛔ REPORT THIS; do "
            f"not add a repair term.")
    elif int(excess_ci["n_episodes"]) < MIN_EPISODES_FOR_CLEAN_BILL:
        verdict = "INCONCLUSIVE"
        notes.append(
            f"TOO FEW EPISODE CLUSTERS for a clean bill: "
            f"{int(excess_ci['n_episodes'])} < "
            f"{MIN_EPISODES_FOR_CLEAN_BILL}. A cluster bootstrap over n "
            f"episodes has only C(2n-1, n) distinct resamples, so at this n "
            f"the percentile interval cannot approximate 95 % and a small SE "
            f"is arithmetic, not precision. The instrument refuses to convert "
            f"that into 'no seam'.")
    elif np.isfinite(mde80) and mde80 <= floor_cmp:
        verdict = "NO_MATERIAL_SEAM"
        notes.append(
            f"WELL-POWERED NULL: the 80 %-power MDE ({mde80:.6g} relative, "
            f"{mde80_phys:.6g} physical) is at or below the materiality floor "
            f"({floor_cmp:.6g} relative, {floor:.6g} physical), so a material "
            f"seam would have been detected and was not. This is a POSITIVE "
            f"result about the seam-free-by-construction claim, not a silence.")
    else:
        verdict = "INCONCLUSIVE"
        notes.append(
            f"UNDER-POWERED: the 80 %-power MDE ({mde80:.6g} relative) exceeds "
            f"the materiality floor ({floor_cmp:.6g}). The absence of a "
            f"detection here is NOT evidence of absence — raise n "
            f"(windows/episodes) or report the bound instead of a verdict.")
    if bool(rank_ci["lo"] > 0.5) and verdict != "SEAM":
        notes.append(
            f"⚠️ the DISTRIBUTION-FREE rank companion DOES separate from its H0 "
            f"of 0.5 (rank {rank_ci['mean']} [{rank_ci['lo']}, "
            f"{rank_ci['hi']}]): the seam boundary is systematically larger "
            f"than the within-band boundaries even though the magnitude is "
            f"sub-material. Worth a look before this row is filed as clean.")

    if verdict != "SEAM" and bool(excess_ci["separated"]) \
            and float(excess_ci["delta"]) > 0 and not flat:
        notes.append(
            "the excess IS separated from zero but is SUB-MATERIAL (CI lower "
            "bound below the floor): a small systematic difference at the "
            "boundary, not an outlier. Reported, not called a seam.")
    if halfwidth is None:
        notes.append(
            "GLOBAL null (every other boundary). If the emitted profile has a "
            "smooth index trend the global null is not exchangeable — read the "
            "LOCAL-null row beside this one before concluding.")
    else:
        notes.append(
            f"LOCAL null |i - {seam}| <= {halfwidth} ({n_null} boundaries): an "
            f"index trend is removed to first order, at the cost of a coarser "
            f"rank resolution (1/{n_null}).")

    return {
        "block": BLOCK, "version": VERSION,
        "channel": channel, "units": units,
        "order": int(order), "order_name": ORDER_NAMES.get(int(order), "d%d" % order),
        "seam_boundary": int(seam),
        "seam_time_s": round(float(seam) * DT, 4),
        "null": "local" if halfwidth is not None else "global",
        "null_halfwidth": (None if halfwidth is None else int(halfwidth)),
        "n_null_boundaries": n_null,
        "seam_mean": round(seam_mean, _ci.DISPLAY_DP),
        "null_scale": round(null_scale, _ci.DISPLAY_DP),
        "null_scale_stat": "mean over the within-band boundaries (NOT the "
                           "median — the median reference is biased upward "
                           "under H0; see the module docstring)",
        "null_median": round(float(np.mean(null_med)), _ci.DISPLAY_DP),
        "ratio": (round(seam_mean / null_scale, _ci.DISPLAY_DP)
                  if null_scale != 0.0 else float("nan")),
        "excess": excess_ci["delta"],
        "excess_ci": excess_ci,
        "excess_rel": (rel_ci["delta"] if rel_ci else float("nan")),
        "excess_rel_ci": rel_ci,
        "excess_rel_units": "multiples of the within-band scale "
                            "(the VERDICT is read on this row)",
        "rank_mean": rank_ci["mean"], "rank_h0": 0.5,
        "rank_above_h0": bool(float(rank_ci["lo"]) > 0.5),
        "rank_ci": rank_ci,
        "top1_rate": top1_ci["mean"],
        "top1_h0": round(top1_null_rate, _ci.DISPLAY_DP),
        "top1_above_h0": bool(float(top1_ci["lo"]) > top1_null_rate),
        "top1_ci": top1_ci,
        "power": {
            "estimator_se": round(se_phys, 6),
            "estimator_se_rel": (round(se_rel, 6) if np.isfinite(se_rel)
                                 else float("nan")),
            "materiality_k": float(materiality_k),
            "materiality_floor": round(floor, 6),
            "materiality_floor_rel": (round(rel_floor, 6)
                                      if np.isfinite(rel_floor)
                                      else float("nan")),
            "mde_power80": (round(mde80_phys, 6) if np.isfinite(mde80_phys)
                            else float("nan")),
            "mde_power80_rel": (round(mde80_rel, 6) if np.isfinite(mde80_rel)
                                else float("nan")),
            "power_at_floor": round(_power_at(floor, se_phys), 4),
            "powered_for_material_seam": bool(
                powered and int(excess_ci["n_episodes"])
                >= MIN_EPISODES_FOR_CLEAN_BILL),
            "min_episodes_for_clean_bill": MIN_EPISODES_FOR_CLEAN_BILL,
            "enough_episode_clusters": bool(
                int(excess_ci["n_episodes"]) >= MIN_EPISODES_FOR_CLEAN_BILL),
            "power_target": POWER_TARGET,
            "mde_z": round(MDE_Z, 6),
            "approximation": "normal approximation to the episode-cluster "
                             "bootstrap distribution of the mean excess; the "
                             "validation of this number is the seam-injection "
                             "test, not the formula",
            "n_windows": int(n),
            "n_episodes": int(excess_ci["n_episodes"]),
        },
        "verdict": verdict,
        "notes": notes,
        "tier": "T1",
        "tier_invariant": True,
        "tier_note": "computed on the EMITTED 60-step plan alone — no recorded "
                     "future actions, no future frames, no ground truth — so "
                     "it is invariant to the eval tier and is stamped T1 (the "
                     "primary tier) rather than left unstamped.",
        "n_windows": int(n),
        "n_episodes": int(excess_ci["n_episodes"]),
        "_read": _READ,
    }


def control_channels(controls=None, waypoints=None, extra=None) -> dict:
    """Assemble the per-step scalar channels an emission dump can offer.

    ``controls`` [N, T, 2] -> ``a`` (m/s²) and ``kappa`` (1/m); ``waypoints``
    [N, T, 2] -> ``wp_x`` / ``wp_y`` (m). ``extra`` merges any further named
    [N, T] series (the CLI adds ``err`` when ground truth is present).

    ⚠️ Waypoints are the INTEGRAL of the controls, so the order correspondence
    shifts by one: a control-level jump (order 1 on ``a``/``kappa``) appears as
    an order-2 feature in position space. Both spaces are reported because they
    fail differently — concatenating two independently ROLLED bands produces a
    position jump (order 1 on ``wp_*``) that the control channels cannot see,
    while concatenating two independent CONTROL sequences integrated by one
    rollout leaves the position continuous and shows only in the controls.
    """
    ch: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    if controls is not None:
        c = _to_numpy(controls)
        if c.ndim != 3 or c.shape[-1] != 2:
            raise ValueError(f"controls must be [N, T, 2] (a, kappa), got "
                             f"{c.shape}")
        ch["a"], units["a"] = c[..., 0], "m/s^2"
        ch["kappa"], units["kappa"] = c[..., 1], "1/m"
    if waypoints is not None:
        w = _to_numpy(waypoints)
        if w.ndim != 3 or w.shape[-1] != 2:
            raise ValueError(f"waypoints must be [N, T, 2], got {w.shape}")
        ch["wp_x"], units["wp_x"] = w[..., 0], "m"
        ch["wp_y"], units["wp_y"] = w[..., 1], "m"
    for k, v in (extra or {}).items():
        a = _to_numpy(v)
        if a.ndim != 2:
            raise ValueError(f"extra channel {k!r} must be [N, T], got "
                             f"{a.shape}")
        ch[k] = a
        units.setdefault(k, "unstated")
    if not ch:
        raise ValueError("no channels: pass controls and/or waypoints")
    return {"channels": ch, "units": units}


def continuity_panel(channels: dict, eid, *, units: dict | None = None,
                     seam: int = SEAM_BOUNDARY, orders=ORDERS,
                     halfwidths=(None, LOCAL_HALFWIDTH),
                     materiality_k: float = MATERIALITY_K,
                     n_boot: int = _ci.DEFAULT_N_BOOT, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """:func:`continuity` over every (channel × order × null), plus a roll-up.

    ⛔ **The roll-up is a COUNT, never a pooled score.** Pooling the rows into
    one number is exactly what the per-family rule forbids one level up: a
    single composite hides which channel and which derivative moved, which is
    the only thing a seam finding is useful for. ``any_seam`` therefore names
    the offending rows rather than averaging them away.
    """
    units = units or {}
    rows, seams, inconclusive, degenerate = [], [], [], []
    for name, x in channels.items():
        for order in orders:
            for hw in halfwidths:
                try:
                    r = continuity(x, eid, seam=seam, order=order, halfwidth=hw,
                                   channel=name, units=units.get(name),
                                   materiality_k=materiality_k, n_boot=n_boot,
                                   seed=seed, alpha=alpha)
                except ValueError as e:      # e.g. order too high for T
                    rows.append({"channel": name, "order": int(order),
                                 "null": "local" if hw is not None else "global",
                                 "verdict": "UNAVAILABLE", "reason": str(e)})
                    continue
                rows.append(r)
                tag = f"{name}/d{order}/{r['null']}"
                if r["verdict"] == "SEAM":
                    seams.append(tag)
                elif r["verdict"] == "INCONCLUSIVE":
                    inconclusive.append(tag)
                elif r["verdict"] == "DEGENERATE":
                    degenerate.append(tag)
    # a row is CONFIRMED only when BOTH nulls agree — the global-only case is
    # an index trend until shown otherwise (see the module docstring).
    by_pair: dict[str, set] = {}
    for t in seams:
        ch_ord, null = t.rsplit("/", 1)
        by_pair.setdefault(ch_ord, set()).add(null)
    confirmed = sorted(k for k, v in by_pair.items() if v == {"global", "local"})
    global_only = sorted(k for k, v in by_pair.items() if v == {"global"})
    local_only = sorted(k for k, v in by_pair.items() if v == {"local"})

    if confirmed:
        headline = "SEAM"
    elif seams:
        headline = "SEAM_ONE_NULL_ONLY"
    elif inconclusive:
        headline = "INCONCLUSIVE"
    elif any(r.get("verdict") == "NO_MATERIAL_SEAM" for r in rows):
        headline = "NO_MATERIAL_SEAM"
    else:
        headline = "DEGENERATE"

    return {
        "block": BLOCK + ".panel", "version": VERSION,
        "seam_boundary": int(seam), "seam_time_s": round(float(seam) * DT, 4),
        "orders": [int(o) for o in orders],
        "channels": sorted(channels),
        "rows": rows,
        "headline": headline,
        "seam_rows_confirmed_both_nulls": confirmed,
        "seam_rows_global_null_only": global_only,
        "seam_rows_local_null_only": local_only,
        "inconclusive_rows": sorted(inconclusive),
        "degenerate_rows": sorted(degenerate),
        "tier": "T1", "tier_invariant": True,
        "_read": _READ + " A row is CONFIRMED only when the GLOBAL and LOCAL "
                         "nulls agree; a global-only firing is an index trend "
                         "until shown otherwise.",
    }


# --------------------------------------------------------------------------- #
# the falsifiability calibration                                               #
# --------------------------------------------------------------------------- #
def boundary_scan(x, eid, *, seam: int = SEAM_BOUNDARY, order: int = 1,
                  channel: str | None = None,
                  materiality_k: float = MATERIALITY_K,
                  n_boot: int = SCAN_N_BOOT, seed: int = 0,
                  alpha: float = 0.05, stride: int = 1) -> dict:
    """Run the identical rule at EVERY boundary — the rule's own FPR, measured.

    ⭐ **This is the C13 defence.** A guard that cannot fail is worthless, and a
    "no seam" reading is only worth something if the same rule DOES fire
    somewhere, or provably fires nowhere on a signal with no structure. The scan
    answers three questions with numbers rather than assurances:

    * ``false_positive_rate`` — how often the rule returns ``SEAM`` at a
      boundary that is NOT the band edge. A rule with a 30 % FPR has not
      detected anything when it fires at 20.
    * ``argmax_boundary`` — where the largest excess actually is. "The hotspot
      is at 43, not 20" is a POSITIVE statement about the seam claim.
    * ``seam_rank_among_boundaries`` — the band edge's rank by excess among all
      boundaries. Under H0 it is uniform.

    The scan uses a cheaper ``n_boot`` (:data:`SCAN_N_BOOT`) because it is a
    CALIBRATION block, not a decision block — stated here and stamped in the
    record, never left for a reader to assume.
    """
    a = _to_numpy(x)
    d, idx = boundary_diffs(a, order)
    per = []
    for i in idx[::max(1, int(stride))]:
        r = continuity(a, eid, seam=int(i), order=order, halfwidth=None,
                       channel=channel, materiality_k=materiality_k,
                       n_boot=n_boot, seed=seed, alpha=alpha)
        per.append({"boundary": int(i), "time_s": round(float(i) * DT, 4),
                    "excess": r["excess"], "lo": r["excess_ci"]["lo"],
                    "excess_rel": r["excess_rel"], "ratio": r["ratio"],
                    "verdict": r["verdict"]})
    others = [p for p in per if p["boundary"] != int(seam)]
    fires = [p["boundary"] for p in others if p["verdict"] == "SEAM"]
    ex = np.array([p["excess"] for p in per], dtype=np.float64)
    order_by_excess = np.argsort(-ex, kind="stable")
    ranked = [per[j]["boundary"] for j in order_by_excess]
    seam_rank = (ranked.index(int(seam)) if int(seam) in ranked else None)
    at_seam = next((p for p in per if p["boundary"] == int(seam)), None)
    return {
        "block": BLOCK + ".scan", "version": VERSION,
        "channel": channel, "order": int(order),
        "seam_boundary": int(seam), "stride": int(stride),
        "n_boundaries_scanned": len(per),
        "false_positive_rate": (round(len(fires) / len(others), 4)
                                if others else float("nan")),
        "false_positive_boundaries": fires,
        "argmax_boundary": (ranked[0] if ranked else None),
        "argmax_excess": (round(float(ex[order_by_excess[0]]), 6)
                          if ex.size else float("nan")),
        "seam_rank_among_boundaries": seam_rank,
        "seam_is_argmax": bool(ranked and ranked[0] == int(seam)),
        "at_seam": at_seam,
        "per_boundary": per,
        "n_boot": int(n_boot),
        "tier": "T1", "tier_invariant": True,
        "_read": "CALIBRATION block (cheaper n_boot than the decision blocks): "
                 "it measures the rule's own false-positive rate and locates "
                 "the true discontinuity hotspot. A SEAM verdict at boundary "
                 f"{int(seam)} is only meaningful against this FPR.",
    }


# --------------------------------------------------------------------------- #
# per-band error — reported SEPARATELY, never pooled                           #
# --------------------------------------------------------------------------- #
def band_errors(pred, gt, eid, *, seam: int = SEAM_BOUNDARY,
                n_boot: int = _ci.DEFAULT_N_BOOT, seed: int = 0,
                alpha: float = 0.05, tier: str | None = None,
                arm: str | None = None) -> dict:
    """Per-band ADE with its OWN episode-cluster CI, plus the paired band delta.

    ⛔ **Never pooled.** ``ade_0_2s`` and ``ade_2_6s`` are two rows, exactly as
    the trainer logs them (§4b: "report the bands SEPARATELY. A pooled 0–6 s
    number cannot show the 2 s seam"). The pooled 0–6 s ADE is emitted too, but
    explicitly as a cross-arm comparability row and never as the headline.

    ⚠️ **The band delta is NOT a seam test.** The tactical band is 2–6 s and the
    operative band 0–2 s, so ``ade_2_6s`` is larger for the ordinary reason that
    prediction error grows with horizon. It is reported because a registry row
    needs it with a paired interval, and it is labelled so nobody reads horizon
    growth as a discontinuity.

    ``tier`` is REQUIRED for this block by the caller — comparing a plan against
    the GT future is a capability-adjacent number and an un-tiered number is
    what the doctrine forbids. This function stamps whatever it is given and
    records ``UNSTAMPED`` if it is given nothing, so the omission is visible.
    """
    p = _to_numpy(pred)
    g = _to_numpy(gt)
    if p.ndim != 3 or p.shape[-1] != 2:
        raise ValueError(f"pred must be [N, T, 2], got {p.shape}")
    if p.shape != g.shape:
        raise ValueError(f"pred/gt shape mismatch: {p.shape} vs {g.shape}")
    n, t, _ = p.shape
    if len(eid) != n:
        raise ValueError(f"eid/pred length mismatch: {len(eid)} vs {n}")
    if not 0 < int(seam) < t:
        raise ValueError(f"seam boundary {seam} is outside the {t}-step plan")
    err = np.linalg.norm(p - g, axis=-1)                       # [N, T]
    op_pw = err[:, :int(seam)].mean(axis=1)
    tac_pw = err[:, int(seam):].mean(axis=1)
    all_pw = err.mean(axis=1)

    def _ci1(v):
        return _ci.episode_cluster_bootstrap(v, eid, reduce="mean",
                                             n_boot=n_boot, seed=seed,
                                             alpha=alpha)

    bands = {
        "ade_0_2s": _ci1(op_pw),
        "ade_2_6s": _ci1(tac_pw),
        "ade_0_6s_pooled": _ci1(all_pw),
    }
    delta = _ci.paired_episode_cluster_bootstrap(
        tac_pw, op_pw, eid, n_boot=n_boot, seed=seed, alpha=alpha)
    return {
        "block": BLOCK + ".bands", "version": VERSION,
        "arm": arm,
        "tier": tier or "UNSTAMPED",
        "seam_boundary": int(seam), "plan_steps": int(t),
        "op_band_s": list(OP_BAND_S), "tac_band_s": list(TAC_BAND_S),
        "bands": bands,
        "band_delta_tac_minus_op": delta,
        "n_windows": int(n), "n_episodes": int(delta["n_episodes"]),
        "_read": "PER-BAND, never pooled into the headline. The band delta is "
                 "HORIZON GROWTH, not a discontinuity — the seam test is the "
                 "continuity panel, not this row.",
    }


def band_errors_paired(pred_a, pred_b, gt, eid, *, seam: int = SEAM_BOUNDARY,
                       n_boot: int = _ci.DEFAULT_N_BOOT, seed: int = 0,
                       alpha: float = 0.05, arm_a=None, arm_b=None) -> dict:
    """Two arms on the SAME windows -> per-band PAIRED deltas.

    This is the row a registry entry carries when two arms are compared at the
    seam: per band, paired, on identical windows. ⛔ Never a quadrature
    combination of two single-arm intervals — the two arm estimates are not
    independent and that combination is invalid (``ci.py`` docstring).
    """
    pa, pb, g = _to_numpy(pred_a), _to_numpy(pred_b), _to_numpy(gt)
    if pa.shape != pb.shape or pa.shape != g.shape:
        raise ValueError(f"arms/gt must be aligned [N, T, 2]: {pa.shape} / "
                         f"{pb.shape} / {g.shape}")
    ea = np.linalg.norm(pa - g, axis=-1)
    eb = np.linalg.norm(pb - g, axis=-1)
    s = int(seam)
    out = {}
    for name, sl in (("ade_0_2s", slice(0, s)), ("ade_2_6s", slice(s, None)),
                     ("ade_0_6s_pooled", slice(None))):
        out[name] = _ci.paired_episode_cluster_bootstrap(
            ea[:, sl].mean(axis=1), eb[:, sl].mean(axis=1), eid,
            n_boot=n_boot, seed=seed, alpha=alpha)
    return {"block": BLOCK + ".bands_paired", "version": VERSION,
            "arm_a": arm_a, "arm_b": arm_b, "seam_boundary": s,
            "paired_delta_a_minus_b": out,
            "_read": "PAIRED episode-cluster bootstrap on identical windows — "
                     "never a quadrature combination of two single-arm CIs."}


# --------------------------------------------------------------------------- #
# rendering                                                                    #
# --------------------------------------------------------------------------- #
def seam_report(panel: dict, bands: dict | None = None,
                scan: dict | None = None) -> str:
    """The panel (+ optional band/scan blocks) as a compact printable block.

    Every seam number is printed WITH its interval, its estimator and its
    verdict on one line — a discontinuity printed bare is exactly the kind of
    number the estimator rule forbids.
    """
    lines = [
        f"X2 SEAM PROBE — boundary {panel['seam_boundary']} "
        f"(t = {panel['seam_time_s']:.1f} s, between plan step "
        f"{panel['seam_boundary'] - 1} and {panel['seam_boundary']})",
        f"  headline: {panel['headline']}   [tier {panel['tier']}, "
        f"tier-invariant]",
        "  " + "-" * 76,
        "  chan       ord null    seam      null      excess/scale "
        "[lo, hi]           MDE80  verdict",
    ]
    for r in panel["rows"]:
        if r.get("verdict") == "UNAVAILABLE":
            lines.append(f"  {r['channel']:<10} d{r['order']}  {r['null']:<7} "
                         f"UNAVAILABLE — {r.get('reason', '')[:40]}")
            continue
        c = r["excess_rel_ci"] or r["excess_ci"]
        mde = r["power"]["mde_power80_rel"]
        lines.append(
            f"  {str(r['channel']):<10} d{r['order']}  {r['null']:<7} "
            f"{r['seam_mean']:<9.4g} {r['null_scale']:<9.4g} "
            f"{c['delta']:<+8.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"
            f"{'':<3} {mde:<6.3f} {r['verdict']}")
    p = next((r["power"] for r in panel["rows"] if "power" in r), None)
    if p:
        lines.append(
            f"  excess/scale and MDE80 are in MULTIPLES of a typical "
            f"within-band step; the materiality floor is "
            f"{p['materiality_k']:.3g}. n = {p['n_windows']} windows / "
            f"{p['n_episodes']} episodes.")
    lines.append("  estimator: paired episode-cluster bootstrap "
                 "(taniteval/ci.py) — NEVER overlapping_holdout_se")
    if panel["seam_rows_confirmed_both_nulls"]:
        lines.append("  ⛔ SEAM CONFIRMED on both nulls: "
                     + ", ".join(panel["seam_rows_confirmed_both_nulls"]))
    if panel["seam_rows_global_null_only"]:
        lines.append("  ⚠️ global-null only (index trend until shown "
                     "otherwise): "
                     + ", ".join(panel["seam_rows_global_null_only"]))
    if scan:
        lines += [
            "  " + "-" * 76,
            f"  scan[{scan['channel']}/d{scan['order']}]: FPR at non-seam "
            f"boundaries {scan['false_positive_rate']} over "
            f"{scan['n_boundaries_scanned']} boundaries; hotspot at "
            f"{scan['argmax_boundary']} (excess {scan['argmax_excess']:.4g}); "
            f"seam rank {scan['seam_rank_among_boundaries']}",
        ]
    if bands:
        b = bands["bands"]
        lines += [
            "  " + "-" * 76,
            f"  bands [tier {bands['tier']}] — reported SEPARATELY, never "
            f"pooled into the headline:",
            f"    ade_0_2s  {b['ade_0_2s']['mean']:.4f} "
            f"[{b['ade_0_2s']['lo']:.4f}, {b['ade_0_2s']['hi']:.4f}]",
            f"    ade_2_6s  {b['ade_2_6s']['mean']:.4f} "
            f"[{b['ade_2_6s']['lo']:.4f}, {b['ade_2_6s']['hi']:.4f}]",
            "    (the tac−op delta is HORIZON GROWTH, not a discontinuity)",
        ]
    lines.append("  ⛔ VERIFY, NEVER REPAIR — a seam finding is reported, "
                 "never fixed with a loss term.")
    return "\n".join(lines)
