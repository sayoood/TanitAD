"""taniteval.pseudosim — PSEUDO-SIMULATION: the protocol that makes our
closed-loop numbers a MEASUREMENT instead of an EXTRAPOLATION.

THE DEFECT THIS MODULE REMOVES (MEASURED, 2026-07-26/27)
--------------------------------------------------------
The sequential closed loop (:mod:`taniteval.clhorizon`) lets the ego walk
arbitrarily far from the logged pose, and the further it walks the less the
substrate can honestly re-render. Out-of-envelope window fractions:
``K=20 -> 12.3 %``, ``K=60 -> 50.7 %``, ``K=185 -> 90.2 %``. The **last horizon
at which the closed loop is a pure MEASUREMENT is 0.4 s (K=4)**, and
``GATE_PROTOCOL`` §0.3 refuses ``K <= 20`` — so **no admissible gate horizon is a
measurement**. Widening cannot rescue it: at yaw ``= inf`` the *lateral* clause
alone still leaves 3.75 % of K=20 windows outside, and ``MEASUREMENT`` requires
zero.

**The cause is ACCUMULATION**, not fidelity. Pseudo-simulation (NAVSIM v2,
*Pseudo-Simulation for Autonomous Driving*, arXiv 2506.04218, CoRL 2025) removes
the mechanism: it **pre-generates a BOUNDED GRID of perturbed observations
before evaluation and never rolls out sequentially**. Deviation is therefore
**CHOSEN, not accumulated**, and **cannot leave the validated envelope by
construction**.

    ``PUBLISHED`` Their protocol reports ``R^2 = 0.80 (r = 0.89), n = 83
    planners`` against nuPlan closed-loop, vs ``R^2 = 0.70 (r = 0.83)`` for the
    best open-loop method. We re-implement the PROTOCOL on our own data and our
    own warp; **no NAVSIM code and no NAVSIM data is vendored** (their code is
    Apache-2.0 but their data is CC-BY-NC-SA).

WHAT THE GRID'S AXES ARE, AND WHY ONE OF THEM IS REFUSED
--------------------------------------------------------
=========================  =========================================  ==========
axis                       substrate                                  status
=========================  =========================================  ==========
**heading** ``dpsi``       ``H = K R K^-1`` — a pure camera rotation   **USED**
                           is exact for ARBITRARY scene depth
                           (``max|dH| = 0.000e+00``, 30 conditions)
**longitudinal** ``dlon``  index offset along the logged path —        **USED**
                           REAL frames, zero synthesis
**lateral** ``dlat``       ground-plane homography under a             ⛔ **REFUSED**
                           FLAT-ROAD assumption
=========================  =========================================  ==========

⛔ **The lateral axis is refused on MEASURED geometry**, not on taste
(``…/incoming/2026-07-27-pseudo-simulation/artifacts/lat_warp_fidelity.json``):
the flat-road warp's relative displacement error is **exactly
``height_above_road / h_cam``** — independent of depth, of ``|dlat|`` and of
focal length. At the camera height (1.50 m) it is **100 %**: a sedan roof that
should move 35.47 px moves **1.18 px**. Above 1.50 m the applied displacement is
**SIGN-INVERTED** (truck roof ``-1.667x``, building ``-4.0x``), and **exactly
50 % of a 256-row frame at pitch 0 lies above the horizon**, where the ground
plane has no preimage at all. At ``|dlat| = 2.0 m`` only **28.3 %** of in-frame
scene points meet the pre-registered ``rel_err < 0.25`` bar against a required
95 %. ⇒ **outcome L-BAD; a 1-D warped axis that is a MEASUREMENT beats a 2-D one
that is an extrapolation.** :class:`GridSpec` therefore raises
:class:`LateralAxisRefused` unless a caller passes an explicit override *and* a
written reason.

WHAT IS ENFORCED HERE (the assertion is the whole point)
---------------------------------------------------------
* :func:`assert_grid_in_envelope` runs on **every** grid, inside
  :func:`pseudo_evaluate`, **before any model is touched**. It is a HARD
  failure (:class:`EnvelopeViolation`), never a report line.
* It **can** fail, and the value that makes it fail is published in its own
  output (``falsifier``): any ``|dpsi| > 12.0 deg`` or ``|dlat| > 3.0 m``. The
  test suite exercises exactly that input.
* ⚠️ **Every emitted node carries ``traffic_mode``.** Our AlpaSim ``trafficsim``
  is disabled (``skip: true``) ⇒ **LITERAL REPLAY**, so every published TanitAD
  closed-loop number ran against non-reactive replayed traffic. That was
  nowhere on record until 2026-07-27. It is on record in every result this
  module emits. *(Reactive traffic is NOT the blocker: NAVSIM v2 itself uses
  rule-based agents on road centrelines, and our own CATK probe measured NOT
  reactive — 155 agents moved 4 mm when a car braked to a dead stop.)*
* :func:`composite` **REFUSES TO EMIT** when no weighted component clears the
  discriminative-range gate (:class:`VacuousMetric`). Comfort is saturated at
  ``>= 99.9 %`` in the published cross-benchmark study — *"essentially zero
  discriminative information"* — and this program has shipped three vacuous
  diagnostics already.
* ⛔ Collision / TTC are **NOT COMPUTABLE** on the current val cache and are
  therefore emitted as ``None`` with a reason, **never as a constant**. See
  :data:`COLLISION_UNAVAILABLE_REASON`.

ESTIMATOR. ``taniteval.ci.episode_cluster_bootstrap`` (``B=2000``), unit = val
episode, paired form for two arms on the same windows.
``overlapping_holdout_se`` appears nowhere — it biases the POINT ESTIMATE as
well as the interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from taniteval import ci as _ci
from taniteval import ego_guard as _egg
from taniteval import ood as _ood
from taniteval.clhorizon import (DT, LEGACY_WARP, LOOKAHEAD_STEP,  # noqa: F401
                                 W as WINDOW, WarpFrameRefused,
                                 assert_warp_frame, sampling_homography,
                                 warp_batch, warp_frames, wrap_angle)
from taniteval.ood import ENV_LAT_MAX, ENV_YAW_MAX

__all__ = [
    "TRAFFIC_MODE_LOG_REPLAY", "TRAFFIC_MODE_NOTE", "PROTOCOL",
    "LATERAL_REFUSAL", "COLLISION_UNAVAILABLE_REASON",
    "EnvelopeViolation", "LateralAxisRefused", "VacuousMetric",
    "GridSpec", "default_grid", "assert_grid_in_envelope",
    "proximity_weights", "pseudo_evaluate", "score_windows",
    "discriminative_range", "composite", "emit",
    "COMFORT_LIMITS", "COMPONENT_WEIGHTS", "COMPONENT_WEIGHTS_PUBLISHED_V1",
    "WEIGHTS_ID", "COMFORT_STATUS",
    "CEIL_FRAC_MAX", "FLOOR_FRAC_MAX", "RANGE_MIN", "LIVE_FRAC_MIN",
    "GATE_VERSIONS", "GATE_VERSION_DEFAULT", "GATE_VERSION_PUBLISHED",
    "UnknownGateVersion",
    "PROGRESS_TERMS", "PROGRESS_TERM_DEFAULT", "PROGRESS_TERM_PUBLISHED",
    "PROGRESS_UNDER_FLOOR_IS_UNFIXABLE", "PROGRESS_OVER_SIDE_TRICHOTOMY",
    "PROGRESS_HUMAN_MIN_M", "progress_ratios_per_step",
    "progress_from_ratio", "metric_id", "UnknownProgressTerm",
    "RECOVERY_TERMS", "RECOVERY_TERM_DEFAULT", "RECOVERY_TERM_PUBLISHED",
    "RECOVERY_TERM_DEFAULT_TARGET", "RECOVERY_TERM_ALIASES",
    "RECOVERY_HOLD_ANCHOR", "RECOVERY_HOLD_ANCHOR_GRID",
    "recovery_from_ratio", "UnknownRecoveryTerm", "saturation",
    # projection-aware re-render, re-exported so a caller of pseudosim never
    # has to reach into clhorizon to state its geometry.
    "LEGACY_WARP", "WarpFrameRefused", "assert_warp_frame", "warp_frames",
]

# --------------------------------------------------------------------------- #
# ⛔ THE PROGRESS TERM IS VERSIONED — v1 was ONE-SIDED (MEASURED 2026-07-28)     #
# --------------------------------------------------------------------------- #
#: ``clamp_v1`` is ``clamp(ratio, 0, 1)``: it charges NOTHING for OVER-travel.
#: MEASURED (…/2026-07-28-tactical-action-input/): v1's tactical plan
#: over-travels on **48.80 %** of windows (p95 ratio **2.430x**), all of which
#: clamp to 1.0. An intervention that cut along-track RMS **8.799 -> 1.557 m
#: (5.65x)** and lifted the +-5 % distance hit-rate **15.00 % -> 58.93 %** moved
#: the composite **+0.0078 [-0.0110, +0.0260], n.s.** — while the SAME estimator
#: on the SAME 15,981 rows SEPARATED a 3.36x DEGRADATION of the same axis
#: (REF-C-XL trained-`v0` ablation, -0.0332 [-0.0433, -0.0243] SEP, replicated
#: at base scale -0.0461). ⇒ **one-sided blindness, not low power: `clamp_v1`
#: punishes going too slow and cannot see going too fast.**
#:
#: ⚠️ EVERY PSS NUMBER PUBLISHED BEFORE 2026-07-28 IS A ``clamp_v1`` NUMBER.
#: It stays exactly computable (``progress_term="clamp_v1"``) and is IDENTIFIED
#: in the composite's own ``name``, so the fix is a new metric id rather than a
#: silent redefinition under a stable name — the failure class this program has
#: logged repeatedly (most recently ``ax``, deliberately NOT redefined).
PROGRESS_TERM_PUBLISHED = "clamp_v1"
#: ``twosided_v2`` is ``clamp(1 - under - OVER_TRAVEL_WEIGHT * over, 0, 1)`` with
#: ``under = max(1 - r, 0)``, ``over = max(r - 1, 0)``.
#:
#: WHY THIS SHAPE, and why the default weight is 1.0:
#:  * It is a **strict refinement**: for every ``r <= 1`` it is IDENTICAL to
#:    ``clamp_v1``, so the under-travel half of the published term is preserved
#:    bit-for-bit and the change is purely *additive information* on the side
#:    the old term was blind to. Pinned by
#:    ``test_twosided_is_IDENTICAL_to_clamp_v1_on_every_under_travelling_row``.
#:  * It is **piecewise-linear and zero-parameter at w = 1**, so the fix cannot
#:    be accused of being tuned to produce a ranking.
#:  * ⚠️ **Two-sided is NOT automatically symmetric.** Over-travel (planning
#:    through space the car will not have) is plausibly MORE dangerous than
#:    under-travel in driving. But this surface has **no collision gate and no
#:    cuboids** (:data:`COLLISION_UNAVAILABLE_REASON`), so danger is NOT
#:    measurable here and any ``w != 1`` would be an ASSUMPTION dressed as a
#:    measurement. ⇒ the default is the minimum-assumption ``w = 1``, and the
#:    ranking's sensitivity to ``w`` is published instead of chosen
#:    (``asym_w0p5`` / ``asym_w2`` / ``asym_w3`` below).
OVER_TRAVEL_WEIGHT = 1.0
#: ⭐ THE DEFAULT IS THE FIXED TERM. A capability that must be passed explicitly
#: is a capability nobody passes — that is exactly the E1 bug this same report
#: found (``assert_ego_is_fed`` existed, was tested, and had never been called).
#: The versioned ``name``/``metric_id`` is what prevents a silent redefinition,
#: not a frozen default.
PROGRESS_TERM_DEFAULT = "twosided_v2"


class UnknownProgressTerm(ValueError):
    """A progress-term name that is not in :data:`PROGRESS_TERMS`.

    Never falls back to a default: a typo'd term would silently produce a
    number under the wrong metric id, which is the whole failure being fixed."""


def _progress_clamp_v1(ratio):
    """⛔ THE PUBLISHED (BLIND) TERM. Kept exactly, forever, for reproduction."""
    return ratio.clamp(0.0, 1.0)


def _progress_twosided(weight):
    """``clamp_v1(r) - w * max(r - 1, 0)``, re-clipped to [0, 1].

    ⭐ Written as *the published term minus an over-travel charge* rather than as
    ``1 - |1 - r|`` on purpose: for every ``r <= 1`` the charge is exactly ``0.0``
    and the result is **BIT-identical** to :func:`_progress_clamp_v1`, not merely
    equal to float tolerance. ``1 - (1 - r)`` is NOT ``r`` in float32, and a
    2.5e-4-level drift on 36 % of rows would have crept into every paired delta
    between the two terms. Pinned by
    ``test_twosided_is_IDENTICAL_to_clamp_v1_on_every_under_travelling_row``."""
    def _f(ratio):
        base = _progress_clamp_v1(ratio)
        over = (ratio - 1.0).clamp_min(0.0)
        return (base - float(weight) * over).clamp(0.0, 1.0)
    return _f


def _progress_hyperbolic(weight):
    """``1 / (1 + w * max(r - 1, 0))`` on the over side; ``clamp_v1`` below.

    ⭐ THE SHAPE THAT ANSWERS *"MOVE THE FLOOR WITHOUT HALVING THE CHARGE RATE"*.
    Its derivative at ``r = 1+`` is exactly ``-w`` — **identical to**
    :func:`_progress_twosided` at the same ``w`` — but it never reaches 0, so it
    has **no floor kink** for a zero-mean jitter to leak through. ``w0p5`` and
    ``w0p3333`` buy their later floor by charging over-travel at half or a third
    of the rate; this family does not.

    ⛔ AND IT IS CONVEX ON THE OVER SIDE, WHICH IS THE PRICE. A convex score
    under zero-mean noise is biased UP by Jensen, so this family cannot SEPARATE
    a zero-mean along-track jitter in the correct direction — it can only be
    n.s. or wrong. MEASURED, and the measurement is why it is not the default;
    see ``…/incoming/2026-07-28-egoprogress-complete/EGOPROGRESS_COMPLETE.md``.

    Strict refinement: ``torch.where(r > 1, tail, clamp_v1(r))`` returns the
    published value BIT-identically for every ``r <= 1``."""
    def _f(ratio):
        base = _progress_clamp_v1(ratio)
        over = (ratio - 1.0).clamp_min(0.0)
        tail = 1.0 / (1.0 + float(weight) * over)
        return torch.where(ratio > 1.0, tail, base)
    return _f


def _progress_exp(weight):
    """``exp(-w * max(r - 1, 0))`` on the over side; ``clamp_v1`` below.

    Same charge rate ``-w`` at ``r = 1+`` as :func:`_progress_twosided` and
    :func:`_progress_hyperbolic`, decays faster than the hyperbolic one and,
    like it, never floors — and is convex on the over side for the same reason.
    Included so the never-flooring class is represented by more than one
    member: C47's amendment is that a shape must be judged against the DENSITY,
    and two decay rates bracket it."""
    def _f(ratio):
        base = _progress_clamp_v1(ratio)
        over = (ratio - 1.0).clamp_min(0.0)
        tail = torch.exp(-float(weight) * over)
        return torch.where(ratio > 1.0, tail, base)
    return _f


def _progress_sqrtlin(weight):
    """``sqrt(max(1 - w * max(r - 1, 0), 0))`` — ⭐ THE ONLY **CONCAVE** MEMBER.

    Floors at exactly the same ``r = 1 + 1/w`` as :func:`_progress_twosided`,
    but approaches that floor with an increasing charge rate instead of a
    constant one. Concavity is the ONLY property that can make a zero-mean
    perturbation cost something: for a linear term ``E[g(r + d)] = g(r)``
    identically when ``E[d] = 0``, so a linear term can never separate a
    zero-mean cell — it can only be n.s.

    ⚠️ Its rate at ``r = 1+`` is ``w/2``, i.e. HALF the linear family's, which
    is the trade this family makes: it charges small over-travel less and large
    over-travel more."""
    def _f(ratio):
        base = _progress_clamp_v1(ratio)
        over = (ratio - 1.0).clamp_min(0.0)
        tail = torch.sqrt((1.0 - float(weight) * over).clamp_min(0.0))
        return torch.where(ratio > 1.0, tail, base)
    return _f


#: ⛔⛔ THE UNDER SIDE (``r <= 0``) CANNOT BE REPAIRED BY A STRICT REFINEMENT,
#: AND THIS IS A PROOF, NOT A PREFERENCE. Any bounded ``g: R -> [0, 1]`` that
#: agrees with ``clamp_v1`` on ``[0, 1]`` has ``g(0) = 0``; combined with
#: ``g >= 0`` and monotonicity it is forced to ``g == 0`` on ``(-inf, 0]``, i.e.
#: it IS the defect. Charging a plan for reversing FURTHER therefore requires
#: ``g(0) = p > 0`` — a RANGE BUDGET with a free parameter, structurally
#: identical to :data:`RECOVERY_HOLD_ANCHOR`.
#:
#: ⛔ AND THAT BUDGET PAYS A PLAN THAT DOES NOT MOVE. ``stand_still`` sits at
#: ``r = 0`` on 100 % of its rows and would rise ``0.0000 -> p`` for free. This
#: programme has already been burned by exactly that trade once: the naive
#: ``v0``-based recovery denominator scored the BLIND arm **+0.597 above the
#: sighted one** because a planner that barely moves has a small cross-track
#: error. *Standing still is not progress.* ⇒ the budget is NAMED, PRICED and
#: REFUSED rather than shipped; the measured price is in
#: ``…/incoming/2026-07-28-egoprogress-complete/raw/under_fix.json``. Pinned by
#: ``test_the_under_side_range_budget_PAYS_STANDING_STILL``.
PROGRESS_UNDER_FLOOR_IS_UNFIXABLE = (
    "A bounded score agreeing with clamp_v1 on [0,1] is forced to 0 on "
    "(-inf, 0]: a plan reversing 1 m and a plan reversing 10 m MUST score the "
    "same. The only escape is a range budget g(0) = p > 0, which pays "
    "stand_still p for not moving. REFUSED — measured, not argued.")


#: name -> ratio-to-score function. Every published number names its key.
PROGRESS_TERMS = {
    "clamp_v1": _progress_clamp_v1,
    "twosided_v2": _progress_twosided(OVER_TRAVEL_WEIGHT),
    # the pre-registered SENSITIVITY grid on the over-travel slope
    # ⚠️ w0p3333 ADDED 2026-07-28 to complete the curve at the low end: the
    # term floors at r = 1 + 1/w, so w = 1/3 puts the floor at r = 4 and is the
    # cheapest point at which the residual E-3 named is arithmetically absent
    # on every scorable arm. It is a SENSITIVITY anchor, not a proposed default
    # — a smaller w charges over-travel MORE SLOWLY, which is the range-budget
    # trade, and "more permissive per unit" is not automatically a fix.
    "twosided_asym_w0p3333": _progress_twosided(1.0 / 3.0),
    "twosided_asym_w0p5": _progress_twosided(0.5),
    "twosided_asym_w1p5": _progress_twosided(1.5),
    "twosided_asym_w2": _progress_twosided(2.0),
    "twosided_asym_w3": _progress_twosided(3.0),
    # ⭐ ADDED 2026-07-28 by the ego_progress over-side repair. Every one of
    # these is a STRICT REFINEMENT (bit-identical to clamp_v1 for r <= 1) and
    # every one is a CANDIDATE, not a default — the acceptance grid is
    # published in `…/2026-07-28-egoprogress-complete/raw/inject_over.json`.
    # The never-flooring pair answer "move the floor without halving the rate";
    # the sqrtlin pair are the only CONCAVE members and are therefore the only
    # ones that can charge a ZERO-MEAN jitter at all.
    "hyp_w1": _progress_hyperbolic(1.0),
    "hyp_w0p5": _progress_hyperbolic(0.5),
    "hyp_w2": _progress_hyperbolic(2.0),
    "exp_w1": _progress_exp(1.0),
    "exp_w0p5": _progress_exp(0.5),
    "sqrtlin_w1": _progress_sqrtlin(1.0),
    "sqrtlin_w0p5": _progress_sqrtlin(0.5),
    "sqrtlin_w0p3333": _progress_sqrtlin(1.0 / 3.0),
    "sqrtlin_w0p25": _progress_sqrtlin(0.25),
}

#: ⛔⛔ THE TRICHOTOMY — why the over-side zero-mean cell is a TRADE and not a
#: search. For any bounded strict refinement ``g`` (``g = clamp_v1`` on
#: ``[0, 1]``) and a zero-mean along-track perturbation:
#:
#:  * **CONVEX tail** (the only way never to floor) ⇒ Jensen makes
#:    ``E[g(r + d)] > g(r)``: the jitter is **REWARDED**. ``hyp_*`` and ``exp_*``.
#:  * **LINEAR tail** ⇒ ``E[g(r + d)] = g(r)`` **exactly**, away from the floor:
#:    the jitter costs **NOTHING** and the cell can only be n.s., never
#:    separated-correct. ``twosided_*``.
#:  * **CONCAVE tail** ⇒ the jitter is **CHARGED** — but concavity with
#:    ``g(1) = 1`` and ``g'(1+) = -w`` forces ``g(r) <= 1 - w(r - 1)``, so the
#:    term reaches 0 no later than ``r = 1 + 1/w`` and that floor is itself a
#:    convex kink that rewards the same jitter beyond it. ``sqrtlin_*``.
#:
#: ⇒ **a concave term whose charge rate at ``r = 1+`` is ``>= 1`` MUST floor by
#: ``r = 2``**, i.e. no shape can simultaneously (a) charge over-travel at the
#: published rate, (b) stay concave, and (c) put its floor beyond the density.
#: The frontier is ``rate(1+) <= 1 / (R - 1)`` for a floor at ``R``. Pinned by
#: ``test_the_over_side_trichotomy_is_arithmetic_not_a_preference``.
PROGRESS_OVER_SIDE_TRICHOTOMY = (
    "convex tail -> zero-mean jitter REWARDED (Jensen); linear tail -> costs "
    "EXACTLY nothing; concave tail -> charged, but floors by r = 1 + 1/w and "
    "the floor kink rewards it again. A concave term with rate >= 1 at r = 1+ "
    "floors by r = 2. The over-side repair is a TRADE, not a search.")


def progress_from_ratio(ratio, progress_term=PROGRESS_TERM_DEFAULT):
    """``ratio = plan_along / human_along`` -> the ``ego_progress`` sub-score."""
    if progress_term not in PROGRESS_TERMS:
        raise UnknownProgressTerm(
            f"unknown progress_term {progress_term!r}; known: "
            f"{sorted(PROGRESS_TERMS)}. Refusing to fall back to a default — a "
            f"typo must not silently produce a number under the wrong metric "
            f"id. Every PSS number published before 2026-07-28 is "
            f"{PROGRESS_TERM_PUBLISHED!r}.")
    return PROGRESS_TERMS[progress_term](ratio)


#: the human 2 s chord below which ``ego_progress`` is undefined. It is the
#: PUBLISHED constant, lifted out of :func:`score_windows` so the per-step
#: resolution can reuse it VERBATIM rather than introduce a second threshold.
PROGRESS_HUMAN_MIN_M = 0.5


def progress_ratios_per_step(pw):
    """⭐ THE RESOLUTION PRIMITIVE — ``ego_progress`` at every step, not just the last.

    :func:`score_windows` computes ``ego_progress`` from the **terminal** step
    alone (``x[:, -1]`` over the human's 2 s chord), so it is **one number per
    row**, ``n_sub = 1`` — the property that made ``lat_heading`` saturate on
    84 % of an arm's rows and that a 20-step mean fixed with **no free
    parameter**. The identical expression exists at every step here: the plan's
    along-track distance at step ``k`` over the human's chord to step ``k``.

    Returns ``(r_steps [n, H], step_defined [n, H], row_defined [n])`` where

    * ``step_defined`` uses :data:`PROGRESS_HUMAN_MIN_M` — the published
      constant, verbatim, so the resolution change introduces NO parameter;
    * ``row_defined`` is EXACTLY the published row mask (the human's **2 s**
      chord above the same constant), so definedness is bit-preserved and a
      resolution change can never silently re-select rows.

    ⚠️ **READ THIS BEFORE USING IT AS A FIX.** ``human_k`` is small at small
    ``k``, so a constant along-track offset ``e`` enters step ``k`` as
    ``e / human_k`` and EXPLODES at the first steps — the structural twin of the
    step-0 plan tangent that destroyed the plain-mean ``lat_heading``
    candidate. MEASURED consequence in
    ``…/incoming/2026-07-28-egoprogress-complete/EGOPROGRESS_COMPLETE.md``; the
    primitive is published because the audit needs it, not because the mean
    resolution is recommended."""
    x, _y, ref_x, ref_y = _cross_and_along(pw)
    H = x.shape[1]
    dx = ref_x[:, 1:H + 1] - ref_x[:, :1]
    dy = ref_y[:, 1:H + 1] - ref_y[:, :1]
    human_k = torch.sqrt(dx ** 2 + dy ** 2)                    # [n, H]
    r = x / human_k.clamp_min(1e-3)
    step_def = human_k > PROGRESS_HUMAN_MIN_M
    row_def = human_k[:, -1] > PROGRESS_HUMAN_MIN_M
    return r, step_def, row_def


# --------------------------------------------------------------------------- #
# ⛔⛔ THE RECOVERY TERM IS VERSIONED TOO — THE SECOND ONE-SIDED CLAMP (C45)     #
# --------------------------------------------------------------------------- #
#: ``clamp_v1`` is ``clamp(1 - r, 0, 1)`` with ``r = |xt_end| / |xt_hold|``.
#:
#: ⛔ IT IS FLOORED ON THE MAJORITY OF EVERY ARM'S ROWS. MEASURED on the
#: 2026-07-27 panel: **55.65 % (`cv_holdv0`) … 92.19 % (`refc_xl_produced`)** of
#: DEFINED rows sit at exactly 0, the **median unclamped ratio exceeds 1.0 for
#: every arm**, and the ratio reaches **34.1**. A row already past the floor
#: cannot be charged more, so an injected degradation that helps a MINORITY of
#: rows RAISES the mean. MEASURED consequence on ``cv_holdv0``: a **2 m constant
#: lateral offset** moves ``PSS@twosided_v2`` **+0.0581 [+0.0473, +0.0691]
#: SEPARATED**, a 5 deg heading error **+0.0747**, and a **ZERO-MEAN** jitter
#: (sigma = 1 m, which cannot re-centre a bias) **+0.0303 SEPARATED**. 8 of 8
#: injections separated in the WRONG direction, on both arms tested — against a
#: published `cv_holdv0`-vs-best-learned-arm gap of only **-0.0090**.
#: ⇒ **the primary paid for the failure it exists to catch.** Class C45.
#:
#: ⚠️ NOTE THE ASYMMETRY WITH ``clamp_v1``'s PROGRESS TWIN: here the CEILING
#: clamp is provably never active (``r >= 0`` by construction, since both terms
#: are absolute values, so ``1 - r <= 1`` always). ``clamp(1-r, 0, 1)`` IS
#: ``max(1-r, 0)``. Only the floor binds, and the floor is the whole defect.
RECOVERY_TERM_PUBLISHED = "clamp_v1"

#: ⭐ THE PARAMETER, AND IT IS THE ONLY ONE: ``q = the score a plan that recovers
#: NOTHING receives`` — i.e. the value of the term at ``r = 1``, where the plan
#: ends exactly as far off the logged path as not steering would have put it.
#:
#: ⛔ A STRICT REFINEMENT IS PROVABLY IMPOSSIBLE HERE, and that is why this fix
#: does not simply copy ``twosided_v2``'s shape. ``twosided_v2`` could subtract
#: an over-travel charge from ``clamp_v1`` because ``clamp_v1(r) = 1`` at
#: ``r = 1`` left a whole unit of range underneath it. ``recovery``'s published
#: term is already **at 0** at ``r = 1``: it has spent its entire range on the
#: half of the domain where the plan is better than hold. Formally — any
#: ``g: [0, inf) -> [0, 1]`` that agrees with ``1 - r`` on ``[0, 1]`` has
#: ``g(1) = 0`` and must therefore be CONSTANT on ``[1, inf)``, i.e. it is the
#: defect. ⇒ **the fix is not a slope choice, it is a RANGE-BUDGET choice**, and
#: ``q`` is that budget: the under-side ``r in [0, 1]`` gets ``[q, 1]`` and the
#: divergence side ``r > 1`` gets ``[0, q]``.
#:  * ``q = 0``   -> the published term. All budget on recovery, none on
#:                  divergence. This is the defect, exactly.
#:  * ``q = 0.5`` -> EVEN SPLIT — the minimum-assumption point, the same role
#:                  ``OVER_TRAVEL_WEIGHT = 1.0`` plays for the progress term.
#:  * ``q -> 1``  -> almost all budget on divergence; the under-side collapses
#:                  into a narrow band and stops discriminating.
#: ⚠️ THERE IS NO "OVER-RECOVERY" SIDE TO BE ASYMMETRIC ABOUT. ``r >= 0``
#: identically, so the domain has exactly two regions (recovering, diverging)
#: and ``q`` is the only asymmetry there is. Its sensitivity is PUBLISHED
#: (``RECOVERY_HOLD_ANCHOR_GRID``), not chosen — see ``…/incoming/
#: 2026-07-28-recovery-twosided/RECOVERY_TWOSIDED.md`` §5.
RECOVERY_HOLD_ANCHOR = 0.5
#: the pre-registered sensitivity grid on ``q``. Every ranking is republished at
#: every point of it, for the same reason ``w`` was swept.
RECOVERY_HOLD_ANCHOR_GRID = (0.0, 0.25, 0.5, 2.0 / 3.0, 0.75)


class UnknownRecoveryTerm(ValueError):
    """A recovery-term name that is not in :data:`RECOVERY_TERMS`.

    Never falls back to a default, for the identical reason
    :class:`UnknownProgressTerm` does not."""


def _recovery_clamp_v1(ratio):
    """⛔ THE PUBLISHED (FLOORED) TERM. Kept exactly, forever, for reproduction."""
    return (1.0 - ratio).clamp(0.0, 1.0)


def _recovery_linear(q):
    """``clamp(1 - (1 - q) * r, 0, 1)`` — the LINEAR BUDGET family.

    ⭐ **Affine-equivalent to the published term on the whole domain the
    published term handled correctly**: for ``r <= 1``,
    ``g(r) = q + (1 - q) * clamp_v1(r)``, a strictly increasing affine map of
    the published value. Therefore

      * no pair of under-recovering rows changes order, ever;
      * the published value is EXACTLY recoverable: ``clamp_v1 = (g - q)/(1-q)``
        wherever ``g >= q``;
      * ``q = 0`` reproduces ``clamp_v1`` BIT-identically (the expression
        collapses to ``clamp(1 - r, 0, 1)``, the published line verbatim).

    That is the closest attainable analogue of ``twosided_v2``'s strict
    refinement, and the impossibility argument above says it is the closest
    ANY bounded term can get.

    ⭐ Its charge rate is a CONSTANT ``1 - q`` across its whole live range,
    which is the property that actually fixes the defect — see
    :func:`_recovery_share` for the shape that never saturates and still fails.

    ⚠️ It still FLOORS, at ``r = 1/(1-q)`` (``q = 2/3`` -> ``r = 3``). MEASURED
    over the 16 non-probe arms at the shipped ``q = 2/3``: **1.40 %
    (`v4_oracle`) to 8.33 % (`v4_blind`)** of defined rows, against the
    published term's 55.65-92.19 %. That is not zero and it is published beside
    every value the term produces; see :func:`saturation`."""
    def _f(ratio):
        return (1.0 - float(1.0 - q) * ratio).clamp(0.0, 1.0)
    return _f


def _recovery_share(q):
    """``q / (q + (1 - q) * r)`` — the SHARE family. It CANNOT saturate…

    ⛔⛔ **…AND IT FAILS THE ACCEPTANCE TEST ANYWAY. THIS WAS MY PROVISIONAL
    DEFAULT AND THE MEASUREMENT REFUTED IT, 0 of 8.** At ``q = 0.5`` it is
    exactly ``|xt_hold| / (|xt_hold| + |xt_end|)`` — the share of the total
    error attributable to the hold baseline — it never floors on ANY row of ANY
    arm (floor fraction **0.0000**, against ``lin_q0p5``'s 0.0482-0.1372), and
    it STILL pays for injected lateral degradation on **8 of 8** injections.
    ``share_q0p25`` 0/8, ``share_q0p6667`` 2/8, ``share_q0p75`` 6/8.

    ⇒ ⭐ **"NEVER SATURATES" IS NOT THE PROPERTY THAT FIXES A SATURATING METRIC.
    THE PROPERTY IS THAT THE CHARGE RATE MUST NOT COLLAPSE WHERE THE DATA
    LIVES.** ``|dg/dr| = q(1-q)/(q + (1-q)r)^2`` decays like ``r^-2``: at
    ``q = 0.5`` it is **1.00 at r = 0, 0.25 at r = 1 and 0.0625 at r = 3**,
    while the linear family charges a **constant 0.5** across its whole live
    range. The MEDIAN ratio on every arm is **>= 1.004**, so the share form
    charges the typical row at a quarter of the rate at which it rewards a row
    that is already nearly perfect — and an injection that pushes a few near-
    zero rows closer to zero then outweighs the many rows it pushes further
    out. **A soft floor is still a floor.**

    ⚠️ It is also NOT affine in the published term (it agrees with ``1 - r``
    only to first order at ``r = 0``), so unlike :func:`_recovery_linear` it
    re-spaces the under-side.

    Kept, published and swept **because it fails** — the rejection is checkable,
    and the reason it fails is the actual finding. Pinned by
    ``test_the_UNSATURATING_share_family_still_pays_for_lateral_degradation``.

    ⛔ ``q = 0`` is degenerate here (the score collapses to 0 for every ``r > 0``)
    and is refused rather than silently emitted."""
    if not 0.0 < q < 1.0:
        raise ValueError(
            f"share family needs 0 < q < 1; got {q}. q = 0 collapses the score "
            f"to 0 for every r > 0 (a metric that cannot fail upward), and "
            f"q = 1 collapses it to 1.")

    def _f(ratio):
        return (q / (q + float(1.0 - q) * ratio.clamp_min(0.0))).clamp(0.0, 1.0)
    return _f


def _q_tag(q):
    return "q" + f"{float(q):.4g}".replace(".", "p")


#: ⭐ THE PRE-REGISTERED SELECTION RULE, written down BEFORE any panel number was
#: computed (``…/2026-07-28-recovery-twosided/code/run_recovery_twosided.py``
#: banks it in ``injections.json`` before the sweep runs):
#:
#:  R1 DISQUALIFY any shape for which the 8 injected lateral degradations are not
#:     ALL separated in the CORRECT (negative) direction on BOTH real arms. A
#:     shape that fails this does not fix the defect and is not a candidate.
#:  R2 DISQUALIFY any shape whose recovery FLOOR fraction is >= 0.50 on any
#:     scorable arm. C45: "a term saturating on the majority of rows is not a
#:     metric; it is a constant with noise."
#:  R3 Among survivors PREFER the LINEAR family — it is affine-equivalent to the
#:     published term on r <= 1, so the under-side ordering is untouched and the
#:     published value stays exactly invertible. Within it prefer the SMALLEST q
#:     (minimum departure from the published term).
#:  R4 If no linear member survives, take the SHARE family at q = 0.5 — the
#:     equal-budget, parameter-free member.
RECOVERY_TERMS = {
    "clamp_v1": _recovery_clamp_v1,
    **{f"lin_{_q_tag(_q)}": _recovery_linear(_q)
       for _q in RECOVERY_HOLD_ANCHOR_GRID},
    **{f"share_{_q_tag(_q)}": _recovery_share(_q)
       for _q in RECOVERY_HOLD_ANCHOR_GRID if _q > 0.0},
}
#: ⭐ the shipped default is an ALIAS onto a swept family member, and which one
#: is PINNED by a test — so the shape can never drift without the test failing.
#:
#: MEASURED OUTCOME OF THE PRE-REGISTERED RULE (20 arms, 15,981 rows, B = 2000,
#: paired episode-cluster bootstrap over the 40 val episodes):
#:   R1 all-8-separated-correct: ``lin_q0p6667`` ✅ 8/8 · ``lin_q0p75`` ✅ 8/8 ·
#:      ⛔ ``clamp_v1`` 0/8 · ``lin_q0p25`` 0/8 · ``share_q0p25`` 0/8 ·
#:      ``share_q0p5`` 0/8 · ``share_q0p6667`` 3/8 · ``share_q0p75`` 6/8 ·
#:      ⚠️ ``lin_q0p5`` **7/8** — its 8th cell is CORRECT IN SIGN but n.s.
#:      (``v1_tactical_follow`` x ``yaw_bias(+5 deg)``: -0.0036 [-0.0075,
#:      +0.0002], stable at 5 seeds). Not broken, under-powered — and R1 says
#:      SEPARATED, so it is disqualified rather than argued in.
#:   R2 max floor fraction over non-probe arms: ``clamp_v1`` 0.9219 ⛔ ·
#:      ``lin_q0p25`` 0.5884 ⛔ · ``lin_q0p5`` 0.1372 ✅ · ``lin_q0p6667``
#:      **0.0833** ✅ · ``lin_q0p75`` 0.0532 ✅ · every ``share`` 0.0000 ✅
#:   R3 prefer LINEAR, smallest surviving q  ⇒  **lin_q0p6667** (q = 2/3)
#:
#: ⭐ ``lin_q0p6667(r) = clamp(1 - r/3, 0, 1)``: the score reaches 0 at three
#: times the hold error, so the divergence half gets 2/3 of the range and the
#: recovery half 1/3.
#:
#: ⚠️⚠️ **THE MINIMUM-ASSUMPTION EVEN SPLIT (q = 0.5) IS THE DIRECT ANALOGUE OF
#: ``OVER_TRAVEL_WEIGHT = 1.0`` AND IT FAILS ITS OWN ACCEPTANCE TEST HERE.**
#: ``lin_q0p5(r) = clamp(1 - r/2, 0, 1) = (clamp(1 - r, -1, +1) + 1)/2`` charges
#: divergence at exactly the rate it credits recovery — the same argument that
#: fixed ``ego_progress`` — but ``recovery``'s ratio distribution has a far
#: heavier tail than the progress ratio's (median >= 1.004 on EVERY arm, p99
#: 3.3-11.1, max 34.1), so a floor at r = 2 still leaves 4.8-13.7 % of rows
#: uncharged and one injection cell fails to separate. ⇒ **the sibling term's
#: parameter cannot be ported; it has to be re-derived against this term's own
#: tail, and the acceptance test — not the analogy — decides.**
RECOVERY_TERM_DEFAULT_TARGET = "lin_q0p6667"
RECOVERY_TERM_DEFAULT = "twosided_v2"
RECOVERY_TERMS[RECOVERY_TERM_DEFAULT] = RECOVERY_TERMS[
    RECOVERY_TERM_DEFAULT_TARGET]
#: ``lin_q0`` and ``clamp_v1`` are the SAME FUNCTION. Both names are kept: the
#: first says where the published term sits in the family, the second is what
#: every pre-2026-07-28 number was computed under and is what must be quoted.
RECOVERY_TERM_ALIASES = {RECOVERY_TERM_DEFAULT: RECOVERY_TERM_DEFAULT_TARGET,
                         "lin_q0": RECOVERY_TERM_PUBLISHED}


def recovery_from_ratio(ratio, recovery_term=None):
    """``ratio = |xt_end| / |xt_hold|`` -> the ``recovery`` sub-score.

    ``ratio = 0`` is a plan that lands on the logged path, ``1`` is a plan
    exactly as far off as not steering at all, ``> 1`` is a plan that ends
    FURTHER off than doing nothing."""
    term = RECOVERY_TERM_DEFAULT if recovery_term is None else recovery_term
    if term not in RECOVERY_TERMS:
        raise UnknownRecoveryTerm(
            f"unknown recovery_term {term!r}; known: {sorted(RECOVERY_TERMS)}. "
            f"Refusing to fall back to a default — a typo must not silently "
            f"produce a number under the wrong metric id. Every PSS number "
            f"published before 2026-07-28 is "
            f"{RECOVERY_TERM_PUBLISHED!r}.")
    return RECOVERY_TERMS[term](ratio)


def metric_id(progress_term=PROGRESS_TERM_DEFAULT, recovery_term=None) -> str:
    """The quotable name. ⛔ Quote THIS, never a bare ``PSS``.

    ⚠️ The recovery suffix is appended ONLY when the recovery term is not the
    published one, so every id published through 2026-07-28 keeps its exact
    string and no pin, log line or report has to be rewritten to still resolve.
    A NON-published recovery term is always visible in the name."""
    if progress_term not in PROGRESS_TERMS:
        raise UnknownProgressTerm(f"unknown progress_term {progress_term!r}")
    rec = RECOVERY_TERM_PUBLISHED if recovery_term is None else recovery_term
    if rec not in RECOVERY_TERMS:
        raise UnknownRecoveryTerm(f"unknown recovery_term {rec!r}")
    base = f"PSS_recovery_progress@{progress_term}"
    if rec == RECOVERY_TERM_PUBLISHED:
        return base
    return f"{base}+rec_{rec}"

# --------------------------------------------------------------------------- #
# provenance strings that must ride along with every number                    #
# --------------------------------------------------------------------------- #
TRAFFIC_MODE_LOG_REPLAY = "log_replay_nonreactive"
TRAFFIC_MODE_NOTE = (
    "Other agents are LOGGED TRACKS REPLAYED WITHOUT REACTION. Our AlpaSim "
    "`trafficsim` is disabled (`skip: true`), i.e. literal replay — so EVERY "
    "published TanitAD closed-loop number, not just this one, ran against "
    "non-reactive traffic. Paired arm-vs-arm comparisons remain valid; absolute "
    "safety numbers are optimistic. NAVSIM v2 (the protocol being reproduced) "
    "also uses non-reactive rule-based agents on road centrelines, and our own "
    "CAT-K reactivity probe measured NOT REACTIVE (155 agents displaced 0.0044 "
    "m when a lead car braked to a dead stop). Reactive traffic is therefore "
    "NOT a precondition for this protocol.")

PROTOCOL = (
    "PSEUDO-SIMULATION. Perturbed observation states are PRE-GENERATED on a "
    "bounded grid and the planner is evaluated ONCE from each. There is no "
    "sequential rollout, so deviation is CHOSEN, not accumulated, and cannot "
    "leave the validated envelope. Re-implemented from the published protocol "
    "(NAVSIM v2 / arXiv 2506.04218, CoRL 2025) on OUR data and OUR warp; no "
    "NAVSIM code or data is vendored.")

LATERAL_REFUSAL = (
    "The LATERAL grid axis is REFUSED on MEASURED geometry. The flat-road "
    "homography's relative displacement error is exactly height_above_road / "
    "h_cam (depth-free, |dlat|-free, f-free): 100 % at the camera height, "
    "SIGN-INVERTED above it, and 50 % of the frame at pitch 0 is above the "
    "horizon where the ground plane has no preimage. At |dlat| = 2.0 m only "
    "28.3 % of in-frame points meet the pre-registered rel_err < 0.25 bar "
    "(required: 95 %). Artifact: TanitAD Research Hub/Benchmarks & Eval/"
    "Implementation/incoming/2026-07-27-pseudo-simulation/artifacts/"
    "lat_warp_fidelity.json (outcome L-BAD). The YAW axis passes the identical "
    "test at max error 0.0 px, so the test is not vacuous.")

COLLISION_UNAVAILABLE_REASON = (
    "NOT COMPUTABLE on the 40-episode val cache. The cache carries only "
    "{frames_u8, actions, poses, maneuvers, episode_id} — no agent cuboids. "
    "`obstacle.offline` (97.4438 % corpus coverage, boxes reference_frame='rig' "
    "on 100 %) would supply them, but (a) the cached `episode_id` is "
    "int.from_bytes(clip_id[:4]) and COLLIDES — 242 clip_index rows map onto the "
    "40 val episode_ids, so episode->clip identity is not resolvable from the "
    "cache alone, and (b) the matching obstacle.offline chunks are not "
    "downloaded. A constant is NOT emitted in its place: a metric that cannot "
    "fail is not a metric.")

# --------------------------------------------------------------------------- #
# the discriminative-range gate (BOOST M8 / C13 applied to METRICS)             #
# --------------------------------------------------------------------------- #
# PROPOSED thresholds, stated before any component is scored.
CEIL_FRAC_MAX = 0.95     # a component pinned at its ceiling this often is dead
#: ⛔⛔ THE MISSING HALF OF THE GATE, ADDED 2026-07-28.
#: ``discriminative_range`` has always COMPUTED ``floor_frac_le_0p001`` and has
#: never USED it, so a component pinned at its FLOOR was admissible while the
#: identical component pinned at its CEILING was refused. That is the same
#: "audited on one side of a two-sided object" class as ``clamp_v1`` (blind
#: above ratio 1.0) and the two-condition ego gate (audited at one condition).
#: MEASURED on the 2026-07-27 panel: ``recovery`` is at its floor on
#: **54.75 % (cv_holdv0) … 92.18 % (refc_xl_produced)** of its DEFINED rows —
#: ``refc_xl`` sits **2.8 points** from being auto-refused by this very gate.
#: Symmetric with :data:`CEIL_FRAC_MAX` on purpose: a gate that is asymmetric in
#: a quantity with no preferred direction is a bug, not a policy.
#:
#: ⛔⛔ AND IT IS STILL NOT ENOUGH — MEASURED 2026-07-28, ONE DAY LATER.
#: ``control.lat_heading`` is floored on **31.22-84.29 %** of defined rows with
#: its ceiling never active (0.0001-0.0023), and **this constant does not refuse
#: a single one of them.** Two structural reasons, and the second is the one that
#: will catch the NEXT term:
#:  1. ⚠️ **RESOLUTION.** 0.95 is a *dead-component* tripwire, implicitly
#:     calibrated against terms that are MEANS over 20 horizon steps — where a
#:     row saturates only if all 20 sub-samples do, so a row-level floor
#:     fraction near 1 really does mean "dead". Applied to a term that is a
#:     SINGLE VALUE PER ROW it is roughly 20x too loose. ``lat_heading`` is such
#:     a term; ``lon_track`` and ``lat_track`` are not, and they passed the audit.
#:  2. ⛔ **IT TESTS EACH END SEPARATELY.** A term 49 % floored AND 49 %
#:     ceilinged clears ``FLOOR_FRAC_MAX`` *and* :data:`CEIL_FRAC_MAX` while
#:     having gradient on 2 % of its rows. The pair of one-sided thresholds is
#:     STRUCTURALLY UNABLE to see combined saturation — the same "audited on one
#:     side of a two-sided object" class as ``clamp_v1`` itself, one level up.
#: ⇒ :data:`LIVE_FRAC_MIN` and gate ``v2`` below. ⚠️ **This constant is NOT
#: lowered**: it is what every published ``@clamp_v1`` composite was gated
#: under, ``recovery@clamp_v1`` floors on 55.65-92.19 % of rows, and lowering it
#: would make that term inadmissible panel-wide and silently redefine every
#: published value. The fix is a VERSIONED gate, not an edited constant.
FLOOR_FRAC_MAX = 0.95
#: ⭐⭐ THE CLAUSE THAT CAN REFUSE A PER-ROW-FLOORED TERM. Gate ``v2`` only.
#: ``live_frac = 1 - floor_frac - ceiling_frac`` — the fraction of defined rows
#: on which the term has ANY gradient at all. It is the TWO-SIDED statistic the
#: pair (``CEIL_FRAC_MAX``, ``FLOOR_FRAC_MAX``) cannot express, and it is set to
#: the same 0.50 as :data:`SATURATION_WARN_FRAC` deliberately: C45's standing
#: consequence said *"a term saturating on the majority of rows is not a metric;
#: it is a constant with noise"*, and a rule that is only ever printed as a
#: warning is a rule the next agent will read past. MEASURED effect: it refuses
#: ``lat_heading@term_lin_q0`` (live 0.157 on ``v4_blind``) and admits
#: ``lat_heading@mean_lin_q0`` (live 0.749) — i.e. it refuses the broken term
#: and passes the fixed one, which is the only test of a guard that counts.
LIVE_FRAC_MIN = 0.50
#: the gate is VERSIONED for the same reason the terms are: ``v1`` is what every
#: published number was gated under and must stay exactly computable; ``v2`` is
#: strictly STRONGER (it can only refuse more, never admit more) and is what any
#: NEW surface should use. ⚠️ The default is ``v1``: a default that silently
#: re-gated every published composite would be the redefinition being fixed.
GATE_VERSION_PUBLISHED = "v1"
GATE_VERSION_DEFAULT = "v1"
GATE_VERSIONS = {
    "v1": {"live_frac_min": None,
           "what": ("ceiling_frac < CEIL_FRAC_MAX AND floor_frac < "
                    "FLOOR_FRAC_MAX AND observed_range >= RANGE_MIN. ⛔ The two "
                    "saturation clauses are ONE-SIDED and 0.95 is a dead-"
                    "component tripwire, so a term floored on 84 % of its rows "
                    "is admissible. This is what every published number was "
                    "gated under and it is kept EXACTLY for reproduction.")},
    "v2": {"live_frac_min": LIVE_FRAC_MIN,
           "what": ("v1 AND live_frac >= LIVE_FRAC_MIN, where live_frac = 1 - "
                    "floor_frac - ceiling_frac. ⭐ STRICTLY STRONGER than v1: "
                    "it can only refuse more. It is two-sided, so it also sees "
                    "a term that splits its saturation between both ends and "
                    "clears both one-sided thresholds.")},
}
RANGE_MIN = 0.05         # observed max - min below this is not a range
# nuPlan/NAVSIM-style comfort bounds. PROPOSED (their exact constants are not
# quotable from the material we verified); every one is published in the output.
COMFORT_LIMITS = {"a_lon_max_mps2": 3.0, "a_lat_max_mps2": 3.0,
                  "jerk_max_mps3": 8.0, "yaw_rate_max_radps": 0.95}
# PDM-Score weights: EP w=5, TTC w=5, Comfort w=2 (PUBLISHED, NAVSIM). TTC is
# unavailable here (no cuboids); RECOVERY is ours and carries the error-recovery
# signal pseudo-simulation exists to produce.
#: ⛔ FROZEN, for exact reproduction of every number published through
#: 2026-07-28. Never edit this dict.
COMPONENT_WEIGHTS_PUBLISHED_V1 = {"ego_progress": 5.0, "recovery": 5.0,
                                  "comfort": 2.0}
#: ⭐ THE LIVE WEIGHTS. ``comfort`` carries **0.0**, not 2.0.
#:
#: WHY, and why this is a PROVABLE NO-OP on every published number:
#: ``comfort`` is dropped by the panel-wide discriminative-range gate for
#: **every arm under every progress term** — it has never contributed to a
#: published composite. And in :func:`composite` a zero weight adds exactly
#: ``0.0`` to both numerator and denominator in *both* branches, so the value is
#: bit-identical whether the term is admitted or dropped. The 16-arm
#: reproduction gate re-runs at ``max|diff| = 0.000000`` with this change in
#: place (``…/2026-07-28-closedloop-control-suite/raw/repro_gate.json``).
#:
#: What changes is the CLAIM, and the claim was wrong: the composite was
#: described as three-term while one of its three terms was information-free.
#: See :data:`COMFORT_STATUS` for the measurement.
COMPONENT_WEIGHTS = {"ego_progress": 5.0, "recovery": 5.0, "comfort": 0.0}
#: quotable id for the weight vector, so a future change cannot be silent.
WEIGHTS_ID = "w_ep5_rec5_comfort0"

#: ⛔ MEASURED 2026-07-28 — the reason ``comfort`` carries no weight.
COMFORT_STATUS = (
    "INFORMATION-FREE ON THIS SURFACE, AND THE PLANS ARE NOT THE REASON. "
    "`comfort` is the AND of four bounds on the finite differences of a "
    "20-waypoint plan. MEASURED: the HUMAN'S OWN LOGGED PATH, differenced "
    "identically, fails the same bounds on the overwhelming majority of the "
    "SAME windows — so the term separates 10 Hz differentiation noise from "
    "smoothness, not good driving from bad. Between two arms differing ONLY in "
    "schedule it moves 720x (0.0004 -> 0.2882), because re-timing at constant "
    "speed smooths a per-waypoint regression. It is dropped by the panel-wide "
    "gate for every arm under every progress term, so removing its weight is a "
    "provable no-op on every published number — what it removes is the false "
    "claim that the composite is three-term. The measurement is still emitted, "
    "under `components.comfort`, as the plan-smoothness flag it actually is.")


class EnvelopeViolation(AssertionError):
    """A grid point lies outside the MEASURED envelope.

    This is an ERROR, not a warning. The entire claim of this protocol is that
    the out-of-envelope fraction is **0 by construction**; a grid that breaks it
    silently would reproduce the exact defect (a too-generous string surviving
    because nobody read the field beside it) in a new costume."""


class LateralAxisRefused(AssertionError):
    """The lateral axis was requested without an explicit, reasoned override."""


class VacuousMetric(AssertionError):
    """No weighted component has usable dynamic range, so no composite is emitted."""


# =========================================================================== #
# the grid                                                                     #
# =========================================================================== #
@dataclass(frozen=True)
class GridSpec:
    """A BOUNDED, PRE-GENERATED set of ego-state perturbations.

    ``dyaw_deg``   heading offsets applied by ``sampling_homography(0, dpsi)`` —
                   geometrically exact for arbitrary depth.
    ``dlon_steps`` frame-index offsets along the logged path. The observation is
                   the REAL frame window at that index: zero synthesis, and the
                   (dlat, dpsi) envelope coordinates of the point are unchanged.
    ``dlat_m``     ⛔ refused by default; see :data:`LATERAL_REFUSAL`.
    """

    dyaw_deg: tuple = (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)
    dlon_steps: tuple = (-10, 0, 10)
    dlat_m: tuple = (0.0,)
    allow_lateral: bool = False
    lateral_override_reason: str = ""
    _meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if any(abs(float(v)) > 0.0 for v in self.dlat_m):
            if not self.allow_lateral or not self.lateral_override_reason.strip():
                raise LateralAxisRefused(LATERAL_REFUSAL)

    def points(self):
        """``[(dlat_m, dyaw_deg, dlon_steps), …]`` in a deterministic order."""
        return [(float(a), float(y), int(s))
                for a in self.dlat_m for y in self.dyaw_deg
                for s in self.dlon_steps]

    @property
    def n_points(self):
        return len(self.dlat_m) * len(self.dyaw_deg) * len(self.dlon_steps)

    def describe(self):
        return {
            "dlat_m": list(self.dlat_m), "dyaw_deg": list(self.dyaw_deg),
            "dlon_steps": list(self.dlon_steps), "n_points": self.n_points,
            "lateral_axis": ("ENABLED (override: "
                             + self.lateral_override_reason + ")"
                             if self.allow_lateral and
                             any(abs(v) > 0 for v in self.dlat_m)
                             else "REFUSED"),
            "lateral_refusal": LATERAL_REFUSAL,
            "warped_axes": ["heading"],
            "unwarped_axes": ["longitudinal (real frames at an index offset)"],
        }


def default_grid(**kw):
    """The shipped grid: heading x longitudinal, lateral refused.

    Heading spans the full MEASURED envelope ``|dpsi| <= 12 deg`` — the widest
    grid that is still 0 % out of envelope. NAVSIM v2's own heading filter is
    20 deg, so **ours is NARROWER than theirs**; that is a disclosable
    limitation, not a defect: our 12 deg is measured, their 20 deg is chosen."""
    return GridSpec(**kw)


def assert_grid_in_envelope(grid, *, _path="pseudosim.grid") -> dict:
    """⭐ THE ASSERTION. 0 % out of envelope, or the run does not start.

    Returns the proof (fractions + verdict + the falsifier), and raises
    :class:`EnvelopeViolation` otherwise. Run **before** any model is loaded, so
    a bad grid costs zero GPU seconds.
    """
    pts = grid.points()
    lat = np.array([[abs(p[0])] for p in pts], dtype=float)      # [n_pts, 1]
    yaw = np.array([[abs(p[1])] for p in pts], dtype=float)
    frac = _ood.envelope_fractions(lat, yaw)
    fw = frac["frac_windows_any_step_out_of_envelope"]
    fs = frac["frac_steps_any"]
    proof = {
        "n_grid_points": len(pts),
        "max_abs_dlat_m": float(lat.max()), "max_abs_dyaw_deg": float(yaw.max()),
        "envelope": frac["envelope"],
        "EXTRAPOLATION_frac_steps_lat_over_3m": frac["frac_steps_lat_over_3m"],
        "EXTRAPOLATION_frac_steps_yaw_over_12deg": frac["frac_steps_yaw_over_12deg"],
        "EXTRAPOLATION_frac_steps_any": fs,
        "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": fw,
        "ood_peak_ratio": None,
        "ratio_is_lower_bound": bool(fs > 0.0 or fw > 0.0),
        "EXTRAPOLATION_VERDICT": _ood._verdict_string(False, fw, fs),
        "why_zero_by_construction": (
            "Deviation is the GRID, not an accumulated rollout state. The "
            "protocol has no mechanism by which a window can drift outside the "
            "envelope, because no window ever advances."),
        "falsifier": {
            "_what": "the assertion CAN fail; these are the values that fail it",
            "smallest_failing_abs_dyaw_deg": ENV_YAW_MAX + 1e-9,
            "smallest_failing_abs_dlat_m": ENV_LAT_MAX + 1e-9,
            "example": (f"GridSpec(dyaw_deg=({ENV_YAW_MAX + 0.5},)) raises "
                        f"EnvelopeViolation; it is exercised in "
                        f"taniteval/tests/test_pseudosim.py."),
        },
    }
    if fw > 0.0 or fs > 0.0:
        raise EnvelopeViolation(
            f"[{_path}] {fs:.4%} of grid points / {fw:.4%} of grid rows lie "
            f"OUTSIDE the MEASURED envelope (|dlat| <= {ENV_LAT_MAX} m, |dyaw| "
            f"<= {ENV_YAW_MAX} deg). Pseudo-simulation's entire claim is that "
            f"this fraction is 0 BY CONSTRUCTION. Shrink the grid; do not widen "
            f"the envelope — P1 proved arithmetically that widening cannot reach "
            f"0 for a sequential rollout, and for a grid it is simply a choice.")
    # belt and braces: the same consistency check the OOD guard applies
    _ood.assert_envelope_verdict_consistent(proof, _path=_path)
    return proof


def proximity_weights(pts, *, yaw_sigma_deg=8.0, lon_sigma_steps=12.0):
    """NAVSIM v2's proximity weighting, on our axes.

    *"A proximity-based weighting scheme assigns higher importance to synthetic
    observations that best match the AV's likely behaviour."* Implemented as a
    separable Gaussian kernel on the perturbation magnitude, normalised to sum 1.

    ⚠️ A weighting choice must not be able to manufacture a result, so
    :func:`emit` always reports the **unweighted** aggregate beside the weighted
    one and flags any disagreement in sign.
    """
    y = np.array([p[1] for p in pts], dtype=float)
    s = np.array([p[2] for p in pts], dtype=float)
    w = np.exp(-0.5 * (y / yaw_sigma_deg) ** 2) * \
        np.exp(-0.5 * (s / lon_sigma_steps) ** 2)
    return w / w.sum()


# =========================================================================== #
# the evaluation — ONE planner call per (window, grid point). No rollout.       #
# =========================================================================== #
def _default_frames(ep, a, b):
    fr = ep.frames[a:b] if hasattr(ep, "frames") else ep["frames_u8"][a:b]
    fr = torch.as_tensor(fr)
    return fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()


def _poses_of(ep):
    p = ep.poses if hasattr(ep, "poses") else ep["poses"]
    return torch.as_tensor(p, dtype=torch.float32)


@torch.no_grad()
def pseudo_evaluate(planner, episodes, grid, *, device="cpu", stride=8,
                    window=WINDOW, horizon=20, frames_of=None, goals=None,
                    batch=16, verbose=False, frame=None) -> dict:
    """Evaluate ``planner`` at every (anchor, grid point). **No rollout.**

    For each anchor ``a`` and grid point ``(dlat, dpsi, dlon)``:

    1. take the **real** frame window ``[a + dlon, a + dlon + window)``;
    2. warp it **once** by ``clhorizon.warp_frames(dlat, dpsi, frame)`` — on the
       DEPLOYED 256x256 pinhole frame that is ``sampling_homography`` verbatim;
       on a cylindrical frame it is the equidistant-azimuth re-render, where a
       yaw is the EXACT pixel shift ``u -> u + f_ref*psi``, ``v -> v``;
    3. ask the planner for a trajectory (ONE call, in the ego frame of the
       perturbed pose);
    4. record it. Nothing is fed back; step 1 is never re-entered.

    Because the observation is synthesised at the *chosen* deviation and the
    loop never advances, the deviation of every evaluated state is exactly the
    grid value — which is why :func:`assert_grid_in_envelope` can guarantee 0 %.

    ``frame`` is the :class:`CanonicalFrame` (or ``to_dict()``) the episodes'
    pixels were BUILT at, read from the cache — never re-derived. ``None`` is
    the deployed 256x256 pinhole frame and is bit-identical to the
    pre-2026-07-27 path. ⛔ A non-256x256 raster with ``frame=None`` is REFUSED
    before any model is touched: that is exactly what a v5 mid-run held-out gate
    would have done silently, and the mid-run gate is what a live run STOPS on.

    Returns per-(window, grid point) arrays plus the envelope proof. Scoring is
    :func:`score_windows`; keeping them separate means any metric can be
    re-derived from the dump with **no GPU** — the arithmetic-only path whose
    absence forced five closed-loop artifacts to be re-driven in July.
    """
    proof = assert_grid_in_envelope(grid)          # BEFORE any model touch
    # ⛔ and the geometry, for the same reason and in the same place.
    warp_prov = assert_warp_frame(
        frame,
        ((frames_of or _default_frames)(episodes[0], 0, window)
         if len(episodes) else None),
        where="pseudosim.pseudo_evaluate")
    # ⛔ E1 (2026-07-28): an ego-TRAINED checkpoint scored ego-BLIND is silent.
    # Refused here, BEFORE any model is touched, for the same reason the
    # envelope assertion is: a bad run must cost zero GPU seconds.
    ego_prov = _egg.assert_adapter_declares_ego(
        planner, where="pseudosim.pseudo_evaluate")
    frames_of = frames_of or _default_frames
    pts = grid.points()
    lon_lo, lon_hi = min(p[2] for p in pts), max(p[2] for p in pts)

    rec = {k: [] for k in ("eid", "anchor", "pt_dlat", "pt_dyaw", "pt_dlon",
                           "v0", "ep_i")}
    trajs, ref_paths, ref_yaw = [], [], []
    n_calls = 0
    for ep_i, ep in enumerate(episodes):
        poses = _poses_of(ep)
        T = int(poses.shape[0])
        lo = max(0, -lon_lo)
        hi = T - window - horizon - max(0, lon_hi)
        anchors = list(range(lo, hi, stride))
        if not anchors:
            continue
        for (dlat, dyaw, dlon) in pts:
            for bi in range(0, len(anchors), batch):
                ch = anchors[bi:bi + batch]
                s = torch.tensor([a + dlon for a in ch])
                last = s + window - 1
                fw = torch.stack([frames_of(ep, int(x), int(x) + window)
                                  for x in s]).to(device)
                fw = warp_frames(fw, dlat, dyaw, frame,
                                 where="pseudosim.pseudo_evaluate")
                v0 = poses[last, 3]
                g = (goals.get(ep_i, last.numpy(), device)
                     if goals is not None else None)
                tj = planner.traj(fw, v0.to(device), g)[:, :horizon].cpu().float()
                n_calls += len(ch)
                trajs.append(tj)
                # the logged path AHEAD of the observed reference pose, in world
                idx = last[:, None] + torch.arange(0, horizon + 1)[None]
                ref_paths.append(poses[idx][..., :2])
                ref_yaw.append(poses[last, 2])
                rec["eid"] += [str(ep_i)] * len(ch)
                rec["anchor"].append(torch.tensor(ch))
                rec["ep_i"].append(torch.full((len(ch),), ep_i))
                rec["v0"].append(v0)
                rec["pt_dlat"].append(torch.full((len(ch),), float(dlat)))
                rec["pt_dyaw"].append(torch.full((len(ch),), float(dyaw)))
                rec["pt_dlon"].append(torch.full((len(ch),), float(dlon)))
        if verbose:
            print(f"    [pseudosim] ep {ep_i + 1}/{len(episodes)} "
                  f"planner calls so far {n_calls}", flush=True)
    if not rec["eid"]:
        return {"_empty": True, "envelope_proof": proof,
                "ego_input": ego_prov, "warp": warp_prov,
                "traffic_mode": TRAFFIC_MODE_LOG_REPLAY}
    out = {k: (v if k == "eid" else torch.cat(v)) for k, v in rec.items()}
    out["traj"] = torch.cat(trajs)                 # [n, horizon, 2] ego frame
    out["ref_path"] = torch.cat(ref_paths)         # [n, horizon+1, 2] world
    out["ref_yaw"] = torch.cat(ref_yaw)            # [n] world heading at ref
    out["envelope_proof"] = proof
    out["ego_input"] = ego_prov
    out["warp"] = dict(warp_prov, max_no_ground_preimage_frac=round(
        float(warp_frames.last_invalid_frac), 6))
    out["grid"] = grid.describe()
    out["horizon_steps"] = int(horizon)
    out["horizon_s"] = round(horizon * DT, 2)
    out["traffic_mode"] = TRAFFIC_MODE_LOG_REPLAY
    out["traffic_mode_note"] = TRAFFIC_MODE_NOTE
    out["protocol"] = PROTOCOL
    out["planner_calls"] = int(n_calls)
    out["rollout_steps_executed"] = 0
    out["_no_accumulation"] = (
        "rollout_steps_executed == 0 BY CONSTRUCTION: the loop never advances, "
        "so deviation cannot accumulate and the envelope cannot be left.")
    return out


# =========================================================================== #
# the map-free composite                                                       #
# =========================================================================== #
def _cross_and_along(pw):
    """Plan endpoint expressed relative to the LOGGED path (map-free).

    The perturbed ego sits at the logged reference pose offset by ``(dlat,
    dpsi)``; the plan is in ITS ego frame. Both are lifted into the reference
    pose's frame, where component 0 is along-track and component 1 is
    cross-track. No map, no lane graph, no drivable-area polygon is used —
    PhysicalAI-AV has none (settled at five probes).
    """
    tj = pw["traj"]                                    # [n, Hh, 2] perturbed ego
    dpsi = torch.deg2rad(pw["pt_dyaw"])
    dlat = pw["pt_dlat"]
    c, s = torch.cos(dpsi), torch.sin(dpsi)
    # perturbed-ego -> reference-ego: rotate by dpsi, then offset by dlat on y
    x = c[:, None] * tj[..., 0] - s[:, None] * tj[..., 1]
    y = s[:, None] * tj[..., 0] + c[:, None] * tj[..., 1] + dlat[:, None]
    # the logged path in the reference-ego frame
    rp = pw["ref_path"]
    ryaw = pw["ref_yaw"]
    dx = rp[..., 0] - rp[:, :1, 0]
    dy = rp[..., 1] - rp[:, :1, 1]
    cr, sr = torch.cos(ryaw)[:, None], torch.sin(ryaw)[:, None]
    ref_x = cr * dx + sr * dy
    ref_y = -sr * dx + cr * dy
    return x, y, ref_x, ref_y


def score_windows(pw, *, comfort_limits=None, dt=DT,
                  progress_term=PROGRESS_TERM_DEFAULT,
                  recovery_term=None) -> dict:
    """Per-(window, grid point) sub-scores. Pure arithmetic — **no GPU**.

    Components, each map-free and each with its discriminative range MEASURED
    (never assumed) by :func:`discriminative_range`:

    ``ego_progress``  along-track distance the plan covers, over the along-track
                      distance the human covered on the same window, scored by
                      the VERSIONED :data:`PROGRESS_TERMS` map. **PUBLISHED: the
                      strongest single predictor of closed-loop Driving Score,
                      Spearman rho = 0.83**, ahead of collision rate (0.45);
                      ADE/L2 is **-0.36, p = 0.43**.

                      ⛔ The published term ``clamp_v1`` is ONE-SIDED: it charges
                      NOTHING for over-travel, and v1 over-travels on 48.80 % of
                      windows (p95 ratio 2.430x), so a 5.65x along-track RMS
                      improvement scored **n.s.** while a 3.36x DEGRADATION on
                      the same axis and rows scored SEPARATED. The default is
                      now ``twosided_v2``; the old value stays computable and is
                      identified in the composite's ``name``.
    ``recovery``      does the plan converge back to the logged path from the
                      perturbed state? A VERSIONED function of
                      ``r = |xt_end| / |xt_hold_matched|`` (:data:`RECOVERY_TERMS`).
                      **This is the error-recovery signal pseudo-simulation
                      exists to produce and that open-loop ADE provably does not
                      measure.** Undefined (NaN) at the unperturbed grid point,
                      by construction.

                      ⛔ The published term ``clamp_v1`` is ONE-SIDED: it is
                      ``max(1 - r, 0)``, so it charges NOTHING for any ``r > 1``
                      — a plan that ends FURTHER off the path than not steering
                      at all. MEASURED: **55.65-92.19 %** of defined rows sit on
                      that floor and the MEDIAN ratio exceeds 1.0 for every arm,
                      so an injected lateral degradation that helps a minority
                      of rows RAISED the composite by **+0.0303 to +0.0747,
                      SEPARATED, 8 of 8 injections, both arms** (C45). The
                      default is now ``twosided_v2``; the old value stays
                      computable and is identified in the composite's ``name``.

                      ⚠️ ``xt_hold_matched`` is computed from **the plan's OWN
                      along-track distance** — ``|dlat + s_along * tan(dpsi)|`` —
                      not from ``v0 * horizon``. That is not cosmetic: MEASURED
                      on the 2-episode smoke of 2026-07-27, the naive
                      ``v0``-based denominator scored the **BLIND** arm
                      **+0.597** above the sighted one, because a planner that
                      barely moves has a small cross-track error and was being
                      paid for it. **Standing still is not recovery.** With the
                      progress-matched denominator a stopped plan yields
                      ``xt_hold -> 0`` and the score is NaN (excluded), not 1.0.
                      Exercised by
                      ``test_recovery_is_not_gameable_by_standing_still``.
    ``comfort``       all of |a_lon|, |a_lat|, |jerk|, |yaw_rate| within bounds.
                      ⚠️ **PUBLISHED as saturated at >= 99.9 %** elsewhere; it is
                      admitted here only if OUR measurement gives it range.
    ``no_collision``  ⛔ ``None`` — see :data:`COLLISION_UNAVAILABLE_REASON`.
    ``ttc``           ⛔ ``None`` — same reason.
    """
    lim = dict(COMFORT_LIMITS if comfort_limits is None else comfort_limits)
    x, y, ref_x, ref_y = _cross_and_along(pw)
    n, Hh = x.shape

    # --- ego progress ------------------------------------------------------ #
    human = torch.sqrt((ref_x[:, -1] - ref_x[:, 0]) ** 2
                       + (ref_y[:, -1] - ref_y[:, 0]) ** 2)
    ego = x[:, -1]                                    # along-track, ref frame
    ratio = ego / human.clamp_min(1e-3)
    ep_score = progress_from_ratio(ratio, progress_term)
    ep_score = torch.where(human > 0.5, ep_score, torch.full_like(ep_score,
                                                                 float("nan")))

    # --- recovery ---------------------------------------------------------- #
    # cross-track of the plan endpoint from the logged path endpoint
    xt_end = (y[:, -1] - ref_y[:, -1]).abs()
    dpsi = torch.deg2rad(pw["pt_dyaw"])
    # ⚠️ PROGRESS-MATCHED denominator: the drift the plan's OWN along-track
    # distance would have produced had it not steered. Using v0 * horizon
    # instead pays a planner for standing still (MEASURED: it put the BLIND arm
    # 0.597 ABOVE the sighted one on the 2026-07-27 smoke).
    s_along = x[:, -1].clamp_min(0.0)
    xt_hold = (pw["pt_dlat"] + s_along * torch.tan(dpsi)).abs()
    # ⛔ THE RATIO IS NOW EMITTED, because the FLOORED HALF OF ITS DOMAIN IS THE
    # DEFECT (C45) and a term's saturation cannot be audited from the score.
    rc_ratio = xt_end / xt_hold.clamp_min(1e-6)
    rc = recovery_from_ratio(rc_ratio, recovery_term)
    rc = torch.where(xt_hold > 0.10, rc, torch.full_like(rc, float("nan")))
    # the naive denominator, kept only as a diagnostic so the defect stays visible
    xt_hold_v0 = (pw["pt_dlat"]
                  + pw["v0"].abs() * (Hh * dt) * torch.sin(dpsi)).abs()

    # --- comfort ----------------------------------------------------------- #
    px = torch.cat([torch.zeros(n, 1), x], 1)
    py = torch.cat([torch.zeros(n, 1), y], 1)
    vx, vy = torch.diff(px, dim=1) / dt, torch.diff(py, dim=1) / dt
    ax, ay = torch.diff(vx, dim=1) / dt, torch.diff(vy, dim=1) / dt
    jx, jy = torch.diff(ax, dim=1) / dt, torch.diff(ay, dim=1) / dt
    head = torch.atan2(vy, vx)
    yr = torch.diff(head, dim=1) / dt
    ok = ((ax.abs().amax(1) <= lim["a_lon_max_mps2"])
          & (ay.abs().amax(1) <= lim["a_lat_max_mps2"])
          & (torch.sqrt(jx ** 2 + jy ** 2).amax(1) <= lim["jerk_max_mps3"])
          & (yr.abs().amax(1) <= lim["yaw_rate_max_radps"]))
    comfort = ok.float()

    return {
        "ego_progress": ep_score.numpy(),
        "ego_progress_raw_ratio": ratio.numpy(),
        "_progress_term": progress_term,
        "_progress_term_note": (
            "clamp_v1 = clamp(r,0,1), the term EVERY pre-2026-07-28 PSS number "
            "was computed under; it charges nothing for over-travel. "
            "twosided_v2 = clamp(1-|1-r|,0,1), identical to clamp_v1 for r<=1."),
        "recovery": rc.numpy(),
        "recovery_raw_ratio": rc_ratio.numpy(),
        "_recovery_term": (RECOVERY_TERM_DEFAULT if recovery_term is None
                           else recovery_term),
        "_recovery_term_note": (
            "clamp_v1 = clamp(1-r,0,1) with r = |xt_end|/|xt_hold| — the term "
            "EVERY pre-2026-07-28 PSS number was computed under. It is FLOORED "
            "on 55.65-92.19 % of DEFINED rows (C45), so it charges nothing for "
            "any further divergence and an injected lateral degradation RAISES "
            "the mean. The floor fraction is emitted beside every value; see "
            "`saturation`."),
        "cross_track_end_m": xt_end.numpy(),
        "cross_track_hold_matched_m": xt_hold.numpy(),
        "along_track_end_m": s_along.numpy(),
        "_cross_track_hold_v0_m_DIAGNOSTIC_NOT_USED": xt_hold_v0.numpy(),
        "comfort": comfort.numpy(),
        "no_collision": None,
        "ttc": None,
        "_unavailable": {"no_collision": COLLISION_UNAVAILABLE_REASON,
                         "ttc": COLLISION_UNAVAILABLE_REASON},
        "_comfort_limits": lim,
        "_map_free": ("No map, lane graph, drivable-area polygon, traffic light "
                      "or route signal is used. PhysicalAI-AV has none — the "
                      "card says verbatim 'we do not include open maps data'. "
                      "DAC / Lane Keeping / Driving-Direction Compliance / "
                      "Traffic-Light Compliance are therefore IMPOSSIBLE here "
                      "and are not faked."),
    }


#: ⚠️ C45's STANDING CONSEQUENCE, as a threshold: a bounded score saturated on
#: at least half its defined rows is reported with an explicit warning wherever
#: it is published. This is NOT the gate (:data:`FLOOR_FRAC_MAX` = 0.95 is); it
#: is the visibility rule, because ``discriminative_range`` COMPUTED the floor
#: fraction for its whole life and never surfaced it beside a score.
SATURATION_WARN_FRAC = 0.50


def saturation(arr, *, floor=1e-3, ceil=0.999,
               warn_frac=SATURATION_WARN_FRAC, n_sub=None) -> dict:
    """⚠️ REPORT THIS BESIDE EVERY BOUNDED SCORE. C45's standing consequence.

    *"For every bounded term, report the FLOOR/CEILING FRACTION beside the
    score."* The floor fraction of ``recovery`` was computable from day one —
    :func:`discriminative_range` calculated it and threw it away — and it is the
    single statistic that would have shown, without any injection experiment,
    that the term could not charge a majority of its own rows.

    A term at its floor on ``f`` of rows has **zero gradient on ``f`` of the
    data**: it is a constant with noise there, and any perturbation that moves
    the remaining ``1 - f`` in the good direction moves the mean the wrong way.

    ⭐⭐ ``live_frac = 1 - floor_frac - ceiling_frac`` IS THE HEADLINE NUMBER,
    added 2026-07-28 after ``lat_heading``. The floor and ceiling fractions are
    each ONE-SIDED, and a rule written on them one at a time is structurally
    unable to see a term that splits its saturation between the two ends — 49 %
    floored plus 49 % ceilinged passes both 0.95 thresholds while having
    gradient on 2 % of its rows. ``live_frac`` is the two-sided statistic and it
    is what gate ``v2`` keys on (:data:`LIVE_FRAC_MIN`).

    ⚠️ ``n_sub`` DECLARES THE TERM'S ROW RESOLUTION — how many sub-samples are
    averaged into each row. It is metadata, not a threshold, and it exists
    because it is the *explanation* of which terms trip the gate: ``n_sub = 1``
    (a single value per row, e.g. ``lat_heading``, ``recovery``,
    ``ego_progress``) saturates at ROW level, while ``n_sub = 20``
    (``lon_track``, ``lat_track``) needs all 20 steps clamped before the row
    saturates at all. Undeclared is reported as UNDECLARED rather than guessed.
    """
    a = np.asarray(arr, dtype=float)
    fin = a[np.isfinite(a)]
    res = ("UNDECLARED" if n_sub is None else
           "PER-ROW (a single sample; its floor bites at ROW level)"
           if int(n_sub) <= 1 else
           f"MEAN over {int(n_sub)} sub-samples (a row saturates only if all "
           f"{int(n_sub)} do)")
    if fin.size == 0:
        return {"n_defined": 0, "defined_frac": 0.0,
                "floor_frac_le_0p001": None, "ceiling_frac_ge_0p999": None,
                "saturated_frac": None, "live_frac": None,
                "n_sub_per_row": n_sub, "row_resolution": res,
                "SATURATION_WARNING": "no finite values"}
    fl = float((fin <= floor).mean())
    ce = float((fin >= ceil).mean())
    live = 1.0 - fl - ce
    node = {"n_defined": int(fin.size),
            "defined_frac": round(float(np.isfinite(a).mean()), 6),
            "floor_frac_le_0p001": round(fl, 6),
            "ceiling_frac_ge_0p999": round(ce, 6),
            "saturated_frac": round(fl + ce, 6),
            # ⭐ the two-sided statistic. Gate v2 keys on THIS.
            "live_frac": round(live, 6),
            "n_sub_per_row": n_sub, "row_resolution": res,
            "SATURATION_WARNING": None}
    if max(fl, ce) >= warn_frac or live <= 1.0 - warn_frac:
        where = ("FLOOR" if fl >= ce else "CEILING") if max(fl, ce) >= warn_frac \
            else "TWO ENDS COMBINED"
        node["SATURATION_WARNING"] = (
            f"⛔ {fl + ce:.2%} of defined rows are SATURATED ({fl:.2%} floor, "
            f"{ce:.2%} ceiling; worst end: {where}), leaving live_frac = "
            f"{live:.4f}. On saturated rows the term has ZERO GRADIENT: it "
            f"cannot charge a further degradation, so an injection that helps a "
            f"minority can move the mean the WRONG WAY. This is class C45; do "
            f"not quote the level without this fraction. Row resolution: {res}.")
    return node


class UnknownGateVersion(ValueError):
    """A gate version that is not in :data:`GATE_VERSIONS`.

    Never falls back, for the identical reason :class:`UnknownProgressTerm`
    does not: a typo'd gate version would silently re-gate a published number."""


def discriminative_range(scores, *, by_arm=None, ceil_frac_max=CEIL_FRAC_MAX,
                         range_min=RANGE_MIN, floor_frac_max=FLOOR_FRAC_MAX,
                         gate_version=GATE_VERSION_DEFAULT,
                         live_frac_min=None, n_sub=None) -> dict:
    """⚠️ BOOST M8 / C13 applied to METRICS: state the range before adopting.

    *"Comfort saturates at >= 99.9 %, contributing essentially zero
    discriminative information"* is a dead clause in someone else's suite. A
    component is **admissible only if it is measured to have range** here.

    ``scores`` maps component name -> 1-D array (NaN allowed).
    ``by_arm`` optionally maps arm name -> {component: array}; the BETWEEN-ARM
    spread is what actually decides an adjudication, so it is reported when
    available and is what ``admissible`` keys on.

    ⭐ ``gate_version`` (2026-07-28). ``v1`` is the PUBLISHED gate and the
    default, so every pre-2026-07-28 composite re-gates identically. ``v2`` adds
    the two-sided ``live_frac >= LIVE_FRAC_MIN`` clause that can refuse a
    per-row-floored term — see :data:`FLOOR_FRAC_MAX` for why the published
    constant is not simply lowered instead. ``v2`` is strictly stronger: it
    never admits anything ``v1`` refuses. Both verdicts are ALWAYS emitted
    (``admissible_v1`` / ``admissible_v2``), whichever version gates, so the
    difference can never be invisible.

    ``n_sub`` optionally maps component name -> sub-samples averaged per row;
    it is passed to :func:`saturation` as metadata (see there).
    """
    if gate_version not in GATE_VERSIONS:
        raise UnknownGateVersion(
            f"unknown gate_version {gate_version!r}; known: "
            f"{sorted(GATE_VERSIONS)}. Refusing to fall back — every number "
            f"published through 2026-07-28 was gated under "
            f"{GATE_VERSION_PUBLISHED!r}.")
    lfm = (GATE_VERSIONS[gate_version]["live_frac_min"] if live_frac_min is None
           else float(live_frac_min))
    n_sub = dict(n_sub or {})
    out = {"_gate": {"ceil_frac_max": ceil_frac_max, "range_min": range_min,
                     "floor_frac_max": floor_frac_max,
                     "gate_version": gate_version,
                     "live_frac_min": lfm,
                     "gate_versions": GATE_VERSIONS,
                     "rule": "admissible iff ceiling_frac < ceil_frac_max AND "
                             "floor_frac < floor_frac_max AND "
                             "(max - min) >= range_min. When >= 2 arms are "
                             "supplied, the between-arm spread must also be "
                             "non-zero. GATE v2 ADDS: live_frac >= "
                             "live_frac_min.",
                     "_live_frac_clause_added": (
                         "2026-07-28, one day after the floor clause and for "
                         "the SAME root cause one level up. floor_frac and "
                         "ceiling_frac are each ONE-SIDED, so the pair cannot "
                         "express 'this term has no gradient left' — 49 % "
                         "floored + 49 % ceilinged clears both. And 0.95 is a "
                         "dead-component tripwire calibrated against terms that "
                         "are MEANS over 20 steps; applied to a SINGLE-VALUE-"
                         "PER-ROW term it is ~20x too loose. MEASURED: "
                         "`control.lat_heading` floors on 31.22-84.29 % of rows "
                         "with its ceiling never active and FLOOR_FRAC_MAX = "
                         "0.95 refuses none of it."),
                     "_floor_clause_added": (
                         "2026-07-28. floor_frac was COMPUTED and never USED, "
                         "so a component pinned at its FLOOR was admissible "
                         "while the same component pinned at its CEILING was "
                         "refused — the 'audited on one side of a two-sided "
                         "object' class. MEASURED: `recovery` is at its floor "
                         "on 54.75-92.18 % of its DEFINED rows across the "
                         "2026-07-27 panel.")}}
    for name, arr in scores.items():
        # ``composite`` reads `_progress_term` / `_recovery_term` out of the
        # SAME dict, so a caller that hands the whole `score_windows` result
        # here would otherwise crash on a metric-id string. Provenance keys are
        # not components; skip them rather than making every caller filter.
        if name.startswith("_"):
            continue
        if arr is None:
            out[name] = {"admissible": False, "reason": "NOT COMPUTABLE",
                         "detail": COLLISION_UNAVAILABLE_REASON}
            continue
        a = np.asarray(arr, dtype=float)
        fin = a[np.isfinite(a)]
        if fin.size == 0:
            out[name] = {"admissible": False, "reason": "no finite values",
                         "n": 0}
            continue
        ceil = float((fin >= 0.999).mean())
        floor = float((fin <= 0.001).mean())
        rng = float(fin.max() - fin.min())
        node = {
            "n": int(fin.size), "n_nan": int(a.size - fin.size),
            "min": round(float(fin.min()), 6), "max": round(float(fin.max()), 6),
            "mean": round(float(fin.mean()), 6),
            "p05": round(float(np.percentile(fin, 5)), 6),
            "p95": round(float(np.percentile(fin, 95)), 6),
            "iqr": round(float(np.percentile(fin, 75)
                               - np.percentile(fin, 25)), 6),
            "ceiling_frac_ge_0p999": round(ceil, 6),
            "floor_frac_le_0p001": round(floor, 6),
            "observed_range": round(rng, 6),
            # ⚠️ C45: the saturation node travels WITH the score, always. The
            # floor fraction was computed here and discarded for this gate's
            # whole life; it is now emitted whether or not it gates.
            "saturation": saturation(a, n_sub=n_sub.get(name)),
        }
        live = 1.0 - ceil - floor
        node["live_frac"] = round(float(live), 6)
        v1 = bool(ceil < ceil_frac_max and floor < floor_frac_max
                  and rng >= range_min)
        # ⭐ BOTH verdicts are always emitted, whichever version gates — a gate
        # whose stricter reading is invisible is a gate nobody will ever adopt.
        v2 = bool(v1 and live >= GATE_VERSIONS["v2"]["live_frac_min"])
        node["admissible_v1"] = v1
        node["admissible_v2"] = v2
        node["admissible"] = v1 if lfm is None else bool(v1 and live >= lfm)
        if not node["admissible"]:
            node["reason"] = (
                "SATURATED at the ceiling" if ceil >= ceil_frac_max else
                "SATURATED at the floor" if floor >= floor_frac_max else
                "range below range_min" if rng < range_min else
                f"NO GRADIENT LEFT: live_frac {live:.4f} < {lfm} — "
                f"{floor:.2%} of rows at the floor and {ceil:.2%} at the "
                f"ceiling. Refused by gate {gate_version}; gate v1 admitted it, "
                f"which is the defect v2 exists to fix (C45 / lat_heading).")
        if by_arm and len(by_arm) >= 2:
            means = {k: float(np.nanmean(np.asarray(v[name], float)))
                     for k, v in by_arm.items()
                     if v.get(name) is not None
                     and np.isfinite(np.asarray(v[name], float)).any()}
            if len(means) >= 2:
                sp = max(means.values()) - min(means.values())
                node["between_arm_mean"] = {k: round(v, 6)
                                            for k, v in means.items()}
                node["between_arm_spread"] = round(float(sp), 6)
                if sp <= 0.0:
                    node["admissible"] = False
                    node["admissible_v1"] = node["admissible_v2"] = False
                    node["reason"] = "zero between-arm spread — cannot adjudicate"
        out[name] = node
    return out


def composite(scores, ranges, *, weights=None, gates=("no_collision",),
              progress_term=None, recovery_term=None) -> dict:
    """PDM-shaped composite over the **admissible** components only.

    ``PDMS = (prod gate_m) x (sum w_x s_x / sum w_x)``. Here every multiplicative
    gate is a collision term and **all of them are unavailable**, so the product
    is empty.

    ⛔ **This is therefore NOT a Driving Score and is not named one.** A
    composite with no collision gate scores *recovery and progress*, and calling
    it anything else would be the same over-claim the OOD verdict string made.

    Raises :class:`VacuousMetric` if no weighted component is admissible —
    refusing to emit is the only honest output when every clause is dead.
    """
    # The progress term rides in the NAME. `scores` carries it when it came from
    # `score_windows`; an explicit argument wins. Never guessed: an unlabelled
    # composite is exactly the silent-redefinition failure being fixed.
    term = (progress_term if progress_term is not None
            else scores.get("_progress_term", PROGRESS_TERM_DEFAULT))
    # ⚠️ SAME RULE FOR THE RECOVERY TERM, and the fallback is the PUBLISHED one,
    # not the default: a caller that hands in a raw dict of arrays with no
    # `_recovery_term` key is reproducing a pre-2026-07-28 number, and naming it
    # after the NEW term would be exactly the silent redefinition being fixed.
    rterm = (recovery_term if recovery_term is not None
             else scores.get("_recovery_term", RECOVERY_TERM_PUBLISHED))
    w = dict(COMPONENT_WEIGHTS if weights is None else weights)
    admitted, dropped, zeroed = {}, {}, {}
    for name, wt in w.items():
        r = ranges.get(name, {})
        if float(wt) == 0.0:
            # ⭐ EXPLICIT, not silent. A zero weight adds exactly 0.0 to both
            # numerator and denominator, so dropping it here is arithmetically
            # identical to leaving it in — but the emitted node then says so.
            zeroed[name] = COMFORT_STATUS if name == "comfort" else "weight 0.0"
        elif r.get("admissible"):
            admitted[name] = wt
        else:
            dropped[name] = r.get("reason", "not admissible")
    if not admitted:
        raise VacuousMetric(
            "REFUSING TO EMIT a composite: no weighted component cleared the "
            f"discriminative-range gate. Dropped: {dropped}. A metric that "
            "cannot fail is not a metric; this program has shipped three "
            "vacuous diagnostics and the correct output here is nothing.")
    num = np.zeros_like(np.asarray(scores[next(iter(admitted))], float))
    den = np.zeros_like(num)
    for name, wt in admitted.items():
        a = np.asarray(scores[name], float)
        m = np.isfinite(a)
        num = num + np.where(m, a * wt, 0.0)
        den = den + np.where(m, wt, 0.0)
    val = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    gate_state = {g: {"available": scores.get(g) is not None,
                      "reason": None if scores.get(g) is not None
                      else COLLISION_UNAVAILABLE_REASON} for g in gates}
    return {
        "name": metric_id(term, rterm),
        "progress_term": term,
        "recovery_term": rterm,
        "_progress_term_warning": (
            "⛔ QUOTE THE VERSIONED NAME. Every PSS number published before "
            f"2026-07-28 is {PROGRESS_TERM_PUBLISHED!r} — a ONE-SIDED term that "
            "charges nothing for over-travel. Two values under different terms "
            "are DIFFERENT METRICS and must never be compared."),
        "_recovery_term_warning": (
            "⛔ BOTH weight-5.0 terms were one-sidedly clamped. Every PSS "
            f"number published through 2026-07-28 also used recovery "
            f"{RECOVERY_TERM_PUBLISHED!r}, which is FLOORED on 55.65-92.19 % of "
            "defined rows and therefore PAID for injected lateral degradation "
            "(+0.0303 to +0.0747 SEPARATED, 8/8 injections, both arms — C45). "
            "The recovery term is part of the metric id whenever it is not the "
            "published one."),
        "component_saturation": {
            k: saturation(scores[k]) for k in admitted
            if scores.get(k) is not None},
        "_component_saturation_note": (
            "⚠️ C45 STANDING CONSEQUENCE: the floor/ceiling fraction ships "
            "beside every bounded term, permanently. A term at its floor on "
            "most rows has zero gradient there and can be moved the WRONG WAY "
            "by a degradation."),
        "_not_a_driving_score": (
            "There is NO collision gate in this composite because the cuboids "
            "are not available (see gates.reason). PDMS-shaped, but it scores "
            "RECOVERY and PROGRESS only. Do not report it as a Driving Score "
            "and do not compare it to a PDMS number."),
        "formula": "(empty gate product) x (sum w_x s_x / sum w_x)",
        "weights_id": (WEIGHTS_ID if weights is None else "custom"),
        "weights_admitted": admitted,
        "components_dropped": dropped,
        "components_zero_weighted": zeroed,
        "n_weighted_terms": len(admitted),
        "gates": gate_state,
        "value": val,
    }


# =========================================================================== #
# aggregation                                                                  #
# =========================================================================== #
def _boot(x, eid, n_boot, seed):
    a = np.asarray(x, dtype=float)
    m = np.isfinite(a)
    if m.sum() < 2 or len(set(np.asarray(eid)[m])) < 2:
        return None
    return _ci.episode_cluster_bootstrap(a[m], list(np.asarray(eid)[m]),
                                         n_boot=n_boot, seed=seed)


def emit(pw, *, arm="unknown", n_boot=None, seed=0, weights=None,
         by_arm_scores=None, progress_term=PROGRESS_TERM_DEFAULT,
         recovery_term=None) -> dict:
    """The full result node for one arm: sub-scores, ranges, composite, CIs.

    Every node carries ``traffic_mode``, the envelope proof, the estimator and
    the refused estimator. Nothing here can be quoted without them.
    """
    n_boot = _ci.DEFAULT_N_BOOT if n_boot is None else int(n_boot)
    sc = score_windows(pw, progress_term=progress_term,
                       recovery_term=recovery_term)
    rterm = sc["_recovery_term"]
    eid = list(pw["eid"])
    comps = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
    comps["no_collision"] = None
    comps["ttc"] = None
    ranges = discriminative_range(comps, by_arm=by_arm_scores)

    node = {
        "arm": arm,
        "metric_id": metric_id(progress_term, rterm),
        "progress_term": progress_term,
        "recovery_term": rterm,
        "ego_input": pw.get("ego_input"),
        "protocol": PROTOCOL,
        "traffic_mode": TRAFFIC_MODE_LOG_REPLAY,
        "traffic_mode_note": TRAFFIC_MODE_NOTE,
        "grid": pw.get("grid"),
        "envelope_proof": pw.get("envelope_proof"),
        "horizon_s": pw.get("horizon_s"),
        "n_evaluations": int(len(eid)),
        "n_episodes": int(len(set(eid))),
        "planner_calls": pw.get("planner_calls"),
        "rollout_steps_executed": pw.get("rollout_steps_executed", 0),
        "_no_accumulation": pw.get("_no_accumulation"),
        "_estimator": "taniteval.ci.episode_cluster_bootstrap "
                      f"(B={n_boot}, unit = val episode)",
        "_refused_estimator": ("overlapping_holdout_se — it biases the POINT "
                               "ESTIMATE as well as the interval"),
        "component_discriminative_range": ranges,
        "components": {},
    }
    for k, v in comps.items():
        node["components"][k] = (
            # ⚠️ C45: `saturation` rides beside EVERY bounded component level,
            # not only inside the gate node. A level quoted without it is not
            # quotable.
            {"ci": _boot(v, eid, n_boot, seed),
             "saturation": saturation(v),
             "admissible": ranges[k].get("admissible")} if v is not None
            else {"ci": None, "admissible": False,
                  "reason": COLLISION_UNAVAILABLE_REASON})
    try:
        comp = composite(comps, ranges, weights=weights,
                         progress_term=progress_term, recovery_term=rterm)
        val = comp.pop("value")
        comp["ci"] = _boot(val, eid, n_boot, seed)
        # the weighted-vs-unweighted disagreement check
        w = proximity_weights([(0.0, float(a), float(b)) for a, b
                               in zip(pw["pt_dyaw"].numpy(),
                                      pw["pt_dlon"].numpy())])
        fin = np.isfinite(val)
        comp["proximity_weighted_mean"] = (
            round(float((val[fin] * w[fin]).sum() / w[fin].sum()), 6)
            if fin.any() else None)
        comp["unweighted_mean"] = (round(float(np.nanmean(val)), 6)
                                   if fin.any() else None)
        node["composite"] = comp
        node["_per_window_composite"] = val
    except VacuousMetric as exc:
        node["composite"] = {"REFUSED_TO_EMIT": str(exc)}
    node["_per_window"] = {k: v for k, v in sc.items()
                           if isinstance(v, np.ndarray)}
    node["_per_window"]["eid"] = np.asarray(eid)
    node["_per_window"]["pt_dyaw"] = pw["pt_dyaw"].numpy()
    node["_per_window"]["pt_dlon"] = pw["pt_dlon"].numpy()
    node["_per_window"]["pt_dlat"] = pw["pt_dlat"].numpy()
    return node
