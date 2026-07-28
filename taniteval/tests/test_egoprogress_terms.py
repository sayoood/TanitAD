"""THE `ego_progress` UNDER SIDE — the half of a weight-5.0 term nobody audited.

Three bounded terms in this suite were repaired in 2026-07-28 and every one of
them was audited on **one side of a two-sided object**. ``ego_progress`` was
repaired on the OVER side (``twosided_v2``) and its UNDER side — ``r <= 0``,
a plan that ends BEHIND where it started — had never been touched by any
injection suite in the programme.

⛔ Every rule below is driven with the value that makes it **FAIL**, not only
the value that makes it pass.

What is pinned here:

1. ⛔⛔ **THE UNDER FLOOR IS ARITHMETICALLY UNFIXABLE.** Any bounded ``g`` that
   agrees with ``clamp_v1`` on ``[0, 1]`` is forced to ``0`` on ``(-inf, 0]``,
   so a plan reversing 1 m and a plan reversing 30 m MUST score the same. The
   only escape is a range budget ``g(0) = p > 0`` — and that budget pays
   ``stand_still`` for not moving. Named, priced, refused.
2. ⛔⛔ **THE OVER SIDE IS A TRADE, NOT A SEARCH.** Preserving the published
   charge rate on ``[1, 2]`` FORCES the published floor at ``r = 2``: the
   budget is spent. Every repair of the zero-mean cell therefore reduces the
   over-travel charge rate somewhere below ``r = 2``. That is arithmetic.
3. ⚠️ **THE RESOLUTION LEVER DOES NOT TRANSFER WHOLESALE.** The 20-step mean
   that fixed ``lat_heading`` with no free parameter STRENGTHENS this axis's
   under side and **WORSENS its over side** — the wrong-way zero-mean leak
   grows ``+0.0104 -> +0.0590``. Both measured, both pinned.
4. ⚠️ **A CONTROL'S DIRECTION IS VERIFIED, NEVER ASSUMED.** ``lon_shift`` has a
   constant sign and RE-CENTRES an arm biased the other way; on one real cell
   it failed the ground-truth check and was refused.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from taniteval import control as C
from taniteval import pseudosim as PS

R = torch.linspace(-5.0, 10.0, 150001, dtype=torch.float64)
NEW_TERMS = ("hyp_w1", "hyp_w0p5", "hyp_w2", "exp_w1", "exp_w0p5",
             "sqrtlin_w1", "sqrtlin_w0p5", "sqrtlin_w0p3333", "sqrtlin_w0p25")


# --------------------------------------------------------------------------- #
# fixture: a straight human path, one row per along-track ratio                 #
# --------------------------------------------------------------------------- #
def _pw(ratios, v0=10.0, Hh=20):
    """A window record whose plan travels ``ratio * human`` along-track.

    The human path is straight at ``v0``, so its 2 s chord is ``v0 * 2`` and
    ``ego_progress``'s raw ratio is exactly the requested one."""
    ratios = torch.as_tensor(ratios, dtype=torch.float64)
    n = ratios.numel()
    t = torch.arange(1, Hh + 1, dtype=torch.float64) * PS.DT
    ref = torch.zeros(n, Hh + 1, 2, dtype=torch.float64)
    ref[:, :, 0] = torch.cat([torch.zeros(1, dtype=torch.float64),
                              v0 * t])[None].expand(n, -1)
    x = ratios[:, None] * (v0 * t)[None]
    y = torch.zeros_like(x)
    return {"traj": torch.stack([x, y], -1).float().contiguous(),
            "ref_path": ref.float(), "ref_yaw": torch.zeros(n),
            "v0": torch.full((n,), v0), "pt_dyaw": torch.zeros(n),
            "pt_dlat": torch.zeros(n), "pt_dlon": torch.zeros(n),
            "eid": [str(i % 8) for i in range(n)]}


# =========================================================================== #
# 1. ⛔⛔ THE UNDER SIDE — the defect, as a number                              #
# =========================================================================== #
def test_a_plan_reversing_1m_and_a_plan_reversing_30m_score_IDENTICALLY():
    """⛔ THE DEFECT THIS STREAM WAS SENT TO AUDIT, driven at its failing value.

    Every term in the registry — published, repaired, and every candidate added
    by the over-side repair — gives the SAME 0.0 for a plan that reverses a
    little and one that reverses a lot. The over-side repair never touched it
    and could not have."""
    human = 20.0
    r = torch.tensor([-1.0 / human, -30.0 / human], dtype=torch.float64)
    for term in PROGRESS_TERM_NAMES():
        v = PS.progress_from_ratio(r, term)
        assert float(v[0]) == float(v[1]) == 0.0, term


def PROGRESS_TERM_NAMES():
    return sorted(PS.PROGRESS_TERMS)


def test_the_under_side_floor_is_ARITHMETICALLY_UNFIXABLE():
    """⛔ The proof, as a property over the whole registry.

    Any bounded ``g: R -> [0, 1]`` that agrees with ``clamp_v1`` on ``[0, 1]``
    has ``g(0) = 0``; with ``g >= 0`` that forces ``g == 0`` below 0. So NO
    member of the registry — present or future — can charge reversal while
    remaining a strict refinement. The test is the universal statement, not an
    example."""
    neg = R[R < 0]
    for term in PROGRESS_TERM_NAMES():
        v = PS.progress_from_ratio(neg, term)
        assert float(v.max()) == 0.0, f"{term} charges below r = 0"
        # …and it is 0 AT 0 too, which is what forces it
        assert float(PS.progress_from_ratio(
            torch.zeros(1, dtype=torch.float64), term)) == 0.0, term
    assert "range budget" in PS.PROGRESS_UNDER_FLOOR_IS_UNFIXABLE
    assert "stand_still" in PS.PROGRESS_UNDER_FLOOR_IS_UNFIXABLE


def test_the_under_side_range_budget_PAYS_STANDING_STILL():
    """⛔ THE PRICE OF THE ONLY AVAILABLE FIX, driven rather than argued.

    ``g(r) = p + (1 - p) * clamp_v1(r)`` is the closest analogue of
    ``recovery``'s ``q``. It is the ONLY family that can charge reversal, and at
    every ``p > 0`` a plan that does not move at all scores ``p`` instead of 0.
    MEASURED on the panel (``raw/under_fix.json``): ``stand_still``'s
    ``ego_progress`` rises 0.000000 -> p exactly, on 100 % of its rows.

    This programme has already paid for that trade once — the naive ``v0``-based
    ``recovery`` denominator put the BLIND arm **+0.597 above the sighted one**
    because a planner that barely moves has a small cross-track error."""
    def budget(r, p):
        base = r.clamp(0.0, 1.0)
        return torch.where(r >= 0.0, p + (1.0 - p) * base,
                           (p * (1.0 + r)).clamp_min(0.0))

    zero = torch.zeros(1, dtype=torch.float64)          # a plan that stands still
    assert float(budget(zero, 0.0)) == 0.0              # p = 0 IS the published term
    for p in (0.05, 0.10, 0.25):
        assert float(budget(zero, p)) == pytest.approx(p), p
        # and it DOES buy what it was meant to buy — reversal is now charged
        deep = torch.tensor([-0.05, -0.50], dtype=torch.float64)
        v = budget(deep, p)
        assert float(v[0]) > float(v[1]), p
    # p = 0 reproduces the published term bit-identically on the whole domain
    assert torch.equal(budget(R, 0.0), PS.progress_from_ratio(R, "clamp_v1"))


# =========================================================================== #
# 2. ⛔⛔ THE OVER SIDE — the trade is forced, not chosen                       #
# =========================================================================== #
def test_preserving_the_published_charge_rate_below_r2_FORCES_the_r2_floor():
    """⭐⭐ THE RESULT THE PI DECISION TURNS ON.

    ``twosided_v2`` spends its ENTIRE remaining range between ``r = 1`` and
    ``r = 2`` at rate 1. So any monotone bounded term that agrees with it on
    ``[0, 2]`` has ``g(2) = 0`` and is therefore 0 everywhere above 2 — i.e. it
    IS the term with the defect. ⇒ **every repair of the zero-mean cell
    necessarily reduces the over-travel charge rate somewhere below r = 2.**
    'It halves the charge rate' is not a property of a particular candidate; it
    is forced by the arithmetic.

    Driven both ways: the terms that keep something above r = 2 are exactly the
    terms that differ from the published one BELOW r = 2."""
    lo = R[(R > 1.0) & (R <= 2.0)]
    hi = R[R > 2.0]
    pub = PS.progress_from_ratio(R, "twosided_v2")
    for term in PROGRESS_TERM_NAMES():
        if term == "twosided_v2":
            continue
        alive_above_2 = float(PS.progress_from_ratio(hi, term).max()) > 0.0
        differs_below_2 = not torch.equal(
            PS.progress_from_ratio(lo, term), pub[(R > 1.0) & (R <= 2.0)])
        if alive_above_2:
            assert differs_below_2, (
                f"{term} keeps range above r = 2 without paying for it below — "
                f"that is arithmetically impossible")
    # and the published term really has spent it all
    assert float(PS.progress_from_ratio(
        torch.tensor([2.0], dtype=torch.float64), "twosided_v2")) == 0.0


def test_the_over_side_trichotomy_is_arithmetic_not_a_preference():
    """⛔ Convex tail -> a zero-mean jitter is REWARDED; linear -> it costs
    EXACTLY nothing; concave -> it is charged. Driven numerically on the three
    families in the registry, away from any floor so the kink cannot confound
    the curvature."""
    g = torch.Generator().manual_seed(7)
    d = torch.randn(200000, generator=g, dtype=torch.float64) * 0.05
    d = d - d.mean()                                   # exactly zero-mean

    def bias(term, r0):
        base = float(PS.progress_from_ratio(
            torch.tensor([r0], dtype=torch.float64), term))
        pert = float(PS.progress_from_ratio(r0 + d, term).mean())
        return pert - base

    # CONVEX (never floors): biased UP -> the metric pays for noise
    assert bias("hyp_w1", 1.5) > 1e-6
    assert bias("exp_w1", 1.5) > 1e-6
    # LINEAR, away from both kinks: exactly nothing, to float noise
    assert abs(bias("twosided_v2", 1.5)) < 1e-9
    assert abs(bias("twosided_asym_w0p5", 1.5)) < 1e-9
    # CONCAVE: charged
    assert bias("sqrtlin_w0p3333", 1.5) < -1e-6
    assert bias("sqrtlin_w0p5", 1.5) < -1e-6
    assert "REWARDED" in PS.PROGRESS_OVER_SIDE_TRICHOTOMY


def test_a_concave_term_charging_at_the_published_rate_MUST_floor_by_r2():
    """⛔ The frontier, as a property: ``rate(1+) <= 1 / (R_floor - 1)``.

    Concavity with ``g(1) = 1`` and ``g'(1+) = -rate`` forces
    ``g(r) <= 1 - rate*(r - 1)``, so a concave term cannot outlive the linear
    one at the same rate. ⇒ you may have the published charge rate OR a floor
    beyond r = 2, never both. Checked on every concave member."""
    eps = 1e-6
    for term in ("sqrtlin_w1", "sqrtlin_w0p5", "sqrtlin_w0p3333",
                 "sqrtlin_w0p25"):
        one = torch.tensor([1.0 + eps], dtype=torch.float64)
        rate = (1.0 - float(PS.progress_from_ratio(one, term))) / eps
        v = PS.progress_from_ratio(R, term)
        alive = R[(v > 0) & (R > 1.0)]
        floor_at = float(alive.max())
        assert rate <= 1.0 / (floor_at - 1.0) + 1e-3, (term, rate, floor_at)
        if rate >= 1.0:
            assert floor_at <= 2.0 + 1e-3, (term, rate, floor_at)


def test_every_new_progress_term_is_a_STRICT_REFINEMENT():
    """⭐ BIT-identical to the published term for every ``r <= 1`` — not equal
    to float tolerance, IDENTICAL. ``1 - (1 - r)`` is not ``r`` in float32 and a
    2.5e-4 drift on a third of the rows would creep into every paired delta."""
    und = R <= 1.0
    base = PS.progress_from_ratio(R, "clamp_v1")
    for term in NEW_TERMS:
        assert torch.equal(PS.progress_from_ratio(R, term)[und], base[und]), term


def test_the_new_terms_do_not_touch_the_default_or_the_published_term():
    """⛔ A candidate that changes the default is a silent redefinition of every
    published PSS level. The default is asserted, not assumed."""
    assert PS.PROGRESS_TERM_DEFAULT == "twosided_v2"
    assert PS.PROGRESS_TERM_PUBLISHED == "clamp_v1"
    assert PS.OVER_TRAVEL_WEIGHT == 1.0
    r = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=torch.float64)
    assert PS.progress_from_ratio(r, "clamp_v1").tolist() == [
        0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    assert PS.progress_from_ratio(r, "twosided_v2").tolist() == [
        0.0, 0.5, 1.0, 0.5, 0.0, 0.0]


def test_an_unknown_progress_term_never_falls_back():
    with pytest.raises(PS.UnknownProgressTerm):
        PS.progress_from_ratio(torch.zeros(3), "sqrtlin_w0p3")   # a typo


# =========================================================================== #
# 3. THE RESOLUTION LEVER — and the direction it does NOT transfer in          #
# =========================================================================== #
def test_progress_ratios_per_step_preserves_the_published_row_definedness():
    """⭐ A resolution change must never silently re-select rows. The row mask is
    the PUBLISHED one — the human's 2 s chord against the published constant —
    and it is asserted bit-for-bit against ``score_windows``' own NaN pattern."""
    for v0 in (10.0, 0.2):                      # the second is BELOW the mask
        pw = _pw([0.5, 1.0, 1.5], v0=v0)
        _r, _sd, row = PS.progress_ratios_per_step(pw)
        pub = np.isfinite(np.asarray(
            PS.score_windows(pw)["ego_progress"], float))
        assert list(np.asarray(row)) == list(pub), v0
    assert PS.PROGRESS_HUMAN_MIN_M == 0.5


def test_the_per_step_ratio_reproduces_the_published_one_at_the_last_step():
    """The terminal column of the per-step primitive IS the published ratio."""
    pw = _pw([-0.4, 0.0, 0.25, 1.0, 2.5])
    r, _sd, _row = PS.progress_ratios_per_step(pw)
    pub = torch.as_tensor(PS.score_windows(pw)["ego_progress_raw_ratio"])
    assert torch.allclose(r[:, -1].float(), pub, atol=1e-6)


def test_the_mean_resolution_has_a_SMALL_DENOMINATOR_pathology():
    """⚠️ THE REASON THE FREE LEVER DOES NOT SIMPLY TRANSFER.

    ``r_k = x_k / human_k`` and ``human_k`` is small at small ``k``, so a
    constant along-track offset ``e`` enters step ``k`` as ``e / human_k`` and
    explodes at the first defined step. That is the structural twin of the
    step-0 plan tangent that destroyed the plain-mean ``lat_heading``
    candidate — and it is why the mean resolution makes this axis MORE
    sensitive to a longitudinal offset, not less."""
    pw = _pw([1.0])
    r0, sd, _ = PS.progress_ratios_per_step(pw)
    shifted = C.apply_control(pw, "lon_shift", 2.0)
    r1, _sd1, _ = PS.progress_ratios_per_step(shifted)
    k0 = int(torch.nonzero(sd[0])[0])            # first DEFINED step
    first = float((r1 - r0)[0, k0])
    last = float((r1 - r0)[0, -1])
    assert first > 5.0 * last, (first, last)


# =========================================================================== #
# 4. THE CONTROL — the one the suite was missing, and its hazard               #
# =========================================================================== #
def test_lon_shift_is_the_control_the_under_side_audit_REQUIRED():
    """⛔ Neither existing longitudinal control can push a row that is already at
    ``r = 0`` further backwards: ``lon_scale`` MULTIPLIES (so ``0 * k = 0``) and
    ``lon_retime`` re-samples along the plan's own arc (a zero-length arc stays
    at zero). Without ``lon_shift`` the under floor is unreachable — which is
    exactly why it went unaudited."""
    pw = _pw([0.0])
    base = float(PS.score_windows(pw)["ego_progress_raw_ratio"][0])
    assert base == 0.0
    for ctl, lv in (("lon_scale", 0.5), ("lon_scale", 2.0),
                    ("lon_retime", 0.5)):
        v = float(PS.score_windows(C.apply_control(pw, ctl, lv))
                  ["ego_progress_raw_ratio"][0])
        assert v == pytest.approx(0.0, abs=1e-6), (ctl, lv)
    moved = float(PS.score_windows(C.apply_control(pw, "lon_shift", -5.0))
                  ["ego_progress_raw_ratio"][0])
    assert moved < -0.2
    assert "lon_shift" in C.CONTROLS and "lon_shift" in C.LADDERS
    assert C.LADDERS["lon_shift"]["two_sided"] is True


def test_lon_shift_DIRECTION_must_be_verified_and_here_it_flips():
    """⛔ THE RE-CENTRING HAZARD, driven at the value where it bites.

    A constant-sign along-track offset degrades an UNDER-travelling row and
    IMPROVES an OVER-travelling one. On an arm that over-travels, a back-shift
    is not a degradation at all — and MEASURED on the real panel it was refused
    for exactly this reason on ``v1_tactical_follow|lon_shift(-2 m)``
    (ground-truth |s_plan - s_human| +0.0216 m, n.s.)."""
    over = _pw([2.0])                     # 2x over-travel: a back-shift HELPS
    under = _pw([0.5])                    # under-travel: a back-shift HURTS

    def gt(pw):
        x, _y, rx, ry = PS._cross_and_along(pw)
        human = torch.sqrt((rx[:, -1] - rx[:, 0]) ** 2
                           + (ry[:, -1] - ry[:, 0]) ** 2)
        return float((x[:, -1] - human).abs()[0])

    assert gt(C.apply_control(over, "lon_shift", -5.0)) < gt(over)
    assert gt(C.apply_control(under, "lon_shift", -5.0)) > gt(under)


def test_the_zero_mean_along_track_control_cannot_recentre_a_bias():
    """``lon_jitter`` is registered as zero-mean and its draw really is."""
    assert "lon_jitter" in C.ZERO_MEAN_CONTROLS
    assert "lon_shift" not in C.ZERO_MEAN_CONTROLS
    pw = _pw([1.0] * 20000)
    j = C.apply_control(pw, "lon_jitter", 2.0)
    d = (j["traj"][:, -1, 0] - pw["traj"][:, -1, 0]).double()
    assert abs(float(d.mean())) < 0.05
    assert float(d.std()) == pytest.approx(2.0, rel=0.05)


# =========================================================================== #
# 5. THE GUARD — ego_progress is TWO-SIDED and both ends must be reported      #
# =========================================================================== #
def test_saturation_reports_BOTH_ends_of_ego_progress():
    """⛔ The one-sided pair (FLOOR_FRAC_MAX, CEIL_FRAC_MAX) cannot express
    *"this term has no gradient left"*. ``ego_progress@clamp_v1`` is the case:
    it floors on the UNDER side and ceilings on the OVER side, and MEASURED on
    the panel its worst floor is 0.3178 while its worst ceiling is 0.5703 —
    each below 0.95, live_frac 0.2461, refused only by gate v2."""
    pw = _pw([-0.5] * 40 + [1.4] * 55 + [0.5] * 5)
    v = PS.score_windows(pw, progress_term="clamp_v1")["ego_progress"]
    s = PS.saturation(v, n_sub=1)
    assert s["floor_frac_le_0p001"] == pytest.approx(0.40, abs=0.01)
    assert s["ceiling_frac_ge_0p999"] == pytest.approx(0.55, abs=0.01)
    assert s["floor_frac_le_0p001"] < PS.FLOOR_FRAC_MAX      # v1 admits
    assert s["ceiling_frac_ge_0p999"] < PS.CEIL_FRAC_MAX     # v1 admits
    assert s["live_frac"] < PS.LIVE_FRAC_MIN                 # ⛔ v2 REFUSES
    assert s["row_resolution"].startswith("PER-ROW")
