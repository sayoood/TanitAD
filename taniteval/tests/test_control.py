"""Tests for :mod:`taniteval.control` — the longitudinal / lateral control suite.

⛔ **THE POINT OF THIS FILE IS THE FAILING DIRECTION.** Every rule below is
exercised with a value that makes it FAIL, not only with one that makes it pass.
The suite exists because this program shipped ``clamp_v1``, a metric that passed
every adversary it had (``stand_still``, ``v4_blind``, ``v1_ego_half`` — all too
*slow*) and was blind on the other side of 1.0 for its whole life.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from taniteval import control as C
from taniteval import pseudosim as PS

DT = C.DT
H = 20


# --------------------------------------------------------------------------- #
# synthetic dumps — a straight road at constant speed, and a curved one        #
# --------------------------------------------------------------------------- #
def _dump(plan_xy, ref_xy, *, n_ep=6, dyaw=0.0, dlon=0):
    """Build a ``pw``-shaped dict from one plan and one reference path."""
    plan = torch.as_tensor(np.asarray(plan_xy, np.float32))
    ref = torch.as_tensor(np.asarray(ref_xy, np.float32))
    n = plan.shape[0]
    eid = [str(i % n_ep) for i in range(n)]
    return {
        "traj": plan, "ref_path": ref,
        "ref_yaw": torch.zeros(n), "v0": torch.full((n,), 10.0),
        "pt_dlat": torch.zeros(n),
        "pt_dyaw": torch.full((n,), float(dyaw)),
        "pt_dlon": torch.full((n,), float(dlon)),
        "anchor": torch.arange(n), "ep_i": torch.arange(n) % n_ep,
        "eid": eid,
    }


def _straight(n=240, v=10.0, jitter=0.0, seed=0):
    """Human drives straight at ``v``; the plan is IDENTICAL to the human."""
    rng = np.random.default_rng(seed)
    t = np.arange(H + 1) * DT
    x = (v * t)[None, :].repeat(n, 0)
    y = np.zeros_like(x)
    if jitter:
        y = y + rng.normal(0, jitter, size=(n, 1))
    ref = np.stack([x, y], -1)
    plan = ref[:, 1:].copy()
    return _dump(plan, ref)


def _curved(n=240, v=10.0, radius=80.0):
    """Human drives a constant-radius arc; the plan follows it exactly."""
    t = np.arange(H + 1) * DT
    th = (v * t / radius)[None, :].repeat(n, 0)
    x = radius * np.sin(th)
    y = radius * (1 - np.cos(th))
    ref = np.stack([x, y], -1)
    return _dump(ref[:, 1:].copy(), ref)


# =========================================================================== #
# 1. geometry
# =========================================================================== #
def test_residuals_are_zero_when_the_plan_IS_the_logged_path():
    """⛔ FAILING VALUE: any non-zero residual for a plan that is the path.

    This pins the ``traj[k] <-> ref_path[k+1]`` index alignment. An off-by-one
    would report a 0.1 s lead — 1.0 m at 10 m/s — as a control error on every
    perfect plan in the program."""
    for pw in (_straight(), _curved()):
        r = C.residuals(pw)
        assert np.abs(r["along_err_m"]).max() < 1e-4
        assert np.abs(r["cross_err_m"]).max() < 1e-4
        assert np.abs(r["xte_m"]).max() < 1e-4
        assert np.nanmax(np.abs(r["heading_err_rad"])) < 1e-4


def test_a_perfect_plan_scores_1_on_every_axis():
    a = C.axes(_straight())
    for k in ("lon_track", "lat_track", "lat_heading"):
        assert np.nanmin(a[k]) > 0.999, k


def test_signed_xte_sign_is_LEFT_positive():
    pw = _straight()
    left = C.apply_control(pw, "lat_shift", +1.0)
    right = C.apply_control(pw, "lat_shift", -1.0)
    assert np.nanmean(C.residuals(left)["xte_m"]) > 0.9
    assert np.nanmean(C.residuals(right)["xte_m"]) < -0.9


def test_xte_is_the_perpendicular_distance_not_the_same_time_index():
    """⭐ The claim that makes ``lat_track`` a LATERAL axis, pinned.

    A plan that drives the correct *line* at 70 % speed has, at matched time
    index, a large apparent cross-track error on a curve — and an arc-matched
    XTE of ~0. ⛔ FAILING VALUE: an arc-matched XTE as large as the
    time-indexed one."""
    pw = _curved()
    slow = C.apply_control(pw, "lon_retime", 0.7)
    r = C.residuals(slow)
    time_indexed = float(np.abs(r["cross_err_m"]).mean())
    arc_matched = float(np.abs(r["xte_m"]).mean())
    assert time_indexed > 0.1, "the control must actually curve away in time"
    assert arc_matched < 0.05 * time_indexed


# =========================================================================== #
# 2. the row mask — the axes must be paired-comparable with the composite
# =========================================================================== #
def test_every_axis_shares_ego_progress_row_mask():
    """⛔ FAILING VALUE: an axis defined on a row where ``ego_progress`` is NaN.

    A paired bootstrap between an axis and the composite is only valid over
    identical rows."""
    pw = _straight()
    a = C.axes(pw)
    ep = np.isfinite(a["ego_progress"])
    for k in ("lon_track", "lat_track", "lat_heading"):
        assert not (np.isfinite(a[k]) & ~ep).any(), k


def test_the_mask_rule_is_bit_identical_to_pseudosim():
    pw = _straight()
    r = C.residuals(pw)
    sc = PS.score_windows(pw)
    assert (r["row_mask"] == np.isfinite(sc["ego_progress"])).all()


# =========================================================================== #
# 3. ⛔ the gaming adversaries — each is a value that makes a naive metric FAIL
# =========================================================================== #
def test_lat_track_is_not_gameable_by_standing_still():
    """⛔ THE MEASURED DEFECT, PINNED.

    On the real 2026-07-27 panel the naive definition gave ``stand_still`` the
    panel's HIGHEST ``lat_track`` (0.8455 vs ``cv_holdv0``'s 0.4414): a plan
    that does not move has almost no cross-track error and was being *paid* for
    it — the identical defect ``recovery``'s progress-matched denominator exists
    to remove. FAILING VALUE: a finite ``lat_track`` for a stopped plan."""
    pw = _straight()
    stopped = C.apply_control(pw, "lon_retime", 0.0)
    a = C.axes(stopped)
    assert not np.isfinite(a["lat_track"]).any()
    assert not np.isfinite(a["lat_heading"]).any()
    # and the LONGITUDINAL axis must still punish it, hard
    assert np.nanmean(a["lon_track"]) < 0.35


def test_lon_track_punishes_BOTH_too_slow_and_too_fast():
    """⛔ THE ``clamp_v1`` FAILURE, PINNED. FAILING VALUE: a 2x over-travelling
    plan scoring as high as a perfect one."""
    pw = _straight()
    base = float(np.nanmean(C.axes(pw)["lon_track"]))
    slow = float(np.nanmean(C.axes(C.apply_control(pw, "lon_retime", 0.5))
                            ["lon_track"]))
    fast = float(np.nanmean(C.axes(C.apply_control(pw, "lon_retime", 2.0))
                            ["lon_track"]))
    assert base > 0.99
    assert slow < base - 0.1
    assert fast < base - 0.1


def test_the_PUBLISHED_progress_term_cannot_see_what_lon_track_sees():
    """⭐ The contrast that justifies the axis: ``clamp_v1`` scores a 2x
    over-travelling plan 1.0000, identical to a perfect one."""
    pw = _straight()
    fast = C.apply_control(pw, "lon_retime", 2.0)
    blind = PS.score_windows(fast, progress_term="clamp_v1")["ego_progress"]
    assert np.nanmean(blind) > 0.999
    assert np.nanmean(C.axes(fast)["lon_track"]) < 0.9


def test_lat_track_punishes_BOTH_signs_of_a_lateral_error():
    pw = _straight()
    base = float(np.nanmean(C.axes(pw)["lat_track"]))
    for d in (-1.0, +1.0):
        v = float(np.nanmean(C.axes(C.apply_control(pw, "lat_shift", d))
                             ["lat_track"]))
        assert v < base - 0.1, d


def test_lat_bias_reveals_the_one_sidedness_lat_track_hides():
    """⭐ ``lat_track`` is sign-blind by design; ``lat_bias_m`` is where the sign
    lives. FAILING VALUE: identical ``lat_bias_m`` for a left and a right drift
    — which would mean the one-sidedness detector cannot detect one-sidedness."""
    pw = _straight()
    left = float(np.nanmean(C.axes(C.apply_control(pw, "lat_shift", +1.0))
                            ["lat_bias_m"]))
    right = float(np.nanmean(C.axes(C.apply_control(pw, "lat_shift", -1.0))
                             ["lat_bias_m"]))
    assert left > 0.9 and right < -0.9
    # …while lat_track is symmetric to within noise
    lt = [float(np.nanmean(C.axes(C.apply_control(pw, "lat_shift", d))
                           ["lat_track"])) for d in (-1.0, 1.0)]
    assert abs(lt[0] - lt[1]) < 0.02


def test_a_constant_shift_can_IMPROVE_a_biased_arm_and_the_suite_shows_it():
    """⛔ THE MEASURED TRAP: a constant-sign lateral control makes a
    one-sidedly biased planner BETTER. It is not a hypothetical — it is why
    :data:`ZERO_MEAN_CONTROLS` exists and why ``admit`` requires one."""
    pw = _straight(jitter=0.0)
    biased = C.apply_control(pw, "lat_shift", +1.0)        # a 1 m left bias
    fixed = C.apply_control(biased, "lat_shift", -1.0)     # the "degradation"
    assert (np.nanmean(C.axes(fixed)["lat_track"])
            > np.nanmean(C.axes(biased)["lat_track"]))
    # ⭐ and the ZERO-MEAN control cannot re-centre it: the RAW lateral error
    # rises (Jensen: E|b + eps| >= |b|), which is the guarantee that makes a
    # zero-mean ladder unambiguous.
    jit = C.apply_control(biased, "lat_jitter", 1.0)
    assert (np.nanmean(np.abs(C.residuals(jit)["xte_m"]))
            > np.nanmean(np.abs(C.residuals(biased)["xte_m"])))
    # …on an UNBIASED arm the score follows the raw error down, which is why the
    # demonstration is run on the near-unbiased reference arm.
    assert (np.nanmean(C.axes(C.apply_control(pw, "lat_jitter", 1.0))
                       ["lat_track"])
            < np.nanmean(C.axes(pw)["lat_track"]))


def test_a_BOUNDED_score_is_not_monotone_in_the_raw_error_on_a_BIASED_arm():
    """⚠️ SELF-REFUTATION, PINNED. My first version of the test above asserted
    that a zero-mean control must lower ``lat_track`` on ANY arm. It does not.

    On an arm with a 1 m standing bias, adding zero-mean noise RAISES the mean
    score even though it raises the mean error, because the score saturates at 0
    and rows pushed further out are charged no more. ⇒ the guarantee a zero-mean
    control gives is about the RAW error, not about a bounded score, and a
    dynamic-range demonstration must be run on a near-UNBIASED reference arm.
    That is exactly why ``run_control_suite.py`` runs its ladders on
    ``cv_holdv0`` and reports the biased arm separately."""
    pw = _straight(jitter=0.0)
    biased = C.apply_control(pw, "lat_shift", +1.0)
    jit = C.apply_control(biased, "lat_jitter", 1.0)
    assert (np.nanmean(np.abs(C.residuals(jit)["xte_m"]))
            > np.nanmean(np.abs(C.residuals(biased)["xte_m"])))
    assert (np.nanmean(C.axes(jit)["lat_track"])
            > np.nanmean(C.axes(biased)["lat_track"]))


# =========================================================================== #
# 4. axis separation — heading vs placement
# =========================================================================== #
def test_a_pure_offset_moves_placement_and_NOT_heading():
    """⭐ The two lateral axes are different instruments. FAILING VALUE: a
    heading response to a pure translation.

    ⚠️ UPDATED 2026-07-28 when ``lat_heading`` was VERSIONED (C45's third
    clamp). The ORIGINAL guarantee is preserved verbatim under the term it was
    written for, and the same guarantee is then re-asserted for the SHIPPED
    term in the only form that survives a range change: the offset response
    must stay far below the heading response. That is rule H0 of the term's
    selection, and it is what killed the ``mean`` resolution — where a
    ``lat_shift(2 m)`` moved the axis **-0.0222 SEPARATED**, 3.4x MORE than the
    smallest heading degradation, because step 0's tangent runs from the
    reference pose to waypoint 0."""
    pw = _curved()
    # (a) the published term, unchanged
    a = C.axes(C.apply_control(pw, "lat_shift", 1.0),
               lat_heading_term="term_lin_q0")
    assert np.nanmean(a["lat_track"]) < 0.9
    assert np.nanmean(a["lat_heading"]) > 0.999
    # (b) the shipped term: placement must move it MUCH less than pointing does
    base = float(np.nanmean(C.axes(pw)["lat_heading"]))
    off = float(np.nanmean(C.axes(C.apply_control(pw, "lat_shift", 1.0))
                           ["lat_heading"]))
    rot = float(np.nanmean(C.axes(C.apply_control(pw, "yaw_bias", 5.0))
                           ["lat_heading"]))
    assert abs(base - off) < abs(base - rot) / 3.0
    # (c) and the FAILING VALUE for the rejected resolution is pinned too
    bad = float(np.nanmean(C.axes(C.apply_control(pw, "lat_shift", 1.0),
                                  lat_heading_term="mean_lin_q0p5")
                           ["lat_heading"]))
    bad_base = float(np.nanmean(C.axes(pw, lat_heading_term="mean_lin_q0p5")
                                ["lat_heading"]))
    assert abs(bad_base - bad) > abs(base - off), (
        "the `mean` resolution must still demonstrate the leak it was rejected "
        "for; if this stops failing, H0 has lost its teeth")


def test_a_rotation_moves_heading():
    """⚠️ UPDATED 2026-07-28: the bar is expressed in units of the term's OWN
    range, because a range-budget term with ``q = 0.5`` cannot move 0.5 by
    construction. The published bar is preserved verbatim under the published
    term, and the direction is asserted under EVERY term in the registry — a
    rotation may never raise a heading score."""
    pw = _straight()
    # the published bar, on the term it was written for
    base = float(np.nanmean(C.axes(pw, lat_heading_term="term_lin_q0")
                            ["lat_heading"]))
    for g in (-10.0, +10.0):
        v = float(np.nanmean(C.axes(C.apply_control(pw, "yaw_bias", g),
                                    lat_heading_term="term_lin_q0")
                             ["lat_heading"]))
        assert v < base - 0.5, g
    # every term: the response is DOWN, and at least half the published bar
    # once rescaled by the term's own live range (1 - q).
    for term in sorted(C.LAT_HEADING_TERMS):
        b = float(np.nanmean(C.axes(pw, lat_heading_term=term)["lat_heading"]))
        for g in (-10.0, +10.0):
            v = float(np.nanmean(C.axes(C.apply_control(pw, "yaw_bias", g),
                                        lat_heading_term=term)["lat_heading"]))
            assert v < b, (term, g)


def test_lon_retime_preserves_the_PATH_and_lon_scale_does_not():
    """⛔ MY OWN FIRST CONTROL WAS NOT AXIS-PURE, and this pins the correction.

    An anisotropic along-axis scale rotates every segment of a curved plan, so
    it contaminates the lateral axes; a re-time walks the SAME polyline.
    FAILING VALUE: a re-time that changes the plan's geometry."""
    pw = _curved()
    r0 = C.residuals(pw)
    r_re = C.residuals(C.apply_control(pw, "lon_retime", 0.6))
    r_sc = C.residuals(C.apply_control(pw, "lon_scale", 0.6))
    # the re-timed plan still lies ON the path; the scaled one does not
    assert np.abs(r_re["xte_m"]).mean() < 0.02
    assert np.abs(r_sc["xte_m"]).mean() > 10 * np.abs(r_re["xte_m"]).mean()
    assert r0 is not None


# =========================================================================== #
# 5. the demonstration machinery
# =========================================================================== #
def test_dynamic_range_reports_an_MDE_on_both_sides():
    pw = _straight(n=360)
    d = C.dynamic_range(pw, pw["eid"], control="lat_shift", axis="lat_track",
                        n_boot=120)
    assert d["mde_up"] is not None and d["mde_down"] is not None
    assert d["both_directions_separate"] is True
    assert d["monotone_up"] and d["monotone_down"]


def test_dynamic_range_refuses_a_ladder_that_omits_its_null():
    with pytest.raises(ValueError):
        pw = _straight(n=60)
        C.dynamic_range(pw, pw["eid"], control="lat_shift", axis="lat_track",
                        levels=(0.5, 1.0), n_boot=10)


def test_admit_REFUSES_an_axis_that_only_separates_on_one_side():
    """⛔ THE ``clamp_v1`` RULE, IN CODE. FAILING VALUE: a one-sided
    demonstration being accepted."""
    one_sided = [{"axis": "fake", "control": "lat_shift", "unit": "m",
                  "two_sided_ladder": True, "zero_mean_control": False,
                  "separates_up": True, "separates_down": False,
                  "monotone_up": True, "monotone_down": None,
                  "mde_up": 0.5, "mde_down": None}]
    with pytest.raises(C.AxisNotDemonstrated):
        C.admit(one_sided)


def test_admit_REFUSES_an_axis_no_control_can_move():
    dead = [{"axis": "fake", "control": "lat_jitter", "unit": "m",
             "two_sided_ladder": False, "zero_mean_control": True,
             "separates_up": False, "separates_down": None,
             "monotone_up": None, "monotone_down": None,
             "mde_up": None, "mde_down": None}]
    with pytest.raises(C.AxisNotDemonstrated) as e:
        C.admit(dead)
    assert "cannot fail" in str(e.value)


def test_admit_REFUSES_when_no_zero_mean_control_separates():
    """⛔ Every separating control has a constant sign ⇒ the movement could be
    re-centring rather than degradation. MEASURED failure mode."""
    only_signed = [{"axis": "fake", "control": "lat_shift", "unit": "m",
                    "two_sided_ladder": True, "zero_mean_control": False,
                    "separates_up": True, "separates_down": True,
                    "monotone_up": True, "monotone_down": True,
                    "mde_up": 0.5, "mde_down": 0.5}]
    with pytest.raises(C.AxisNotDemonstrated) as e:
        C.admit(only_signed)
    assert "ZERO-MEAN" in str(e.value)
    # …and it PASSES once a zero-mean control is added
    ok = only_signed + [{"axis": "fake", "control": "lat_jitter", "unit": "m",
                         "two_sided_ladder": False, "zero_mean_control": True,
                         "separates_up": True, "separates_down": None,
                         "monotone_up": True, "monotone_down": None,
                         "mde_up": 0.25, "mde_down": None}]
    assert C.admit(ok)["admissible"] is True


def test_admit_REFUSES_a_non_monotone_response():
    bad = [{"axis": "fake", "control": "lat_jitter", "unit": "m",
            "two_sided_ladder": False, "zero_mean_control": True,
            "separates_up": True, "separates_down": None,
            "monotone_up": False, "monotone_down": None,
            "mde_up": 0.25, "mde_down": None}]
    with pytest.raises(C.AxisNotDemonstrated) as e:
        C.admit(bad)
    assert "monotone" in str(e.value)


# =========================================================================== #
# 6. the gate and the composite
# =========================================================================== #
def _live_and_dead():
    """Two arms with REAL range on ``lat_track``, plus one saturated arm."""
    a = C.axes(C.apply_control(_straight(jitter=1.2, seed=1), "lat_jitter", 0.8))
    b = C.axes(C.apply_control(_straight(jitter=1.2, seed=2), "lat_jitter", 1.2))
    dead = dict(a)
    dead["lat_track"] = np.full_like(np.asarray(a["lat_track"], float), 1.0)
    return a, b, dead


def test_panel_gate_is_panel_wide_not_per_arm():
    """One arm with a dead axis must drop that axis for EVERY arm."""
    a, b, dead = _live_and_dead()
    ok = C.panel_gate({"a": a, "b": b}, names=("lat_track",))
    assert "lat_track" in ok["admitted"], ok["dropped"]
    g = C.panel_gate({"a": a, "b": b, "c": dead}, names=("lat_track",))
    assert "lat_track" in g["dropped"]
    assert g["gate"] == "PANEL-WIDE"


def test_panel_gate_excludes_probes_from_voting():
    a, b, dead = _live_and_dead()
    g = C.panel_gate({"a": a, "b": b, "stand_still": dead},
                     probes=("stand_still",), names=("lat_track",))
    assert "lat_track" in g["admitted"], g["dropped"]


def test_control_score_REFUSES_an_axis_with_no_demonstrated_range():
    a = C.axes(_straight())
    admitted = {k: True for k in C.CONTROL_WEIGHTS}
    s = C.control_score(a, admitted, demonstrated={"lon_track"})
    assert "lat_track" in s["weights_dropped"]
    assert "lon_track" in s["weights_admitted"]


def test_control_score_refuses_entirely_when_nothing_is_admitted():
    a = C.axes(_straight())
    with pytest.raises(PS.VacuousMetric):
        C.control_score(a, {k: False for k in C.CONTROL_WEIGHTS},
                        weights={"lat_track": 2.0})


# =========================================================================== #
# 7. ⛔ the two refusals, and the pseudosim changes that carry them
# =========================================================================== #
def test_comfort_carries_ZERO_weight_and_the_published_vector_is_frozen():
    assert PS.COMPONENT_WEIGHTS["comfort"] == 0.0
    assert PS.COMPONENT_WEIGHTS_PUBLISHED_V1 == {"ego_progress": 5.0,
                                                 "recovery": 5.0,
                                                 "comfort": 2.0}
    assert PS.WEIGHTS_ID == "w_ep5_rec5_comfort0"


def test_zeroing_comfort_is_a_PROVABLE_no_op_on_the_composite():
    """⛔ FAILING VALUE: any difference between the composite computed with
    ``comfort`` at weight 2.0 and at weight 0.0 when ``comfort`` is admissible.

    This is what licenses the weight change: it removes a false CLAIM (that the
    composite is three-term) and changes no number."""
    n = 400
    rng = np.random.default_rng(0)
    sc = {"ego_progress": rng.uniform(0, 1, n),
          "recovery": rng.uniform(0, 1, n),
          # deliberately given real range, so it WOULD be admitted
          "comfort": rng.uniform(0, 1, n)}
    rng_ = PS.discriminative_range(sc)
    assert rng_["comfort"]["admissible"] is True
    with_zero = PS.composite(sc, rng_)["value"]
    only_two = PS.composite(sc, rng_, weights={"ego_progress": 5.0,
                                               "recovery": 5.0})["value"]
    assert np.nanmax(np.abs(with_zero - only_two)) == 0.0
    assert PS.composite(sc, rng_)["n_weighted_terms"] == 2
    assert "comfort" in PS.composite(sc, rng_)["components_zero_weighted"]


def test_the_range_gate_now_refuses_a_FLOOR_saturated_component():
    """⛔ THE MISSING HALF OF THE GATE. ``floor_frac`` was computed and never
    used, so a component pinned at its floor was admissible while the identical
    component pinned at its ceiling was refused.

    FAILING VALUE: a component that is 0.0 on 99 % of rows being admissible."""
    n = 1000
    floored = np.zeros(n)
    floored[:10] = np.linspace(0.2, 1.0, 10)      # real range, but 99 % at 0
    r = PS.discriminative_range({"x": floored})
    assert r["x"]["floor_frac_le_0p001"] >= 0.95
    assert r["x"]["observed_range"] >= PS.RANGE_MIN     # range alone passes it
    assert r["x"]["admissible"] is False
    assert r["x"]["reason"] == "SATURATED at the floor"
    # …and the ceiling clause still fires, so the gate is symmetric
    r2 = PS.discriminative_range({"x": 1.0 - floored})
    assert r2["x"]["reason"] == "SATURATED at the ceiling"


def test_missing_gates_are_stated_and_never_faked():
    assert C.MISSING_GATES["no_collision"] == PS.COLLISION_UNAVAILABLE_REASON
    assert "IMPOSSIBLE" in C.MISSING_GATES["no_drivable_area"]
    a = C.axes(_straight())
    admitted = {k: True for k in C.CONTROL_WEIGHTS}
    s = C.control_score(a, admitted)
    assert "not a Driving Score" in s["_not_a_driving_score"] \
        or "NOTHING computed" in s["_not_a_driving_score"]


# =========================================================================== #
# 8. provenance
# =========================================================================== #
def test_the_suite_id_carries_every_constant_that_can_move_a_number():
    assert C.SUITE_ID(1.0, 1.75, 0.2, 10.0) != C.SUITE_ID(2.0, 1.75, 0.2, 10.0)
    assert C.SUITE_ID(1.0, 1.75, 0.2, 10.0) != C.SUITE_ID(1.0, 3.5, 0.2, 10.0)
    assert C.SUITE_ID(1.0, 1.75, 0.2, 10.0) != C.SUITE_ID(1.0, 1.75, 0.4, 10.0)
    assert C.SUITE_ID(1.0, 1.75, 0.2, 10.0) != C.SUITE_ID(1.0, 1.75, 0.2, 20.0)


def test_unknown_control_raises_and_never_falls_back():
    pw = _straight(n=20)
    with pytest.raises(KeyError):
        C.apply_control(pw, "lat_shif", 1.0)          # a typo
    with pytest.raises(KeyError):
        C.ladder_levels("nope")


def test_every_two_sided_ladder_straddles_its_null():
    """⛔ A ladder validated on one side is the ``clamp_v1`` failure repeated."""
    for name, spec in C.LADDERS.items():
        if not spec["two_sided"]:
            continue
        lv = [float(x) for x in spec["levels"]]
        null = float(spec["null"])
        assert any(x < null for x in lv), name
        assert any(x > null for x in lv), name
        assert null in lv, name


def test_block_emits_its_estimator_and_refuses_the_deprecated_one():
    b = C.block(_straight(n=120), arm="synthetic", n_boot=50)
    assert "episode_cluster_bootstrap" in b["estimator"]
    assert "overlapping_holdout_se" in b["refused_estimator"]
    assert b["traffic_mode"] == PS.TRAFFIC_MODE_LOG_REPLAY
    assert "no_collision" in b["missing_gates"]
