"""THE THIRD ONE-SIDED CLAMP (``lat_heading``) AND THE GUARD THAT MISSED IT.

⛔ Every rule below is driven with **the value that makes it FAIL**, not only
the value that makes it pass. Three bounded terms in a row were audited on one
side of a two-sided object; a test suite that only exercises the fixed direction
would be the fourth.

The two defects pinned here:

1. ``lat_heading = clamp(1 - |dpsi_end| / PSI_TOL, 0, 1)`` is a **single value
   per row**, so its floor bites at ROW level: MEASURED floored on
   **31.22-84.29 %** of defined rows with the ceiling never active
   (0.0001-0.0023). ``lon_track`` and ``lat_track`` — the two bounded axes that
   PASSED the same audit — are MEANS over 20 steps.
2. ⛔⛔ ``FLOOR_FRAC_MAX = 0.95`` refuses none of it, for a reason that
   generalises: **it tests each end separately**, so a term 49 % floored AND
   49 % ceilinged clears ``FLOOR_FRAC_MAX`` *and* ``CEIL_FRAC_MAX`` while having
   gradient on 2 % of its rows. That is the case ``test_the_guard_ALSO_misses_a
   _term_that_splits_its_saturation`` drives, and it is the "next one" the brief
   said would slip through for the same reason.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from taniteval import control as C
from taniteval import pseudosim as PS

U = np.linspace(0.0, 16.0, 1601)


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _pw(n=64, Hh=20, v0=10.0, curve=0.0, drift=0.0):
    """A minimal ``pseudo_evaluate``-shaped record. No model, no corpus.

    Rows differ in heading error so ``lat_heading`` has real range; the human
    path may curve so the arc-matched reference is genuinely exercised."""
    t = torch.arange(1, Hh + 1, dtype=torch.float64) * PS.DT
    ref = torch.zeros(n, Hh + 1, 2, dtype=torch.float64)
    s = torch.cat([torch.zeros(1, dtype=torch.float64), v0 * t])
    ref[:, :, 0] = s[None].expand(n, -1)
    ref[:, :, 1] = (curve * s ** 2)[None].expand(n, -1)
    # each row aims a different angle off the reference tangent
    ang = torch.linspace(-0.6, 0.6, n, dtype=torch.float64)[:, None]
    x = torch.cos(ang) * (v0 * t)[None]
    y = torch.sin(ang) * (v0 * t)[None] + drift
    return {"traj": torch.stack([x, y], -1).float().contiguous(),
            "ref_path": ref.float(), "ref_yaw": torch.zeros(n),
            "v0": torch.full((n,), v0), "pt_dyaw": torch.zeros(n),
            "pt_dlat": torch.zeros(n), "pt_dlon": torch.zeros(n),
            "eid": [str(i % 8) for i in range(n)]}


# =========================================================================== #
# 1. THE PUBLISHED TERM — driven with the value that makes it FAIL             #
# =========================================================================== #
def test_the_PUBLISHED_lat_heading_cannot_tell_at_tolerance_from_backwards():
    """⛔ THE DEFECT, as a number.

    A plan 11.5 deg off (exactly at PSI_TOL) and a plan pointing **exactly
    backwards** (u = pi/0.2 = 15.71, the structural maximum of a wrapped angle)
    are scored IDENTICALLY at 0. MEASURED: 46.3 % of the panel's pooled rows sit
    in that zero-gradient half, and 84.29 % of ``v4_blind``'s."""
    g = C.LAT_HEADING_SHAPES["lin_q0"]
    assert list(g(np.array([1.0, 2.0, 6.07, 15.708]))) == [0.0, 0.0, 0.0, 0.0]
    # the shipped shape at the same points is strictly ordered until ITS floor
    fixed = C.LAT_HEADING_SHAPES[C.LAT_HEADING_TERMS[
        C.LAT_HEADING_TERM_DEFAULT_TARGET][1]]
    v = fixed(np.array([0.0, 0.5, 1.0]))
    assert v[0] > v[1] > v[2]


def test_a_term_at_its_floor_CANNOT_BE_CHARGED_MORE_and_that_is_the_mechanism():
    """⛔ Zero gradient on the floored half is the whole failure mode, and the
    residual is asserted rather than glossed: the shipped term still floors."""
    eps = 1e-7
    g = C.LAT_HEADING_SHAPES["lin_q0"]
    for u in (1.5, 6.0, 15.0):
        assert float((g(np.array([u])) - g(np.array([u + eps])))[0]) == 0.0
    # and the linear family's floor moves exactly to 1/(1-q), never to infinity
    for q, floor_at in ((0.5, 2.0), (2.0 / 3.0, 3.0), (0.9, 10.0)):
        gq = C.LAT_HEADING_SHAPES[f"lin_q{f'{q:.4g}'.replace('.', 'p')}"]
        assert float(gq(np.array([floor_at - 1e-6]))[0]) > 0.0
        assert float(gq(np.array([floor_at + 1e-6]))[0]) == 0.0


def test_the_published_lat_heading_expression_is_BIT_IDENTICAL_under_the_term():
    """✅ The reproduction half. Not ``approx`` — BIT-identical, over a dense
    grid: a 1e-7 drift on 40 % of rows would creep into every paired delta
    between the two terms."""
    pw = _pw(curve=0.002)
    r = C.residuals(pw)
    old = np.clip(1.0 - np.abs(r["heading_err_rad"]) / C.PSI_TOL_RAD, 0.0, 1.0)
    new = C.lat_heading_from_err(r["heading_err_rad"],
                                 r["heading_err_rad_steps"],
                                 lat_heading_term="term_lin_q0")
    m = np.isfinite(old) & np.isfinite(new)
    assert m.sum() > 0
    assert bool((old[m] == new[m]).all())


def test_the_published_axis_id_and_suite_id_DO_NOT_MOVE():
    """⚠️ Every pin, log line and report published through 2026-07-28 must keep
    resolving; a NON-published term must always be visible in the name."""
    assert C.lat_heading_axis_id("term_lin_q0") == "lat_heading"
    assert C.lat_heading_axis_id(C.LAT_HEADING_TERM_PUBLISHED) == "lat_heading"
    assert C.SUITE_ID(lat_heading_term="term_lin_q0") == C.SUITE_ID(
        lat_heading_term=C.LAT_HEADING_TERM_PUBLISHED)
    assert "+lath_" not in C.SUITE_ID(lat_heading_term="term_lin_q0")
    for t in ("mean_lin_q0", "term_lin_q0p5", C.LAT_HEADING_TERM_DEFAULT):
        assert C.lat_heading_axis_id(t) == f"lat_heading@{t}"
        assert C.SUITE_ID(lat_heading_term=t).endswith(f"+lath_{t}")


def test_an_unknown_lat_heading_term_RAISES_and_never_falls_back():
    with pytest.raises(C.UnknownLatHeadingTerm):
        C.lat_heading_from_err(np.zeros(3), np.zeros((3, 4)),
                               lat_heading_term="mean_lin_q05")
    with pytest.raises(C.UnknownLatHeadingTerm):
        C.axes(_pw(), lat_heading_term="nope")


def test_the_shipped_default_is_an_ALIAS_onto_a_swept_member_and_it_is_PINNED():
    """⭐ So the shape can never drift without this test failing."""
    assert C.LAT_HEADING_TERM_DEFAULT_TARGET in C.LAT_HEADING_TERMS
    assert (C.LAT_HEADING_TERM_ALIASES[C.LAT_HEADING_TERM_DEFAULT]
            == C.LAT_HEADING_TERM_DEFAULT_TARGET)
    assert C.LAT_HEADING_TERM_DEFAULT_TARGET == "mean1_lin_q0p5"
    res, shape = C.LAT_HEADING_TERMS[C.LAT_HEADING_TERM_DEFAULT_TARGET]
    assert res == "mean1" and shape == "lin_q0p5"


def test_the_ZERO_PARAMETER_candidate_was_REJECTED_and_that_is_recorded():
    """⛔ THE SELF-REFUTATION, pinned so it cannot quietly disappear.

    ``mean1_lin_q0`` — resolution change only, published shape verbatim, NO
    free parameter — is what H3 prefers and what I expected to ship. MEASURED
    at B = 2000 it scored **9/10**: correct in sign on every cell and not
    SEPARATED on one, so H1 disqualified it. The two levers do different jobs —
    the resolution fixes the saturation, the shape fixes the power — and a
    term that changed only the resolution would have been a half-repair."""
    assert "mean1_lin_q0" in C.LAT_HEADING_TERMS
    assert C.LAT_HEADING_TERM_DEFAULT_TARGET != "mean1_lin_q0"
    # and the reason is arithmetic: q = 0 charges half as fast as q = 0.5
    eps = 1e-4

    def rate(name):
        g = C.LAT_HEADING_SHAPES[name]
        return float((g(np.array([0.5])) - g(np.array([0.5 + eps])))[0]) / eps
    assert rate("lin_q0p5") == pytest.approx(rate("lin_q0") / 2.0, rel=1e-3)


def test_the_parameter_that_FAILED_on_recovery_is_the_one_that_WINS_here():
    """⭐⭐ C47 from the other direction, as an assertion.

    ``q = 0.5`` scored 7/8 and was DISQUALIFIED on ``recovery``; ``q = 2/3``
    won there. Here ``q = 0.5`` passes and is selected. Both terms use the
    identical one-parameter budget family, so nothing but the DENSITY differs —
    which is exactly why an inherited constant is a hypothesis."""
    assert PS.RECOVERY_TERM_DEFAULT_TARGET == "lin_q0p6667"
    assert C.LAT_HEADING_TERM_DEFAULT_TARGET.endswith("lin_q0p5")
    # the two families are the same algebra on their own normalised quantity
    u = np.linspace(0.0, 4.0, 401)
    lath = C.LAT_HEADING_SHAPES["lin_q0p5"](u)
    rec = PS.recovery_from_ratio(torch.as_tensor(u, dtype=torch.float64),
                                 "lin_q0p5").numpy()
    assert np.allclose(lath, rec, atol=1e-12)


# =========================================================================== #
# 2. THE RESOLUTION LEVER — and the claim it would have destroyed              #
# =========================================================================== #
def test_the_MEAN_resolution_needs_all_steps_clamped_before_a_row_saturates():
    """⭐ THE STRUCTURAL FIX, reduced to arithmetic. One bad step out of 20
    floors a per-row term and moves a mean by 1/20."""
    steps = np.zeros((1, 20))
    steps[0, 7] = 1.0                       # one step 57 deg off
    end = np.array([1.0])                   # terminal step also 57 deg off
    per_row = C.lat_heading_from_err(end, steps, lat_heading_term="term_lin_q0")
    meaned = C.lat_heading_from_err(end, steps, lat_heading_term="mean_lin_q0")
    assert float(per_row[0]) == 0.0                     # floored, no gradient
    assert float(meaned[0]) == pytest.approx(19.0 / 20.0)


def test_mean1_is_TRANSLATION_INVARIANT_where_mean_is_NOT():
    """⛔⛔ THE VALUE THAT REFUTED THE OBVIOUS RESOLUTION FIX.

    ``lat_heading`` exists as a SECOND lateral axis only because a pure lateral
    OFFSET moves ``lat_track`` and leaves it unchanged. Step 0's plan tangent
    runs from the REFERENCE POSE to waypoint 0, so translating the whole plan
    rotates that one segment enormously — MEASURED on the panel, ``mean`` moved
    **-0.0222 SEPARATED** under ``lat_shift(+2 m)``. Every waypoint-to-waypoint
    segment is translation-invariant by construction, which is what ``mean1``
    keeps."""
    pw = _pw()
    shifted = C.apply_control(pw, "lat_shift", 2.0)
    a, b = C.residuals(pw), C.residuals(shifted)
    d0 = np.abs(a["heading_err_rad_steps"][:, 0]
                - b["heading_err_rad_steps"][:, 0])
    dk = np.abs(a["heading_err_rad_steps"][:, 1:]
                - b["heading_err_rad_steps"][:, 1:])
    # step 0 moves a LOT; the plan-internal segments do not move at all.
    # ⚠️ 1e-5 rather than 0: the plan tangents are EXACTLY invariant, but the
    # arc-matched REFERENCE segment `signed_xte` picks can change by one index
    # on a curving path, so a float-level residual survives. It is five orders
    # of magnitude below step 0's, which is the whole claim.
    assert np.nanmax(d0) > 0.3
    assert np.nanmax(dk) < 1e-5
    # ⇒ the axis follows: mean1 is unchanged, mean is not
    for term, invariant in (("mean1_lin_q0", True), ("mean_lin_q0", False)):
        x = C.lat_heading_from_err(a["heading_err_rad"],
                                   a["heading_err_rad_steps"],
                                   lat_heading_term=term)
        y = C.lat_heading_from_err(b["heading_err_rad"],
                                   b["heading_err_rad_steps"],
                                   lat_heading_term=term)
        moved = abs(float(np.nanmean(x) - np.nanmean(y)))
        assert (moved < 1e-5) is invariant, term


def test_the_mean_ignores_NaN_steps_and_refuses_a_row_with_no_heading_at_all():
    """⚠️ A step with no segment length has no heading. Scoring it 1.0 would
    repeat the ``recovery`` defect verbatim (*standing still is not aim*)."""
    steps = np.full((2, 20), np.nan)
    steps[0, :5] = 0.0
    v = C.lat_heading_from_err(np.array([0.0, np.nan]), steps,
                               lat_heading_term="mean_lin_q0")
    assert float(v[0]) == pytest.approx(1.0)
    assert np.isnan(v[1])


def test_the_row_set_is_IDENTICAL_across_every_lat_heading_term():
    """⛔ Otherwise no paired delta between two terms is valid."""
    pw = _pw(curve=0.002)
    ref = None
    for name in sorted(C.LAT_HEADING_TERMS):
        m = np.isfinite(np.asarray(C.axes(pw, lat_heading_term=name)[
            "lat_heading"], float))
        if ref is None:
            ref = m
        assert bool((m == ref).all()), f"{name} changed the row set"


# =========================================================================== #
# 3. THE SHAPE FAMILIES AND C47's DISCRIMINATOR                               #
# =========================================================================== #
def test_the_linear_family_is_AFFINE_in_the_published_shape_below_tolerance():
    """⭐ So no pair of in-tolerance rows changes order and the published value
    is exactly recoverable."""
    u = np.linspace(0.0, 1.0, 501)
    base = C.LAT_HEADING_SHAPES["lin_q0"](u)
    for q in (0.25, 0.5, 2.0 / 3.0, 0.75, 0.85, 0.9):
        g = C.LAT_HEADING_SHAPES[f"lin_q{f'{q:.4g}'.replace('.', 'p')}"](u)
        assert np.allclose(g, q + (1.0 - q) * base, atol=1e-12)


def test_the_share_family_NEVER_saturates_and_that_is_NOT_the_property():
    """⛔ C47, carried as a checkable rejection rather than a paragraph.

    ``share`` floors on nothing at all, and its charge rate still collapses:
    at u = 3 it charges **less than a sixth** of what the linear family does."""
    assert float(C.LAT_HEADING_SHAPES["share_q0p5"](np.array([1e6]))[0]) > 0.0
    eps = 1e-4

    def rate(name, u):
        g = C.LAT_HEADING_SHAPES[name]
        return float((g(np.array([u])) - g(np.array([u + eps])))[0]) / eps
    assert rate("share_q0p5", 3.0) < rate("lin_q0p5", 1.0) / 6.0
    # ⭐ THE DISCRIMINATOR: > 1 means the shape rewards polishing a good row
    # harder than it charges a typical one.
    bias = {n: rate(n, 0.05) / rate(n, 0.91)      # panel median u = 0.910
            for n in ("lin_q0p5", "share_q0p5", "cos_q0p5")}
    assert bias["lin_q0p5"] == pytest.approx(1.0, abs=1e-6)
    assert bias["share_q0p5"] > 1.0
    assert bias["cos_q0p5"] < 1.0        # the angle-native one charges harder


def test_the_cos_family_hits_its_anchor_and_its_cost_is_pinned_too():
    """⚠️ Its slope is ZERO at u = 0, so it cannot tell an excellent plan from a
    perfect one. That cost is asserted, not left in prose."""
    for q in (0.25, 0.5, 0.75):
        g = C.LAT_HEADING_SHAPES[f"cos_q{f'{q:.4g}'.replace('.', 'p')}"]
        assert float(g(np.array([1.0]))[0]) == pytest.approx(q, abs=1e-9)
        assert float(g(np.array([0.0]))[0]) == pytest.approx(1.0)
    g = C.LAT_HEADING_SHAPES["cos_q0p5"]
    assert float((g(np.array([0.0])) - g(np.array([0.01])))[0]) < 1e-4


def test_the_share_family_REFUSES_its_degenerate_anchor():
    with pytest.raises(ValueError):
        C._lath_share(0.0)
    assert "share_q0" not in C.LAT_HEADING_SHAPES


# =========================================================================== #
# 4. ⛔⛔ THE GUARD — driven with the values that make it FAIL                   #
# =========================================================================== #
def test_the_published_gate_ADMITS_a_term_floored_on_84_percent_of_rows():
    """⛔ THE DEFECT IN THE GUARD, as a number. This is `v4_blind`'s measured
    ``lat_heading`` shape, and ``FLOOR_FRAC_MAX = 0.95`` passes it."""
    x = np.concatenate([np.zeros(8429), np.linspace(0.01, 0.99, 1571)])
    r = PS.discriminative_range({"x": x}, gate_version="v1")["x"]
    assert r["floor_frac_le_0p001"] == pytest.approx(0.8429, abs=1e-4)
    assert r["admissible"] is True and r["admissible_v1"] is True
    # …and gate v2 refuses exactly that array
    assert r["admissible_v2"] is False
    r2 = PS.discriminative_range({"x": x}, gate_version="v2")["x"]
    assert r2["admissible"] is False
    assert "live_frac" in r2["reason"] or "GRADIENT" in r2["reason"]


def test_the_guard_ALSO_misses_a_term_that_SPLITS_its_saturation():
    """⛔⛔ THE "NEXT ONE", and it is why the fix is not a smaller constant.

    49 % floored plus 49 % ceilinged: gradient on **2 %** of rows, and it clears
    ``FLOOR_FRAC_MAX`` *and* ``CEIL_FRAC_MAX``, because the pair of one-sided
    thresholds is STRUCTURALLY UNABLE to express 'no gradient left'. Lowering
    either constant does not fix this case; the two-sided ``live_frac`` does."""
    x = np.concatenate([np.zeros(490), np.ones(490),
                        np.linspace(0.01, 0.99, 20)])
    r = PS.discriminative_range({"x": x}, gate_version="v1")["x"]
    assert r["floor_frac_le_0p001"] < PS.FLOOR_FRAC_MAX
    assert r["ceiling_frac_ge_0p999"] < PS.CEIL_FRAC_MAX
    assert r["observed_range"] >= PS.RANGE_MIN
    assert r["admissible"] is True, "gate v1 admits a 98 %-dead term"
    assert r["live_frac"] == pytest.approx(0.02, abs=1e-3)
    assert PS.discriminative_range({"x": x},
                                   gate_version="v2")["x"]["admissible"] is False


def test_gate_v2_is_STRICTLY_STRONGER_and_can_never_admit_more_than_v1():
    """⚠️ A 'fix' that makes a gate more permissive is not a fix. Property test
    over 400 random shapes, including degenerate ones."""
    rng = np.random.default_rng(20260728)
    for _ in range(400):
        f = rng.integers(0, 1000) / 1000.0
        c = rng.integers(0, int((1.0 - f) * 1000) + 1) / 1000.0
        mid = max(1, 1000 - int(f * 1000) - int(c * 1000))
        x = np.concatenate([np.zeros(int(f * 1000)), np.ones(int(c * 1000)),
                            rng.uniform(0.01, 0.99, mid)])
        d = PS.discriminative_range({"x": x}, gate_version="v1")["x"]
        assert not (d["admissible_v2"] and not d["admissible_v1"])


def test_the_published_gate_thresholds_are_NOT_LOWERED_and_here_is_why():
    """⛔ ``recovery@clamp_v1`` floors on 55.65-92.19 % of rows. Lowering
    ``FLOOR_FRAC_MAX`` to the warning line would make it inadmissible panel-wide
    and change every published PSS value — the silent redefinition being fixed.
    All four constants are asserted together so none can drift into another."""
    assert PS.FLOOR_FRAC_MAX == 0.95
    assert PS.CEIL_FRAC_MAX == 0.95
    assert PS.LIVE_FRAC_MIN == 0.50
    assert PS.SATURATION_WARN_FRAC == 0.50
    assert PS.GATE_VERSION_DEFAULT == "v1" == PS.GATE_VERSION_PUBLISHED
    # the value that makes it fail: recovery's own measured floor fraction
    x = np.concatenate([np.zeros(9219), np.linspace(0.01, 0.99, 781)])
    assert PS.discriminative_range({"x": x})["x"]["admissible"] is True


def test_an_unknown_gate_version_RAISES_and_never_falls_back():
    with pytest.raises(PS.UnknownGateVersion):
        PS.discriminative_range({"x": np.linspace(0, 1, 10)},
                                gate_version="v3")


def test_BOTH_gate_verdicts_are_always_emitted_whichever_version_gates():
    """⭐ A stronger reading that is invisible is a reading nobody adopts."""
    x = np.concatenate([np.zeros(700), np.linspace(0.01, 0.99, 300)])
    for gv in ("v1", "v2"):
        n = PS.discriminative_range({"x": x}, gate_version=gv)["x"]
        assert set(("admissible", "admissible_v1", "admissible_v2")) <= set(n)
        assert n["admissible_v1"] is True and n["admissible_v2"] is False
        assert n["admissible"] is (gv == "v1")


# =========================================================================== #
# 5. SATURATION REPORTING — resolution and the two-sided statistic             #
# =========================================================================== #
def test_saturation_publishes_live_frac_and_the_ROW_RESOLUTION():
    """⚠️ ``n_sub`` is the fact that EXPLAINS which terms trip the gate, and it
    was nowhere in the code while three terms were being audited."""
    x = np.concatenate([np.zeros(400), np.ones(400), np.linspace(.01, .99, 200)])
    n1 = PS.saturation(x, n_sub=1)
    n20 = PS.saturation(x, n_sub=20)
    assert n1["live_frac"] == pytest.approx(0.2, abs=1e-9)
    assert n1["saturated_frac"] == pytest.approx(0.8, abs=1e-9)
    assert "PER-ROW" in n1["row_resolution"]
    assert "MEAN over 20" in n20["row_resolution"]
    assert PS.saturation(x)["row_resolution"] == "UNDECLARED"


def test_the_warning_fires_on_COMBINED_saturation_not_only_on_one_end():
    """⛔ 40 % floor + 40 % ceiling warns nothing under a one-sided rule."""
    x = np.concatenate([np.zeros(400), np.ones(400), np.linspace(.01, .99, 200)])
    w = PS.saturation(x)["SATURATION_WARNING"]
    assert w and "TWO ENDS COMBINED" in w
    # and the one-sided cases still name their end, as before
    assert "FLOOR" in PS.saturation(np.concatenate(
        [np.zeros(800), np.linspace(.01, .99, 200)]))["SATURATION_WARNING"]
    assert "CEILING" in PS.saturation(np.concatenate(
        [np.ones(800), np.linspace(.01, .99, 200)]))["SATURATION_WARNING"]
    assert PS.saturation(np.linspace(0.05, 0.95, 500))[
        "SATURATION_WARNING"] is None


def test_comfort_is_100_percent_saturated_BY_CONSTRUCTION_and_range_is_a_lie():
    """⚠️ C46 from a third direction. ``observed_range = 1.0`` records only that
    both values OCCUR — it is 1.0 for ANY binary array containing one of each,
    whatever the mixture — so ``RANGE_MIN`` cannot see the difference between a
    balanced indicator and one that is 99.9 % constant.

    ⚠️ AND THE ONE-SIDED THRESHOLDS ONLY CATCH THE EXTREMES: at a 0.1 %/99.9 %
    mixture gate v1 does refuse it (one end exceeds 0.95), but everywhere in
    between — including the arms MEASURED here — it sails through with zero
    gradient on every single row."""
    for p in (0.001, 0.5, 0.999):
        x = (np.arange(1000) < int(p * 1000)).astype(float)
        n = PS.discriminative_range({"comfort": x})["comfort"]
        assert n["observed_range"] == 1.0
        assert n["live_frac"] == 0.0
        assert n["admissible_v2"] is False, "v2 refuses it on every mixture"
        # v1's verdict depends on the MIXTURE, not on the (zero) gradient
        assert n["admissible_v1"] is (0.05 <= p <= 0.95)
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        x = (np.arange(1000) < int(p * 1000)).astype(float)
        assert PS.discriminative_range({"comfort": x})["comfort"][
            "admissible_v1"] is True, "v1 cannot see a 100 %-saturated term"


def test_axis_n_sub_declares_the_resolution_of_every_bounded_axis():
    assert C.AXIS_N_SUB["lon_track"] == 20 and C.AXIS_N_SUB["lat_track"] == 20
    assert C.AXIS_N_SUB["lat_heading_terminal"] == 1
    assert C.AXIS_N_SUB["recovery"] == 1 and C.AXIS_N_SUB["ego_progress"] == 1


# =========================================================================== #
# 6. WIRING — the axis, the summary, the panel gate                            #
# =========================================================================== #
def test_axes_publishes_the_term_the_axis_id_and_the_terminal_twin():
    a = C.axes(_pw(curve=0.001))
    assert a["_lat_heading_term"] == C.LAT_HEADING_TERM_DEFAULT
    assert a["_lat_heading_axis_id"].startswith("lat_heading@")
    assert "lat_heading_terminal" in a
    pub = C.axes(_pw(curve=0.001), lat_heading_term="term_lin_q0")
    m = np.isfinite(pub["lat_heading"]) & np.isfinite(pub["lat_heading_terminal"])
    assert bool((pub["lat_heading"][m]
                 == pub["lat_heading_terminal"][m]).all())


def test_axis_summary_carries_saturation_with_the_terms_own_resolution():
    pw = _pw(curve=0.001)
    for term, want in (("term_lin_q0", "PER-ROW"), ("mean1_lin_q0", "MEAN")):
        a = C.axes(pw, lat_heading_term=term)
        s = C.axis_summary(a, a and pw["eid"], n_boot=20)["lat_heading"]
        assert want in s["saturation"]["row_resolution"]
        assert s["lat_heading_term"] == term
        assert s["saturation"]["live_frac"] is not None


def test_the_control_panel_gate_defaults_to_v2_and_reports_what_v1_would_say():
    """⭐ THE ONLY TEST OF A GUARD THAT COUNTS: it must refuse the broken term
    and admit the fixed one, on the same data."""
    by = {}
    for arm, drift in (("straight", 0.0), ("aimed_off", 0.0)):
        pw = _pw(curve=0.002 if arm == "aimed_off" else 0.0)
        by[arm] = C.axes(pw, lat_heading_term="term_lin_q0")
    g = C.panel_gate(by, names=("lat_heading",))
    assert g["gate_version"] == C.CONTROL_GATE_VERSION == "v2"
    assert "admitted_under_gate_v1" in g
    g1 = C.panel_gate(by, names=("lat_heading",), gate_version="v1")
    assert g1["gate_version"] == "v1"


# =========================================================================== #
# 7. ego_progress ABOVE r = 2 — the residual, pinned as arithmetic             #
# =========================================================================== #
def test_twosided_v2_floors_at_ratio_2_and_the_w_grid_moves_that_floor():
    """⚠️ The range-budget trade, as numbers. A bounded [0,1] score with a fixed
    under-side CANNOT both charge over-travel at rate 1 and keep charging past
    r = 2; a later floor is bought with a lower charge rate, never for free."""
    r = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    assert PS.progress_from_ratio(r, "twosided_v2").tolist() == [0, 0, 0, 0]
    w05 = PS.progress_from_ratio(r, "twosided_asym_w0p5")
    w33 = PS.progress_from_ratio(r, "twosided_asym_w0p3333")
    assert float(w05[0]) == pytest.approx(0.5) and float(w05[1]) == 0.0
    assert float(w33[0]) == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert float(w33[2]) == pytest.approx(0.0, abs=1e-9)
    # ⛔ and the cost: at r = 1.5 the later-flooring term charges LESS
    a = PS.progress_from_ratio(torch.tensor([1.5], dtype=torch.float64),
                               "twosided_v2")
    b = PS.progress_from_ratio(torch.tensor([1.5], dtype=torch.float64),
                               "twosided_asym_w0p3333")
    assert float(b) > float(a)


def test_the_new_w_anchor_does_not_touch_the_default_or_the_published_term():
    """✅ Additive only: a new sensitivity anchor may not move a published id."""
    assert PS.PROGRESS_TERM_DEFAULT == "twosided_v2"
    assert PS.OVER_TRAVEL_WEIGHT == 1.0
    assert PS.metric_id() == "PSS_recovery_progress@twosided_v2"
    assert PS.metric_id("clamp_v1", "clamp_v1") == "PSS_recovery_progress@clamp_v1"
    u = torch.linspace(0.0, 4.0, 4001, dtype=torch.float64)
    assert bool((PS.progress_from_ratio(u, "twosided_v2")
                 == PS.progress_from_ratio(u, "twosided_v2")).all())
