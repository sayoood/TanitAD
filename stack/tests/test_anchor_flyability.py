"""Pin the exact Kamm instrument, and prove it catches what the old one missed.

The controls here are the shipped refcv4b geometry: a 13 x 9 product grid of
(a_lon, a_lat) with a_lat in [-3, +3] m/s^2 at 0.75 steps, rolled 60 steps at
dt = 0.1 s (6 s) with kappa_cap 0.12 and an alat speed floor of 4.0 m/s.
"""

import math

import pytest
import torch

from tanitad.instruments.flyability import (
    G, derive_kappa, friction_load, kamm_report, slot_gradient_load_DEPRECATED)

STEPS, DT = 60, 0.1
KAPPA_CAP, V_FLOOR = 0.12, 4.0
A_LAT = torch.arange(-3.0, 3.01, 0.75)                      # 9 values
A_LON = torch.cat([torch.tensor([-4.0]),
                   torch.arange(-6, 6) * (7.0 / 12.0)])     # 13, contains 0


def shipped_controls() -> torch.Tensor:
    """[117, 2] = (a_lon, a_lat), the shipped product grid."""
    lon, lat = torch.meshgrid(A_LON, A_LAT, indexing="ij")
    return torch.stack([lon.reshape(-1), lat.reshape(-1)], dim=-1)


def test_grid_shape_and_zero_action_present():
    """The no-action control must exist EXACTLY, not approximately.

    Every trivial floor in the programme is defined against it, and a grid
    that merely passes NEAR zero silently deletes the control. MEASURED
    2026-09-04: ``np.linspace(-4, 3, 13)`` has step 7/12 and does NOT contain
    zero, while a report asserted that every grid did.
    """
    c = shipped_controls()
    assert c.shape == (117, 2)
    assert (A_LAT.abs() < 1e-12).any(), "a_lat grid must contain 0 exactly"
    assert (A_LON.abs() < 1e-12).any(), "a_lon grid must contain 0 exactly"
    exact = ((c[:, 0].abs() < 1e-12) & (c[:, 1].abs() < 1e-12)).sum()
    assert int(exact) == 1, "exactly one straight-ahead {a=0, kappa=0} control"


def test_constant_speed_is_exactly_v_squared_kappa():
    """a_lon = 0 => v is constant => a_lat is EXACTLY v^2 * kappa, all steps."""
    v0 = torch.tensor([20.0])
    ctrl = torch.tensor([[0.0, 3.0]])                 # a_lon 0, a_lat 3 m/s^2
    kap = derive_kappa(ctrl, v0, units="alat", kappa_cap=KAPPA_CAP,
                       alat_v_floor=V_FLOOR)
    load = friction_load(ctrl, v0, units="alat", steps=STEPS, dt=DT,
                         kappa_cap=KAPPA_CAP, alat_v_floor=V_FLOOR)
    want = float(v0) ** 2 * float(kap)
    assert torch.allclose(load["a_lat"], torch.full_like(load["a_lat"], want),
                          atol=1e-4)
    assert load["a_lon"].abs().max() < 1e-5
    # and the requested lateral acceleration is recovered, because kappa was
    # derived from it at this very speed and 3.0 is under the clamp.
    assert want == pytest.approx(3.0, abs=1e-4)


def test_straight_ahead_control_carries_no_load():
    load = friction_load(torch.tensor([[0.0, 0.0]]), torch.tensor([27.0]),
                         units="alat", steps=STEPS, dt=DT)
    assert float(load["a_tot"].abs().max()) == 0.0
    assert float(load["peak_g"].max()) == 0.0


def test_stopped_vehicle_carries_no_lateral_load():
    """yaw_rate = v * kappa, so v = 0 turns nothing -- and a_lat must follow."""
    load = friction_load(torch.tensor([[0.0, 3.0]]), torch.tensor([0.0]),
                         units="alat", steps=STEPS, dt=DT)
    assert float(load["a_lat"].abs().max()) == 0.0


def test_realised_decel_is_not_the_commanded_one_past_a_stop():
    """The integrator clamps v at 0, so a closed form over-reports braking."""
    v0 = torch.tensor([5.0])
    load = friction_load(torch.tensor([[-4.0, 0.0]]), v0, units="alat",
                         steps=STEPS, dt=DT)
    a_lon = load["a_lon"][0, 0]
    assert float(a_lon.min()) == pytest.approx(-4.0, abs=1e-4)
    assert float(a_lon[-1]) == pytest.approx(0.0, abs=1e-6), \
        "after the stop the commanded decel is no longer realised"
    assert float(load["v_start"][0, 0, -1]) == 0.0


def test_lateral_load_grows_along_the_path_when_accelerating():
    """kappa is held constant while v rises, so a_lat GROWS -- the fact the
    slot-gradient instrument could not see."""
    load = friction_load(torch.tensor([[2.0, 3.0]]), torch.tensor([10.0]),
                         units="alat", steps=STEPS, dt=DT)
    a_lat = load["a_lat"][0, 0].abs()
    assert float(a_lat[-1]) > float(a_lat[0]) * 2.0


def test_units_change_the_answer_by_orders_of_magnitude():
    """The same bytes under the wrong units are not slightly wrong."""
    ctrl = shipped_controls()
    v0 = torch.tensor([36.0])
    alat = kamm_report(ctrl, v0, units="alat", steps=STEPS, dt=DT)
    kappa = kamm_report(ctrl, v0, units="kappa", steps=STEPS, dt=DT)
    assert alat["peak_g_overall"] < 1.0
    assert kappa["peak_g_overall"] > 100.0
    assert kappa["peak_g_overall"] > 100.0 * alat["peak_g_overall"]


def test_shipped_vocabulary_breaks_the_friction_circle_at_low_speed():
    """⛔ A REAL DEFECT IN THE SHIPPED refcv4b VOCABULARY, pinned here.

    MEASURED 2026-09-04 by this instrument. The load is WORST AT LOW SPEED and
    falls monotonically above 4 m/s -- the opposite of the intuition, and the
    reason every previous gate missed it: they sampled v0 in {10, 18, 27, 36}
    and never looked below 10.

        v0 (m/s)     0.5    2.0    4.0    8.0   10.0   18.0   27.0   36.0
        peak (g)    3.85   4.52   5.51   3.05   2.28   1.21   0.87   0.73
        over mu=0.7   22     30     30     24     18     10      4      2

    THE MECHANISM, and it is not a rounding effect. ``kappa`` is derived ONCE
    from v0 and then HELD CONSTANT over the roll. At v0 = 4 m/s a 3 m/s^2
    lateral request needs kappa = 3/16 = 0.1875, which the 0.12 cap binds. The
    same anchor then accelerates at +2.9167 m/s^2 for 6 s, reaching 21.2 m/s,
    and ``a_lat = v(t)^2 * kappa`` carries that capped curvature up to
    54.0 m/s^2 = 5.51 g. The cap bounds the CURVATURE, which is not the same
    thing as bounding the LOAD.

    ⚠️ 32 of 117 anchors break mu = 0.7 somewhere in the speed range; only 18
    do so at v0 >= 10. This is a candidate-pool defect, not a selection defect
    -- the model still chooses -- but a fan containing paths no vehicle can
    drive is the exact complaint that condemned refcv3's synthetic pool, and
    nothing prunes them here (``--sel-accel-max 2.0`` is MEASURED inert:
    0.00 % killed, 117.0 survivors/window).

    ⇒ The fix belongs in the next build, not in a restart: clamp kappa against
    the REALISED speed along the path (require ``max_t v(t)^2 * kappa`` under a
    design load), not against v0 alone.
    """
    ctrl = shipped_controls()
    rep_low = kamm_report(ctrl, torch.tensor([4.0]), units="alat",
                          steps=STEPS, dt=DT)
    rep_high = kamm_report(ctrl, torch.tensor([27.0]), units="alat",
                           steps=STEPS, dt=DT)
    assert rep_low["n_anchors"] == 117
    assert rep_low["peak_g_overall"] == pytest.approx(5.51, abs=0.02)
    assert rep_low["per_speed"][0]["over_mu_0.7"] == 30
    # and it really is worse low than high -- the counter-intuitive direction
    assert rep_low["peak_g_overall"] > 5.0 * rep_high["peak_g_overall"]
    assert rep_high["per_speed"][0]["over_mu_0.7"] == 4


def test_agrees_with_the_independent_turn_coverage_measurement():
    """Cross-check: a sibling agent measured 18/117 over mu=0.7 with peak
    2.28 g at v0 = 10 m/s by an independent route. This instrument must
    reproduce that, or one of the two is wrong."""
    rep = kamm_report(shipped_controls(), torch.tensor([10.0]), units="alat",
                      steps=STEPS, dt=DT)
    assert rep["per_speed"][0]["over_mu_0.7"] == 18
    assert rep["per_speed"][0]["peak_g"] == pytest.approx(2.28, abs=0.02)


def test_slot_gradient_arm_under_reports_the_peak():
    """⛔ THE DELIBERATE-REGRESSION ARM.

    A gate that has never been shown to FAIL a known-bad input has not been
    tested. Here the known-bad input is the OLD INSTRUMENT: differencing the
    eight slot waypoints twice must come out materially BELOW the exact
    per-step load, which is the defect that let an under-report pass as a gate.
    """
    ctrl = shipped_controls()
    v0 = torch.tensor([10.09])
    exact = friction_load(ctrl, v0, units="alat", steps=STEPS, dt=DT)
    peak_exact = exact["peak_g"][0]

    kap = derive_kappa(ctrl, v0, units="alat", kappa_cap=KAPPA_CAP,
                       alat_v_floor=V_FLOOR)
    seq = torch.stack([ctrl[:, 0], kap[0]], dim=-1)[:, None, :]
    seq = seq.expand(ctrl.shape[0], STEPS, 2)
    st0 = torch.zeros(ctrl.shape[0], 4)
    st0[:, 3] = float(v0)
    from tanitad.models.kinematic import rollout_unicycle
    path = rollout_unicycle(st0, seq, dt=DT)[..., :2]

    # the LIVE run's own slots: config.json horizons = [5,10,15,20,30,40,50,60]
    # step numbers, i.e. 0.5-1.0 s apart against a 0.1 s integration step --
    # which is the whole reason the second difference smooths the peak away.
    horizons = torch.tensor([5, 10, 15, 20, 30, 40, 50, 60])
    slots = horizons - 1
    slot_t = horizons.to(torch.float64) * DT
    peak_slot = slot_gradient_load_DEPRECATED(path[:, slots], slot_t)

    assert float(peak_slot.max()) < float(peak_exact.max()), \
        "the slot-gradient instrument must under-report the peak"
    ratio = float(peak_exact.max()) / max(float(peak_slot.max()), 1e-9)
    assert ratio > 1.15, f"under-report ratio {ratio:.2f} is smaller than the " \
                         "1.21-1.85x measured on this vocabulary"


def test_report_is_per_speed_not_pooled():
    """A v0-conditioned vocabulary's load IS speed-dependent; a single pooled
    count is uninterpretable and must not be the only thing reported."""
    rep = kamm_report(shipped_controls(), torch.tensor([4.0, 36.0]),
                      units="alat", steps=STEPS, dt=DT)
    assert len(rep["per_speed"]) == 2
    assert rep["per_speed"][0]["v0_ms"] == 4.0
    assert all("over_mu_0.7" in r for r in rep["per_speed"])


def test_derive_kappa_rejects_unknown_units():
    with pytest.raises(ValueError, match="units must be"):
        derive_kappa(shipped_controls(), torch.tensor([10.0]), units="guess",
                     kappa_cap=KAPPA_CAP, alat_v_floor=V_FLOOR)


def test_kappa_cap_binds_at_low_speed():
    """At the speed floor a 3 m/s^2 request needs kappa = 3/16 = 0.1875, above
    the 0.12 cap -- so the cap, not the request, sets the geometry."""
    kap = derive_kappa(torch.tensor([[0.0, 3.0]]), torch.tensor([2.0]),
                       units="alat", kappa_cap=KAPPA_CAP, alat_v_floor=V_FLOOR)
    assert float(kap) == pytest.approx(KAPPA_CAP, abs=1e-7)
    assert 3.0 / (V_FLOOR ** 2) > KAPPA_CAP


def test_g_constant_is_standard_gravity():
    assert G == pytest.approx(9.81, abs=1e-9)
    assert math.isclose(G, 9.81)
