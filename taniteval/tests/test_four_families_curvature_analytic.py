"""ANALYTIC pins for ``four_families`` curvature — an ABSOLUTE anchor, not an invariance.

⛔ THE HOLE THIS CLOSES. ``test_four_families_dt.py`` pins curvature only by INVARIANCE
(``a["curvature"] == b["curvature"]`` under a dt change) and by RATIOS. A metric that is
uniformly wrong passes every one of those. Nothing in this repo ever compared a curvature
against a value known in closed form, which is exactly why a suspicion that curvature was
corrupted by the non-uniform v3 horizon set could not be settled by reading the tests.

⭐ A circular arc of radius R has curvature EXACTLY ``1/R`` everywhere, and a straight line
EXACTLY ``0``. Those are the anchors. Measured 2026-09-07 on the real module:

    v3 NON-UNIFORM horizon (5,10,15,20,30,40,50,60) x 0.1 s, R = 20 m : 1.0067 x (1/R)
    UNIFORM 0.5 s control,                            R = 20 m : 1.0026 x (1/R)
    dense 0.1 s grid,                                 R = 20 m : 1.0001 x (1/R)

⇒ the residual is CHORD DISCRETISATION (O(phi^2), present on the uniform grid too), NOT the
non-uniform spacing: ``_seq_geometry`` divides heading change by ARC LENGTH, never by ``dt``.

⛔⛔ AND THE DEFECT THAT IS REAL. Curvature is ``dh / (ds_mid + eps)``: on a STOPPED path
``ds_mid -> 0`` and the quantity is UNBOUNDED. ``pair_valid`` is what makes it safe, and it is
NOT optional. A local reimplementation in
``Research/2026-09-06-p4-p13-p14-validation/raw/p14_banked_fan.py`` dropped the mask and
published ``kappa MAE = 2.30973`` for refc-base-30k against ``0.02737`` for a straight-line
floor — the "84x worse on curvature than a plan that never steers" figure. Re-derived from
``taniteval/results/fan_refc-base-30k.pt`` (n = 881, grid [5,10,15,20] = 0.5 s UNIFORM): the
whole 84x comes from 43 windows (4.9 %) whose ego is STOPPED (``v0 <= 0.5 m/s``); on the 838
moving windows the ratio is 0.56, i.e. the model is BETTER than the floor. A straight-line
floor has ``kappa == 0`` EXACTLY at any speed, so it is structurally immune to the
singularity the model is exposed to — the contrast is not paired in the way it was read.
"""
import math

import pytest
import torch

from taniteval import four_families as FF

TICK = 0.1
V3_HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)   # refc_v3.py:121 — STEPS, 0.5 s then 1.0 s stride
V3_TIMES = [h * TICK for h in V3_HORIZONS]      # 0.5 1.0 1.5 2.0 3.0 4.0 5.0 6.0 s
UNIFORM_TIMES = [0.5 * i for i in range(1, 9)]  # the discriminating control


def _arc(R, v, times, n=3):
    """Circular arc from the ego origin, tangent +x, curving LEFT.
    x = R sin(vt/R), y = R(1 - cos(vt/R)).  TRUE CURVATURE = 1/R EXACTLY."""
    t = torch.tensor(times, dtype=torch.float64)
    th = v * t / R
    wp = torch.stack([R * torch.sin(th), R * (1 - torch.cos(th))], dim=-1)
    return wp.unsqueeze(0).expand(n, -1, -1).contiguous().float()


def _line(v, times, n=3):
    t = torch.tensor(times, dtype=torch.float64)
    wp = torch.stack([v * t, torch.zeros_like(t)], dim=-1)
    return wp.unsqueeze(0).expand(n, -1, -1).contiguous().float()


# ------------------------------------------------- 1. the absolute anchor: 1/R on a circle
@pytest.mark.parametrize("R", [20.0, 50.0, 200.0, 1000.0])
def test_curvature_recovers_one_over_R_on_the_v3_NON_UNIFORM_horizon(R):
    """⭐ THE TEST THE SUSPICION NEEDED. Feed the REAL v3 slot times — whose spacing changes
    from 0.5 s to 1.0 s at the operative/tactical seam — and demand 1/R back."""
    g = FF._seq_geometry(_arc(R, 10.0, V3_TIMES), 0.5)
    k = g["curvature"][g["pair_valid"]].mean().item()
    assert k == pytest.approx(1.0 / R, rel=0.01), (
        "curvature on the non-uniform v3 horizon must recover 1/R to 1 %%; got %.6f vs %.6f"
        % (k, 1.0 / R))


@pytest.mark.parametrize("R", [20.0, 50.0, 200.0])
def test_straight_line_has_exactly_zero_curvature_on_every_grid(R):
    """The null. A plan that never steers must read 0 — this is also WHY a straight-line
    floor is immune to the ds -> 0 singularity and cannot be compared naively against a model."""
    for times in (V3_TIMES, UNIFORM_TIMES):
        g = FF._seq_geometry(_line(10.0, times), 0.5)
        assert g["curvature"].abs().max().item() == 0.0


def test_non_uniform_spacing_is_not_the_error_source():
    """⭐ THE DISCRIMINATING CONTROL. Same arc, same estimator, UNIFORM grid. If the
    non-uniform set were the defect, this would recover 1/R and the v3 set would not.
    Both recover it; the small residual is chord discretisation and is present in BOTH."""
    R = 20.0                                   # tightest turn -> largest discretisation bias
    k_v3 = FF._seq_geometry(_arc(R, 10.0, V3_TIMES), 0.5)["curvature"].mean().item()
    k_uni = FF._seq_geometry(_arc(R, 10.0, UNIFORM_TIMES), 0.5)["curvature"].mean().item()
    assert k_v3 == pytest.approx(1.0 / R, rel=0.01)
    assert k_uni == pytest.approx(1.0 / R, rel=0.01)
    # the EXTRA error the non-uniformity buys is under half a percent
    assert abs(k_v3 - k_uni) / (1.0 / R) < 0.005


def test_curvature_does_not_depend_on_the_dt_argument_at_all():
    """``curvature = dh / ds_mid`` — arc length, never time. Pins the claim in the source
    comment, so a future edit that reintroduces a /dt is caught here and not in a briefing."""
    wp = _arc(50.0, 10.0, V3_TIMES)
    a = FF._seq_geometry(wp, 0.1)["curvature"]
    b = FF._seq_geometry(wp, 0.5)["curvature"]
    c = FF._seq_geometry(wp, 1.0)["curvature"]
    assert torch.equal(a, b) and torch.equal(b, c)


def test_yaw_rate_DOES_need_the_grid_and_says_so():
    """The contrast that keeps the two straight: yaw-rate IS a per-time rate and IS wrong on
    a mis-declared grid, while curvature is not. On a circle the true yaw rate is v/R."""
    R, v = 50.0, 10.0
    g = FF._seq_geometry(_arc(R, v, UNIFORM_TIMES), 0.5)     # correct grid
    assert g["yaw_rate"].mean().item() == pytest.approx(v / R, rel=0.01)
    bad = FF._seq_geometry(_arc(R, v, UNIFORM_TIMES), 0.1)   # mis-declared grid
    assert bad["yaw_rate"].mean().item() == pytest.approx(5.0 * v / R, rel=0.01)


# ------------------------------------------------- 2. the singularity guard is NOT optional
def test_curvature_is_unbounded_without_pair_valid_on_a_stopped_path():
    """⛔ THE REAL DEFECT, reproduced. A stopped ego whose plan wobbles by millimetres has
    ds -> 0, and dh/ds explodes. ``pair_valid`` is the only thing standing between that and a
    published number. This test FAILS if someone widens the gate or drops the mask."""
    stopped = torch.zeros(4, 8, 2)
    t = torch.arange(1, 9, dtype=torch.float32)
    stopped[:, :, 0] = 0.002 * t                      # 2 mm per slot: parked
    stopped[:, :, 1] = 0.001 * torch.sin(3.0 * t)     # 1 mm of lateral wobble
    g = FF._seq_geometry(stopped, 0.5)
    assert g["curvature"].abs().max().item() > 50.0, \
        "a millimetre-scale wobble at zero speed must produce an ABSURD curvature"
    assert not bool(g["pair_valid"].any()), \
        "pair_valid must exclude EVERY pair of a stopped path — it is the guard"
    assert g["min_ds_m"] == pytest.approx(0.25)       # 0.5 m/s x 0.5 s grid


def test_the_gate_can_fail_a_moving_path_is_kept():
    """⛔ A guard that never admits anything is not a guard. Same estimator, a MOVING path:
    every pair must survive, and the curvature must still be 1/R."""
    R = 50.0
    g = FF._seq_geometry(_arc(R, 10.0, V3_TIMES), 0.5)
    assert bool(g["pair_valid"].all())
    assert g["curvature"][g["pair_valid"]].mean().item() == pytest.approx(1.0 / R, rel=0.01)


def test_straight_floor_is_immune_to_the_singularity_at_any_speed():
    """⛔ WHY THE 84x COMPARISON WAS NOT A FAIR ONE. A straight-line floor reads EXACTLY 0
    however slowly it moves, so ``|k_floor - k_gt|`` is bounded by the GT's own curvature
    while ``|k_model - k_gt|`` is not. Any unmasked model-vs-straight-floor curvature contrast
    is decided by the slowest windows in the corpus, not by smoothness."""
    for v in (10.0, 1.0, 0.1, 0.01, 0.0):
        g = FF._seq_geometry(_line(v, V3_TIMES), 0.5)
        assert g["curvature"].abs().max().item() == 0.0


def test_menger_style_estimator_and_seq_geometry_agree_when_masked():
    """Cross-estimator agreement on the anchor: the parametric formula
    kappa = |x'y'' - y'x''| / (x'^2+y'^2)^{3/2} and ``dh/ds`` must land on the same 1/R.
    Two independent implementations agreeing on a closed-form target is the strongest
    available evidence that neither is silently mis-scaled."""
    import numpy as np
    R = 50.0
    times0 = np.concatenate([[0.0], np.array(V3_TIMES)])
    th = 10.0 * times0 / R
    p = np.stack([R * np.sin(th), R * (1 - np.cos(th))], -1)[None]
    d1 = np.gradient(p, times0, axis=-2)
    d2 = np.gradient(d1, times0, axis=-2)
    num = np.abs(d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0])
    den = np.power(d1[..., 0] ** 2 + d1[..., 1] ** 2, 1.5)
    k_param = float((num / np.maximum(den, 1e-6)).mean())
    k_seq = FF._seq_geometry(_arc(R, 10.0, V3_TIMES), 0.5)["curvature"].mean().item()
    # both land on 1/R; the parametric one carries np.gradient's one-sided end bias (~17 %)
    assert k_seq == pytest.approx(1.0 / R, rel=0.01)
    assert k_param == pytest.approx(1.0 / R, rel=0.20)
    assert math.copysign(1, k_seq) == math.copysign(1, k_param)
