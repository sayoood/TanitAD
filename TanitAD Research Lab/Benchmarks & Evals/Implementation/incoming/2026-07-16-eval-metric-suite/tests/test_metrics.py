"""Sanity tests for the custom metric suite (gate G-B2): every metric is checked
on a synthetic fixture whose answer is known analytically, plus its direction of
goodness and its baseline (E2E-like) case. No trained model, no simulator.

Analytic ground truth is derived in comments next to each assertion so a reviewer
can recompute by hand.
"""

import numpy as np
import pytest

from tanitad_metrics import (
    LAL_NO_REACTION,
    ScenarioTelemetry,
    ade,
    compute_cnce,
    compute_lal,
    compute_lops,
    compute_okri,
    compute_tms,
    fde,
    miss_rate,
    rmse_xy,
    run_scenario_suite,
    trajectory_extra_metrics,
)


# --------------------------------------------------------------------------- #
# Fixture builder                                                              #
# --------------------------------------------------------------------------- #
def _telemetry(T=10, dt=0.1, **overrides):
    """A benign default clip; override any field for a specific metric test."""
    base = dict(
        ego_v=np.full(T, 10.0),
        ego_jerk=np.zeros(T),
        steer_rate=np.zeros(T),
        latency_ms=np.full(T, 50.0),
        hazard_los_flag=np.zeros(T, dtype=bool),
        dist_to_blind_spot=np.full(T, 100.0),      # far -> no blind spot by default
        is_occluded_flag=np.zeros(T, dtype=bool),
        wm_hazard_xy=np.full((T, 2), np.nan),
        gt_hazard_xy=np.zeros((T, 2)),
        dt=dt,
    )
    base.update(overrides)
    return ScenarioTelemetry(**base)


# --------------------------------------------------------------------------- #
# 1. LAL                                                                       #
# --------------------------------------------------------------------------- #
def test_lal_proactive_positive():
    # dt=0.1: LoS first at idx 50 (t=5.0), braking first at idx 46 (t=4.6) -> +0.4
    T = 60
    los = np.zeros(T, dtype=bool); los[50:] = True
    jerk = np.zeros(T); jerk[46] = -2.0            # one prophylactic brake before LoS
    lal = compute_lal(_telemetry(T=T, hazard_los_flag=los, ego_jerk=jerk))
    assert lal == pytest.approx(0.4, abs=1e-9)


def test_lal_reactive_negative():
    # brake only AFTER LoS: LoS idx 50 (5.0), brake idx 53 (5.3) -> -0.3 (reactive E2E)
    T = 60
    los = np.zeros(T, dtype=bool); los[50:] = True
    jerk = np.zeros(T); jerk[53] = -3.0
    lal = compute_lal(_telemetry(T=T, hazard_los_flag=los, ego_jerk=jerk))
    assert lal == pytest.approx(-0.3, abs=1e-9)
    assert lal < 0                                  # direction: reactive is non-positive


def test_lal_no_reaction_sentinel():
    T = 60
    los = np.zeros(T, dtype=bool); los[50:] = True  # hazard seen, ego never brakes
    assert compute_lal(_telemetry(T=T, hazard_los_flag=los)) == LAL_NO_REACTION


def test_lal_no_hazard_is_zero():
    assert compute_lal(_telemetry()) == 0.0         # LoS never occurs -> nothing to anticipate


# --------------------------------------------------------------------------- #
# 2. TMS                                                                       #
# --------------------------------------------------------------------------- #
def test_tms_perfectly_smooth_is_one():
    assert compute_tms(_telemetry()) == pytest.approx(1.0)  # zero jerk, zero steer


def test_tms_known_integral():
    # dt=1, |jerk|=2 const over N=6 -> integral 2*1*5=10; steer 0 -> TMS=1/(1+1*10)=1/11
    tel = _telemetry(T=6, dt=1.0, ego_jerk=np.full(6, 2.0))
    assert compute_tms(tel) == pytest.approx(1.0 / 11.0, rel=1e-9)


def test_tms_monotone_in_jerk():
    smooth = compute_tms(_telemetry(T=6, dt=1.0, ego_jerk=np.full(6, 1.0)))
    rough = compute_tms(_telemetry(T=6, dt=1.0, ego_jerk=np.full(6, 4.0)))
    assert rough < smooth                            # more jerk -> less stable


# --------------------------------------------------------------------------- #
# 3. OKRI                                                                      #
# --------------------------------------------------------------------------- #
def test_okri_known_integral():
    # dt=1, v=10, m=1500 -> KE=75000; d_blind=5 (<30), eps=0.1 -> risk=75000/5.1
    # N=3 -> integral = risk*1*2; /1000 -> expected
    T = 3
    tel = _telemetry(T=T, dt=1.0, ego_v=np.full(T, 10.0),
                     dist_to_blind_spot=np.full(T, 5.0), ego_mass_kg=1500.0)
    expected = (75000.0 / 5.1) * 2.0 / 1000.0        # ~= 29.4118
    assert compute_okri(tel) == pytest.approx(expected, rel=1e-9)


def test_okri_no_blind_spot_is_zero():
    assert compute_okri(_telemetry()) == 0.0         # d_blind=100 everywhere (>30)


def test_okri_lower_when_slower():
    fast = compute_okri(_telemetry(T=3, dt=1.0, ego_v=np.full(3, 10.0),
                                   dist_to_blind_spot=np.full(3, 5.0)))
    slow = compute_okri(_telemetry(T=3, dt=1.0, ego_v=np.full(3, 3.0),
                                   dist_to_blind_spot=np.full(3, 5.0)))
    assert slow < fast                               # throttling near a blind spot is safer


# --------------------------------------------------------------------------- #
# 4. CNCE                                                                      #
# --------------------------------------------------------------------------- #
def test_cnce_known_value_no_collision():
    # dt=1, v=10, N=3 -> dist=10*1*2=20; lat=50ms=0.05s; P=4 -> denom=0.2; 20/0.2=100
    tel = _telemetry(T=3, dt=1.0, ego_v=np.full(3, 10.0),
                     latency_ms=np.full(3, 50.0), params_billions=4.0, collisions=0)
    assert compute_cnce(tel) == pytest.approx(100.0, rel=1e-9)


def test_cnce_collision_penalty():
    tel = _telemetry(T=3, dt=1.0, ego_v=np.full(3, 10.0),
                     latency_ms=np.full(3, 50.0), params_billions=4.0, collisions=1)
    assert compute_cnce(tel) == pytest.approx(100.0 * np.exp(-2.0), rel=1e-9)


def test_cnce_penalizes_bigger_model():
    small = compute_cnce(_telemetry(T=3, dt=1.0, ego_v=np.full(3, 10.0),
                                    params_billions=4.0))
    big = compute_cnce(_telemetry(T=3, dt=1.0, ego_v=np.full(3, 10.0),
                                  params_billions=15.0))
    assert big < small                               # compute bloat lowers efficacy


# --------------------------------------------------------------------------- #
# 5. LOPS                                                                      #
# --------------------------------------------------------------------------- #
def test_lops_perfect_tracking_is_one():
    T = 6
    occ = np.ones(T, dtype=bool)
    gt = np.tile(np.array([3.0, 4.0]), (T, 1))
    tel = _telemetry(T=T, is_occluded_flag=occ, wm_hazard_xy=gt.copy(), gt_hazard_xy=gt)
    assert compute_lops(tel) == pytest.approx(1.0)   # exact WM = GT -> exp(0)=1


def test_lops_known_error():
    # constant 2 m tracking error, gamma=0.5 -> exp(-1) for every occluded step
    T = 6
    occ = np.ones(T, dtype=bool)
    gt = np.zeros((T, 2))
    wm = np.tile(np.array([2.0, 0.0]), (T, 1))       # ||.||=2
    tel = _telemetry(T=T, is_occluded_flag=occ, wm_hazard_xy=wm, gt_hazard_xy=gt)
    assert compute_lops(tel) == pytest.approx(np.exp(-1.0), rel=1e-9)


def test_lops_e2e_baseline_is_zero():
    # no occlusion sampled / no WM estimate -> 0.0 (standard E2E has no latent track)
    assert compute_lops(_telemetry()) == 0.0


def test_lops_ignores_unoccluded_and_nan():
    # only the occluded, WM-estimated steps count
    T = 4
    occ = np.array([True, False, True, False])
    wm = np.array([[0.0, 0.0], [9.0, 9.0], [np.nan, np.nan], [0.0, 0.0]])
    gt = np.zeros((T, 2))
    tel = _telemetry(T=T, is_occluded_flag=occ, wm_hazard_xy=wm, gt_hazard_xy=gt)
    # step0: occluded, wm==gt -> exp(0)=1 ; step2 occluded but NaN wm -> excluded
    # steps 1,3 not occluded -> excluded. Only step0 counts -> 1.0
    assert compute_lops(tel) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #
def test_run_scenario_suite_keys():
    out = run_scenario_suite(_telemetry(), model_name="unit")
    for k in ("LAL_s", "TMS", "OKRI", "CNCE", "LOPS"):
        assert k in out
    assert out["model"] == "unit"


# --------------------------------------------------------------------------- #
# Trajectory seam (extra_metrics for the D1-D3 gate runner)                    #
# --------------------------------------------------------------------------- #
def test_ade_fde_identical_is_zero():
    xy = np.random.RandomState(0).randn(8, 3, 2)
    assert ade(xy, xy) == pytest.approx(0.0)
    assert fde(xy, xy) == pytest.approx(0.0)
    assert rmse_xy(xy, xy) == pytest.approx(0.0)


def test_ade_constant_offset():
    # every point offset by (3,4) -> displacement 5 everywhere -> ade=fde=rmse=5
    true = np.zeros((5, 2, 2))
    pred = true + np.array([3.0, 4.0])
    assert ade(pred, true) == pytest.approx(5.0)
    assert fde(pred, true) == pytest.approx(5.0)
    assert rmse_xy(pred, true) == pytest.approx(5.0)


def test_miss_rate_threshold():
    # final errors [1, 3]; threshold 2 -> one miss -> 0.5
    true = np.zeros((2, 1, 2))
    pred = np.array([[[1.0, 0.0]], [[3.0, 0.0]]])
    assert miss_rate(pred, true, thresh_m=2.0) == pytest.approx(0.5)


def test_extra_metrics_seam_accepts_torch():
    # the gate runner passes torch tensors as (pred_xy, true_xy); the seam must
    # accept them and return plain floats.
    torch = pytest.importorskip("torch")
    metrics = trajectory_extra_metrics()             # default rmse + miss_rate
    assert set(metrics) == {"rmse", "miss_rate"}
    pred = torch.zeros(4, 2, 2)
    true = torch.ones(4, 2, 2)
    for name, fn in metrics.items():
        val = fn(pred, true)
        assert isinstance(val, float)
    # rmse of all-0 vs all-1: each component err 1 -> norm sqrt(2) everywhere
    assert metrics["rmse"](pred, true) == pytest.approx(np.sqrt(2.0), rel=1e-6)
