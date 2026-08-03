"""IDM four-family instrument: definition parity with taniteval, the cadence
trap, and — the point of the whole file — that each family can DISCRIMINATE."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# repo pattern (test_heldout_gate.py:38): taniteval is a sibling checkout, not an
# installed package. The definition-parity test below must RUN, not skip — it is
# what keeps the IDM and the world model reporting the same geometry.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "taniteval"))

from tanitad.eval.idm_families import (
    IDM_DT_S,
    all_families,
    balanced_accuracy,
    confusion,
    geometry,
    lateral,
    longitudinal,
    manoeuvre_classes,
    strategic,
    tactical,
)


# --------------------------------------------------------------------------- #
# helpers: synthetic trajectories with a KNOWN manoeuvre                       #
# --------------------------------------------------------------------------- #
def arc(n, speed, yaw_rate, dt=IDM_DT_S, H=4, accel=0.0, seed=None):
    """Ego-frame waypoints of a constant-turn-rate, constant-accel motion."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        speed = speed + rng.normal(scale=0.5, size=n)
        yaw_rate = yaw_rate + rng.normal(scale=0.02, size=n)
    speed = np.broadcast_to(np.asarray(speed, float), (n,)).astype(float)
    yaw_rate = np.broadcast_to(np.asarray(yaw_rate, float), (n,)).astype(float)
    out = np.zeros((n, H, 2))
    x = np.zeros(n); y = np.zeros(n); psi = np.zeros(n); v = speed.copy()
    sub = 10                                   # integrate finely, sample at H
    for h in range(H):
        for _ in range(sub):
            ddt = dt / sub
            x += v * np.cos(psi) * ddt
            y += v * np.sin(psi) * ddt
            psi += yaw_rate * ddt
            v += accel * ddt
        out[:, h, 0] = x
        out[:, h, 1] = y
    return out


# --------------------------------------------------------------------------- #
# 1. definition parity + the cadence trap                                      #
# --------------------------------------------------------------------------- #
def test_geometry_matches_four_families_at_10hz():
    """Same definitions as the program's instrument — only DT_S is a parameter.
    If this drifts, the IDM and the world model stop being comparable."""
    ff = pytest.importorskip("taniteval.four_families")
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    wp = np.cumsum(rng.normal(scale=0.8, size=(50, 6, 2)), axis=1)
    mine = geometry(wp, dt=ff.DT_S)
    theirs = ff._seq_geometry(torch.from_numpy(wp))
    for key in ("speed", "heading", "yaw_rate", "curvature", "accel",
                "along", "cross"):
        assert np.allclose(mine[key], theirs[key].numpy(), atol=1e-9), key
    assert np.array_equal(mine["valid"], theirs["valid"].numpy())
    assert np.array_equal(mine["pair_valid"], theirs["pair_valid"].numpy())


def test_cadence_is_load_bearing():
    """THE trap: IDM waypoints are 0.5 s apart. Scoring them at four_families'
    hard-coded 0.1 s reads every speed 5x too large."""
    wp = arc(20, speed=10.0, yaw_rate=0.0)
    assert geometry(wp, dt=0.5)["speed"].mean() == pytest.approx(10.0, rel=0.02)
    assert geometry(wp, dt=0.1)["speed"].mean() == pytest.approx(50.0, rel=0.02)


def test_geometry_rejects_bad_shape_and_dt():
    with pytest.raises(ValueError):
        geometry(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        geometry(np.zeros((4, 3, 2)), dt=0)


# --------------------------------------------------------------------------- #
# 2. manoeuvre classes are correct AND factored                                #
# --------------------------------------------------------------------------- #
def test_manoeuvre_classes_recover_the_known_motion():
    left = manoeuvre_classes(arc(10, 10.0, +0.25))
    right = manoeuvre_classes(arc(10, 10.0, -0.25))
    straight = manoeuvre_classes(arc(10, 10.0, 0.0))
    assert (left["lateral"] == 2).all()
    assert (right["lateral"] == 0).all()
    assert (straight["lateral"] == 1).all()
    assert (manoeuvre_classes(arc(10, 10.0, 0.0, accel=+1.5))["longitudinal"] == 2).all()
    assert (manoeuvre_classes(arc(10, 10.0, 0.0, accel=-1.5))["longitudinal"] == 0).all()
    assert (straight["longitudinal"] == 1).all()


def test_factored_classes_survive_a_SIMULTANEOUS_lat_and_lon_manoeuvre():
    """The defect the factoring exists for: a vehicle braking INTO a left turn.
    The factored axes must report both; the legacy mixed 5-way must lose the
    longitudinal decision entirely. If mixed ever kept it, the program's
    diagnosis of longitudinal blindness would be wrong."""
    m = manoeuvre_classes(arc(10, 12.0, +0.25, accel=-1.5))
    assert (m["lateral"] == 2).all(), "lateral lost"
    assert (m["longitudinal"] == 0).all(), "longitudinal lost by the FACTORED axes"
    assert set(np.unique(m["mixed"])) == {3}, "mixed should collapse to 'left'"


def test_balanced_accuracy_punishes_a_constant_predictor():
    """Plain accuracy would give a 'always straight' labeller ~0.9 here."""
    gt = np.array([1] * 90 + [0] * 5 + [2] * 5)
    const = np.ones_like(gt)
    C = confusion(const, gt, 3)
    assert float(np.trace(C) / C.sum()) == pytest.approx(0.9)
    assert balanced_accuracy(C) == pytest.approx(1 / 3)


def test_balanced_accuracy_require_all_blocks_the_bootstrap_degeneracy():
    """REGRESSION. Turns are <1 % of a highway corpus, so many episode resamples
    contain no turn at all. With only one class present, 'mean recall over
    present classes' scores a BLIND CONSTANT predictor at 1.0 — which is how the
    first run produced a blind control with CI [0.3333, 1.0000]. Such draws must
    return nan so the bootstrap drops and counts them."""
    gt = np.ones(50, dtype=np.int64)                    # a draw with no turns
    C = confusion(np.ones(50, dtype=np.int64), gt, 3)
    assert balanced_accuracy(C) == pytest.approx(1.0)   # the trap
    assert math.isnan(balanced_accuracy(C, require_all=True))
    # a draw with every class present is unaffected
    gt2 = np.array([0] * 5 + [1] * 40 + [2] * 5)
    C2 = confusion(gt2, gt2, 3)
    assert balanced_accuracy(C2, require_all=True) == pytest.approx(1.0)


def test_confusion_orientation_is_gt_rows_pred_cols():
    C = confusion(pred=np.array([0, 0]), gt=np.array([1, 1]), k=2)
    assert C[1, 0] == 2 and C[0, 1] == 0


# --------------------------------------------------------------------------- #
# 3. DISCRIMINATION — the negative controls that make the numbers quotable     #
# --------------------------------------------------------------------------- #
def _oracle_vs_degraded():
    rng = np.random.default_rng(3)
    n = 400
    yr = rng.normal(scale=0.25, size=n)
    ac = rng.normal(scale=1.2, size=n)
    gt = arc(n, 12.0, yr, accel=ac)
    good = arc(n, 12.0, yr + rng.normal(scale=0.02, size=n),
               accel=ac + rng.normal(scale=0.1, size=n))
    blind = arc(n, 12.0, 0.0, accel=0.0)          # always "straight + cruise"
    return gt, good, blind


def test_tactical_separates_a_good_labeller_from_a_blind_one():
    """NEGATIVE CONTROL. A labeller that always says 'straight, cruise' must
    collapse to chance balanced accuracy. If the instrument cannot see that, no
    tactical number it produces is worth quoting."""
    gt, good, blind = _oracle_vs_degraded()
    tg = tactical(good, gt)
    tb = tactical(blind, gt)
    for axis in ("lateral", "longitudinal"):
        assert tg[axis]["balanced_accuracy"] > 0.80, (axis, tg[axis])
        assert tb[axis]["balanced_accuracy"] == pytest.approx(1 / 3, abs=0.02), (
            axis, tb[axis])
        assert tg[axis]["balanced_accuracy"] > tb[axis]["balanced_accuracy"] + 0.4


def test_tactical_exposes_the_mixed_class_blindness_on_real_shaped_data():
    """The mixed 5-way must be MEASURABLY worse at longitudinal recall than the
    factored axis — that is the defect made visible rather than re-created."""
    gt, good, _ = _oracle_vs_degraded()
    t = tactical(good, gt)
    gm = manoeuvre_classes(gt)
    turning = gm["lateral"] != 1
    assert turning.sum() > 20, "fixture must contain turns"
    # on turning windows the mixed class carries no longitudinal information
    assert len(np.unique(gm["mixed"][turning])) <= 2
    assert t["longitudinal"]["balanced_accuracy"] > 0.8


def test_lateral_separates_a_wrong_curvature_that_ADE_barely_sees():
    """The family's whole justification: a path can be smooth and wrong."""
    gt, good, blind = _oracle_vs_degraded()
    lg = lateral(good, gt)
    lb = lateral(blind, gt)
    assert lb["curvature_mae_inv_m"] > 3 * lg["curvature_mae_inv_m"]
    assert lb["yaw_rate_mae_rad_s"] > 3 * lg["yaw_rate_mae_rad_s"]
    assert lg["n_curvature"] > 0 and lg["n_pair_valid"] > 0


def test_longitudinal_separates_a_speed_error():
    gt = arc(200, 12.0, 0.0)
    slow = arc(200, 8.0, 0.0)
    ok = arc(200, 12.1, 0.0)
    assert (longitudinal(slow, gt)["traj_speed_mae_mps"]
            > 10 * longitudinal(ok, gt)["traj_speed_mae_mps"])
    assert longitudinal(slow, gt)["traj_speed_bias_mps"] < 0     # too slow
    assert longitudinal(slow, gt)["along_final_bias_m"] < 0      # behind


def test_scalar_channels_are_reported_when_supplied():
    gt = arc(30, 10.0, 0.1)
    s_gt = np.stack([np.full(30, 10.0), np.full(30, 0.1),
                     np.zeros(30), np.zeros(30)], 1)
    s_pr = s_gt + 0.5
    f = all_families(gt, gt, pred_scalars=s_pr, gt_scalars=s_gt)
    assert f["LONGITUDINAL"]["scalar_speed_mae_mps"] == pytest.approx(0.5)
    assert f["LATERAL"]["scalar_yaw_rate_mae_rad_s"] == pytest.approx(0.5)
    assert f["LATERAL"]["scalar_steer_mae"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 4. the contract: unavailable families are DECLARED, never dropped            #
# --------------------------------------------------------------------------- #
def test_strategic_is_unavailable_with_a_reason_and_n():
    s = strategic(1234)
    assert s["status"] == "UNAVAILABLE" and s["n"] == 1234
    assert "route" in s["reason"] and len(s["reason"]) > 60
    with pytest.raises(NotImplementedError):
        strategic(10, route_labels_available=True)


def test_all_families_declares_every_family_and_flags_the_gaps():
    gt, good, _ = _oracle_vs_degraded()
    f = all_families(good, gt)
    assert set(f) >= {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}
    assert f["LONGITUDINAL"]["distance_keeping"]["status"] == "UNAVAILABLE"
    assert f["STRATEGIC"]["status"] == "UNAVAILABLE"
    assert "STRATEGIC" in f["_unavailable"]
    assert "never pooled" in f["_contract"]
    # no family may be silently collapsed into a single score
    assert not any(k.lower() in ("score", "total", "composite") for k in f)


def test_perfect_prediction_scores_perfectly_everywhere():
    gt, _, _ = _oracle_vs_degraded()
    f = all_families(gt, gt)
    assert f["LONGITUDINAL"]["traj_speed_mae_mps"] == pytest.approx(0.0, abs=1e-9)
    assert f["LATERAL"]["curvature_mae_inv_m"] == pytest.approx(0.0, abs=1e-9)
    for axis in ("lateral", "longitudinal"):
        assert f["TACTICAL"][axis]["balanced_accuracy"] == pytest.approx(1.0)
