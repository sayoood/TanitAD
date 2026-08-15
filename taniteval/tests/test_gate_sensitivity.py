"""The direction gate is a free parameter of the instrument — pin that it is swept.

⛔ THE FAILURE THIS EXISTS TO PREVENT (RETRACTION_LOG R-2026-08-06-yawgate). The panel
published `maneuver_vs_trajectory.kappa` at a single unswept `DIR_YAW_RAD = 0.15`. On the
Alpamayo comparison that gate is ~6.5x the typical 2 s turn — the HUMAN's own median
|net yaw| is 0.023 rad and only 17.9 % of windows exceed it — and the ranking of two arms
REVERSED between 0.15 and 0.10. A number whose value is set by an unstated analysis choice
is not a measurement.

⭐ The load-bearing test is `test_verdict_instability_is_reported`: a panel whose coherence
call flips inside the swept range must SAY SO, because that is precisely the case where
quoting the published gate is misleading and where a reader would otherwise see a clean
verdict.
"""
import numpy as np

from taniteval.hierarchy import (DIR_YAW_RAD, GATE_SWEEP, KAPPA_VERDICT_LADDER,
                                 R_LEFT, R_RIGHT, R_STRAIGHT, _gate_sensitivity)


def _cls(v, g):
    return np.where(v > g, R_LEFT, np.where(v < -g, R_RIGHT, R_STRAIGHT))


def test_published_gate_is_inside_its_own_sweep():
    """A sweep that does not contain the published value cannot show whether the
    published value is an outlier."""
    assert DIR_YAW_RAD in GATE_SWEEP


def test_turn_magnitude_exposes_a_mis_scaled_gate():
    """⭐ THE DIAGNOSTIC. Small turns + a coarse gate => almost nothing is 'turning',
    and the panel must make that visible rather than report a confident kappa."""
    rng = np.random.default_rng(0)
    gt = rng.normal(0.0, 0.03, 400)            # typical 2 s turn, as measured
    out = _gate_sensitivity(gt.tolist(), gt.tolist(), _cls(gt, DIR_YAW_RAD).tolist())
    tm = out["gt_turn_magnitude"]
    assert tm["median_abs_net_yaw_rad"] < DIR_YAW_RAD / 3
    assert tm["frac_above_published_gate"] < 0.05, tm
    assert out["per_gate"][f"{DIR_YAW_RAD:.2f}"]["frac_gt_turning"] < 0.05


def test_verdict_instability_is_reported():
    """⛔ THE LOAD-BEARING TEST. A declaration that tracks only FINE turns looks
    incoherent at a coarse gate and coherent at a fine one. The panel must report
    verdict_stable=False, not a clean number at 0.15."""
    rng = np.random.default_rng(1)
    traj = rng.normal(0.0, 0.03, 600)
    # declaration agrees with the trajectory at fine resolution only
    man = _cls(traj, 0.01)
    out = _gate_sensitivity(traj.tolist(), traj.tolist(), man.tolist())
    k_coarse = out["per_gate"][f"{DIR_YAW_RAD:.2f}"]["maneuver_vs_trajectory_kappa"]
    k_fine = out["per_gate"]["0.01"]["maneuver_vs_trajectory_kappa"]
    assert k_fine > k_coarse, (k_fine, k_coarse)
    assert out["verdict_stable"] is False, out["kappa_range"]


def test_stable_agreement_reports_stable():
    """The negative control: a declaration that genuinely matches the driven path at
    every scale must NOT be flagged unstable, or the flag means nothing.

    ⚠️ Asserts the PUBLISHED band, not a bare number. This test used to read
    ``min(kappa_range) >= 0.2`` — a threshold in no published ladder, and the same
    private cut that made ``verdict_stable`` untrustworthy (see
    ``test_kappa_ladder_single_source.py``). Every swept gate must land in the
    SAME band, and here that band is the top one."""
    rng = np.random.default_rng(2)
    traj = rng.normal(0.0, 0.4, 600)           # big, unambiguous turns
    out = _gate_sensitivity(traj.tolist(), traj.tolist(),
                            _cls(traj, DIR_YAW_RAD).tolist())
    assert out["verdict_stable"] is True
    assert out["verdicts_across_sweep"] == ["SUBSTANTIAL"], out["kappa_range"]
    assert min(out["kappa_range"]) >= KAPPA_VERDICT_LADDER[1][0], out["kappa_range"]


def test_every_swept_gate_is_present_and_ordered():
    rng = np.random.default_rng(3)
    v = rng.normal(0.0, 0.1, 200)
    out = _gate_sensitivity(v.tolist(), v.tolist(), _cls(v, 0.1).tolist())
    assert list(out["per_gate"]) == [f"{g:.2f}" for g in GATE_SWEEP]
    # frac_gt_turning must be MONOTONE NON-DECREASING as the gate tightens —
    # if it is not, the classifier is not doing what its name says.
    fracs = [out["per_gate"][f"{g:.2f}"]["frac_gt_turning"] for g in GATE_SWEEP]
    assert fracs == sorted(fracs), fracs


def test_degenerate_single_class_returns_none_not_a_fake_kappa():
    """⛔ kappa is UNDEFINED when one side has a single class (pe = 1). Returning 0.0
    would read as 'no agreement' when the truth is 'not computable'."""
    flat = [0.0] * 100
    out = _gate_sensitivity(flat, flat, [R_STRAIGHT] * 100)
    assert out["per_gate"][f"{DIR_YAW_RAD:.2f}"]["maneuver_vs_trajectory_kappa"] is None
    assert out["kappa_range"] is None


def test_assemble_guard_survives_numpy_arrays_and_missing_keys():
    """⛔ THE BUG THIS PINS, and it cost a full 40-episode GPU pass. `_assemble` turns
    every banked list into a numpy array, so the guard `if A.get("traj_net_yaw")` raised
    *"truth value of an array with more than one element is ambiguous"* — it did NOT fall
    through to the UNAVAILABLE branch, it killed the panel AFTER the model had run.

    The guard must behave correctly for all three real shapes: present-and-populated,
    absent (a pre-2026-08-06 panel), and present-but-empty."""
    populated = {"traj_net_yaw": np.zeros(7), "gt_net_yaw": np.zeros(7)}
    absent = {"gt_dir": np.zeros(7)}
    empty = {"traj_net_yaw": np.zeros(0), "gt_net_yaw": np.zeros(0)}
    for A, expect in ((populated, True), (absent, False), (empty, False)):
        got = all(len(A.get(k, ())) for k in ("traj_net_yaw", "gt_net_yaw"))
        assert bool(got) is expect, (A.keys(), got, expect)
