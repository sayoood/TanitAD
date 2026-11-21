"""D-YAWMASK-1 -- the paired yaw-rate cell is scored ONLY on step pairs that have a tangent.

WHAT BROKE. `taniteval/tools/refav1_arm.py::_components` emitted
`LAT_yaw_rate_mae_radps` as `|yaw_rate_pred - yaw_rate_gt|.mean(1)` over EVERY
step pair, while `four_families.lateral` masks the same term with
`pred.pair_valid & gt.pair_valid` (a step shorter than `min_ds` has no path
tangent, so its atan2 heading is noise). MEASURED by the EvalFlyWheel battery at
refcv6 step 5000: refcv4b 0.2034 rad/s unmasked vs 0.0318 on valid pairs, and a
paired `os - refcv4b` cell of -0.1683 "separated" that is -0.0042 on valid pairs.
`_components` feeds `_paired_families` (refav1_arm, refcv3_arm), paired_openloop,
stratified_openloop, refav1_paired_delta and openloop_suite.

WHAT IS PINNED, every expectation a LITERAL (an analytic target), never an
expression over the code under test:
  (1) a window whose GT is stopped, whose prediction is stopped, or whose GT
      jitters while the prediction drives, carries NO valid pair -> NaN;
  (2) a fully valid window is untouched: exact plan 0.0, a constant 0.1 rad/step
      turn on the 0.5 s grid reads exactly 0.2 rad/s;
  (3) a PARTIAL window is averaged over its valid pairs only: 0.2 rad/s, where
      the unmasked mean of the same pairs is 2.16106 rad/s;
  (4) the paired cell: two arms of IDENTICAL lateral skill that differ only in
      standstill jitter read exactly 0.0 [0.0, 0.0], not separated, with the
      three stopped windows dropped and COUNTED (the defect read pi/2 separated);
  (5) every `_paired_families` block names the mask it used.

DELIBERATE REGRESSIONS (run by the 2026-09-26-yaw-rate-mask package's
`code/run_regression_arms.py`, which swaps `taniteval/tools/refav1_arm.py` in a
clean scratch tree and runs this file unchanged): the historical defect (the
pre-fix unmasked line), a GT-only mask and a PRED-only mask each turn this file
RED; the fixed file turns it GREEN.

ASCII-only assertion text: this dev box's console is cp1252.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools" / "refav1_arm.py"

DT = 0.5                      # the 4-waypoint 0.5 s grid; min_ds = 0.5 m/s x 0.5 s = 0.25 m
KEY = "LAT_yaw_rate_mae_radps"


def _load_tool():
    spec = importlib.util.spec_from_file_location("_refav1_arm_yawmask_ut", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_refav1_arm_yawmask_ut"] = mod
    spec.loader.exec_module(mod)
    return mod


ra = _load_tool()


def _turn(step_m: float, dtheta: float, n: int = 4) -> np.ndarray:
    """n waypoints: steps of `step_m` metres whose heading grows by `dtheta` per step
    (step k has heading k*dtheta), starting from the ego origin along +x."""
    pts, x, y = [], 0.0, 0.0
    for k in range(n):
        x += step_m * math.cos(k * dtheta)
        y += step_m * math.sin(k * dtheta)
        pts.append([x, y])
    return np.asarray(pts, dtype=np.float32)


#: 10 m/s straight line on the 0.5 s grid: every step 5 m > min_ds 0.25 m.
STRAIGHT = np.array([[5.0, 0.0], [10.0, 0.0], [15.0, 0.0], [20.0, 0.0]], np.float32)
#: a constant turn of 0.1 rad per 0.5 s step = 0.2 rad/s on every pair, every step valid.
TURN = _turn(5.0, 0.1)
#: fully stopped.
STOPPED = np.zeros((4, 2), np.float32)
#: 0.1 m steps (< min_ds) whose heading rotates pi/2 every step: yaw-rate pi rad/s on
#: every pair, and NO step clears min_ds.
JITTER = np.array([[0.1, 0.0], [0.1, 0.1], [0.0, 0.1], [0.0, 0.0]], np.float32)
#: GT that drives 2 steps then creeps 0.1 m sideways and stops: steps (5,0),(5,0),
#: (0,0.1),(0,0) -> valid T,T,F,F -> pair_valid T,F,F; GT yaw-rate 0, pi, -pi rad/s.
GT_PARTIAL = np.array([[5.0, 0.0], [10.0, 0.0], [10.0, 0.1], [10.0, 0.1]], np.float32)
#: GT pulling away from standstill: steps (0,0.1),(5,0),(5,0),(5,0) -> valid F,T,T,T ->
#: pair_valid F,T,T; GT yaw-rate -pi, 0, 0 rad/s. A mask built from the SECOND step of
#: each pair alone (valid[:, 1:]) keeps pair 0 and scores its -pi.
GT_START = np.array([[0.0, 0.1], [5.0, 0.1], [10.0, 0.1], [15.0, 0.1]], np.float32)


def _yaw(P, G):
    return np.asarray(ra._components(np.stack(P), np.stack(G), DT)[KEY], dtype=np.float64)


# --------------------------------------------------------------------------- #
# (1) no valid pair -> NaN, whichever side has no tangent                       #
# --------------------------------------------------------------------------- #
def test_gt_stopped_window_is_not_scored():
    v = _yaw([JITTER], [STOPPED])
    assert math.isnan(v[0]), (
        "GT stopped, prediction jittering: no pair has a tangent, so the window must be "
        "NaN (dropped), not pi rad/s (the unmasked defect)")


def test_pred_stopped_window_is_not_scored_even_when_gt_drives():
    # the discriminator for a GT-ONLY mask: GT is valid on every pair here.
    v = _yaw([JITTER], [STRAIGHT])
    assert math.isnan(v[0]), (
        "prediction jittering at standstill against a driving GT: the PREDICTION has no "
        "tangent, so no pair is scored. A GT-only mask scores pi rad/s here.")


def test_gt_jitter_window_is_not_scored_even_when_pred_drives():
    # the discriminator for a PRED-ONLY mask: the prediction is valid on every pair.
    v = _yaw([STRAIGHT], [JITTER])
    assert math.isnan(v[0]), (
        "GT jittering at standstill against a driving prediction: the GT has no tangent, "
        "so no pair is scored. A pred-only mask scores pi rad/s here.")


# --------------------------------------------------------------------------- #
# (2) fully valid windows are untouched                                         #
# --------------------------------------------------------------------------- #
def test_exact_plan_reads_zero():
    v = _yaw([STRAIGHT], [STRAIGHT])
    assert v[0] == 0.0


def test_constant_turn_reads_its_analytic_yaw_rate():
    # 0.1 rad per step / 0.5 s per step = 0.2 rad/s on each of the 3 pairs.
    v = _yaw([TURN], [STRAIGHT])
    assert abs(v[0] - 0.2) < 1e-5, f"expected 0.2 rad/s, got {v[0]!r}"


# --------------------------------------------------------------------------- #
# (3) a partial window averages its VALID pairs only                            #
# --------------------------------------------------------------------------- #
def test_partial_window_averages_only_its_valid_pairs():
    # valid pair 0 only: |0.2 - 0| = 0.2. The unmasked mean of the same three pairs is
    # (0.2 + |0.2 - pi| + |0.2 + pi|) / 3 = 2.16106 rad/s.
    v = _yaw([TURN], [GT_PARTIAL])
    assert abs(v[0] - 0.2) < 1e-5, f"expected 0.2 rad/s on the one valid pair, got {v[0]!r}"
    assert abs(v[0] - 2.16106) > 1.0, "the partial window was averaged over invalid pairs"


def test_gt_pulling_away_scores_only_pairs_whose_both_steps_have_a_tangent():
    # pairs 1 and 2 are valid and exact (0.0); pair 0 straddles the standstill step.
    # Unmasked, or masked on the pair's second step only: (pi + 0 + 0) / 3 = 1.0472.
    v = _yaw([STRAIGHT], [GT_START])
    assert v[0] == 0.0, f"expected exactly 0.0 on the two valid pairs, got {v[0]!r}"


def test_heading_cell_is_unchanged_on_the_partial_window():
    # heading is masked by single-step validity: steps 0,1 valid, errors 0 and 0.1 rad
    # -> 0.05 rad = 2.8647890 deg. The yaw fix must not move it.
    c = ra._components(np.stack([TURN]), np.stack([GT_PARTIAL]), DT)
    assert abs(float(c["LAT_heading_mae_deg"][0]) - 2.8647890) < 1e-4


def test_the_component_keys_are_unchanged():
    c = ra._components(np.stack([STRAIGHT]), np.stack([STRAIGHT]), DT)
    assert sorted(c) == sorted([
        "ade_m", "fde_m", "LON_speed_mae_mps", "LON_along_mae_m", "LON_accel_mae_mps2",
        "LAT_cross_mae_m", "LAT_heading_mae_deg", "LAT_yaw_rate_mae_radps",
        "TAC_traj_lat_correct", "TAC_traj_lon_correct"])


# --------------------------------------------------------------------------- #
# (4) the paired cell                                                           #
# --------------------------------------------------------------------------- #
def _paired_block():
    # 3 episodes x (one driving window, one stopped-GT window). Arm a is exact on both;
    # arm b is exact on the driving window and jitters at standstill. Their lateral
    # skill on every window that HAS a tangent is identical.
    G = np.stack([STRAIGHT, STOPPED] * 3)
    A = np.stack([STRAIGHT, STOPPED] * 3)
    B = np.stack([STRAIGHT, JITTER] * 3)
    eid = [0, 0, 1, 1, 2, 2]
    comps = {"a": ra._components(A, G, DT), "b": ra._components(B, G, DT)}
    return ra._paired_families(comps, "a", "b", eid, {"a": "T1", "b": "T1"}, 200, 0)


def test_standstill_jitter_is_not_a_separated_yaw_rate_difference():
    cell = _paired_block()["families"]["lateral"][KEY]
    # the defect read delta pi/2 = 1.5708 rad/s, separated, from the jitter alone
    assert cell["delta"] == 0.0 and cell["lo"] == 0.0 and cell["hi"] == 0.0, cell
    assert cell["separated"] is False
    assert cell["n_dropped_nonfinite"] == 3
    assert cell["n_windows"] == 3


def test_every_paired_block_names_its_yaw_mask():
    blk = _paired_block()
    assert "pred.pair_valid AND gt.pair_valid" in blk["yaw_rate_cell"]
