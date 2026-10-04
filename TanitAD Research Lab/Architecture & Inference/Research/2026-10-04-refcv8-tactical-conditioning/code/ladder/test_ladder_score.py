"""Pins for the ladder harness (CPU). Expectations are LITERALS or ANALYTIC targets, never an expression over the code
under test; every bar-reading test carries a deliberate-regression arm that must go RED.
Run:  PYTHONPATH=<tree>/stack;<tree>/taniteval  python -m pytest -q test_ladder_score.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_arms as LA  # noqa: E402
import ladder_score as LS  # noqa: E402

SLOT_T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])


def straight(v, n=1):
    P = np.zeros((n, 8, 2))
    P[:, :, 0] = v * SLOT_T
    return P


def arc(R, v, n=1, left=True):
    s = v * SLOT_T
    th = s / R
    P = np.zeros((n, 8, 2))
    P[:, :, 0] = R * np.sin(th)
    P[:, :, 1] = (1 if left else -1) * R * (1 - np.cos(th))
    return P


# ------------------------------------------------------------------ analytic geometry ------------------------------ #
def test_straight_plan_against_itself_reads_zero_error_and_full_credit():
    P = straight(10.0, 4)
    z = mkpass(P, P)
    wm = LS.window_metrics(z)
    for k in ("cross_abs_6s", "along_abs_6s", "speed_mae_0_2s", "speed_mae_2_6s", "curv_mae_0_2s", "yaw_mae_0_2s",
              "progress_rel_err_6s", "ade_all"):
        num, den = wm[k]
        assert den.sum() > 0 and float(num.sum()) == pytest.approx(0.0, abs=1e-9), k
    num, den = wm["head15_all"]
    assert num.sum() == den.sum() == 4.0


def test_speed_mae_2_6s_is_the_literal_speed_gap():
    """a plan at 10 m/s against a GT at 12 m/s: every 1-s segment differs by EXACTLY 2 m/s."""
    wm = LS.window_metrics(mkpass(straight(10.0, 3), straight(12.0, 3)))
    num, den = wm["speed_mae_2_6s"]
    assert float(num.sum() / den.sum()) == pytest.approx(2.0, abs=1e-9)
    num, den = wm["progress_rel_err_6s"]
    assert float(num.sum() / den.sum()) == pytest.approx(2.0 / 12.0, abs=1e-9)


def test_turn_direction_and_curvature_on_a_circle():
    """GT = a left arc of R 30 m at 8 m/s (48 m, 91.7 deg): a GT TURN. A straight plan is direction-WRONG and its
    curvature error is EXACTLY 1/30 on every scored pair; the same arc is direction-correct with zero curvature error."""
    gt = arc(30.0, 8.0, 2)
    wm = LS.window_metrics(mkpass(straight(8.0, 2), gt))
    n, d = wm["dir_correct_turn"]
    assert d.sum() == 2 and n.sum() == 0
    n, d = wm["curv_mae_0_2s"]
    assert float(n.sum() / d.sum()) == pytest.approx(1 / 30.0, rel=2e-3)
    wm = LS.window_metrics(mkpass(arc(30.0, 8.0, 2), gt))
    n, d = wm["dir_correct_turn"]
    assert n.sum() == d.sum() == 2


def test_ceiling_violation_uses_the_fed_ceiling_and_ignores_unlimited_windows():
    P = straight(20.0, 3)
    z = mkpass(P, P, v_lim=np.array([15.0, 25.0, np.inf]))
    n, d = LS.window_metrics(z)["ceil_violation"]
    assert d.tolist() == [1.0, 1.0, 0.0] and n.tolist() == [1.0, 0.0, 0.0]


# ------------------------------------------------------------------ k sizing (SPEC sec. 4.2 worked example) -------- #
def test_k_reproduces_the_registered_worked_example():
    assert LS.k_from(-0.000506, 3.282, 872.4) == 51.0          # SPEC_WPB_LADDER sec. 4.2: k ~= 51


def test_k_is_capped_and_none_without_a_root():
    assert LS.k_from(-0.5, 0.01, 100.0) in (100.0, None)
    assert LS.k_from(0.5, 0.0, 1.0) is None


# ------------------------------------------------------------------ macro-F1 / bootstrap ----------------------------- #
def test_macro_f1_literal():
    C = np.array([[8, 2, 0], [1, 1, 0], [0, 0, 0]], float)   # class 2 absent -> mean over 2 classes
    f0 = 2 * (8 / 9) * (8 / 10) / ((8 / 9) + (8 / 10))
    f1 = 2 * (1 / 3) * (1 / 2) / ((1 / 3) + (1 / 2))
    assert LS.macro_f1_from_conf(C) == pytest.approx((f0 + f1) / 2, abs=1e-12)


def test_identical_arms_pair_to_exactly_zero():
    rng = np.random.default_rng(1)
    inv = np.repeat(np.arange(10), 5)
    cnt = LS.counts_of(LS.make_draws(10, b=200), 10)
    x = rng.normal(size=50)
    a = dict(zip(("pt", "bs", "n"), LS.boot_ratio(x, np.ones(50), inv, cnt)))
    c = LS.paired(a, a, +1)
    assert c["delta"] == 0.0 and c["ci"] == [0.0, 0.0] and not c["sep_better"] and not c["sep_worse"]


# ------------------------------------------------------------------ end to end: the L1 reading ---------------------- #
def mkpass(P, gt, v_lim=None, sha=None, p_lat=None, lat_v9=None):
    W = len(P)
    z = {"traj": np.asarray(P, np.float32), "gt": np.asarray(gt, np.float32), "gt_valid": np.ones((W, 8), bool),
         "v_lim": np.full(W, np.inf) if v_lim is None else v_lim,
         "p_lat": np.tile(np.eye(8)[0], (W, 1)) if p_lat is None else p_lat,
         "lat_v9": np.zeros(W, np.int64) if lat_v9 is None else lat_v9,
         "win_sha12": np.array([f"{i // 4:012d}" for i in range(W)] if sha is None else sha)}
    return z


def write_pass(w, arm, row, seed, z):
    d = w / "eval" / arm
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(d / f"{row}_s{seed}.npz"), **z)
    (d / f"{row}_s{seed}.json").write_text(json.dumps({"window_sha12_t_sha256": "D"}), encoding="utf-8")


def l1_world(tmp_path, vr8d_like="V0"):
    """40 episodes x 4 windows: half GT left turns, half straight. V0 / V0r drive straight everywhere (wrong on
    turns); V-R8 / V-R8r follow the GT; V-R8d copies `vr8d_like`."""
    W = 160
    gt = np.concatenate([arc(30.0, 8.0, W // 2), straight(8.0, W // 2)])
    plans = {"V0": straight(8.0, W), "V0r": straight(8.0, W), "V-R8": gt.copy(), "V-R8r": gt.copy()}
    plans["V-R8d"] = plans[vr8d_like].copy()
    for arm, P in plans.items():
        for s in (0, 1):
            write_pass(tmp_path, arm, "base", s, mkpass(P, gt))
            if arm == "V-R8":
                for r in ("legal", "rc_off", "rc_shuf"):
                    write_pass(tmp_path, arm, r, s, mkpass(P, gt))
    (tmp_path / "I0.json").write_text(json.dumps({"PASS": True}), encoding="utf-8")
    return tmp_path


def test_L1_clears_when_the_recipe_drives_and_the_regression_arm_does_not(tmp_path):
    res = LS.score(l1_world(tmp_path))
    l1 = res["rungs"]["L1"]
    assert l1["status"] == "SCORED"
    assert l1["VERDICT"] == "CLEARS", json.dumps(LS._json_safe(l1), indent=1)[:2000]


def test_MUT_L1_regression_arm_that_still_drives_makes_the_rung_NOT_QUOTABLE(tmp_path):
    """deliberate regression: V-R8d keeps V-R8's skill (the instrument cannot see the derangement) -> RED."""
    res = LS.score(l1_world(tmp_path, vr8d_like="V-R8"))
    assert res["rungs"]["L1"]["VERDICT"].startswith("NOT QUOTABLE")


def test_no_score_without_a_passed_I0(tmp_path):
    w = l1_world(tmp_path)
    (w / "I0.json").write_text(json.dumps({"PASS": False}), encoding="utf-8")
    assert LS.score(w)["VERDICT"].startswith("STOP")


def test_incomplete_rung_is_not_read(tmp_path):
    w = l1_world(tmp_path)
    for f in (w / "eval" / "V0r").glob("base_s1.*"):
        f.unlink()
    assert LS.score(w)["rungs"]["L1"]["status"] == "INCOMPLETE"


def test_passes_on_different_windows_are_refused(tmp_path):
    w = l1_world(tmp_path)
    (w / "eval" / "V0" / "base_s0.json").write_text(json.dumps({"window_sha12_t_sha256": "OTHER"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="NOT on the same windows"):
        LS.score(w)


# ------------------------------------------------------------------ the arm table ------------------------------------ #
R7 = ["--size", "large", "--trunk-compile", "--speed-max-sidecar-v6", "x", "--speed-max-sidecar-v6-eval", "y",
      "--max-speed-input-v6", "--v7-labels", "l", "--out", "o", "--seed", "0", "--steps", "50400"]


def test_arm_table_audits_one_variable_and_V0_is_speed_only():
    A = LA.arms(R7, 3000, "51", "/r")
    rep = LA.audit(A)
    assert rep["PASS"] and rep["I0_assert1_V0_speed_only"]["PASS"] and rep["no_v8_sidecar_anywhere"]["PASS"]
    assert set(A) == set(LA.ORDER)


def test_MUT_a_second_difference_fails_the_audit():
    A = LA.arms(R7, 3000, "51", "/r")
    ps, base, delta, role = A["V-MAP4"]
    A["V-MAP4"] = (LA.setf(ps, "--lr", ["2e-4"]), base, delta, role)
    assert not LA.audit(A)["V-MAP4"]["PASS"] and not LA.audit(A)["PASS"]


def test_MUT_a_live_refcv8_flag_on_V0_fails_assertion_1():
    A = LA.arms(R7, 3000, None, "/r")
    ps, base, delta, role = A["V0"]
    A["V0"] = (ps + [("--w-r8-cons", ["0.05"])], base, delta, role)
    assert not LA.audit(A)["I0_assert1_V0_speed_only"]["PASS"]


def test_every_arm_V0_included_trains_on_the_corrected_clock():
    A = LA.arms(R7, 3000, "51", "/r")
    assert all(dict(ps).get("--pose-sync-sidecar") == [LA.POSE_SYNC] for ps, *_r in A.values())
    assert LA.audit(A)["pose_sync_on_every_arm"]["PASS"]


def test_MUT_V0_on_the_uncorrected_clock_fails_the_audit():
    """MM 2026-10-05: V0 without --pose-sync-sidecar would make L1 a two-variable comparison -> RED."""
    A = LA.arms(R7, 3000, None, "/r")
    ps, base, delta, role = A["V0"]
    A["V0"] = (LA.drop(ps, "--pose-sync-sidecar"), base, delta, role)
    rep = LA.audit(A)
    assert not rep["pose_sync_on_every_arm"]["PASS"] and "V0" in rep["pose_sync_on_every_arm"]["arms_without"]
    assert not rep["PASS"]
