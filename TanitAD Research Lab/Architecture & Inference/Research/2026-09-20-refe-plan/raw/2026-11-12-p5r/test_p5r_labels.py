"""Unit tests for p5r_labels (card SG/p5r/PREREG_P5R.md §8). Synthetic tracker: no nuPlan, no scipy, CPU only.

The controls C-R1 (lam -> inf), C-R2 (lam = 0), C-R3 (identity tracker), the zero control, and the reachability
mutation: with the lam term mutated out of the objective, the C-R1 check must go RED.
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import p5r_labels as R  # noqa: E402
import navsim_dac as ND  # noqa: E402  (module level: math + numpy only)


def lag_tracker(P):
    """Smooth, invertible, mildly nonlinear stand-in for LQR + bicycle: a first-order lag along the knots."""
    P = np.asarray(P, np.float64)
    prev = np.concatenate([np.zeros_like(P[:, :1]), P[:, :-1]], 1)
    s = P - 0.3 * (P - prev)
    s[..., 1] = s[..., 1] + 0.02 * np.sin(P[..., 2])
    s[..., 2] = P[..., 2] - 0.2 * (P[..., 2] - prev[..., 2])
    return s


def identity_tracker(P):
    return np.asarray(P, np.float64).copy()


def make_row(seed=0, curve=0.03):
    rng = np.random.default_rng(seed)
    t = np.arange(1, 21) * 0.2
    v = 8.0 + rng.normal(0, 0.3)
    h = curve * t ** 2 / 2.0
    x = np.cumsum(v * 0.2 * np.cos(h))
    y = np.cumsum(v * 0.2 * np.sin(h))
    traj = np.stack([x, y, h], -1)
    d0 = np.stack([0.05 * np.sin(t), 0.08 * t / 4.0, 0.01 * np.cos(t)], -1)
    return traj, d0


def c_r1_check(res, traj, tol=1e-6):
    return float(np.max(np.abs(res["label"] - traj))) <= tol


def test_d_is_invertible_unit_lower_triangular():
    D = R.dmat()
    assert np.allclose(np.tril(D), D)
    assert np.all(np.diag(D) == 1.0)
    assert abs(np.linalg.det(D) - 1.0) < 1e-9


def test_lambda_inf_is_bit_exact_identity():
    traj, d0 = make_row()
    out = R.solve(traj, d0, [math.inf], lag_tracker, ND.to_navsim)
    assert np.array_equal(out[math.inf]["label"], traj)


def test_c_r1_green_and_drop_lambda_mutation_red():
    traj, d0 = make_row(1)
    ok = R.solve(traj, d0, [1e12], lag_tracker, ND.to_navsim)[1e12]
    assert c_r1_check(ok, traj), float(np.max(np.abs(ok["label"] - traj)))
    bad = R.solve(traj, d0, [1e12], lag_tracker, ND.to_navsim, mutate="drop_lambda")[1e12]
    assert not c_r1_check(bad, traj), "dropping lambda from the objective must turn C-R1 RED"


def test_zero_control_and_lambda_reaches_objective():
    traj, d0 = make_row(2)
    out = R.solve(traj, d0, [0.0, 1.0], lag_tracker, ND.to_navsim)
    assert out[0.0]["J_reg"] == 0.0                                  # zero control: exactly 0.0
    r1 = out[1.0]
    assert np.max(np.abs(r1["d"])) > 0 and r1["J_reg"] > 0.0           # lam reaches J (S5)
    g = R.reg_vec(r1["d"])
    assert abs(r1["J_reg"] - 1.0 * float(g @ g)) <= 1e-9 * max(1.0, r1["J_reg"])
    assert abs(r1["J"] - (r1["J_track"] + r1["J_reg"])) <= 1e-9 * max(1.0, r1["J"])
    mut = R.solve(traj, d0, [1.0], lag_tracker, ND.to_navsim, mutate="drop_lambda")[1.0]
    assert mut["J_reg"] == 0.0                                        # S5 must flag the mutation


def test_c_r2_lambda_zero_descends_from_warm_start():
    traj, d0 = make_row(3)
    T = ND.to_navsim(traj)
    e0 = R.resid(T, lag_tracker(ND.to_navsim(traj + d0)[None])[0])
    out = R.solve(traj, d0, [0.0], lag_tracker, ND.to_navsim)[0.0]
    assert out["J_track"] <= float(e0 @ e0)
    assert out["J_track"] < 0.01 * float(e0 @ e0)                    # an invertible tracker is nearly inverted


def test_tracking_error_does_not_fall_with_lambda():
    traj, d0 = make_row(4, curve=0.08)
    out = R.solve(traj, d0, list(R.LAMBDA_GRID), lag_tracker, ND.to_navsim)
    jt = [out[lm]["J_track"] for lm in R.LAMBDA_GRID]
    assert jt[-1] >= jt[0]
    rough = [float(R.reg_vec(out[lm]["d"]) @ R.reg_vec(out[lm]["d"])) for lm in R.LAMBDA_GRID]
    assert rough[-1] <= rough[0]


def test_c_r3_identity_tracker_known_value():
    traj, d0 = make_row(5)
    out = R.solve(traj, d0, [1.0], identity_tracker, ND.to_navsim)[1.0]
    assert float(np.max(np.abs(out["label"] - traj))) <= 1e-6


def test_tracker_call_count_matches_card_formula():
    traj, d0 = make_row(6)
    cnt = R.Counter(lag_tracker)
    out = R.solve(traj, d0, [1.0], cnt, ND.to_navsim)[1.0]
    assert cnt.calls == 1 + out["iters"]
    assert cnt.props == 25 + 3 * out["iters"]
    assert out["iters"] <= R.K_MAX


def test_joint_lambda_batch_equals_solo():
    traj, d0 = make_row(7)
    joint = R.solve(traj, d0, [0.1, 10.0], lag_tracker, ND.to_navsim)
    solo = R.solve(traj, d0, [10.0], lag_tracker, ND.to_navsim)
    assert np.array_equal(joint[10.0]["label"], solo[10.0]["label"])


def test_tb6_is_ratio_of_medians_and_projection_reduces_to_it():
    rng = np.random.default_rng(8)
    trajs = np.stack([make_row(i, curve=c)[0] for i, c in enumerate(rng.uniform(0, 0.1, 60))])
    labels = trajs.copy()
    labels[:, :, 1] += 0.01 * np.sin(np.arange(20))[None]
    r = R.tb6(labels, trajs)
    ll = np.array([R.learnability(q) for q in labels])
    lt = np.array([R.learnability(q) for q in trajs])
    assert r["R_acc"] == pytest.approx(np.median(ll[:, 0]) / np.median(lt[:, 0]))
    assert r["R_hr"] == pytest.approx(np.median(ll[:, 1]) / np.median(lt[:, 1]))
    assert R.tb6(trajs, trajs)["R_acc"] == 1.0
    p = R.projection(lt[:, 1], lt[:, 1], lt[:, 1])                      # sample == bank, label == raw
    assert p["R_proj"] == pytest.approx(1.0, abs=0.08)
    assert p["dropped_mass"] == 0.0


def test_weights_and_estimator_are_the_p5_ones():
    import p5_target_labels as P5
    assert R.paired_ci is P5.paired_ci and R.boot_weights is P5.boot_weights and R.learnability is P5.learnability
    assert (P5.N_BOOT, P5.SEED) == (10_000, 20260927)
    assert R.L_M_PER_RAD == ND.HALF_LENGTH


def test_module_imports_no_scipy_and_no_nuplan():
    import subprocess
    code = ("import sys; sys.path.insert(0, r'%s'); import p5r_labels; "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('scipy', 'nuplan', 'torch')]; "
            "print('BAD', bad); sys.exit(1 if bad else 0)") % HERE
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
