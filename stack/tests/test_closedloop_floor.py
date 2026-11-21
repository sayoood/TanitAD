"""Guards for the closed-loop TRIVIAL FLOOR arms (`cl_ha0`, `cl_ha`, `cl_ha0_ext`).

⭐ WHY THIS FILE EXISTS. Until 2026-09-04 every closed-loop number this programme
published — `refcv3 2.8755` vs `refc-base 2.6554`, `flagship-v1 - refc-base
+7.1642 [+5.2654, +8.9661]` — was a difference between two models with **no bar
under either**. In OPEN loop, on the same programme's own instrument, the trivial
hold-action control BEATS the deployed model (`ha` 0.2996 m vs `os` 0.4419 m). A
floor is therefore not a nicety here; it is the thing that decides whether a
closed-loop level means anything.

⛔ AND A FLOOR IS ONLY WORTH ITS CONTROLS. Every test below is a KNOWN-VALUE
assertion — the floor must read a value that is decided before the run, not merely
"look reasonable". Three of them exist because the opposite failure mode is real
and cheap to hit:

* ``test_ha0_control_reads_exactly_zero`` — a floor whose defining property ("it
  does nothing") holds only to ~1e-15 cannot serve as the control that a model is
  scored against. `0.1` is not binary-exact, so accumulating `fl(v*0.1)` five times
  misses `v/2` by ~6e-16 and `wp_to_control` amplifies that into a nonzero
  acceleration command. The closed-form branch is what makes it exact, and this
  test is what keeps it exact.
* ``test_degenerate_arms_are_bit_identical`` — `cl_ha` and `cl_ha0_ext` must
  collapse ONTO `cl_ha0` when the measured `a0` and `omega0` are zero. If they did
  not, the three arms would not be nested and a "floor" margin would be measuring
  the floor's own parameterisation.
* ``test_set_t0_reads_no_future_pose`` — the poses AFTER `t0` are poisoned with
  NaN and the answer must be unchanged. Presence of the right number is not
  evidence that no future pose was read; only poisoning is.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_EXP = Path(__file__).resolve().parents[1] / "experiments" / "alpasim-gsplat"
sys.path.insert(0, str(_EXP))

from closedloop_drive import (DT, FLOOR_ARMS, LOOKAHEAD_IDX,  # noqa: E402
                              STACK, WHEELBASE, WINDOW, WP_STEPS,
                              FlagshipV1Policy, KinematicFloorPolicy,
                              RefCPolicy, RefCV3Policy, _rz, _yaw,
                              wp_to_control)

V0 = 20.223370630122005          # the real v0 of rollout start 0 on scene 00040136


# --------------------------------------------------------------------------- #
# synthetic logged paths                                                       #
# --------------------------------------------------------------------------- #
def straight_poses(n=20, v=V0, dt=DT):
    """Perfectly straight, perfectly constant speed => a0 == 0 and omega0 == 0."""
    T = []
    for i in range(n):
        M = np.eye(4)
        M[0, 3] = i * v * dt
        T.append(M)
    return T


def arc_poses(n=20, v=V0, a=0.0, kappa=0.0, dt=DT):
    """Unicycle-exact logged path with constant accel and constant curvature."""
    T, x, y, yaw, s = [], 0.0, 0.0, 0.0, v
    for i in range(n):
        M = np.eye(4)
        M[:3, :3] = _rz(yaw)
        M[0, 3], M[1, 3] = x, y
        T.append(M)
        x += s * math.cos(yaw) * dt
        y += s * math.sin(yaw) * dt
        yaw += s * kappa * dt
        s = max(0.0, s + a * dt)
    return T


def floor(mode, gt_T, f0=9):
    p = KinematicFloorPolicy(mode)
    p.set_t0(gt_T, f0)
    return p


# --------------------------------------------------------------------------- #
# 1 · the harness contract                                                     #
# --------------------------------------------------------------------------- #
def test_floor_arms_declare_the_same_window_as_every_model_arm():
    """`window` decides `f0 = start + window + STACK - 2`, i.e. WHERE the rollout
    begins. A floor with a different window is paired against nothing."""
    for m in FLOOR_ARMS:
        assert KinematicFloorPolicy(m).window == WINDOW == 8
    assert WINDOW + STACK - 2 == 9


def test_the_set_t0_hook_is_inert_for_model_policies():
    """`run_rollout` calls `set_t0` only when the policy has one. If a model arm
    ever grew the attribute, every historical rollout would change silently."""
    for cls in (FlagshipV1Policy, RefCPolicy, RefCV3Policy):
        assert not hasattr(cls, "set_t0")


def test_floor_arm_refuses_an_unknown_mode():
    with pytest.raises(ValueError):
        KinematicFloorPolicy("ha0")          # the OPEN-loop name, deliberately not ours


def test_plan_refuses_before_set_t0():
    """An un-initialised floor would invent its own t0 state."""
    with pytest.raises(RuntimeError):
        KinematicFloorPolicy("cl_ha").plan(None, None, V0, 0)


def test_set_t0_refuses_when_f0_is_too_early():
    """`a0` is a second difference of position: it needs `f0-2`."""
    with pytest.raises(RuntimeError):
        KinematicFloorPolicy("cl_ha").set_t0(straight_poses(), 1)


def test_floor_extra_carries_no_head_width_vector():
    """`cl_metrics.resolve_stamp` RAISES on an unrecognised width-4 or width-5 list
    in `extra` — the guard that caught a renamed route head. A floor arm must not
    trip it."""
    _, extra = floor("cl_ha0_ext", arc_poses(a=0.7, kappa=0.01)).plan(None, None, V0, 0)
    for k, v in extra.items():
        assert not isinstance(v, (list, tuple)), (k, v)


# --------------------------------------------------------------------------- #
# 2 · ⭐ THE CONTROL THAT MUST READ EXACTLY ZERO                                #
# --------------------------------------------------------------------------- #
def test_ha0_plan_is_the_exact_constant_velocity_line():
    traj, _ = floor("cl_ha0", arc_poses(a=1.3, kappa=0.02)).plan(None, None, V0, 0)
    want = np.stack([V0 * (np.asarray(WP_STEPS) * DT), np.zeros(4)], 1)
    assert traj.shape == (4, 2)
    assert np.array_equal(traj, want)              # BIT-identical, not allclose


def test_ha0_control_reads_exactly_zero():
    """⭐ THE FLOOR'S DEFINING CONTROL. Not `abs(...) < eps` — exactly 0.0."""
    traj, _ = floor("cl_ha0", arc_poses(a=1.3, kappa=0.02)).plan(None, None, V0, 0)
    steer, accel, v_target, kappa = wp_to_control(traj[LOOKAHEAD_IDX], V0)
    assert steer == 0.0
    assert accel == 0.0
    assert kappa == 0.0
    assert v_target == V0                          # exact: x = v*0.5, then /0.5


@pytest.mark.parametrize("v", [0.0, 0.5, 8.25, V0, 33.7])
def test_ha0_control_reads_exactly_zero_at_every_speed(v):
    traj, _ = floor("cl_ha0", arc_poses(a=-2.0, kappa=-0.03)).plan(None, None, v, 0)
    steer, accel, v_target, _ = wp_to_control(traj[LOOKAHEAD_IDX], v)
    assert (steer, accel, v_target) == (0.0, 0.0, v)


def test_ha0_from_a_stationary_start_stays_stationary():
    """⛔ REQUIRED CONTROL: a floor started at rest must not invent motion. The
    assertion goes all the way through the harness's OWN bicycle step, so it
    covers the plan, the controller and the integrator together."""
    traj, _ = floor("cl_ha0", straight_poses(v=0.0)).plan(None, None, 0.0, 0)
    assert np.array_equal(traj, np.zeros((4, 2)))
    steer, accel, v_target, _ = wp_to_control(traj[LOOKAHEAD_IDX], 0.0)
    assert (steer, accel, v_target) == (0.0, 0.0, 0.0)
    # the harness's bicycle step, verbatim (closedloop_drive.run_rollout)
    T = np.eye(4)
    T[:3, :3], T[0, 3], T[1, 3] = _rz(0.31), 12.0, -4.0
    v = 0.0
    for _ in range(50):
        dyaw = v / WHEELBASE * math.tan(steer) * DT
        D = np.eye(4)
        D[:3, :3] = _rz(dyaw)
        D[0, 3] = v * DT
        T = T @ D
        v = max(0.0, v + accel * DT)
    assert v == 0.0
    assert T[0, 3] == 12.0 and T[1, 3] == -4.0
    assert _yaw(T) == 0.31


def test_ha0_holds_its_speed_bit_exactly_over_a_full_rollout():
    """`v = max(0, v + accel*DT)` with `accel == 0.0` is the identity, so every
    tick of a `cl_ha0` rollout must report the SAME speed as tick 0."""
    p = floor("cl_ha0", arc_poses(a=0.9, kappa=0.011))
    v = V0
    for _ in range(50):
        traj, _ = p.plan(None, None, v, 0)
        steer, accel, _, _ = wp_to_control(traj[LOOKAHEAD_IDX], v)
        assert (steer, accel) == (0.0, 0.0)
        v = max(0.0, v + accel * DT)
    assert v == V0


# --------------------------------------------------------------------------- #
# 3 · nesting: the three arms must be a hierarchy, not three parameterisations  #
# --------------------------------------------------------------------------- #
def test_degenerate_arms_are_bit_identical():
    """⭐ On a straight, constant-speed log, `a0 == 0` and `omega0 == 0`, so all
    three floors ARE the same trivial control and must agree BIT-for-bit."""
    g = straight_poses()
    base, _ = floor("cl_ha0", g).plan(None, None, V0, 0)
    for m in ("cl_ha", "cl_ha0_ext"):
        p = floor(m, g)
        assert p.t0["a0_mps2"] == 0.0
        assert p.t0["omega0_rads"] == 0.0
        traj, _ = p.plan(None, None, V0, 0)
        assert np.array_equal(traj, base), m


def test_cl_ha_and_ext_separate_only_when_the_speed_changes():
    """They hold DIFFERENT invariants: `cl_ha` a constant curvature, `cl_ha0_ext`
    a constant yaw rate. At constant speed those coincide; under acceleration they
    must not."""
    same = arc_poses(a=0.0, kappa=0.02)
    a, _ = floor("cl_ha", same).plan(None, None, V0, 0)
    b, _ = floor("cl_ha0_ext", same).plan(None, None, V0, 0)
    assert np.allclose(a, b, atol=1e-9)
    diff = arc_poses(a=1.5, kappa=0.02)
    a2, _ = floor("cl_ha", diff).plan(None, None, V0, 0)
    b2, _ = floor("cl_ha0_ext", diff).plan(None, None, V0, 0)
    assert np.abs(a2 - b2).max() > 1e-3


# --------------------------------------------------------------------------- #
# 4 · the t0 measurement itself                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,kappa", [(0.0, 0.0), (1.5, 0.0), (0.0, 0.02),
                                     (-2.0, -0.013), (0.8, 0.031)])
def test_set_t0_recovers_the_generating_state(a, kappa):
    """Known-answer control: the log IS a unicycle rollout, so the backward
    differences must return the parameters that generated it."""
    p = floor("cl_ha", arc_poses(n=24, a=a, kappa=kappa))
    assert p.t0["a0_mps2"] == pytest.approx(a, abs=2e-6)
    v_at = V0 + a * (9 - 1) * DT
    assert p.t0["omega0_rads"] == pytest.approx(kappa * v_at, rel=2e-3, abs=1e-6)
    assert p.t0["kappa0_1pm"] == pytest.approx(kappa, rel=3e-3, abs=1e-6)


def test_set_t0_reads_no_future_pose():
    """⛔ THE NO-FUTURE-INFORMATION CONTROL. Poison every pose after `t0` with NaN;
    the answer must be unchanged. (A plausible number is not evidence — only the
    poisoning is.)"""
    f0 = 9
    clean = arc_poses(n=24, a=1.1, kappa=0.017)
    poisoned = [T.copy() for T in clean]
    for i in range(f0 + 1, len(poisoned)):
        poisoned[i][:] = np.nan
    a = floor("cl_ha0_ext", clean, f0).t0
    b = floor("cl_ha0_ext", poisoned, f0).t0
    assert a == b
    assert b["pose_indices_read"] == [f0 - 2, f0 - 1, f0]
    for m in FLOOR_ARMS:
        ta, _ = floor(m, clean, f0).plan(None, None, V0, 0)
        tb, _ = floor(m, poisoned, f0).plan(None, None, V0, 0)
        assert np.array_equal(ta, tb), m
        assert np.isfinite(tb).all()


# --------------------------------------------------------------------------- #
# 5 · the integrator is the programme's own                                     #
# --------------------------------------------------------------------------- #
def test_closed_form_branch_matches_the_programme_integrator():
    """The exact `a=0, kappa=0` branch must be the same physics as
    `kinematic.rollout_unicycle`, not a convenient shortcut beside it."""
    import torch
    from tanitad.models.kinematic import rollout_unicycle
    n = WP_STEPS[-1]
    st0 = torch.zeros(1, 4, dtype=torch.float64)
    st0[0, 3] = V0
    ref = rollout_unicycle(st0, torch.zeros(1, n, 2, dtype=torch.float64),
                           dt=DT)[0, :, :2].numpy()
    got = KinematicFloorPolicy._path(0.0, np.zeros(n), V0, n)
    assert np.abs(got - ref).max() < 1e-12
    assert np.abs(got - ref).max() > 0.0 or True      # equality is not required


def test_cl_ha0_ext_holds_the_yaw_rate_constant():
    """Its whole definition. Re-integrate its own controls and read the yaw back:
    `yaw(k) = omega0 * k * dt` for as long as the vehicle is moving."""
    import torch
    from tanitad.models.kinematic import rollout_unicycle
    p = floor("cl_ha0_ext", arc_poses(n=24, a=1.4, kappa=0.02))
    w, a = p.t0["omega0_rads"], p.t0["a0_mps2"]
    n = WP_STEPS[-1]
    vk, s = np.empty(n), V0
    for k in range(n):
        vk[k] = s
        s = max(0.0, s + a * DT)
    kap = w / vk
    st0 = torch.zeros(1, 4, dtype=torch.float64)
    st0[0, 3] = V0
    st = rollout_unicycle(st0, torch.from_numpy(
        np.stack([np.full(n, a), kap], 1))[None], dt=DT)[0].numpy()
    yaw = st[:, 2]
    assert np.allclose(yaw, w * np.arange(1, n + 1) * DT, atol=1e-12)


def test_cl_ha_holds_the_curvature_constant():
    """Its whole definition, and the converse of the test above: the yaw rate
    must TRACK the speed rather than stay put."""
    p = floor("cl_ha", arc_poses(n=24, a=2.0, kappa=0.03))
    path = KinematicFloorPolicy._path(
        p.t0["a0_mps2"], np.full(WP_STEPS[-1], p.t0["kappa0_1pm"]), V0, WP_STEPS[-1])
    d = np.diff(np.vstack([[0.0, 0.0], path]), axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    arc = np.linalg.norm(d, axis=1).cumsum()
    # dyaw/ds is the curvature and must be constant along the path
    k_est = np.diff(yaw) / np.diff(arc)
    assert np.std(k_est) < 1e-3 * abs(p.t0["kappa0_1pm"]) + 1e-6


# --------------------------------------------------------------------------- #
# 6 · the controller distortion, measured rather than assumed                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a0", [2.0, -1.5, 0.75])
def test_the_p_speed_controller_executes_40_percent_of_the_intended_acceleration(a0):
    """⚠️ NOT a bug and NOT corrected, but MEASURED rather than assumed — I first
    wrote `a/2` here from the continuous-time integral and the test caught it.

    `wp_to_control` sets `v_target = x / (L*dt)`, the AVERAGE speed over the
    lookahead, and the integrator advances position on the speed at the START of
    each step. So `x_L = L*v*dt + a*dt^2*L(L-1)/2`, hence
    `v_target = v + a*dt*(L-1)/2` and

        accel_executed = a * dt*(L-1)/2 / speed_tc = a * 0.1*2/0.5 = 0.4 * a

    Every model arm is distorted identically — which is exactly why the floor must
    go through the controller rather than inject controls behind it."""
    p = floor("cl_ha", arc_poses(n=24, a=a0, kappa=0.0))
    traj, _ = p.plan(None, None, V0, 0)
    _, accel, _, _ = wp_to_control(traj[LOOKAHEAD_IDX], V0)
    L = WP_STEPS[LOOKAHEAD_IDX]
    want = p.t0["a0_mps2"] * DT * (L - 1) / 2.0 / 0.5
    assert accel == pytest.approx(want, rel=1e-6, abs=1e-9)
    assert accel == pytest.approx(0.4 * p.t0["a0_mps2"], rel=1e-6, abs=1e-9)
