"""Known-value tests for ``taniteval/floors_kin.py`` (H-V8F-FLOOR-1, stream S1).

Every test is an IDENTITY or a value known in advance, because the floor is what the programme's drivers will be
read against: ``X - X`` is exactly 0; a constant-velocity input IS ``ha0``; ``damp50`` IS ``0.5*ha0 + 0.5*ha0_ext``;
the vectorised ``retime`` IS the original scalar ``build_explore.py::retime`` (a verbatim copy lives below as the
reference); the integrator IS ``kinematic.rollout_unicycle``.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from taniteval import floors_kin as fk                                    # noqa: E402


# ---- verbatim copy of build_explore.py::arclen / retime (the reference) -------------------------------------- #
def _arclen_ref(p):
    P = np.concatenate([np.zeros((1, 2), p.dtype), p], 0)
    seg = np.diff(P, axis=0)
    L = np.linalg.norm(seg, axis=-1)
    return P, seg, L, np.concatenate([[0.0], np.cumsum(L)])


def _retime_ref(path, lon_src):
    P, seg, L, S = _arclen_ref(path.astype(np.float64))
    s_src = _arclen_ref(lon_src.astype(np.float64))[3][1:]
    out = np.zeros((len(s_src), 2))
    nz = np.nonzero(L > 1e-9)[0]
    for i, s in enumerate(s_src):
        if S[-1] < 1e-6 or len(nz) == 0:
            out[i] = (s, 0.0)
            continue
        if s <= S[-1]:
            j = int(np.clip(np.searchsorted(S, s, side="right") - 1, 0, len(L) - 1))
            while L[j] <= 1e-9 and j + 1 < len(L):
                j += 1
            out[i] = P[j] + (s - S[j]) / max(L[j], 1e-12) * seg[j] if L[j] > 1e-9 else P[j]
        else:
            jj = nz[-1]
            out[i] = P[-1] + (s - S[-1]) * seg[jj] / L[jj]
    return out.astype(np.float32)


def _windows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    v0 = rng.uniform(0.0, 20.0, n)
    a0 = rng.uniform(-4.0, 3.0, n)
    st = rng.uniform(-0.15, 0.15, n)
    return np.stack([v0, a0, st], 1)


# ---- identities ----------------------------------------------------------------------------------------------- #
def test_x_minus_x_is_exactly_zero():
    from taniteval.ci import paired_episode_cluster_bootstrap as pb
    kin = _windows()
    g = fk.ha0_paths(kin[:, 0], 0.5, 4) + 0.3
    p = fk.kd_x(fk.ha0_paths(kin[:, 0], 0.1, 20), fk.ha0_ext_paths(kin, 0.1, 20, action_units="steer"))[:, 4::5]
    a = fk.ade(p, g)
    r = pb(a, a, np.repeat(np.arange(8), 5), n_boot=200)
    assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0


def test_constant_velocity_input_reproduces_ha0_exactly():
    kin = _windows()
    kin[:, 1] = 0.0
    kin[:, 2] = 0.0                                       # a0 = 0, k0 = 0
    for units in ("kappa", "steer"):
        ext = fk.ha0_ext_paths(kin, 0.1, 30, action_units=units)
        h0 = fk.ha0_paths(kin[:, 0], 0.1, 30)
        assert np.array_equal(ext, h0)
        assert np.array_equal(fk.damp50(h0, ext), h0)
    # and ha0 is a straight, constant-speed line: y == 0, x = v0 * t
    assert np.all(h0[..., 1] == 0.0)
    np.testing.assert_allclose(h0[..., 0], kin[:, :1] * 0.1 * np.arange(1, 31)[None], rtol=1e-12)


def test_kd_x_of_constant_velocity_is_ha0():
    kin = _windows()
    kin[:, 1:] = 0.0
    h0 = fk.ha0_paths(kin[:, 0], 0.1, 30)
    out = fk.kd_x(h0, fk.ha0_ext_paths(kin, 0.1, 30))
    np.testing.assert_allclose(out, h0.astype(np.float32), atol=2e-5)


def test_damp50_is_half_ha0_plus_half_ha0_ext_exactly():
    kin = _windows()
    h0 = fk.ha0_paths(kin[:, 0], 0.5, 4)
    he = fk.ha0_ext_paths(kin, 0.5, 4, action_units="steer")
    assert np.array_equal(fk.damp50(h0, he), 0.5 * h0 + 0.5 * he)
    # synthetic hand case: v0 = 10, a0 = -2 m/s^2, straight, dt = 1 s: ha0 x = 10, 20 ; ha0_ext x = 10, 18 (start-of-step speed)
    k = np.array([[10.0, -2.0, 0.0]])
    h0 = fk.ha0_paths(k[:, 0], 1.0, 2)
    he = fk.ha0_ext_paths(k, 1.0, 2)
    np.testing.assert_allclose(he[0, :, 0], [10.0, 18.0])
    np.testing.assert_allclose(fk.damp50(h0, he)[0, :, 0], [10.0, 19.0])


def test_retime_vectorised_equals_original_scalar_retime():
    rng = np.random.default_rng(1)
    for K in (4, 20, 30):
        kin = _windows(60, seed=K)
        kin[:7, 0] = 0.0                                  # standstill windows (zero-length segments)
        kin[7:12, 0] = rng.uniform(0, 0.3, 5)             # creeping
        h0 = fk.ha0_paths(kin[:, 0], 0.2, K)
        he = fk.ha0_ext_paths(kin, 0.2, K, action_units="kappa")
        damp = fk.damp50(h0, he)
        fast = fk.retime(damp, he, chunk=16)
        ref = np.stack([_retime_ref(damp[i], he[i]) for i in range(len(damp))])
        np.testing.assert_allclose(fast, ref, atol=1e-5)
    # beyond-the-end extrapolation + a degenerate (zero-length) damped path
    path = np.array([[[1.0, 0.0], [2.0, 0.0], [2.0, 0.0]], [[0.0, 0.0]] * 3])
    src = np.array([[[3.0, 0.0], [6.0, 0.0], [9.0, 0.0]], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    ref = np.stack([_retime_ref(path[i], src[i]) for i in range(2)])
    np.testing.assert_allclose(fk.retime(path, src), ref, atol=1e-6)
    np.testing.assert_allclose(fk.retime(path[1], src[1]), [[1, 0], [2, 0], [3, 0]], atol=1e-6)   # single-window API


def test_kd_x_self_retime_reproduces_damp50():
    """Control (build_explore.py ``kd_self``): re-timing the damped path by ITSELF must return damp50."""
    kin = _windows(50)
    h0 = fk.ha0_paths(kin[:, 0], 0.1, 30)
    he = fk.ha0_ext_paths(kin, 0.1, 30, action_units="steer")
    damp = fk.damp50(h0, he)
    np.testing.assert_allclose(fk.retime(damp, damp), damp.astype(np.float32), atol=2e-4)


def test_kd_x_longitudinal_profile_is_ha0_exts_and_geometry_is_damp50s():
    kin = _windows(30, seed=5)
    h0 = fk.ha0_paths(kin[:, 0], 0.1, 30)
    he = fk.ha0_ext_paths(kin, 0.1, 30, action_units="steer")
    out = fk.kd_x(h0, he).astype(np.float64)
    s_out = np.concatenate([np.zeros((30, 1)), np.cumsum(np.linalg.norm(np.diff(
        np.concatenate([np.zeros((30, 1, 2)), out], 1), axis=1), axis=-1), axis=1)], 1)[:, 1:]
    s_ext = np.concatenate([np.zeros((30, 1)), np.cumsum(np.linalg.norm(np.diff(
        np.concatenate([np.zeros((30, 1, 2)), he], 1), axis=1), axis=-1), axis=1)], 1)[:, 1:]
    np.testing.assert_allclose(s_out, s_ext, rtol=2e-3, atol=2e-3)


# ---- the integrator and the GT agree with the programme's own ---------------------------------------------------- #
def test_integrator_matches_rollout_unicycle():
    torch = pytest.importorskip("torch")
    try:
        from tanitad.refs.refa_v1_plan import unicycle_paths
    except Exception:                                      # pragma: no cover
        pytest.skip("tanitad not importable")
    kin = _windows(25, seed=3)
    for units in ("steer", "kappa"):
        mine = fk.ha0_ext_paths(kin, 0.1, 40, action_units=units)
        ctr = torch.tensor(kin[:, 1:], dtype=torch.float32)[:, None, :].expand(-1, 40, 2).contiguous()
        theirs = unicycle_paths(ctr, torch.tensor(kin[:, 0], dtype=torch.float32), 0.1, action_units=units).numpy()
        np.testing.assert_allclose(mine, theirs, atol=2e-4)


def test_gt_waypoints_matches_metric_dynamics():
    torch = pytest.importorskip("torch")
    try:
        from tanitad.models.metric_dynamics import gt_ego_waypoints
    except Exception:                                      # pragma: no cover
        pytest.skip("tanitad not importable")
    rng = np.random.default_rng(0)
    T = 80
    yaw = np.cumsum(rng.normal(0, 0.02, T))
    xy = np.cumsum(np.stack([np.cos(yaw), np.sin(yaw)], 1) * 1.0, 0)
    poses = np.concatenate([xy, yaw[:, None], np.full((T, 1), 10.0)], 1).astype(np.float32)
    t0 = np.array([5, 17, 30])
    mine = fk.gt_waypoints(poses, t0, [5, 10, 20])
    for i, t in enumerate(t0):
        ref = gt_ego_waypoints(torch.tensor(poses[t])[None], torch.tensor(poses[t + 1:t + 21])[None], [5, 10, 20])[0].numpy()
        np.testing.assert_allclose(mine[i], ref, atol=1e-4)


# ---- features ----------------------------------------------------------------------------------------------------- #
def test_kin_features_known_value():
    f = fk.kin_features(np.array([[2.0, 3.0, 5.0]]))
    np.testing.assert_array_equal(f[0], [2.0, 3.0, 5.0, 4.0, 6.0, 10.0, 20.0, 15.0])


def test_t0_kinematics_constant_acceleration_is_exact():
    dt, a = 0.1, 1.5
    v = 4.0 + a * dt * np.arange(50)
    k = fk.t0_kinematics(v, np.full(50, 0.03), [7, 20], stride=1, dt=dt)
    np.testing.assert_allclose(k[:, 1], a, atol=1e-9)
    np.testing.assert_allclose(k[:, 0], v[[7, 20]])
    np.testing.assert_allclose(k[:, 2], 0.03)
    k2 = fk.t0_kinematics(v, np.zeros(50), [8], stride=2, dt=0.2)
    np.testing.assert_allclose(k2[:, 1], a, atol=1e-9)
    with pytest.raises(ValueError):
        fk.t0_kinematics(v, np.zeros(50), [0], stride=1, dt=dt)       # no closed step before t0


def test_nothing_after_t0_enters_the_features():
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 20, 60)
    s = rng.uniform(-0.1, 0.1, 60)
    a = fk.t0_kinematics(v, s, [25], stride=1, dt=0.1)
    v2, s2 = v.copy(), s.copy()
    v2[26:] += 100.0
    s2[26:] += 1.0
    assert np.array_equal(a, fk.t0_kinematics(v2, s2, [25], stride=1, dt=0.1))


# ---- acts check ---------------------------------------------------------------------------------------------------- #
def test_acts_check_constant_velocity_vs_accelerating():
    v0 = np.full(6, 10.0)
    cv = fk.acts_check(fk.ha0_paths(v0, 0.5, 4), 0.5)
    assert cv["mean_abs_a_mps2"] < 1e-9 and cv["zero_accel_frac"] == 1.0
    acc = fk.ha0_ext_paths(np.stack([v0, np.full(6, 2.0), np.zeros(6)], 1), 0.5, 4)
    r = fk.acts_check(acc, 0.5)
    np.testing.assert_allclose(r["mean_abs_a_mps2"], 2.0, rtol=1e-6)
    assert r["zero_accel_frac"] == 0.0


# ---- P10 ----------------------------------------------------------------------------------------------------------- #
def _synthetic(n_ep=24, per=20, seed=0):
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(n_ep), per)
    kin = np.stack([rng.uniform(0, 20, n_ep * per), rng.uniform(-3, 3, n_ep * per), rng.uniform(-0.1, 0.1, n_ep * per)], 1)
    base = fk.ha0_paths(kin[:, 0], 0.5, 4).astype(np.float32)
    # a learnable, purely kinematic residual
    resid = np.stack([np.stack([0.4 * kin[:, 1] * (j + 1), 30.0 * kin[:, 2] * (j + 1)], -1) for j in range(4)], 1)
    return eid, kin, base, base.astype(np.float64) + resid


def test_p10_learns_a_kinematic_residual_and_is_deterministic():
    eid, kin, base, g = _synthetic()
    kf = fk.kin_features(kin)
    y = (g - base).reshape(len(g), -1)
    tr, te = eid < 18, eid >= 18
    p1, best1 = fk.fit_predict_p10(kf[tr], y[tr], base[tr], g[tr], eid[tr], kf[te], base[te], epochs=40)
    p2, best2 = fk.fit_predict_p10(kf[tr], y[tr], base[tr], g[tr], eid[tr], kf[te], base[te], epochs=40)
    assert np.array_equal(p1, p2) and best1 == best2                      # same RNG draws -> same fit
    assert fk.ade(p1, g[te]).mean() < 0.5 * fk.ade(base[te], g[te]).mean()  # it learned
    assert best1[0] == 0 and best1[1] in (1e-4, 1e-2)


def test_p10_training_seed_default_is_zero_and_replicates_differ():
    eid, kin, base, g = _synthetic()
    kf = fk.kin_features(kin)
    y = (g - base).reshape(len(g), -1)
    tr, te = eid < 18, eid >= 18
    a, _ = fk.fit_predict_p10(kf[tr], y[tr], base[tr], g[tr], eid[tr], kf[te], base[te], epochs=20)
    b, _ = fk.fit_predict_p10(kf[tr], y[tr], base[tr], g[tr], eid[tr], kf[te], base[te], epochs=20, seed=0)
    c, _ = fk.fit_predict_p10(kf[tr], y[tr], base[tr], g[tr], eid[tr], kf[te], base[te], epochs=20, seed=1)
    assert np.array_equal(a, b)                                           # default == the banked fit_mlp_kin seed
    assert not np.array_equal(a, c)                                       # a replicate is a different draw


def test_p10_starts_at_the_prior():
    eid, kin, base, g = _synthetic(n_ep=8)
    torch = pytest.importorskip("torch")
    net = fk._make_mlp(8, 8)
    out = net(torch.randn(5, 8))
    assert float(out.detach().abs().max()) == 0.0                         # zero-init output layer: prediction == base


def test_grouped_folds_never_split_an_episode():
    eid = np.repeat(np.arange(23), 7)
    f = fk.grouped_folds(eid, 5)
    for e in range(23):
        assert len(set(f[eid == e])) == 1
    assert set(f) == set(range(5))


def test_late_fusion_endpoints_and_alpha_fit():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(30, 4, 2))
    os_ = g.copy()
    p10 = g + 1.0
    assert np.array_equal(fk.late_fusion(os_, p10, 1.0), os_)
    assert np.array_equal(fk.late_fusion(os_, p10, 0.0), p10)
    assert fk.fit_alpha(os_, p10, g) == 1.0                                # os is perfect -> alpha = 1
    assert fk.fit_alpha(p10, os_, g) == 0.0
