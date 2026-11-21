"""Pins for a0k0_headroom_r8.py: analytic fit targets (literals), w = 1 / w = 0 identities, cross-fit leak detection with a DELIBERATE
REGRESSION arm (a leaky cross-fit must be caught), the ha0_imu integrator against analytic targets and against the library unicycle in
the a0 = 0 limit, and the real deliverable's controls (skipped only if A0K0_HEADROOM.json is absent).

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -m pytest -q test_a0k0_headroom_r8.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a0k0_headroom_r8 as M  # noqa: E402

OUT_JSON = HERE.parent / "raw" / "a0k0_headroom" / "A0K0_HEADROOM.json"


def _ob(n=3000, T=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T, 2)).astype(np.float32), rng.normal(size=(n, T, 2)).astype(np.float32)


# ------------------------------------------------------------------ analytic targets (expectations are LITERALS)
@pytest.mark.parametrize("tgt,box,want", [(0.3, (0.0, 1.0), 0.3), (0.3, (0.0, 2.0), 0.3), (1.5, (0.0, 2.0), 1.5), (1.5, (0.0, 1.0), 1.0),
                                          (1.0, (0.0, 2.0), 1.0), (0.0, (0.0, 2.0), 0.0)])
def test_fit_hits_analytic_target(tgt, box, want):
    O, B = _ob()
    G = tgt * O.astype(np.float64) + (1 - tgt) * B.astype(np.float64)
    for per_axis in (True, False):
        W = M.fit_gains(O, B, G, box=box, per_axis=per_axis)
        assert np.allclose(W, want, atol=1e-4), (per_axis, W)


def test_fit_separates_the_axes():
    O, B = _ob()
    G = np.empty(O.shape)
    G[..., 0] = 0.2 * O[..., 0] + 0.8 * B[..., 0]
    G[..., 1] = 0.8 * O[..., 1] + 0.2 * B[..., 1]
    W = M.fit_gains(O, B, G, box=(0.0, 2.0))
    assert np.allclose(W[:, 0], 0.2, atol=1e-4) and np.allclose(W[:, 1], 0.8, atol=1e-4)


def test_blend_identities_are_bit_exact():
    O, B = _ob(seed=3)
    O *= 37.0
    B *= 11.0
    assert np.array_equal(M.blend(O, B, np.ones((4, 2))), O)
    assert np.array_equal(M.blend(O, B, np.zeros((4, 2))), B)
    assert M.blend(O, B, np.ones((4, 2))).dtype == np.float32


# ------------------------------------------------------------------ cross-fitting: no leak, and the DELIBERATE REGRESSION is caught
def _episodic(n_ep=20, per=40, seed=1):
    rng = np.random.default_rng(seed)
    clip = np.repeat(np.arange(n_ep), per)
    O, B = _ob(n=n_ep * per, seed=seed)
    w_ep = rng.uniform(0.2, 0.9, size=n_ep)[clip][:, None, None]       # episode-level truth -> fold weights genuinely differ
    G = w_ep * O + (1 - w_ep) * B + 0.05 * rng.normal(size=O.shape)
    return O, B, G, clip


def _held_out_weights(O, B, G, clip, leak):
    folds = M.episode_folds(clip)
    f0 = folds == 0
    W1, _ = M.crossfit(O, B, G, folds, leak=leak)
    G2 = G.copy()
    G2[f0] = 50.0 * O[f0] - 49.0 * B[f0]                                  # wreck fold 0's own GT
    W2, _ = M.crossfit(O, B, G2, folds, leak=leak)
    return float(np.abs(W1[f0] - W2[f0]).max())


def test_crossfit_never_sees_the_held_out_fold():
    O, B, G, clip = _episodic()
    assert _held_out_weights(O, B, G, clip, leak=False) == 0.0


def test_deliberate_regression_leaky_crossfit_is_detected():
    """MUTATION: the held-out fold inside its own fit.  The same detector must go RED (its weights move)."""
    O, B, G, clip = _episodic()
    assert _held_out_weights(O, B, G, clip, leak=True) > 0.1


def test_insample_scores_better_than_crossfit_on_synthetic():
    O, B, G, clip = _episodic()
    Wx, _ = M.crossfit(O, B, G, M.episode_folds(clip))
    Wi, _ = M.insample(O, B, G)
    assert M.ade_np(M.blend(O, B, Wi), G).mean() < M.ade_np(M.blend(O, B, Wx), G).mean()


def test_episode_folds_are_episode_pure_and_balanced():
    clip = np.repeat(np.arange(139), 3)
    f = M.episode_folds(clip)
    for c in np.unique(clip):
        assert len(set(f[clip == c].tolist())) == 1
    counts = np.bincount([f[clip == c][0] for c in np.unique(clip)])
    assert counts.tolist() == [28, 28, 28, 28, 27]


# ------------------------------------------------------------------ ha0_imu integrator
def test_imu_straight_constant_speed():
    P = M.imu_path(np.zeros(3), np.zeros(3), np.array([0.0, 5.0, 20.0]), [5, 10, 20]).numpy()
    assert np.allclose(P[..., 0], np.array([0.0, 5.0, 20.0])[:, None] * np.array([0.5, 1.0, 2.0])[None], atol=1e-4)
    assert np.abs(P[..., 1]).max() == 0.0


def test_imu_constant_yaw_rate_matches_discrete_closed_form():
    v0, psi, dt = 10.0, 0.2, 0.1
    P = M.imu_path(np.zeros(1), np.full(1, psi), np.full(1, v0), [20, 60]).numpy()[0]
    for j, k in enumerate((20, 60)):
        xs = sum(v0 * dt * math.cos(i * psi * dt) for i in range(k))
        ys = sum(v0 * dt * math.sin(i * psi * dt) for i in range(k))
        assert abs(P[j, 0] - xs) < 1e-3 and abs(P[j, 1] - ys) < 1e-3


def test_imu_speed_floors_at_zero():
    P = M.imu_path(np.full(1, -10.0), np.zeros(1), np.full(1, 1.0), [1, 2, 10, 20]).numpy()[0]
    assert abs(P[0, 0] - 0.1) < 1e-6 and np.allclose(P[1:, 0], 0.1, atol=1e-6)


def test_imu_equals_library_unicycle_in_the_a0_zero_limit():
    torch = pytest.importorskip("torch")
    from tanitad.models import kinematic_prior as KP
    rng = np.random.default_rng(5)
    v0 = torch.from_numpy(rng.uniform(0, 30, 200).astype(np.float32))
    k0 = torch.from_numpy(rng.uniform(-0.1, 0.1, 200).astype(np.float32))
    z = torch.zeros_like(v0)
    a = M.imu_path(z, k0 * v0, v0, [5, 10, 20, 60]).numpy()
    b = KP.prior_path(z, k0, v0, [5, 10, 20, 60]).numpy()
    assert np.abs(a - b).max() <= 1e-4


# ------------------------------------------------------------------ the deliverable
def _cells(r):
    for tree, tr in r["trees"].items():
        for sd, s in tr["seeds"].items():
            for surf, ss in s["surfaces"].items():
                for st, x in ss["sets"].items():
                    yield (tree, sd, surf, st), ss, x


@pytest.fixture(scope="module")
def R():
    if not OUT_JSON.exists():
        pytest.skip("A0K0_HEADROOM.json not written yet")
    return json.loads(OUT_JSON.read_text(encoding="utf-8"))


def test_deliverable_controls_all_pass(R):
    assert R["controls_summary"]["ALL_PASS"] is True
    assert set(R["trees"]) == {"step30000ema_unc", "step30000_unc"}
    n = 0
    for key, ss, x in _cells(R):
        n += 1
        a = x["controls"]["a_g1_reproduces_banked_os"]
        assert a["PASS"] and a["n_exact"] == a["n_rows"] == 15, key
        assert x["controls"]["b_g0_is_prior"]["PASS"], key
        assert ss["validation"]["V1_ha0_ext_rebuilt_vs_panel_max_abs_m"] <= 1e-3, key
        assert ss["validation"]["V2_imu_with_psidot_kappa0v0_a0zero_vs_unicycle_kappa0_max_abs_m"] <= 1e-4, key
        assert ss["decisions_check"]["PASS"], key
    assert n == 16


def test_deliverable_reproduces_the_2026_09_26_prior_measurement(R):
    assert R["controls_summary"]["R0926_exact"] is True
    c = R["controls_summary"]["R0926_S2_all_ha0_ext_pose_minus_ha0_ext"]
    assert (c["delta"], c["lo"], c["hi"], c["n_windows"], c["n_episodes"]) == (0.0055, 0.0036, 0.0076, 4754, 139)


def test_deliverable_compliant_e1_is_the_banked_literal(R):
    x = R["trees"]["step30000ema_unc"]["seeds"]["0"]["surfaces"]["S2"]["sets"]["compliant"]["refs"]["E1_registered_os_minus_ha0_ext"]
    assert (x["delta"], x["lo"], x["hi"], x["n_windows"], x["n_episodes"]) == (-0.0085, -0.0229, 0.0065, 4159, 138)


def test_deliverable_in_sample_beats_cross_fit_somewhere_and_is_sign_bounded(R):
    """Non-cross-fitted weights must score better than cross-fitted on >= 1 cell (the optimism the cross-fit removes is real), and the
    in-sample arm can never be worse than either endpoint it contains (g = 0 and g = 1 are feasible)."""
    lt = [k for k, _ss, x in _cells(R) if x["controls"]["c_d_optimism"]["IS_mean_ade_lt_XF_mean_ade"]]
    assert lt, "in-sample never beat cross-fitted: the cross-fit is not removing anything -- suspect a leak"
    for k, _ss, x in _cells(R):
        assert x["controls"]["c_d_optimism"]["IS_known_sign_le_prior_and_os"], k


# ------------------------------------------------------------------ summary direction (the defect the first summary had)
def _fake_tree(PR, sign):
    cellv = {"delta": 0.5 * sign, "lo": 0.1 * sign if sign > 0 else -0.9, "hi": 0.9 if sign > 0 else -0.1, "n_windows": 10, "n_episodes": 5}
    keys = [m[1] for m in PR.METRICS]
    arm = {"mean_ade_m": 1.0, "minus_ha0_ext": cellv, "minus_ha0_ext_pose": cellv,
           "NR_minus_refcv7": {k: cellv for k in keys}, "minus_os": {k: cellv for k in keys}}
    st = {"ade_means": {"os": 1.0}, "refs": {"E1_registered_os_minus_ha0_ext": cellv, "os_minus_ha0_ext_pose": {"ade_m": cellv},
                                              "os_minus_ha0_imu": cellv}, "arms": {"RS_XF": arm, "GATED_RS_XF": arm}}
    return {"seeds": {s: {"surfaces": {"S2": {"sets": {"compliant": st}}}} for s in ("0", "1")}}


def test_summary_is_direction_aware_for_accuracy_rows():
    """Every cell separated POSITIVE: error rows are WORSE, the two accuracy rows (lower_is_better False) are BETTER.  The first summary's
    rule (bar_NR_metric on every row) would have listed the accuracy rows as WORSE -- this pin goes RED on that mutation."""
    import pair_read_r8 as PR
    s = M.summarise(PR, _fake_tree(PR, +1), ("S2",), ("compliant",), (0, 1))["S2|compliant"]["RS_XF"]
    acc = {f"{k} ({fam})" for fam, k, _l, _u, lower, *_r in PR.METRICS if not lower}
    assert acc == {"LON_speed_within_0p5 (LONGITUDINAL)", "TAC_traj_lat_correct (TACTICAL)", "TAC_traj_lon_correct (TACTICAL)"}
    assert acc <= set(s["vs_os_separated_better_both_seeds_all_rows"])
    assert not (acc & set(s["vs_os_separated_worse_both_seeds_all_rows"]))
    assert "ade_m (ADE)" in s["vs_os_separated_worse_both_seeds_all_rows"]
    # the NR rollup reads only the S2 headline rows, which are all error metrics
    assert s["NR_rollup_headline_rows"]["ADE"].startswith("FAIL")
