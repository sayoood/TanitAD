"""Analytic tests for the TanitEval diagnostic panel (bench.py): kinematic FLOOR,
ego-status / latent CEILING, both-convention L2, and skill_score.

Mirrors the style of stack/tests/test_driving_diagnostic.py: synthetic inputs
with hand-known answers, CPU-only, no checkpoint. pytest is NOT installed on the
eval pod, so these run standalone (``python tests/test_bench_diagnostic.py``);
they are also plain ``test_*`` functions so pytest can collect them if present.

Run:  PYTHONPATH=/root/taniteval:/root/TanitAD/stack python tests/test_bench_diagnostic.py
"""
import math
import sys

import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import bench  # noqa: E402
from driving_diagnostic import WP_STEPS  # noqa: E402

N_WP = len(WP_STEPS)


def _approx(a, b, tol=1e-6):
    assert abs(float(a) - float(b)) <= tol, f"{a} != {b} (tol {tol})"


def _straight_poses(T=40, theta=0.3, s=2.0):
    """Constant-velocity straight line at heading theta, speed s/step. [T,4]."""
    t = torch.arange(T, dtype=torch.float64)
    d = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float64)
    xy = t[:, None] * s * d[None, :]
    yaw = torch.full((T,), theta, dtype=torch.float64)
    v = torch.full((T,), s, dtype=torch.float64)
    return torch.cat([xy, yaw[:, None], v[:, None]], dim=1)


# --------------------------------------------------------------------------- #
# ego_status_features — perception-free kinematic vector                       #
# --------------------------------------------------------------------------- #
def test_ego_status_on_straight_constant_velocity():
    """On a straight constant-velocity path (any heading), every ego-frame step
    displacement is [s, 0]; v0=s, a0=omega=alpha=0 — exact."""
    for theta in (0.0, 0.3, -1.1):
        poses = _straight_poses(theta=theta, s=2.5)
        last = torch.tensor([10, 20, 30])
        f = bench.ego_status_features(poses, last)
        assert f.shape == (3, 2 * bench.EGO_HIST + 4)
        # the EGO_HIST displacement pairs are all (s, 0)
        for j in range(bench.EGO_HIST):
            _approx(f[:, 2 * j].mean(), 2.5, 1e-6)       # forward = s
            _approx(f[:, 2 * j + 1].abs().max(), 0.0, 1e-6)  # lateral = 0
        _approx(f[:, -4].mean(), 2.5, 1e-6)              # v0 = s
        _approx(f[:, -3].abs().max(), 0.0, 1e-6)         # a0 = 0
        _approx(f[:, -2].abs().max(), 0.0, 1e-6)         # omega0 = 0
        _approx(f[:, -1].abs().max(), 0.0, 1e-6)         # alpha0 = 0


# --------------------------------------------------------------------------- #
# both-convention open-loop L2                                                  #
# --------------------------------------------------------------------------- #
def test_l2_conventions_known():
    de = torch.tensor([[1.0, 2.0, 3.0, 4.0]])           # one window, 4 waypoints
    m = bench._l2_conventions(de)
    _approx(m["cumulative_avg_0_2s"], 2.5)              # mean(1,2,3,4)
    _approx(m["endpoint_avg"], 2.5)                     # mean of the 4 endpoints
    _approx(m["l2@1s"], 2.0)                            # de at waypoint index 1
    _approx(m["l2@2s"], 4.0)                            # de at waypoint index 3
    _approx(m["ade@1s"], 1.5)                           # mean(1,2)
    _approx(m["ade@2s"], 2.5)


def test_l2_endpoint_vs_cumulative_diverge():
    """Two windows where the conventions differ at an intermediate horizon."""
    de = torch.tensor([[0.0, 0.0, 0.0, 8.0], [0.0, 4.0, 0.0, 0.0]])
    m = bench._l2_conventions(de)
    # endpoint = mean over the 4 per-horizon means = mean(0, 2, 0, 4) = 1.5
    _approx(m["endpoint_avg"], 1.5)
    _approx(m["cumulative_avg_0_2s"], 1.5)              # equals at the 2 s tail
    _approx(m["l2@1s"], 2.0)                            # mean(0,4) at index 1


# --------------------------------------------------------------------------- #
# kinematic floor (best-of-3)                                                   #
# --------------------------------------------------------------------------- #
def _const_wp(n, x):
    """[n,4,2] with every point at (x, 0)."""
    w = torch.zeros(n, N_WP, 2)
    w[..., 0] = x
    return w


def _floor_win(n=30):
    """gt at origin; cv/gs/cyr with known constant errors 2/3/5 (cv is best)."""
    return {
        "gt": torch.zeros(n, N_WP, 2),
        "constant_velocity": _const_wp(n, 2.0),
        "go_straight": _const_wp(n, 3.0),
        "constant_yaw_rate": _const_wp(n, 5.0),
        "cv": _const_wp(n, 2.0),
        "speed": torch.ones(n),
        "head_deg": torch.zeros(n),                     # all 'straight'
        "eid": [i // 3 for i in range(n)],
    }


def test_kinematic_floor_best_of_3():
    f = bench.kinematic_floor(_floor_win())
    _approx(f["per_baseline_ade_0_2s"]["constant_velocity"], 2.0)
    _approx(f["per_baseline_ade_0_2s"]["go_straight"], 3.0)
    _approx(f["per_baseline_ade_0_2s"]["constant_yaw_rate"], 5.0)
    _approx(f["best_of_3_ade_0_2s"], 2.0)               # per-window min == cv
    assert f["which_baseline_wins"]["constant_velocity"] == 30
    assert f["which_baseline_wins"]["go_straight"] == 0
    assert f["by_curvature"]["straight"]["n"] == 30


def test_floor_per_window_min_beats_any_single():
    """Per-window best-of-3 can beat every single baseline's mean when the best
    predictor differs window-to-window."""
    n = 20
    gt = torch.zeros(n, N_WP, 2)
    cv = _const_wp(n, 1.0)
    cyr = _const_wp(n, 1.0)
    cv[10:, :, 0] = 5.0    # cv great on first half (err 1), awful on second (5)
    cyr[:10, :, 0] = 5.0   # cyr the reverse
    win = {"gt": gt, "constant_velocity": cv, "go_straight": _const_wp(n, 9.0),
           "constant_yaw_rate": cyr, "cv": cv, "speed": torch.ones(n),
           "head_deg": torch.zeros(n), "eid": list(range(n))}
    f = bench.kinematic_floor(win)
    _approx(f["per_baseline_ade_0_2s"]["constant_velocity"], 3.0)   # mean(1,5)
    _approx(f["per_baseline_ade_0_2s"]["constant_yaw_rate"], 3.0)
    _approx(f["best_of_3_ade_0_2s"], 1.0)              # every window has a 1.0


# --------------------------------------------------------------------------- #
# skill_score — known ratio on a straight stratum                              #
# --------------------------------------------------------------------------- #
def test_skill_score_known_ratio_on_straights():
    """model error 4, kinematic floor 2 (cv) => skill_vs_floor == 2.0, and the
    straight-stratum near-trivial flag fires (2 <= 3)."""
    n = 30
    win = _floor_win(n)
    win["pred"] = _const_wp(n, 4.0)                     # model error 4
    win["ego_status"] = torch.empty(n, 0)              # no ego ceiling
    win["states"] = torch.empty(n, 0)                  # no latent ceiling
    win["wp_steps"] = list(WP_STEPS)
    d = bench.diagnostic(win)
    _approx(d["model_ade_0_2s"], 4.0)
    _approx(d["kinematic_floor"]["best_of_3_ade_0_2s"], 2.0)
    straight = d["skill_score"]["by_curvature"]["straight"]
    _approx(straight["skill_vs_floor"], 2.0, 1e-3)
    assert d["ego_status_ceiling"] is None
    assert d["latent_ceiling"] is None
    fal = d["falsifiers"]["skill_on_straights"]
    assert fal["near_trivial_competitive"] is True     # 2.0 <= 3.0


def test_skill_zero_when_model_is_perfect():
    n = 30
    win = _floor_win(n)
    win["pred"] = win["gt"].clone()                    # perfect model
    win["ego_status"] = torch.empty(n, 0)
    win["states"] = torch.empty(n, 0)
    d = bench.diagnostic(win)
    _approx(d["model_ade_0_2s"], 0.0)
    _approx(d["skill_score"]["by_curvature"]["straight"]["skill_vs_floor"], 0.0)


# --------------------------------------------------------------------------- #
# ridge ceilings + ridge-vs-CTRV falsifier                                     #
# --------------------------------------------------------------------------- #
def _linear_win(n_ep=10, per=20, f_dim=6, seed=0, noise=False):
    """Windows where gt is an EXACT linear map of an ego-status feature block —
    a held-out ridge must recover it (ceiling ADE ~ 0). If noise=True, gt is
    independent of the features (ridge cannot beat the mean)."""
    g = torch.Generator().manual_seed(seed)
    n = n_ep * per
    feats = torch.randn(n, f_dim, generator=g)
    w_true = torch.randn(f_dim, 2 * N_WP, generator=g)
    gt_flat = feats @ w_true
    if noise:
        gt_flat = torch.randn(n, 2 * N_WP, generator=g)
    gt = gt_flat.reshape(n, N_WP, 2)
    eid = [i // per for i in range(n)]
    return feats, gt, eid


def test_ridge_ceiling_recovers_linear_map():
    feats, gt, eid = _linear_win(seed=1)
    _wp, de, alpha, r2 = bench._best_ridge_ceiling(feats, gt, eid)
    assert float(de.mean()) < 0.05, float(de.mean())   # near-exact recovery
    assert r2 > 0.99


def test_ridge_ceiling_fails_on_noise():
    feats, gt, eid = _linear_win(seed=2, noise=True)
    _wp, de_noise, _a, _r2 = bench._best_ridge_ceiling(feats, gt, eid)
    fe, ge, ee = _linear_win(seed=1)
    _wp2, de_lin, _a2, _r22 = bench._best_ridge_ceiling(fe, ge, ee)
    # a real linear signal is decoded >10x better than pure noise
    assert float(de_noise.mean()) > 10 * float(de_lin.mean())


def test_ridge_beats_ctrv_flag():
    """Falsifier plumbing: ego-status ridge that decodes gt well BEATS a
    deliberately-bad CTRV baseline; the flag and note report it honestly."""
    feats, gt, eid = _linear_win(seed=3)
    n = feats.shape[0]
    win = {
        "pred": gt.clone(), "gt": gt, "cv": _const_wp(n, 9.0),
        "constant_velocity": _const_wp(n, 9.0), "go_straight": _const_wp(n, 9.0),
        "constant_yaw_rate": _const_wp(n, 9.0),        # awful CTRV (ade 9)
        "ego_status": feats, "states": torch.empty(n, 0),
        "speed": torch.ones(n), "head_deg": torch.zeros(n), "eid": eid,
        "wp_steps": list(WP_STEPS),
    }
    d = bench.diagnostic(win)
    eb = d["ego_status_ceiling"]
    assert eb is not None
    assert eb["held_out_ade_0_2s"] < eb["ctrv_ade_0_2s"]
    assert eb["ridge_beats_ctrv"] is True
    assert "BEATS" in eb["note"]


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
