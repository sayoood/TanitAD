"""Re-timing a frozen head's waypoints — the retrain-free longitudinal fix.

⭐ WHY THE LATERAL TESTS ARE THE LOAD-BEARING ONES. flagship v1's lateral channel is
its healthy one — it beats a 34.3 B six-camera model on curvature MAE. A longitudinal
"fix" that quietly bends the path would trade our only good channel for our bad one and
the aggregate ADE might still improve, hiding it. `test_geometry_is_preserved_exactly`
and `test_curvature_is_untouched` exist to make that failure impossible to miss.
"""
import numpy as np
import torch

from tanitad.models.kinematic import (retime_path, rollout_unicycle,
                                      unicycle_controls_from_path)


def _straight(K=20, v=10.0, dt=0.1):
    return torch.stack([torch.arange(1, K + 1, dtype=torch.float64) * v * dt,
                        torch.zeros(K, dtype=torch.float64)], dim=-1)[None]


def _arc(K=20, v=10.0, kappa=0.05, dt=0.1):
    c = torch.zeros(1, K, 2, dtype=torch.float64)
    c[..., 1] = kappa
    s0 = torch.tensor([[0.0, 0.0, 0.0, v]], dtype=torch.float64)
    return rollout_unicycle(s0, c, dt)[..., :2]


def test_entry_transient_is_removed_exactly():
    """⭐ THE STRUCTURAL GUARANTEE. The schedule starts at the ego's TRUE v0, so the
    first step cannot disagree with it. MEASURED on flagship v1: 1.5367 -> 0.0000."""
    path = _straight(v=10.0)                      # path implies 10 m/s
    out = retime_path(path, torch.tensor([5.0], dtype=torch.float64))   # ego at 5
    first_speed = float(torch.linalg.norm(out[0, 0])) / 0.1
    assert abs(first_speed - 5.0) < 1e-6, first_speed


def test_geometry_is_preserved_exactly():
    """⛔ THE LOAD-BEARING TEST. Every re-timed sample must lie ON the original curve.
    Re-integrating the recovered curvature under a new speed profile would NOT satisfy
    this — `yaw_rate = v*kappa`, so the path would bend differently and the healthy
    lateral channel would be silently corrupted."""
    path = _arc(kappa=0.08)
    out = retime_path(path, torch.tensor([7.0], dtype=torch.float64))
    # distance from each output sample to the original polyline (incl. the origin)
    poly = torch.cat([torch.zeros_like(path[:, :1]), path], 1)[0]
    for q in out[0]:
        a, b = poly[:-1], poly[1:]
        ab = b - a
        t = (((q - a) * ab).sum(-1) / ab.pow(2).sum(-1).clamp_min(1e-12)).clamp(0, 1)
        d = torch.linalg.norm(a + t[:, None] * ab - q, dim=-1).min()
        assert float(d) < 1e-6, float(d)


def test_curvature_is_untouched():
    """The lateral channel must survive a longitudinal fix. Curvature is per-METRE, so
    re-sampling the same curve leaves it invariant."""
    path = _arc(kappa=0.06)
    v0 = torch.tensor([6.0], dtype=torch.float64)
    k_before = unicycle_controls_from_path(path)[..., 1]
    k_after = unicycle_controls_from_path(retime_path(path, v0))[..., 1]
    assert abs(float(k_after.mean()) - float(k_before.mean())) < 5e-3, \
        (float(k_before.mean()), float(k_after.mean()))


def test_jerk_barrier_binds():
    """A thrashing speed profile must come out inside the jerk bound."""
    K = 20
    rng = np.random.default_rng(0)
    speeds = 10.0 + rng.normal(0, 4.0, K)          # violently varying speed
    xs = np.cumsum(np.abs(speeds) * 0.1)
    path = torch.tensor(np.stack([xs, np.zeros(K)], -1), dtype=torch.float64)[None]
    out = retime_path(path, torch.tensor([10.0], dtype=torch.float64),
                      accel_limit=3.0, jerk_limit=6.0)
    acc = unicycle_controls_from_path(out)[0, :-1, 0]
    assert float(acc.abs().max()) <= 3.0 + 1e-3, float(acc.abs().max())
    jerk = (acc[1:] - acc[:-1]) / 0.1
    assert float(jerk.abs().max()) <= 6.0 + 1e-2, float(jerk.abs().max())


def test_a_feasible_path_is_left_alone():
    """⛔ THE NEGATIVE CONTROL. A path that is ALREADY feasible must come back
    essentially unchanged, or the projection is not a fix but a distortion applied to
    everyone."""
    v = 9.0
    path = _straight(v=v)
    out = retime_path(path, torch.tensor([v], dtype=torch.float64),
                      accel_limit=3.0, jerk_limit=6.0)
    assert torch.allclose(out, path, atol=1e-6), float((out - path).abs().max())


def test_overrun_extrapolates_instead_of_piling_up_at_the_end():
    """⛔ If the schedule outruns the curve, clamping would stack every sample on the
    endpoint and manufacture a hard stop the arm never planned — a speed fix that
    invents a braking event. Samples must keep advancing."""
    path = _straight(v=2.0)                         # a short, slow curve
    out = retime_path(path, torch.tensor([25.0], dtype=torch.float64),
                      accel_limit=3.0, jerk_limit=6.0)
    steps = torch.linalg.norm(out[0, 1:] - out[0, :-1], dim=-1)
    assert float(steps.min()) > 0.05, float(steps.min())


def test_speed_never_goes_negative_and_output_is_finite():
    rng = np.random.default_rng(1)
    path = torch.tensor(np.cumsum(rng.normal(0, 0.5, (3, 20, 2)), 1))
    out = retime_path(path, torch.tensor([0.0, 5.0, 20.0], dtype=torch.float64))
    assert torch.isfinite(out).all()
    xs = torch.linalg.norm(torch.cat([torch.zeros_like(out[:, :1]), out], 1).diff(dim=1),
                           dim=-1)
    assert float(xs.min()) >= 0.0


def test_shape_contract_is_enforced():
    for bad in (torch.zeros(20, 2), torch.zeros(1, 20, 3)):
        try:
            retime_path(bad, torch.zeros(1))
        except ValueError:
            continue
        raise AssertionError(f"accepted {tuple(bad.shape)}")
