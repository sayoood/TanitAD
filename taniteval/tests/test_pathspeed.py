"""Analytic tests for the decoupled long/lat planning metrics (pathspeed.py).

Synthetic inputs with hand-known answers, CPU-only, no checkpoint. Covers the
geometry core (tangents, arc length, speed/accel profiles, frenet & axis
decomposition, arc-length-resampled fixed-distance path geometry) and the
per-horizon / stratified assembly. The two decompositions are validated to
AGREE on straight windows and the frenet split is validated to isolate a pure
lateral offset EVEN ON A CURVE (where the fixed axes mix), and the fixed-
distance path-geometry metric is validated to be SPEED-DECOUPLED (a pure speed
error gives ~0 path-geometry error).

Run: PYTHONPATH=/root/taniteval:/root/TanitAD/stack python tests/test_pathspeed.py
"""
import math
import sys

import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import pathspeed as P  # noqa: E402

K = 20


def _approx(a, b, tol=1e-4):
    assert abs(float(a) - float(b)) <= tol, f"{a} != {b} (tol {tol})"


def _straight(dx, dy=0.0, n=1):
    """n windows, straight path: step k at (dx*k, dy*k). [n,K,2]."""
    k = torch.arange(1, K + 1, dtype=torch.float32)
    p = torch.stack([dx * k, dy * k], dim=-1)            # [K,2]
    return p.unsqueeze(0).expand(n, K, 2).clone()


# --------------------------------------------------------------------------- #
# geometry core                                                               #
# --------------------------------------------------------------------------- #
def test_arclength_straight():
    p = _straight(2.0)                                   # 2 m/step forward
    al = P.arclength(p)                                  # [1,K+1]
    _approx(al[0, 0], 0.0)
    _approx(al[0, -1], 2.0 * K)                          # 40 m total


def test_step_speed_constant():
    p = _straight(2.0)                                   # 2 m/step @10Hz = 20 m/s
    v = P.step_speed(p)                                  # [1,K]
    assert torch.allclose(v, torch.full_like(v, 20.0), atol=1e-4)


def test_step_accel_zero_at_constant_speed():
    p = _straight(2.0)
    a = P.step_accel(p)
    assert float(a.abs().max()) < 1e-3                   # constant speed => a~0


def test_step_speed_ramp_is_accel():
    # position = 0.5*a*t^2 sampled at t=k*dt => quadratic ramp, accel > 0
    k = torch.arange(1, K + 1, dtype=torch.float32)
    x = 0.5 * 4.0 * (k * P.DT) ** 2                      # a = 4 m/s^2
    p = torch.stack([x, torch.zeros_like(x)], dim=-1).unsqueeze(0)
    a = P.step_accel(p)                                  # ~4 m/s^2 (interior)
    _approx(a[0, 5:].mean(), 4.0, tol=0.2)


def test_segment_tangents_degenerate_no_nan():
    p = torch.zeros(2, K, 2)                             # fully stationary
    t = P.segment_tangents(p)
    assert torch.isfinite(t).all()
    assert torch.allclose(t[:, 0], torch.tensor([1.0, 0.0]))  # ego-forward fb


def test_heading_deg_known():
    _approx(P.heading_deg(_straight(2.0, 0.0))[0, -1], 0.0)     # +x -> 0 deg
    _approx(P.heading_deg(_straight(0.0, 2.0))[0, -1], 90.0)    # +y -> +90 deg


# --------------------------------------------------------------------------- #
# decomposition — straight (axis == frenet)                                    #
# --------------------------------------------------------------------------- #
def test_axis_pure_longitudinal():
    gt = _straight(2.0)                                  # 20 m/s
    pred = _straight(1.5)                                # 15 m/s (slower)
    lon, lat = P.axis_residual(pred, gt)
    _approx(lon[0, -1], -0.5 * K)                        # -10 m short at 2 s
    assert float(lat.abs().max()) < 1e-5                 # zero lateral


def test_frenet_pure_longitudinal_straight():
    gt, pred = _straight(2.0), _straight(1.5)
    along, cross = P.frenet_residual(pred, gt)
    _approx(along[0, -1], -0.5 * K)                      # behind by 10 m
    assert float(cross.abs().max()) < 1e-5
    # long_frac of the endpoint sq-error == 1
    sq_a, sq_c = float(along[:, -1] ** 2), float(cross[:, -1] ** 2)
    _approx(sq_a / (sq_a + sq_c + P.EPS), 1.0, tol=1e-3)


def test_frenet_pure_lateral_straight():
    gt = _straight(2.0)
    pred = gt.clone()
    pred[..., 1] += 0.5                                  # constant left offset
    along, cross = P.frenet_residual(pred, gt)
    assert float(along.abs().max()) < 1e-5              # no along-track error
    assert torch.allclose(cross, torch.full_like(cross, 0.5), atol=1e-4)


def test_frenet_orthonormal_identity():
    torch.manual_seed(0)
    gt = torch.randn(5, K, 2).cumsum(1)                 # random smooth-ish path
    pred = gt + 0.3 * torch.randn(5, K, 2)
    along, cross = P.frenet_residual(pred, gt)
    lhs = along ** 2 + cross ** 2
    rhs = (pred - gt).pow(2).sum(-1)                    # ||r||^2
    assert torch.allclose(lhs, rhs, atol=1e-4)          # orthonormal basis


# --------------------------------------------------------------------------- #
# the honest split on a CURVE (frenet isolates lateral; fixed axes mix)        #
# --------------------------------------------------------------------------- #
def _arc(n=1):
    """A left-curving GT arc (heading sweeps ~90 deg over the horizon)."""
    dphi = (math.pi / 2) / K
    xs, ys, x, y, th = [], [], 0.0, 0.0, 0.0
    for _ in range(K):
        x += 2.0 * math.cos(th)
        y += 2.0 * math.sin(th)
        th += dphi
        xs.append(x); ys.append(y)
    p = torch.tensor([xs, ys], dtype=torch.float32).T   # [K,2]
    return p.unsqueeze(0).expand(n, K, 2).clone()


def test_frenet_isolates_lateral_on_curve():
    gt = _arc()
    t = P.segment_tangents(gt)                           # [1,K,2]
    nvec = torch.stack([-t[..., 1], t[..., 0]], dim=-1)
    pred = gt + 0.4 * nvec                               # pure lateral offset
    along, cross = P.frenet_residual(pred, gt)
    assert float(along.abs().max()) < 1e-4              # frenet: ZERO along
    assert torch.allclose(cross, torch.full_like(cross, 0.4), atol=1e-4)
    # the FIXED ego axes, in contrast, smear the same offset into BOTH x and y
    lon, lat = P.axis_residual(pred, gt)
    assert float(lon.abs().max()) > 0.1                 # axis long is non-zero
    assert float(lat.abs().max()) > 0.1                 # (the arc turned)


# --------------------------------------------------------------------------- #
# fixed-distance path geometry — SPEED-DECOUPLED (refbpatch idea)              #
# --------------------------------------------------------------------------- #
def test_path_geometry_speed_decoupled():
    # same straight GEOMETRY, very different SPEED => path-geometry error ~ 0
    gt = _straight(3.0)                                  # fast: L = 60 m
    pred = _straight(1.5)                                # slow: L = 30 m
    pg, Lc = P.path_geometry_crosstrack(pred, gt)
    _approx(Lc[0], 30.0, tol=1e-3)                      # common length = min
    assert float(pg[0]) < 1e-3                          # geometry error ~ 0
    # ...yet the along-track (longitudinal) error is huge
    along, _ = P.frenet_residual(pred, gt)
    assert abs(float(along[0, -1])) > 25.0


def test_path_geometry_detects_lateral_offset():
    gt = _straight(2.5)
    pred = gt.clone()
    pred[..., 1] += 0.5                                  # parallel, shifted 0.5 m
    pg, _ = P.path_geometry_crosstrack(pred, gt)
    _approx(pg[0], 0.5, tol=0.05)                       # recovers the offset


# --------------------------------------------------------------------------- #
# per-horizon block + assembly                                                 #
# --------------------------------------------------------------------------- #
def test_metric_block_identical_is_zero():
    gt = _straight(2.0, 0.1)
    blk = P.metric_block(gt.clone(), gt, torch.arange(1))
    h = blk["per_horizon"]["2s"]
    _approx(h["de_at_h_m"], 0.0, tol=1e-4)
    _approx(h["planned_speed_err_mps"], 0.0, tol=1e-3)
    _approx(blk["trajectory"]["path_geometry_crosstrack_rmse_m"], 0.0, tol=1e-3)


def test_metric_block_pure_longitudinal():
    gt = _straight(2.0, 0.0, n=4)                       # 20 m/s
    pred = _straight(1.5, 0.0, n=4)                     # 15 m/s (under)
    blk = P.metric_block(pred, gt, torch.arange(4))
    h = blk["per_horizon"]["2s"]
    _approx(h["long_frac_of_sqerr"], 1.0, tol=1e-3)     # all error is long
    assert h["lat_rmse_m"] < 1e-4
    assert h["planned_speed_bias_mps"] < 0              # under-predicts speed
    _approx(h["planned_speed_bias_mps"], -5.0, tol=0.2)  # 15 - 20 m/s
    assert blk["trajectory"]["underpredicts_speed"] is True
    _approx(blk["trajectory"]["long_frac_of_sqerr_2s"], 1.0, tol=1e-3)


def test_metric_block_pure_lateral():
    gt = _straight(2.0, 0.0, n=4)
    pred = gt.clone()
    pred[..., 1] += 0.6
    blk = P.metric_block(pred, gt, torch.arange(4))
    h = blk["per_horizon"]["2s"]
    _approx(h["long_frac_of_sqerr"], 0.0, tol=1e-3)     # no longitudinal error
    _approx(h["lat_rmse_m"], 0.6, tol=1e-3)


def test_run_assembles_and_classifies_high_speed():
    # 100 straight windows, speeds spread 6..30 m/s, model always 20% too slow
    # => pure-longitudinal everywhere; the fast stratum must read 'longitudinal'.
    n = 100
    speeds = torch.linspace(6.0, 30.0, n)
    dx = speeds * P.DT                                   # m/step
    k = torch.arange(1, K + 1, dtype=torch.float32)
    gt = (dx[:, None, None] * k[None, :, None] *
          torch.tensor([1.0, 0.0]))                     # [n,K,2] straight
    pred = gt * 0.8                                      # 20% short (slow)
    col = {"pred": pred, "gt": gt, "ctrv": gt.clone(), "cv": gt.clone(),
           "gs": gt.clone(), "eid": list(range(n)),
           "speed": speeds, "head_deg": torch.zeros(n), "v0": speeds}
    res = P.run(col)
    hd = res["headline"]
    assert hd["high_speed_loss_is"] == "longitudinal"
    assert hd["high_speed_long_frac_of_2s_sqerr"] > 0.95
    assert hd["high_speed_underpredicts_speed"] is True
    # per-horizon error must GROW with the horizon (compounding read present)
    g = res["per_horizon_de_all_m"]
    assert g["2s"] > g["0.5s"]
    assert res["compounding_ratio_2s_over_0p5s"] > 1.0


def test_run_lateral_stratum_classifies_lateral():
    n = 60
    speeds = torch.linspace(6.0, 24.0, n)
    dx = speeds * P.DT
    k = torch.arange(1, K + 1, dtype=torch.float32)
    gt = (dx[:, None, None] * k[None, :, None] * torch.tensor([1.0, 0.0]))
    pred = gt.clone()
    pred[..., 1] += 0.7                                  # pure lateral drift
    col = {"pred": pred, "gt": gt, "ctrv": gt.clone(), "cv": gt.clone(),
           "gs": gt.clone(), "eid": list(range(n)),
           "speed": speeds, "head_deg": torch.zeros(n), "v0": speeds}
    res = P.run(col)
    assert res["strata"]["all"]["read"]["dominant_component"] == "lateral"
