"""CPU smoke for the supervised non-causal IDM head (scripts/idm_head.py).

Guards the pipeline that produces the IDM/YouTube go/no-go number: window
construction (target alignment), forward shapes, a finite/differentiable loss,
and that a tiny fit runs and yields finite metrics. Fast (CPU, synthetic).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import idm_head as ih  # noqa: E402


def test_build_windows_target_alignment():
    # A hand-built episode: speed/steer/accel are exact functions of the index so
    # the derived targets at the window CENTER must match the source rows exactly.
    T, D = 60, 16
    z = torch.randn(T, D)
    yaw = 0.01 * torch.arange(T).float()                # constant yaw-rate 0.1 rad/s
    v = 5.0 + 0.0 * torch.arange(T).float()
    x = torch.cumsum(v * torch.cos(yaw) * ih.DT, 0)
    y = torch.cumsum(v * torch.sin(yaw) * ih.DT, 0)
    poses = torch.stack([x, y, yaw, v], 1).float()
    actions = torch.stack([0.02 * torch.arange(T).float(),
                           0.5 * torch.ones(T)], 1).float()
    Z, scal, traj = ih.build_windows(z, poses, actions, k=4, stride=1)
    assert Z.shape[1] == 9 and Z.shape[2] == D
    assert scal.shape[1] == 4 and traj.shape[1:] == (4, 2)
    # first valid center is t=4 (needs 4 past frames); check its targets
    t0 = 4
    assert torch.allclose(scal[0, 0], poses[t0, 3])                 # speed
    assert torch.allclose(scal[0, 2], actions[t0, 0])              # steer
    assert torch.allclose(scal[0, 3], actions[t0, 1])              # accel
    # centered yaw-rate ~ 0.1 rad/s (constant-rate construction)
    assert abs(float(scal[0, 1]) - 0.1) < 1e-3
    # trajectory forward at 2 s ~ v*2 = 10 m ahead (ego +x), lateral small
    assert traj[0, -1, 0] > 8.0


def test_forward_shapes_and_differentiable():
    # the REAL (default) head on the flagship state_dim is "a few M" params
    assert 1e6 < ih.count_params(ih.IDMHead(state_dim=2048)) < 1e7
    head = ih.IDMHead(state_dim=32, d_model=64, depth=2, n_heads=4)
    z = torch.randn(7, 9, 32)
    out = head(z)
    assert out["scalars"].shape == (7, 4) and out["traj"].shape == (7, 4, 2)
    scal = torch.randn(7, 4)
    traj = torch.randn(7, 4, 2)
    std = ih.Standardizer.fit(scal)
    ld = ih.idm_loss(out, scal, traj, std)
    assert torch.isfinite(ld["loss"])
    ld["loss"].backward()
    g = sum(float(p.grad.norm()) for p in head.parameters() if p.grad is not None)
    assert g > 0 and g == g                                        # finite, nonzero


def test_train_head_runs_and_metrics_finite():
    D = 32
    def cat(lst):
        return (torch.cat([a[0] for a in lst]), torch.cat([a[1] for a in lst]),
                torch.cat([a[2] for a in lst]))
    tr = [ih.build_windows(*ih._synthetic_episode(80, D, s), k=4) for s in range(6)]
    va = [ih.build_windows(*ih._synthetic_episode(80, D, s), k=4) for s in (6, 7)]
    res = ih.train_head(cat(tr), {"val": cat(va)}, state_dim=D, epochs=2,
                        batch=64, log=lambda *_: None)
    m = res["val"]["val"]
    assert m["n"] > 0
    for name in ih.SCALAR_NAMES:
        assert m["r2"][name] == m["r2"][name]                     # not NaN
    assert m["ade_2s"] == m["ade_2s"] and m["ade_2s"] >= 0.0
