"""kp_cpu_probe.py -- TREEDRIFT CPU-only deterministic sub-output probe: the kinematic prior / plan-rolling functions of ONE tree on fixed synthetic pose windows.
Usage: PYTHONPATH=<tree>/stack python kp_cpu_probe.py <tree> <out.npz>.  No GPU, no RNG beyond a fixed numpy seed for the INPUTS."""
import sys, hashlib
from pathlib import Path
import numpy as np
import torch
tree = Path(sys.argv[1]).resolve()
import tanitad.models.kinematic_prior as kp
assert tree in Path(kp.__file__).resolve().parents, kp.__file__
rs = np.random.RandomState(0)
B, T = 96, 10
v0 = rs.uniform(0.0, 32.0, B); yawrate = rs.uniform(-0.35, 0.35, B); acc = rs.uniform(-3, 2, B); yaw0 = rs.uniform(-3, 3, B)
dt = 0.1
poses = np.zeros((B, T, 4), np.float64)
for b in range(B):
    x = y = 0.0; yaw = yaw0[b]; v = max(v0[b] - acc[b] * dt * (T - 1), 0.0)
    for t in range(T):
        poses[b, t] = (x, y, yaw, v)
        x += v * dt * np.cos(yaw); y += v * dt * np.sin(yaw); yaw += yawrate[b] * dt + 0.01 * rs.randn(); v = max(v + acc[b] * dt, 0.0)
P = torch.from_numpy(poses).to(torch.float32)
H = [5, 10, 15, 20, 30, 40, 50, 60]
out = {}
for mode in ("ha0_ext_pose", "cv_yawrate"):
    a0, k0 = kp.prior_controls(mode, P, T, None, dt=dt)
    v = P[:, T - 1, 3]
    g = torch.Generator().manual_seed(1)
    delta = (torch.randn(B, 5, len(H), 2, generator=g) * torch.tensor([1.0, 2.0])).to(torch.float32)
    for units in ("alat",):
        out[f"{mode}.a0"] = a0.numpy(); out[f"{mode}.k0"] = k0.numpy()
        out[f"{mode}.prior_path"] = kp.prior_path(a0, k0, v, H).numpy()
        out[f"{mode}.roll_plan.{units}"] = kp.roll_plan(delta, a0, k0, v, H, control_units=units).numpy()
        out[f"{mode}.compose_ticks.{units}"] = kp.compose_ticks(delta, a0, k0, v, H, control_units=units).numpy()
        ab = kp.absolute_controls(delta, a0, k0, v, control_units=units)
        out[f"{mode}.absolute_controls.{units}"] = ab.numpy()
        out[f"{mode}.residual_controls.{units}"] = kp.residual_controls(ab, a0, k0, v, control_units=units).numpy()
np.savez(sys.argv[2], **out)
print(tree.name, {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:10] for k, v in out.items()})
