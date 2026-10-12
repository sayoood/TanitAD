"""CPU cost of refe/field_loss.py (the dev-box GPU is shared; this does NOT time the A40). MEASURED on random trajectories: a
timing of the arithmetic, not of any real batch. Run: python raw/2026-10-12-field-loss/bench_cpu.py"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "refe"))
import field_loss as FL  # noqa: E402

torch.set_num_threads(4)
for B in (4, 8):
    g = torch.Generator().manual_seed(0)
    traj = (torch.randn(B, 64, 20, 3, generator=g) * 0.5 + torch.tensor([10.0, 0, 0])).cumsum(2).clamp(-40, 60).requires_grad_(True)
    sdf = torch.randint(-127, 127, (B, 160, 128), dtype=torch.int8, generator=g)
    for _ in range(3):
        FL.field_loss(traj, sdf).backward()
    n, t = 30, time.perf_counter()
    for _ in range(n):
        traj.grad = None
        FL.field_loss(traj, sdf).backward()
    ms = (time.perf_counter() - t) / n * 1e3
    print(f"CPU (4 threads) fwd+bwd, B={B} K=64: {ms:.1f} ms per micro-batch; peak live tensors ~ {B*64*40*4*2*4/1e6:.2f} MB of corners (fp32)")
