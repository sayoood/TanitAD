#!/usr/bin/env python3
"""The on-policy scorer path (`REFe.forward(score_extra=...)`) must score a set EXACTLY as the proposals are scored.

Three checks on a random-init ViT-S REFe, CPU, fp32, eval mode (no dropout, deterministic):
  DEFAULT   forward() without score_extra returns the same two tensors as forward(score_extra=...)
            returns first -- the new argument does not perturb the shipped path
  IDENTITY  feeding a first pass's OWN proposals back as score_extra reproduces `score` BIT FOR BIT
  MUTATION  feeding the proposals in a SHUFFLED slot order must permute the scores with them (a
            set scorer is permutation-EQUIVARIANT), and feeding DIFFERENT trajectories (x + 1 m)
            must change them -- so the identity check cannot pass by ignoring its input
Prints ZZSCOREXTRA_OK or ZZSCOREXTRA_FAIL <which>; exit 0 / 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import REFe, REFeConfig  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    cfg = REFeConfig.for_backbone("vits16")
    cfg.img_h, cfg.img_w = 128, 256                  # small frames: this checks wiring, not vision
    m = REFe(cfg).eval()
    B = 2
    img = torch.randn(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w)
    ego = torch.randn(B, cfg.ego_dim)
    goal = torch.randn(B, 2 * cfg.n_goal_points)
    fails = []
    with torch.no_grad():
        traj0, score0 = m(img, ego, goal)
        traj1, score1, sx = m(img, ego, goal, score_extra=traj0)
        if not (torch.equal(traj0, traj1) and torch.equal(score0, score1)):
            fails.append("DEFAULT")
        d_id = (sx - score0).abs().max().item()
        if not torch.equal(sx, score0):
            fails.append(f"IDENTITY(max|d|={d_id:.3g})")
        perm = torch.randperm(cfg.n_proposals)
        _, _, sp = m(img, ego, goal, score_extra=traj0[:, perm])
        d_perm = (sp - score0[:, perm]).abs().max().item()
        if d_perm > 1e-5:
            fails.append(f"EQUIVARIANCE(max|d|={d_perm:.3g})")
        moved = traj0.clone()
        moved[..., 0] += 1.0
        _, _, sm = m(img, ego, goal, score_extra=moved)
        d_mov = (sm - score0).abs().max().item()
        if d_mov < 1e-4:
            fails.append(f"MUTATION_INERT(max|d|={d_mov:.3g})")
    print(f"  identity max|d| {d_id:.3g} · permuted-set max|d| {d_perm:.3g} · moved-by-1m max|d| {d_mov:.3g}")
    print("ZZSCOREXTRA_OK" if not fails else f"ZZSCOREXTRA_FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
