"""ad hoc: the OLD frozen formula (straight continuation) must FAIL the AMENDMENT_1 known answer on a curved path."""
import sys
sys.path[:0] = ["F:/TanitAD-scratch/recover_ft/tree/stack"]
import torch
from tanitad.data import v8f_recover as R

k = torch.arange(1, 31, dtype=torch.float64)
psi = 0.03 * k + 0.001 * k * k
g = torch.stack([torch.cumsum(1.2 * torch.cos(psi), 0), torch.cumsum(1.2 * torch.sin(psi), 0)], -1)[None]
z = torch.zeros(1, dtype=torch.float64)


def old(gt, dy, dpsi, K):
    c, s = torch.cos(dpsi)[:, None], torch.sin(dpsi)[:, None]
    gx, gy = gt[..., 0], gt[..., 1] - dy[:, None]
    gp = torch.stack([c * gx + s * gy, -s * gx + c * gy], -1)
    L = torch.diff(torch.cat([torch.zeros_like(gt[:, :1]), gt], 1), dim=1).norm(dim=-1).cumsum(1)
    st = torch.stack([L, torch.zeros_like(L)], -1)
    w = torch.ones(30, dtype=gt.dtype)
    w[:K] = R.smoothstep((torch.arange(K, dtype=gt.dtype) + 1.0) / K)
    return (1.0 - w)[None, :, None] * st + w[None, :, None] * gp


for K in (1, 10):
    print("K", K, "OLD max|target-gt| =", float((old(g, z, z, K) - g).abs().max()), " NEW =", float((R.rejoin_target(g, z, z, steps=K) - g).abs().max()))
