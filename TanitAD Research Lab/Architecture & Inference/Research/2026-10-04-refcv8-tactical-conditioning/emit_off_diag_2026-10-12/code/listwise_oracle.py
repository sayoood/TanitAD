"""Q3 (zero-training): what does the X1 LISTWISE TARGET itself rank best? From a --save-cands base dump, compute the
training cost of `refcv8_conditioning.listwise_selection_loss` for every fan row against GT (speed-profile term +
terminal-heading term, list_t=1, 1 m, 15 deg -- V-R8's config.json), and write the argmin-cost row as an ordinary
eval dump `lworacle_s{s}` so ladder_score scores it with its own definitions. A selector that learned the target
PERFECTLY would emit this row. keep = every fan row (the eval dump carries no reach_keep).

Cross-check (not a re-run of my own derivation): the REAL loss function, fed score = -cost_mine/t, must return
H(p_mine) = -sum p_mine log p_mine to 1e-4 on every window batch; a mutated cost (lateral offset added) must FAIL it.
Also prints the target's effective support exp(H(p*)) -- how many rows the soft target spreads its mass over.
usage: python listwise_oracle.py <arm dir with base_s{0,1}.npz>"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from tanitad.refs import refcv8_conditioning as r8c

T, SP, DIR = 1.0, 1.0, 15.0


def cost_mine(fan, gt, gv):
    v = gv.to(torch.float32)
    n = v.sum(-1)

    def cum(p):
        q = torch.cat([torch.zeros_like(p[..., :1, :]), p], -2)
        return torch.linalg.vector_norm(q[..., 1:, :] - q[..., :-1, :], dim=-1).cumsum(-1)
    e_sp = ((cum(fan) - cum(gt)[:, None]).abs() * v[:, None]).sum(-1) / n.clamp_min(1)[:, None] / SP
    dth = r8c._wrap(r8c._terminal_heading(fan) - r8c._terminal_heading(gt)[:, None])
    e_dir = dth.abs() / math.radians(DIR) * gv[:, -1:].to(torch.float32)
    return e_sp + e_dir, n > 0


def check(fan, gt, gv, cost, ok):
    keep = torch.ones(cost.shape, dtype=torch.bool)
    p = torch.softmax(-cost / T, -1)
    h = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)[ok].mean()
    real = r8c.listwise_selection_loss(-cost / T, fan, gt, gv, keep, t=T, speed_scale_m=SP, dir_scale_deg=DIR)
    return float(real), float(h)


d = Path(sys.argv[1])
for s in (0, 1):
    src = d / f"base_s{s}.npz"
    z = dict(np.load(src, allow_pickle=False))
    rec = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
    fan = torch.from_numpy(z["cands"].astype(np.float32))
    gt = torch.from_numpy(z["gt"].astype(np.float32))
    gv = torch.from_numpy(z["gt_valid"].astype(bool))
    nb = int(z["n_base"][0])
    cost, ok = cost_mine(fan, gt, gv)
    worst = 0.0
    for i in range(0, len(fan), 64):
        sl = slice(i, i + 64)
        if not bool(ok[sl].any()):
            continue
        real, h = check(fan[sl], gt[sl], gv[sl], cost[sl], ok[sl])
        worst = max(worst, abs(real - h))
    # mutation arm: a cost with a lateral term the real target does NOT have must break the identity
    lat = (fan[..., 1] - gt[:, None, :, 1]).abs().mean(-1)
    real_m, h_m = check(fan[:64], gt[:64], gv[:64], (cost + lat)[:64], ok[:64])
    assert worst < 1e-4, f"my cost != the real target (max |dH| {worst})"
    assert abs(real_m - h_m) > 1e-3, "mutation did not break the identity -- the check is inert"
    p = torch.softmax(-cost / T, -1)
    eff = torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum(-1))
    idx = cost.argmin(1).numpy()
    err = np.linalg.norm(z["cands"] - z["gt"][:, None], axis=-1)
    w = z["gt_valid"][:, None, :].astype(float)
    ade = (err * w).sum(-1) / np.maximum(w.sum(-1), 1.0)
    ar = np.arange(len(idx))
    print(f"{d.name} s{s}: identity max|dH|={worst:.2e} (mutation |dH|={abs(real_m - h_m):.3f}); "
          f"p* effective support median {float(eff.median()):.1f} of {fan.shape[1]} rows; "
          f"lworacle extra-row frac {(idx >= nb).mean():.3f}; mean ADE lworacle {ade[ar, idx].mean():.3f} "
          f"vs ADE-oracle {ade.min(1).mean():.3f}")
    v = dict(z)
    v["traj"] = z["cands"][ar, idx]
    v["variant_idx"] = idx
    np.savez_compressed(str(d / f"lworacle_s{s}") + ".npz", **v)
    r = dict(rec, row="lworacle", variant_of="base", variant_rule="argmin X1 listwise cost (t=1, 1 m, 15 deg)")
    (d / f"lworacle_s{s}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
