#!/usr/bin/env python3
"""toy_slot_diversity.py -- in the presence-probe toy, do the Q slots predict DIFFERENT boxes (diversity) or collapse?

Trains the toy (as toy_presence_probe.py) and reports, on all 16 frames: the mean per-frame std of the slots' predicted
centres, the mean distance from each target to its NEAREST slot and to its SECOND-nearest slot (duplicates), and how
often the Hungarian winner of a target changes between two consecutive evaluations (assignment stability).
usage: python toy_slot_diversity.py <focal|bce> <lr> <A> <Q> <steps> <every>
"""
import sys

import torch

from tanitad.models import box3d_head as B3
from tanitad.models.slot_presence import refined_box3d_losses


def batch_schedule(n=16, b=4, steps=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < steps:
        p = torch.randperm(n, generator=g).tolist()
        out += [p[k:k + b] for k in range(0, n, b)]
    return out[:steps]


def main():
    mode, lr, A, Q, steps, every = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), \
        int(sys.argv[5]), int(sys.argv[6])
    g = torch.Generator().manual_seed(1)
    mem = torch.randn(16, 8, 16, generator=g)
    box = torch.zeros(16, A, 4)
    box[..., 0] = torch.rand(16, A, generator=g) * 50 + 5
    box[..., 1] = torch.rand(16, A, generator=g) * 20 - 10
    box[..., 2], box[..., 3] = 4.5, 1.9
    v = torch.ones(16, A, dtype=torch.bool)
    tgt = {"box": box, "yaw": torch.zeros(16, A), "cls": torch.zeros(16, A, dtype=torch.long), "valid": v,
           "occ": torch.full((16, A), -1.0), "rates": torch.zeros(16, A, 3),
           "rates_mask": torch.zeros(16, A, dtype=torch.bool), "cz": torch.full((16, A), 0.8),
           "h": torch.full((16, A), 1.6), "zh_mask": v.clone()}
    vis = {"n_full": torch.full((16, A), 1000, dtype=torch.int32),
           "n_vis": torch.full((16, A), 900, dtype=torch.int32), "known": v.clone()}
    torch.manual_seed(0)
    dec = B3.Box3DSlotDecoder(16, 8, n_queries=Q, d_model=64, depth=3, n_heads=4, enforce_band=False,
                              presence_prior=0.01)
    dec.deep_supervision = True
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=0.0)
    prev = None
    print(f"mode={mode} lr={lr} A={A} Q={Q}", flush=True)
    for step, idx in enumerate(batch_schedule(steps=steps), 1):
        opt.zero_grad()
        t = {k: x[idx] for k, x in tgt.items()}
        vv = {k: x[idx] for k, x in vis.items()}
        refined_box3d_losses(dec(mem[idx]), t, presence_loss=mode, vis1=True, vis=vv)["total"].backward()
        opt.step()
        if step % every == 0:
            with torch.no_grad():
                pred = dec(mem)
                r = refined_box3d_losses(pred, tgt, presence_loss=mode, vis1=True, vis=vis)
                xy = pred["box"][..., :2]                                   # [16, Q, 2]
                spread = xy.std(dim=1).mean().item()
                d = torch.cdist(tgt["box"][..., :2], xy)                    # [16, A, Q]
                s = d.sort(dim=-1).values
                win = torch.stack([torch.as_tensor(r["match"]["rows"][b])[torch.argsort(r["match"]["cols"][b])]
                                   for b in range(16)])
                stab = float((win == prev).float().mean()) if prev is not None else float("nan")
                prev = win
                print(f"{step} slot centre spread {spread:.2f} m | target->nearest slot {s[..., 0].mean():.2f} m, "
                      f"->2nd {s[..., 1].mean():.2f} m | assignment unchanged since last eval {stab:.2f}", flush=True)


if __name__ == "__main__":
    main()
