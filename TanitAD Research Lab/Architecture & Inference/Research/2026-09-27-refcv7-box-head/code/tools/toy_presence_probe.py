#!/usr/bin/env python3
"""toy_presence_probe.py -- does a small 3-D slot decoder's PRESENCE specialise (matched slots up, unmatched down)
when it has more queries than targets? INFORMATIVE; the finding that made A10.1 pre-register R6.

A toy: 16 frames, frame-specific random memory [8, 16], A targets per frame (random boxes in the field), Q queries,
Box3DSlotDecoder (d 64, 3 layers, prior 0.01, per-layer supervision), the refined box loss with VIS-1 (every target
visible), AdamW, batches of 4 by the harness's schedule. Prints, every K steps, the presence term and the median /
tail of sigma(presence) over Hungarian-matched vs unmatched slots on all 16 frames.

usage: python toy_presence_probe.py <focal|bce> <lr> <A> <Q> <steps> <every>     (the stack on PYTHONPATH)
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
    print(f"mode={mode} lr={lr} A={A} Q={Q} steps={steps}", flush=True)
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
                m = r["match"]
                p = torch.sigmoid(pred["presence_logit"])
                pm = torch.cat([p[b][m["rows"][b]] for b in range(16)])
                mask = torch.ones_like(p, dtype=torch.bool)
                for b in range(16):
                    mask[b, m["rows"][b]] = False
                pu = p[mask] if bool(mask.any()) else torch.zeros(1)
                print(f"{step} pres {float(r['loss_presence']):.4f} centre {float(r['loss_centre']):.3f} "
                      f"p_matched med {float(pm.median()):.3f} frac>=.5 {float((pm >= .5).float().mean()):.3f} | "
                      f"p_unmatched med {float(pu.median()):.3f} p90 {float(pu.quantile(0.9)):.3f} "
                      f"frac>=.5 {float((pu >= .5).float().mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
