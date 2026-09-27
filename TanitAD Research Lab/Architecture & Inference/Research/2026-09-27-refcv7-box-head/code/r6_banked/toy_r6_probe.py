#!/usr/bin/env python3
"""toy_r6_probe.py -- the toy of toy_presence_probe.py WITH +R6 denoising queries (INFORMATIVE only).

Same 16 frames / A targets / Q queries / decoder as the presence probe; adds the DN pass (DN_GROUPS_ARM groups,
positives + negatives, per-layer reconstruction) to every step's loss. Reports matched vs unmatched presence.

usage: python toy_r6_probe.py <lr> <A> <Q> <steps> <every> [groups]   (a tree with slot_denoise on PYTHONPATH)
"""
import sys

import torch

from tanitad.models import box3d_head as B3
from tanitad.models import slot_denoise as DN
from tanitad.models.slot_presence import refined_box3d_losses


def batch_schedule(n=16, b=4, steps=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < steps:
        p = torch.randperm(n, generator=g).tolist()
        out += [p[k:k + b] for k in range(0, n, b)]
    return out[:steps]


def main():
    lr, A, Q, steps, every = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    groups = int(sys.argv[6]) if len(sys.argv) > 6 else DN.DN_GROUPS_ARM
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
    emb = DN.DenoiseQueryEmbed(64, 10, x_range=60.0, y_range=16.0, z_range=4.0)
    opt = torch.optim.AdamW(list(dec.parameters()) + list(emb.parameters()), lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(7919)
    print(f"R6 toy: lr={lr} A={A} Q={Q} steps={steps} groups={groups}", flush=True)
    for step, idx in enumerate(batch_schedule(steps=steps), 1):
        opt.zero_grad()
        t = {k: x[idx] for k, x in tgt.items()}
        vv = {k: x[idx] for k, x in vis.items()}
        m_in = mem[idx]
        L = refined_box3d_losses(dec(m_in), t, presence_loss="focal", vis1=True, vis=vv)["total"]
        q = DN.make_dn_queries(emb, t, groups=groups, gen=gen)
        if q["q"] is not None:
            L = L + DN.dn_losses(DN.dn_forward(dec, m_in, q), t, q)["total"]
        L.backward()
        opt.step()
        if step % every == 0:
            with torch.no_grad():
                pred = dec(mem)
                r = refined_box3d_losses(pred, tgt, presence_loss="focal", vis1=True, vis=vis)
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
