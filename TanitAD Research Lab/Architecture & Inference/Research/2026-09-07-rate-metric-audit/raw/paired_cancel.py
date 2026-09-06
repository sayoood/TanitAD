"""DOES THE DEFECT CANCEL IN A PAIRED CONTRAST?

Two contrast SHAPES, on the same dump, same grid:

  (A) ANCHOR vs ANCHOR   (shipped - oracle)  -- both arms drawn from the SAME fan,
      so both are equally exposed to the ds->0 singularity. Expect: CANCELS.
  (B) ANCHOR vs FLOOR    (shipped - straight_floor) -- the floor has kappa == 0
      EXACTLY at any speed and is structurally IMMUNE. Expect: DOES NOT CANCEL.

If (A) cancels and (B) does not, the defect is an INTERACTION, not a shared bias,
and only floor-referenced curvature verdicts move.

ZERO GPU.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import four_families as ff

DT = 0.1


def pwo(p):
    z = np.zeros(p.shape[:-2] + (1, 2), dtype=p.dtype)
    return np.concatenate([z, p], axis=-2)


def kappa_p14(p, ts):
    d1 = np.gradient(p, ts, axis=-2); d2 = np.gradient(d1, ts, axis=-2)
    num = np.abs(d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0])
    den = np.power(d1[..., 0] ** 2 + d1[..., 1] ** 2, 1.5)
    return num / np.maximum(den, 1e-6)


def boot(dif, eids, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    ue = np.unique(eids); idx = {e: np.where(eids == e)[0] for e in ue}
    obs = float(np.nanmean(dif)); st = np.empty(B)
    for i in range(B):
        sel = np.concatenate([idx[e] for e in rng.choice(ue, len(ue), replace=True)])
        st[i] = np.nanmean(dif[sel])
    lo, hi = np.percentile(st, [2.5, 97.5])
    return {"delta": round(obs, 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "separated": bool(lo > 0 or hi < 0)}


out = {}
for tag, fn in (("refc-base-30k", "fan_refc-base-30k.pt"), ("refc-xl-30k", "fan_refc-xl-30k.pt")):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    fan, gt = d["fan"].float().numpy(), d["gt"].float().numpy()
    logits = d["logits"].float().numpy(); eid = np.asarray(d["eid"]); n = gt.shape[0]
    ts0 = np.concatenate([[0.0], np.array([w * DT for w in d["wp_steps"]], dtype=np.float64)])
    gdt = float(np.diff(ts0)[0])
    straight = np.zeros_like(gt); straight[..., 0] = np.linalg.norm(gt, axis=-1)
    ship = fan[np.arange(n), logits.argmax(1)]
    oracle = fan[np.arange(n), np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1).argmin(1)]
    kg = kappa_p14(pwo(gt), ts0)
    G = ff._seq_geometry(torch.as_tensor(gt).float(), gdt)

    def unmasked(t):
        return np.abs(kappa_p14(pwo(t), ts0) - kg).mean(-1)

    def masked(t):
        P = ff._seq_geometry(torch.as_tensor(t).float(), gdt)
        m = (P["pair_valid"] & G["pair_valid"]).numpy()
        dk = (P["curvature"] - G["curvature"]).abs().numpy()
        return np.nanmean(np.where(m, dk, np.nan), axis=-1)

    res = {}
    for label, a, b in (("A_anchor_vs_anchor(shipped-oracle)", ship, oracle),
                        ("B_anchor_vs_floor(shipped-straight)", ship, straight)):
        u, mk = boot(unmasked(a) - unmasked(b), eid), boot(masked(a) - masked(b), eid)
        # does the VERDICT move? sign + separation
        sign = lambda r: 0 if not r["separated"] else (1 if r["delta"] > 0 else -1)
        res[label] = {"UNMASKED_published_style": u, "MASKED_canonical": mk,
                      "verdict_moves": bool(sign(u) != sign(mk)),
                      "sign_unmasked": sign(u), "sign_masked": sign(mk)}
    out[tag] = res

print(json.dumps(out, indent=1))
