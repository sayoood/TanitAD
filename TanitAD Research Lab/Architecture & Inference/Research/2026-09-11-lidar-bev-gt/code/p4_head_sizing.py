#!/usr/bin/env python3
"""P4 - parameter cost of the candidate BEV heads, COUNTED not estimated.

⛔ A head's parameter count is an arithmetic claim, and arithmetic claims in this
programme get built and counted. Every number this script prints comes from
`sum(p.numel() for p in module.parameters())` on a module that forward-passes on a
real-shaped tensor, not from a formula in prose.

REF-C / refcv5 context (MEASURED, `E-BEV-AUX-1`'s shipped record):
  * trunk total with the WP-D aux head        108,440,118
  * trunk total with it stripped              108,257,502
  * the WP-D polar aux head                       182,616  (6 keys)
  * ResNet stride 32 on 256x640  ->  feature map 8 x 20, feat dim 704
  * programme ceiling                         sub-300 M

Usage:  python p4_head_sizing.py --out raw/head_sizing.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn

TRUNK_DEPLOYED = 108_257_502          # MEASURED: D0's param_breakdown.total
WPD_AUX_HEAD = 182_616                # MEASURED: 108,440,118 - 108,257,502
CEILING = 300_000_000
FEAT, GH, GW = 704, 8, 20             # refcv5's ResNet map at --image-hw 256 640


# --------------------------------------------------------------------------
# Candidate A - LIFT-SPLAT-SHOOT (the cheap one)
# --------------------------------------------------------------------------
class LiftSplatBEV(nn.Module):
    """Per-feature-cell depth distribution + context, splatted into a metric BEV.

    ⭐ Why this is the cheap candidate: the lift is ONE 1x1 convolution, and all the
    geometry (which BEV cell a (row, col, depth-bin) triple falls into) is a FIXED
    precomputed index built from the CYLINDRICAL ray model and the rig extrinsics --
    zero learned parameters, and it is where the metric scale enters.
    """

    def __init__(self, feat: int = FEAT, n_depth: int = 48, ctx: int = 64,
                 bev_ch: int = 64, n_out: int = 2, bev_layers: int = 3):
        super().__init__()
        self.lift = nn.Conv2d(feat, n_depth + ctx, 1)
        enc: list[nn.Module] = []
        c = ctx
        for _ in range(bev_layers):
            enc += [nn.Conv2d(c, bev_ch, 3, padding=1), nn.BatchNorm2d(bev_ch), nn.ReLU(True)]
            c = bev_ch
        self.bev_enc = nn.Sequential(*enc)
        self.out = nn.Conv2d(bev_ch, n_out, 1)


# --------------------------------------------------------------------------
# Candidate B - BEV CROSS-ATTENTION (the expressive one)
# --------------------------------------------------------------------------
class BEVCrossAttn(nn.Module):
    """Learned BEV queries cross-attending the image feature map (BEVFormer-style).

    ⚠️ The QUERY EMBEDDING is the term that scales with the grid, and it is the term
    a dense Cartesian grid blows up: 128x128 queries at d=256 is 4.2 M parameters
    before a single transformer layer. A POLAR 24x20 grid registered column-to-column
    onto the 20 feature columns costs 480 queries -- and the corpus being CYLINDRICAL
    is what makes that registration exact (HFOV 120.000 deg, column linear in azimuth).
    """

    def __init__(self, feat: int = FEAT, d: int = 256, n_layers: int = 4,
                 n_heads: int = 8, d_ff: int = 1024,
                 bev_h: int = 24, bev_w: int = 20, n_out: int = 2):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(bev_h * bev_w, d))
        self.kv_proj = nn.Linear(feat, d)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d, nhead=n_heads,
                                       dim_feedforward=d_ff, batch_first=True,
                                       norm_first=True)
            for _ in range(n_layers)])
        self.out = nn.Linear(d, n_out)


def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def breakdown(m: nn.Module) -> dict:
    return {n: int(sum(p.numel() for p in c.parameters()))
            for n, c in m.named_children()} | {
        "_bare_parameters": int(sum(p.numel() for n, p in m.named_parameters()
                                    if "." not in n))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw/head_sizing.json")
    args = ap.parse_args()
    torch.manual_seed(0)

    cands = {
        "A_liftsplat_polar24x20": LiftSplatBEV(n_depth=48, ctx=64, bev_ch=64,
                                               n_out=2, bev_layers=3),
        "A_liftsplat_cart120x64_wide": LiftSplatBEV(n_depth=64, ctx=128, bev_ch=128,
                                                    n_out=2, bev_layers=4),
        "B_xattn_polar24x20_d256_L4": BEVCrossAttn(bev_h=24, bev_w=20, d=256, n_layers=4),
        "B_xattn_cart120x64_d256_L4": BEVCrossAttn(bev_h=120, bev_w=64, d=256, n_layers=4),
        "B_xattn_cart128x128_d256_L6": BEVCrossAttn(bev_h=128, bev_w=128, d=256, n_layers=6),
        "B_xattn_polar48x40_d192_L3": BEVCrossAttn(bev_h=48, bev_w=40, d=192, n_layers=3,
                                                   n_heads=6, d_ff=768),
    }

    out = {
        "schema": "tanitad.bev_head_sizing/1",
        "trunk_deployed_params": TRUNK_DEPLOYED,
        "wpd_aux_head_params": WPD_AUX_HEAD,
        "ceiling_params": CEILING,
        "feature_map": {"feat": FEAT, "gh": GH, "gw": GW,
                        "why": "refcv5 ResNet stride 32 on --image-hw 256 640"},
        "candidates": {},
    }
    for name, m in cands.items():
        n = count(m)
        out["candidates"][name] = {
            "params": n,
            "params_M": round(n / 1e6, 4),
            "pct_of_trunk": round(100.0 * n / TRUNK_DEPLOYED, 3),
            "x_wpd_aux_head": round(n / WPD_AUX_HEAD, 1),
            "total_with_trunk": TRUNK_DEPLOYED + n,
            "headroom_to_ceiling": CEILING - (TRUNK_DEPLOYED + n),
            "breakdown": breakdown(m),
        }

    # ---- a forward pass, so "it has these parameters" is not the only claim ----
    x = torch.zeros(2, FEAT, GH, GW)
    m = cands["A_liftsplat_polar24x20"]
    y = m.lift(x)
    out["forward_probe"] = {
        "input_shape": list(x.shape),
        "lift_output_shape": list(y.shape),
        "note": "the splat index is precomputed geometry (0 learned params) and is "
                "where the CYLINDRICAL ray model and the rig extrinsics enter",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["candidates"].items():
        print(f"{k:34s} {v['params']:>12,}  {v['params_M']:>7.3f} M  "
              f"{v['pct_of_trunk']:>6.2f} % of trunk  {v['x_wpd_aux_head']:>7.1f}x WP-D aux")
    print(f"[p4] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
