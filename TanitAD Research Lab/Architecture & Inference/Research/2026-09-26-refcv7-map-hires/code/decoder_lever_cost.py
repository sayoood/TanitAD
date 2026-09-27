"""§9 lever (3) design input: the ANALYTIC cost at b16 of decoder variants STACKED on the R2
near lift (near_lift_x_m 20), vs R2 as built. CPU, fp32, batch 1 x 16, grad ckpt ON.

  R2         -- the R2 branch (A12): NEW-2 + the 0.1 m near lift (0-20 m)
  3c_near    -- + a residual 0.1 m block (3x3 conv + GN + GELU + 3x3 conv, d_up) on the NEAR
                rows only, after the skip is added, before conv1 (zero-init last conv)
  3b_deeper  -- + one more 3x3 conv + GN + GELU at 0.1 m on the WHOLE map (after conv2)
  3a_wider   -- d_up 32 -> 64 (the whole 0.1 m decoder twice as wide)
"""
import json
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, r"C:/Users/Admin/nb2r3_tree_2374/stack")
import tanitad  # noqa: E402

assert "nb2r3_tree_2374" in tanitad.__file__
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.models import bev_lift as L  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.models.trunk_shapes import FRAME_416x1024  # noqa: E402

torch.set_num_threads(8)
CAM = RigCamera.nominal(FRAME_416x1024, height_m=1.5, x_m=1.5)


def cfg(**kw):
    return H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0, grad_ckpt=True,
                            near_lift_x_m=20.0, **kw)


class _Res(nn.Module):
    def __init__(self, d, g):
        super().__init__()
        self.c1 = nn.Conv2d(d, d, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(g, d)
        self.c2 = nn.Conv2d(d, d, 3, padding=1, bias=False)
        nn.init.zeros_(self.c2.weight)

    def forward(self, x):
        return x + self.c2(F.gelu(self.n1(self.c1(x))))


def patch_refine(br, mode):
    r = br.refine
    d, g = int(br.cfg.d_up), int(br.cfg.norm_groups)
    if mode == "3c_near":
        r.near_block = _Res(d, g)
    elif mode == "3b_deeper":
        r.conv3 = nn.Conv2d(d, d, 3, padding=1, bias=False)
        r.norm3 = nn.GroupNorm(g, d)

    def fwd(x, near=None, near_refine=None, near_block_zero_input=False):
        x = r.inp(x)
        x = F.interpolate(x, size=r.out_hw, mode="bilinear", align_corners=False)
        if near is not None:
            n = int(near.shape[2])
            top = x[:, :, :n] + near
            if mode == "3c_near":
                top = r.near_block(top)
            x = torch.cat([top, x[:, :, n:]], dim=2)
        x = r.act(r.norm1(r.conv1(x)))
        x = r.act(r.norm2(r.conv2(x)))
        if mode == "3b_deeper":
            x = r.act(r.norm3(r.conv3(x)))
        return r.cls(x)
    r.forward = fwd
    return br


def saved_bytes(fn, exclude):
    skip = {t.untyped_storage().data_ptr() for t in exclude}
    seen = {}

    def pack(t):
        p = t.untyped_storage().data_ptr()
        if p not in skip:
            seen[p] = t.untyped_storage().nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return sum(seen.values()), out


def measure(name, br):
    torch.manual_seed(0)
    c = br.cfg
    g = L.build_lift_geometry(CAM, frame=FRAME_416x1024, stride=8, grid=c.lift_grid)
    grid, valid = g.grid.unsqueeze(0), g.valid.unsqueeze(0)
    f8 = torch.randn(1, 512, 52, 128, requires_grad=True)
    pool = PB.PlannerBEVPool(int(c.d_model), 96, c.lift_grid)
    codes = torch.randint(0, 8, (1,) + tuple(c.out_hw), dtype=torch.uint8)
    codes[:, :60] = 255

    def fwd():
        o = br(f8, grid, valid)
        lm = H.map_hires_loss_row(o["map_hires_logits"], codes,
                                  lift_valid_025=o["map_hires_lift_valid"],
                                  with_metrics=False)["loss"]
        return lm + pool(o["map_hires_bev"]).square().mean()
    excl = list(br.parameters()) + list(pool.parameters()) + [f8, grid, valid, codes]
    nb, loss = saved_bytes(fwd, excl)
    loss.backward()
    with FlopCounterMode(display=False) as fc:
        fwd()
    ts = []
    for _ in range(7):
        br.zero_grad(set_to_none=True)
        t = time.perf_counter()
        fwd().backward()
        ts.append(time.perf_counter() - t)
    # the decoder's largest live 0.1 m activation (transient, not saved under ckpt)
    d_up = int(c.d_up)
    row = {"arm": name, "params_branch": int(sum(p.numel() for p in br.parameters())),
           "saved_GiB_b16": round(16 * nb / 2 ** 30, 3),
           "fwd_GFLOPs_b1": round(fc.get_total_flops() / 1e9, 2),
           "cpu_fwd_bwd_s_b1_median": round(statistics.median(ts), 3),
           "decoder_0p1m_activation_GiB_b16": round(16 * d_up * 1000 * 600 * 4 / 2 ** 30, 3)}
    print(json.dumps(row), flush=True)
    return row


rows = []
torch.manual_seed(0)
rows.append(measure("R2", H.MapHiresBranch(cfg(), d_image=512, image_hw=(52, 128))))
torch.manual_seed(0)
rows.append(measure("R3_A15_real", H.MapHiresBranch(cfg(near_refine_blocks=1), d_image=512,
                                                     image_hw=(52, 128))))
for mode in ("3b_deeper",):
    torch.manual_seed(0)
    rows.append(measure(mode, patch_refine(H.MapHiresBranch(cfg(), d_image=512,
                                                            image_hw=(52, 128)), mode)))
torch.manual_seed(0)
rows.append(measure("3a_wider", H.MapHiresBranch(cfg(d_up=64), d_image=512, image_hw=(52, 128))))
print(json.dumps({"rows": rows, "torch": torch.__version__}, indent=1))
