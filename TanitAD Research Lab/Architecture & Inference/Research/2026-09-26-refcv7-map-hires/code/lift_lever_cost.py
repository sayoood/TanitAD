"""A12 design input: the ANALYTIC cost at b16 of the three pre-listed lift levers, against
the NEW-2 A7 branch as landed (cef9709 == this tree), all with the decoder's gradient
checkpointing ON (the launch config). CPU, fp32, batch 1, scaled x16. No GPU.

* saved bytes: autograd saved tensors (saved_tensors_hooks), DISTINCT storages, parameters
  and inputs excluded -- the method of code/cost_map_hires.py --mode analytic_a6;
* FLOPs: torch.utils.flop_counter (conv / matmul; grid_sample is not counted -- its sample
  count is reported beside it);
* CPU fwd+bwd wall time at b1 (median of 3) as a RELATIVE cost only.

(a) Z = 0 road-plane channel: +32 channels sampled at z = 0 only (unsummed), concatenated
    into the SHARED encoder's stem (64 -> 96 input channels).
(b) 0.1 m near-range lift: the stride-8 map lifted at 0.1 m over x 0-20 m x y +-30 m
    (200 x 600), the main lift's 4 heights, 32 channels (= d_up), ADDED to the 10 cm
    decoder's upsampled input on those rows, inside the decoder's checkpoint. Map-only.
(c) stride-4 features: the ONE lift reads resnet101 layer1 (256 x 104 x 256) instead of
    layer2 (512 x 52 x 128); plus the trunk tap's own cost (stem..layer1 vs stem..layer2).
"""
import json
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, r"C:/Users/Admin/nb2_tree_4797ffb/stack")
import tanitad  # noqa: E402

assert "nb2_tree_4797ffb" in tanitad.__file__
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.models import bev_lift as L  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.models.trunk_shapes import FRAME_416x1024  # noqa: E402
from tanitad.data.semantic_map_gt_fine import BEVGrid  # noqa: E402

torch.set_num_threads(8)
CAM = RigCamera.nominal(FRAME_416x1024, height_m=1.5, x_m=1.5)
CFG = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0, grad_ckpt=True)


def geom(stride, grid):
    g = L.build_lift_geometry(CAM, frame=FRAME_416x1024, stride=stride, grid=grid)
    return g.grid.unsqueeze(0).contiguous(), g.valid.unsqueeze(0).contiguous()


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


class NearSkipBranch(H.MapHiresBranch):
    """(b) prototype: the 0.1 m near lift added to the decoder's upsampled input."""

    def __init__(self, cfg, *, d_image, image_hw, near_rows=200):
        super().__init__(cfg, d_image=d_image, image_hw=image_hw)
        self.near_rows = near_rows
        self.near = H.BEVLiftProjectFirst(d_in=d_image, d_out=int(cfg.d_up),
                                          n_heights=len(cfg.heights_m), feat_hw=image_hw)

    def _refine_with_near(self, feats, f8, g01, v01):
        r = self.refine
        x = r.inp(feats)
        x = F.interpolate(x, size=r.out_hw, mode="bilinear", align_corners=False)
        near = self.near(f8, g01, v01)
        x = torch.cat([x[:, :, :self.near_rows] + near, x[:, :, self.near_rows:]], dim=2)
        x = r.act(r.norm1(r.conv1(x)))
        x = r.act(r.norm2(r.conv2(x)))
        return r.cls(x)

    def forward(self, f8, grid, valid, g01=None, v01=None):
        bev = self.lift(f8, grid, valid)
        feats = self._run(self.encoder, bev)
        logits = torch.utils.checkpoint.checkpoint(self._refine_with_near, feats, f8, g01, v01,
                                                   use_reentrant=False)
        return {"map_hires_logits": logits, "map_hires_lift_valid": valid.any(dim=1),
                "map_hires_bev": feats}


class Z0Branch(H.MapHiresBranch):
    """(a) prototype: +32 z = 0 channels concatenated into the shared encoder's stem."""

    def __init__(self, cfg, *, d_image, image_hw, d0=32):
        super().__init__(cfg, d_image=d_image, image_hw=image_hw)
        self.z0 = H.BEVLiftProjectFirst(d_in=d_image, d_out=d0, n_heights=1, feat_hw=image_hw)
        st = self.encoder.stem
        st[0] = nn.Conv2d(int(cfg.d_lift) + d0, int(cfg.d_model), 3, padding=1, bias=False)

    def forward(self, f8, grid, valid, g01=None, v01=None):
        bev = torch.cat([self.lift(f8, grid, valid),
                         self.z0(f8, grid[:, :1], valid[:, :1])], dim=1)
        feats = self._run(self.encoder, bev)
        logits = self._run(self.refine, feats)
        return {"map_hires_logits": logits, "map_hires_lift_valid": valid.any(dim=1),
                "map_hires_bev": feats}


def measure(name, br, f8, grid, valid, g01=None, v01=None):
    torch.manual_seed(0)
    pool = PB.PlannerBEVPool(int(CFG.d_model), 96, CFG.lift_grid)
    codes = torch.randint(0, 8, (1,) + tuple(CFG.out_hw), dtype=torch.uint8)
    codes[:, :60] = 255

    def fwd():
        o = br(f8, grid, valid, g01, v01) if g01 is not None or isinstance(br, Z0Branch) \
            else br(f8, grid, valid)
        lm = H.map_hires_loss_row(o["map_hires_logits"], codes,
                                  lift_valid_025=o["map_hires_lift_valid"],
                                  with_metrics=False)["loss"]
        return lm + pool(o["map_hires_bev"]).square().mean()
    excl = list(br.parameters()) + list(pool.parameters()) + [f8, grid, valid, codes] + \
        ([g01, v01] if g01 is not None else [])
    nb, loss = saved_bytes(fwd, excl)
    loss.backward()
    with FlopCounterMode(display=False) as fc:
        fwd()
    flops = fc.get_total_flops()
    ts = []
    for _ in range(3):
        br.zero_grad(set_to_none=True)
        t = time.perf_counter()
        fwd().backward()
        ts.append(time.perf_counter() - t)
    n_params = int(sum(p.numel() for p in br.parameters()))
    row = {"arm": name, "params_branch": n_params, "saved_bytes_b1": nb,
           "saved_GiB_b16": round(16 * nb / 2 ** 30, 3), "fwd_GFLOPs_b1": round(flops / 1e9, 2),
           "cpu_fwd_bwd_s_b1_median": round(statistics.median(ts), 3)}
    print(json.dumps(row), flush=True)
    return row


rows = []
g8, v8 = geom(8, CFG.lift_grid)
f8 = torch.randn(1, 512, 52, 128, requires_grad=True)
torch.manual_seed(0)
rows.append(measure("baseline_A7", H.MapHiresBranch(CFG, d_image=512, image_hw=(52, 128)),
                    f8, g8, v8))
g01, v01 = geom(8, BEVGrid(x_fwd_m=20.0, y_half_m=30.0, cell_m=0.1))
torch.manual_seed(0)
rows.append(measure("b_near_0p1m_0_20m", NearSkipBranch(CFG, d_image=512, image_hw=(52, 128)),
                    f8, g8, v8, g01, v01))
torch.manual_seed(0)
rows.append(measure("a_z0_channel", Z0Branch(CFG, d_image=512, image_hw=(52, 128)),
                    f8, g8, v8))
g4, v4 = geom(4, CFG.lift_grid)
f4 = torch.randn(1, 256, 104, 256, requires_grad=True)
torch.manual_seed(0)
rows.append(measure("c_stride4", H.MapHiresBranch(CFG, d_image=256, image_hw=(104, 256)),
                    f4, g4, v4))
# the trunk tap: stem..layer1 vs stem..layer2 of resnet101 on ONE frame (unchunked), and the
# output each hands to the lift (what survives under --trunk-chunk-ckpt)
import timm  # noqa: E402
net = timm.create_model("resnet101.a1_in1k", pretrained=False, features_only=True,
                        out_indices=(1, 2))
net.eval()
x = torch.rand(1, 3, 416, 1024)
for p in net.parameters():
    p.requires_grad_(True)
tap = {}
for idx, key in ((0, "stride4_layer1"), (1, "stride8_layer2")):
    nb, outs = saved_bytes(lambda: net(x), list(net.parameters()) + [x])
    o = outs[idx]
    tap[key] = {"out_shape": list(o.shape), "out_MiB_b16": round(16 * o.numel() * 4 / 2 ** 20, 1)}
    with FlopCounterMode(display=False) as fc:
        net(x)
    tap[key]["note"] = "flops counted for stem..layer2 (both outputs) -- see below"
tap["stem_to_layer2_fwd_GFLOPs_b1"] = round(fc.get_total_flops() / 1e9, 2)
m1 = timm.create_model("resnet101.a1_in1k", pretrained=False, features_only=True, out_indices=(1,))
with FlopCounterMode(display=False) as fc1:
    m1(x)
tap["stem_to_layer1_fwd_GFLOPs_b1"] = round(fc1.get_total_flops() / 1e9, 2)
out = {"method": __doc__.split("\n")[0], "rows": rows, "trunk_tap": tap,
       "grid_sample_outputs_b1": {
           "main_lift": 4 * 64 * 400 * 240, "b_near": 4 * 32 * 200 * 600,
           "a_z0": 1 * 32 * 400 * 240, "c_stride4": 4 * 64 * 400 * 240},
       "torch": torch.__version__}
print(json.dumps(out, indent=1))
