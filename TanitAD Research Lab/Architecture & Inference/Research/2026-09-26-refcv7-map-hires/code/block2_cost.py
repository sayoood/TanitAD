"""The second near refine block's ANALYTIC cost at b16 (the PI's decision input, prepared per the
Master Mind after A17.1's edge FAIL): the REAL branch with near_refine_blocks 1 (A15 / A17.1) vs
2, both stacked on the 20 m near lift; CPU, fp32, batch 1 x 16, grad ckpt ON -- decoder_lever_cost's
exact method (FLOPs by FlopCounterMode; saved bytes by saved_tensors_hooks, parameters excluded).
Usage: block2_cost.py <candidate tree> <out json>"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

TREE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(TREE / "stack"))
import tanitad  # noqa: E402

assert str(TREE).replace("\\", "/").lower() in tanitad.__file__.replace("\\", "/").lower()
from tanitad.data.rig_projection import RigCamera  # noqa: E402
from tanitad.models import bev_lift as L  # noqa: E402
from tanitad.models import map_head_hires as H  # noqa: E402
from tanitad.models import refcv6_perception_branch as PB  # noqa: E402
from tanitad.models.trunk_shapes import FRAME_416x1024  # noqa: E402

torch.set_num_threads(8)
CAM = RigCamera.nominal(FRAME_416x1024, height_m=1.5, x_m=1.5)


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


def measure(blocks):
    torch.manual_seed(0)
    c = H.MapHiresConfig(w_map_hires=1.0, x_max_m=100.0, y_half_m=30.0, grad_ckpt=True,
                         near_lift_x_m=20.0, near_refine_blocks=blocks)
    br = H.MapHiresBranch(c, d_image=512, image_hw=(52, 128))
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
    for _ in range(5):
        br.zero_grad(set_to_none=True)
        t = time.perf_counter()
        fwd().backward()
        ts.append(time.perf_counter() - t)
    row = {"near_refine_blocks": blocks,
           "params_branch": int(sum(p.numel() for p in br.parameters())),
           "params_near_refine": int(sum(p.numel() for p in br.near_refine.parameters())),
           "saved_GiB_b16": round(16 * nb / 2 ** 30, 3),
           "fwd_GFLOPs_b1": round(fc.get_total_flops() / 1e9, 2),
           "cpu_fwd_bwd_s_b1_median": round(statistics.median(ts), 3),
           "receptive_field_added_m": round(sum(2 * (d1 + d2) for d1, d2 in
                                                [H.NearRefineBlock.DILATIONS] * blocks) * 0.1, 2)}
    print(json.dumps(row), flush=True)
    return row


rows = [measure(1), measure(2)]
d = {"rows": rows, "torch": torch.__version__,
     "delta_2_vs_1": {"fwd_GFLOPs_b1": round(rows[1]["fwd_GFLOPs_b1"] - rows[0]["fwd_GFLOPs_b1"], 2),
                      "fwd_pct": round(100 * (rows[1]["fwd_GFLOPs_b1"] / rows[0]["fwd_GFLOPs_b1"] - 1), 2),
                      "params": rows[1]["params_branch"] - rows[0]["params_branch"],
                      "saved_GiB_b16": round(rows[1]["saved_GiB_b16"] - rows[0]["saved_GiB_b16"], 3)},
     "note": ("ANALYTIC (FLOPs, saved bytes, params); CPU wall time is a noisy relative proxy only. "
              "A15's one block measured 0.711 s/step at b4 on Thor against A12's 0.712 (its "
              "cost was not measurable in step time).")}
OUT.write_bytes((json.dumps(d, indent=1) + "\n").encode("utf-8"))
print(json.dumps(d["delta_2_vs_1"]))
