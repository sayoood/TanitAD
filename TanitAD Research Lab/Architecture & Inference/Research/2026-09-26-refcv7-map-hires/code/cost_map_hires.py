"""refcv7 NEW-2 -- the COST of the 10 cm branch (SPEC_REFCV7 §6.2 item 5).

Two instruments, one file:

* ``--mode analytic`` (CPU, no GPU needed): the parameter count, and the bytes of
  every tensor autograd SAVES for the backward (``torch.autograd.graph.
  saved_tensors_hooks``) in one forward of the branch at the REAL feature shape --
  resnet101 stride 8 at 416 x 1024 = 512 x 52 x 128 -- at batch 1, scaled linearly
  to batch 16. This is the activation memory the branch adds to a training step
  (transient peaks inside an op are NOT counted; the GPU mode measures those).
  Also: the trunk tap (stem..layer2 of resnet101 on the CURRENT frame), whose saved
  bytes are counted the same way (pretrained=False: the weights do not change the
  bytes).
* ``--mode analytic_a6``: SPEC_REFCV7 §11 / §12 (A6/A7) -- option (c) at the /2 and
  the A7 extent (60 x +-16 and 100 x +-30 m), the planner pool included, with and
  without the decoder's gradient checkpointing; plus the REMOVED stride-16 path.
* ``--mode gpu``: ``torch.cuda.max_memory_allocated()`` and forward+backward time of
  the branch ALONE at batch 4 and 8 on the same shape (fp32, the loss included).
  ⛔ Runs only when the dev-box GPU gate is open (the caller checks nvidia-smi and
  host RAM first) and ASSERTS residency: every tensor on cuda, no host spill.

Writes one JSON to ``--out``. No clip data is read; geometry is a nominal camera
(the branch's cost does not depend on which clip).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from tanitad.data.rig_projection import RigCamera
from tanitad.models import bev_lift as L
from tanitad.models import map_head_hires as H
from tanitad.models.trunk_shapes import FRAME_416x1024

C8, HW8 = 512, (52, 128)             # resnet101 layer2 @ 416x1024 (feature_info, MEASURED)


def _geom(cfg, b, device):
    g = L.build_lift_geometry(RigCamera.nominal(FRAME_416x1024, height_m=1.5, x_m=1.5),
                              frame=FRAME_416x1024, stride=8, grid=cfg.lift_grid)
    return (g.grid.unsqueeze(0).expand(b, *g.grid.shape).contiguous().to(device),
            g.valid.unsqueeze(0).expand(b, *g.valid.shape).contiguous().to(device))


def _saved_bytes(fn, exclude=()) -> tuple[int, object]:
    """Bytes of the DISTINCT storages autograd saves during ``fn()``, excluding the
    storages of ``exclude`` (parameters and inputs, which are resident anyway)."""
    skip = {t.untyped_storage().data_ptr() for t in exclude}
    seen = {}

    def pack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in skip:
            seen[ptr] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return int(sum(seen.values())), out


def analytic(out_path: Path) -> dict:
    torch.manual_seed(0)
    cfg = H.MapHiresConfig(w_map_hires=1.0)
    br = H.MapHiresBranch(cfg, d_image=C8, image_hw=HW8)
    grid, valid = _geom(cfg, 1, "cpu")
    f8 = torch.randn(1, C8, *HW8, requires_grad=True)
    codes = torch.randint(0, 8, (1, 600, 320), dtype=torch.uint8)
    codes[:, :60] = 255

    def fwd():
        o = br(f8, grid, valid)
        return H.map_hires_loss_row(o["map_hires_logits"], codes,
                                    lift_valid_025=o["map_hires_lift_valid"],
                                    with_metrics=False)["loss"]
    t = time.perf_counter()
    excl = list(br.parameters()) + [f8, grid, valid, codes]
    nbytes, loss = _saved_bytes(fwd, excl)
    t_fwd = time.perf_counter() - t
    t = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t
    rec = {"params": br.param_breakdown(), "config": cfg.as_dict(),
           "feature_shape": [C8, *HW8],
           "saved_bytes_b1": nbytes,
           "saved_GiB_b16": round(16 * nbytes / 2 ** 30, 3),
           "cpu_fwd_s_b1": round(t_fwd, 3), "cpu_bwd_s_b1": round(t_bwd, 3)}
    cfg_ck = H.MapHiresConfig(w_map_hires=1.0, grad_ckpt=True)
    br2 = H.MapHiresBranch(cfg_ck, d_image=C8, image_hw=HW8)
    br2.load_state_dict(br.state_dict())
    n2, _ = _saved_bytes(lambda: H.map_hires_loss_row(
        br2(f8, grid, valid)["map_hires_logits"], codes, with_metrics=False)["loss"],
        list(br2.parameters()) + [f8, grid, valid, codes])
    rec["saved_bytes_b1_grad_ckpt"] = n2
    rec["saved_GiB_b16_grad_ckpt"] = round(16 * n2 / 2 ** 30, 3)
    # the trunk tap: stem..layer2 of resnet101 on ONE current frame
    from tanitad.models import timm_trunk as TT
    tr = TT.TimmResNetTrunk(TT.TimmTrunkConfig(
        model_name="resnet101.a1_in1k", pretrained=False, verify_imagenet_stats=False,
        frames=3, image_hw=(416, 1024), frozen_bn=True))
    rec["tap_stage"] = tr.enable_s8_tap()
    x = torch.rand(1, 9, 416, 1024)
    xn = tr.normalise(x)
    nt, s8 = _saved_bytes(lambda: tr.s8_from_normalised(xn, [0]),
                          list(tr.parameters()) + [x, xn])
    rec["tap_saved_bytes_b1_unchunked"] = nt
    rec["tap_saved_GiB_b16_unchunked"] = round(16 * nt / 2 ** 30, 3)
    rec["tap_output_bytes_b1"] = int(s8.numel() * 4)
    rec["tap_note"] = ("with --trunk-chunk-ckpt the tap runs checkpointed like the main "
                       "pass: only its [b, 512, 52, 128] fp32 output is kept, "
                       f"{s8.numel() * 4 * 16 / 2 ** 20:.0f} MiB at b16")
    rec["host"] = {"python": platform.python_version(), "torch": torch.__version__}
    out_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def gpu(out_path: Path, batches=(4, 8), reps: int = 5) -> dict:
    assert torch.cuda.is_available(), "no CUDA"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    cfg = H.MapHiresConfig(w_map_hires=1.0)
    br = H.MapHiresBranch(cfg, d_image=C8, image_hw=HW8).to(dev)
    rec = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "params": br.param_breakdown(), "rows": []}
    for b in batches:
        grid, valid = _geom(cfg, b, dev)
        f8 = torch.randn(b, C8, *HW8, device=dev, requires_grad=True)
        codes = torch.randint(0, 8, (b, 600, 320), dtype=torch.uint8, device=dev)
        codes[:, :60] = 255
        for p in br.parameters():
            assert p.is_cuda
        assert f8.is_cuda and grid.is_cuda and valid.is_cuda and codes.is_cuda

        def step():
            o = br(f8, grid, valid)
            loss = H.map_hires_loss_row(o["map_hires_logits"], codes,
                                        lift_valid_025=o["map_hires_lift_valid"],
                                        with_metrics=False)["loss"]
            loss.backward()
            return loss

        step()                                         # warm-up (cuDNN, allocator)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        ts = []
        for _ in range(reps):
            br.zero_grad(set_to_none=True)
            f8.grad = None
            torch.cuda.synchronize()
            t = time.perf_counter()
            step()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        peak = torch.cuda.max_memory_allocated()
        free, total = torch.cuda.mem_get_info()
        rec["rows"].append({"batch": b, "fwd_bwd_s_median": round(sorted(ts)[len(ts) // 2], 4),
                            "fwd_bwd_s_all": [round(v, 4) for v in ts],
                            "max_memory_allocated_GiB": round(peak / 2 ** 30, 3),
                            "allocated_before_GiB": round(base / 2 ** 30, 3),
                            "peak_minus_resident_inputs_GiB": round((peak - base) / 2 ** 30, 3),
                            "device_free_GiB_after": round(free / 2 ** 30, 2),
                            "device_total_GiB": round(total / 2 ** 30, 2)})
        del grid, valid, f8, codes
        torch.cuda.empty_cache()
    out_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# --------------------------------------------------------------------------- #
# refcv7 A6/A7: option (c) at the /2 and the A7 extent -- saved bytes, analytic  #
# --------------------------------------------------------------------------- #
def analytic_a6(out_path: Path, extents=((60.0, 16.0), (100.0, 30.0))) -> dict:
    """Option (c) (SPEC_REFCV7 §11.1): ONE lift + ONE encoder at 0.25 m feed the 10 cm
    decoder AND the planner pool (crop 60 m x +-16 m, 2 x 2 avg, 1x1 + GN -> 96). Saved
    bytes (``saved_tensors_hooks``, distinct storages, parameters and inputs excluded)
    of ONE sample's forward, the map loss plus a consumer-proxy loss on the pooled BEV
    (so the pool's saved tensors are counted), with and without the decoder's gradient
    checkpointing; scaled x16. Beside it, the REMOVED refcv6 stride-16 path (BEVLift +
    0.5 m encoder + map head) at the real shape, i.e. what A6 gives back."""
    from tanitad.models import refcv6_perception_branch as PB
    rec = {"method": "saved_tensors_hooks, distinct storages, fp32, batch 1 x 16",
           "rows": []}
    for x_max, y_half in extents:
        for ck in (False, True):
            torch.manual_seed(0)
            cfg = H.MapHiresConfig(w_map_hires=1.0, x_max_m=x_max, y_half_m=y_half,
                                   grad_ckpt=ck)
            br = H.MapHiresBranch(cfg, d_image=C8, image_hw=HW8)
            pool = PB.PlannerBEVPool(int(cfg.d_model), 96, cfg.lift_grid)
            grid, valid = _geom(cfg, 1, "cpu")
            f8 = torch.randn(1, C8, *HW8, requires_grad=True)
            codes = torch.randint(0, 8, (1,) + tuple(cfg.out_hw), dtype=torch.uint8)
            codes[:, :60] = 255

            def fwd():
                o = br(f8, grid, valid)
                lm = H.map_hires_loss_row(o["map_hires_logits"], codes,
                                          lift_valid_025=o["map_hires_lift_valid"],
                                          with_metrics=False)["loss"]
                return lm + pool(o["map_hires_bev"]).square().mean()
            excl = list(br.parameters()) + list(pool.parameters()) + [f8, grid, valid,
                                                                       codes]
            t = time.perf_counter()
            nb, loss = _saved_bytes(fwd, excl)
            t_f = time.perf_counter() - t
            t = time.perf_counter()
            loss.backward()
            t_b = time.perf_counter() - t
            rec["rows"].append({
                "extent_m": [x_max, y_half], "grad_ckpt": ck,
                "lift_grid": list(cfg.lift_grid.shape), "out_hw": list(cfg.out_hw),
                "params_branch": br.param_breakdown(),
                "params_pool": int(sum(p.numel() for p in pool.parameters())),
                "saved_bytes_b1": nb, "saved_GiB_b16": round(16 * nb / 2 ** 30, 3),
                "cpu_fwd_s_b1": round(t_f, 3), "cpu_bwd_s_b1": round(t_b, 3)})
            print(json.dumps(rec["rows"][-1]), flush=True)
            del br, pool, grid, valid, f8, codes, loss
    # the REMOVED stride-16 path, at the real resnet101 stride-16 shape (1024 x 26 x 64)
    torch.manual_seed(0)
    pcfg = PB.PerceptionBranchConfig(w_map=1.0, w_box3d=0.0)
    s16 = PB.PerceptionBranch(pcfg, d_image=1024, image_hw=(26, 64))
    from tanitad.data.bev_raster import GRID_DEFAULT
    g = L.build_lift_geometry(RigCamera.nominal(FRAME_416x1024, height_m=1.5, x_m=1.5),
                              frame=FRAME_416x1024, stride=16, grid=GRID_DEFAULT)
    f16 = torch.randn(1, 1024, 26, 64, requires_grad=True)
    gg, gv = g.grid.unsqueeze(0), g.valid.unsqueeze(0)

    def fwd16():
        o = s16(f16, gg, gv)
        return o["map_logits"].square().mean() + o["bev_feats"].square().mean()
    n16, l16 = _saved_bytes(fwd16, list(s16.parameters()) + [f16, gg, gv])
    rec["removed_refcv6_s16_path"] = {
        "params": s16.param_breakdown(), "saved_bytes_b1": n16,
        "saved_GiB_b16": round(16 * n16 / 2 ** 30, 3)}
    rec["host"] = {"python": platform.python_version(), "torch": torch.__version__}
    out_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("analytic", "analytic_a6", "gpu"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = {"analytic": analytic, "analytic_a6": analytic_a6, "gpu": gpu}[a.mode](a.out)
    print(json.dumps(r, indent=1))
