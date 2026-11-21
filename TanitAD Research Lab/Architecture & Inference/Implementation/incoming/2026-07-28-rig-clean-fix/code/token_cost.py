"""Compute delta of the rig-clean frame — MEASURED forward+backward, not counted.

The brief asks for the new token count and the compute delta. The token count is
arithmetic; the compute delta is not (attention is quadratic in tokens, the MLP
linear, and the readout's adaptive pooling is neither), so it is measured on the
real `ViTEncoder` at the real width.
"""
import json, sys, time
import torch
from tanitad.config import base250cam_config
from tanitad.geometry import apply_frame, geometry_report
from tanitad.data.calib import (CanonicalFrame, PHYSICALAI_WIDE120_256x640,
                                PHYSICALAI_RIG_CLEAN_176x624,
                                PHYSICALAI_RIG_CLEAN_128x576)
from tanitad.models.encoder import ViTEncoder

dev = "cuda" if torch.cuda.is_available() else "cpu"
B, REP = 8, 12
out = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
       "batch": B, "reps": REP, "frames": []}
for frame in (PHYSICALAI_WIDE120_256x640, PHYSICALAI_RIG_CLEAN_176x624,
              PHYSICALAI_RIG_CLEAN_128x576,
              CanonicalFrame(176, 640, PHYSICALAI_WIDE120_256x640.f_ref,
                             "cylindrical")):
    cfg = base250cam_config()
    apply_frame(cfg, frame)
    rep = geometry_report(cfg)
    enc = ViTEncoder(cfg.encoder).to(dev)
    x = torch.randn(B, cfg.encoder.in_channels, frame.height, frame.width,
                    device=dev)
    for _ in range(3):
        enc(x).sum().backward()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(REP):
        enc(x).sum().backward()
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / REP
    mem = (torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out["frames"].append({
        "tag": frame.tag(), "hw": [frame.height, frame.width],
        "n_tokens": rep["n_tokens"], "token_grid": rep["token_grid"],
        "tiles_exactly": (rep["token_grid"][0] % cfg.readout.grid == 0 and
                          rep["token_grid"][1] % (cfg.readout.grid_w or
                                                  cfg.readout.grid) == 0),
        "state_dim": rep["state_dim"],
        "s_per_fwdbwd": round(dt, 5), "peak_MiB": round(mem, 1) if mem else None,
        "params_M": round(sum(p.numel() for p in enc.parameters()) / 1e6, 3)})
    del enc, x
base = out["frames"][0]["s_per_fwdbwd"]
for f in out["frames"]:
    f["vs_parent"] = round(f["s_per_fwdbwd"] / base, 4)
    f["tokens_vs_parent"] = round(f["n_tokens"] / out["frames"][0]["n_tokens"], 4)
print(json.dumps(out, indent=1))
