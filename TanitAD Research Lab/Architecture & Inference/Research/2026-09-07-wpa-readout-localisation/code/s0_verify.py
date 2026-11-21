# -*- coding: utf-8 -*-
"""WP-A step 0 -- verify every seam BEFORE banking anything.

Prints, from SOURCE/ARTIFACT (never from prose):
  * the v6 trunk's ENCODER token grid and its READOUT grid  (the 16x40 vs 4x? fact)
  * one decoded episode's frame geometry
  * one frame's token tensor shape
  * one frame's GT BEV raster occupancy, with the all-zero baseline stated
"""
import json
import lzma
import os
import sys

sys.path.insert(0, r"C:\Users\Admin\wpa-readout\stack_snapshot")
sys.path.insert(0, r"C:\Users\Admin\wpa-readout\stack_snapshot\scripts")

import numpy as np
import torch

from tanitad.data._contract import to_float_frames
from tanitad.data.v2_dataset import decode_full_episode
from tanitad.data.bev_raster import (GRID_DEFAULT, agents_to_array, fov_census,
                                     fov_mask, rasterize)
from tanitad.eval.v6_probe_trunk import load_trunk_auto

CKPT = r"C:\Users\Admin\k8-pull\ckpt.pt"
EPS = r"C:\Users\Admin\tanitad-data\refav1-eval141\eps"
JOIN = r"C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz"

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("ckpt keys:", list(ck.keys())[:8], "step:", ck.get("step"))
trunk, grounding, step = load_trunk_auto(ck, dev, ckpt_path=CKPT)
st = trunk.stack
print("ENCODER token grid  :", trunk.token_grid)
print("READOUT grid_shape  :", trunk.grid_shape, "d_readout:", trunk.d_readout,
      "n_cells:", trunk.n_cells, "d_op(state_dim):", trunk.state_dim)
print("in_channels:", trunk.in_channels, "window:", trunk.window)
print("model frame:", trunk.frame)
# positive assertion from the WEIGHTS, not the config
ro = st.readout
print("readout module     :", type(ro).__name__,
      "token_h/w:", (ro.token_h, ro.token_w),
      "grid/grid_w:", (ro.grid, ro.grid_w),
      "exact_pool:", ro.exact_pool,
      "pool:", ro.pool)
print("readout.proj.weight:", tuple(ro.proj.weight.shape))
print("encoder.pos        :", tuple(st.encoder.pos.shape))

# ---- one episode -----------------------------------------------------------
clip = "01be5919-6e61-45ba-a193-5f447efb4b51"
ep = decode_full_episode(os.path.join(EPS, clip + ".v2ep.pt"))
print("episode frames:", tuple(ep.frames.shape), ep.frames.dtype,
      "poses:", tuple(ep.poses.shape))
d = torch.load(os.path.join(EPS, clip + ".v2ep.pt"), map_location="cpu",
               weights_only=False)
n_raw = len(d["jpeg_len"]); n_stack = int(d["n_stack"])
print("n_raw:", n_raw, "n_stack:", n_stack, "codec:", d.get("codec"),
      "frame:", d.get("frame"))

f = to_float_frames(ep.frames[:2]).to(dev)           # [2, 9, 256, 640]
print("float frames:", tuple(f.shape), float(f.min()), float(f.max()))
with torch.no_grad():
    z, tok = st.encode_window(f[None], return_tokens=True)
print("z (compact state):", tuple(z.shape), " tokens:", tuple(tok.shape))

# ---- the join --------------------------------------------------------------
rows = 0
mine = []
with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
    for line in fh:
        rows += 1
        r = json.loads(line)
        if r["clip_id"] == clip:
            mine.append(r)
        if rows > 60000:
            break
print("join lines scanned:", rows, "for this clip:", len(mine))
print("join line keys:", sorted(mine[0].keys()))
print("agent keys:", sorted(mine[0]["agents"][0].keys()) if mine[0]["agents"] else None)
frames_present = sorted(r["frame"] for r in mine)
print("frame idx range:", frames_present[0], "..", frames_present[-1],
      "n:", len(frames_present))

occ_fracs = []
for r in mine[:200]:
    arr = agents_to_array(r["agents"])
    ras = rasterize(arr, GRID_DEFAULT)
    occ_fracs.append(float(ras.mean()))
occ = np.array(occ_fracs)
print("GRID:", GRID_DEFAULT.shape, "cells:", GRID_DEFAULT.shape[0] * GRID_DEFAULT.shape[1])
print("occupancy frac  mean %.6f  median %.6f  max %.6f"
      % (occ.mean(), np.median(occ), occ.max()))
print("ALL-ZERO predictor accuracy on this clip: %.4f %%" % (100.0 * (1 - occ.mean())))

m = fov_mask(GRID_DEFAULT, 120.0)
print("fov_mask in-field frac: %.4f" % float(m.mean()))
print("fov_census:", json.dumps(fov_census(GRID_DEFAULT, 120.0, 4), default=str)[:600])
print("OK")
