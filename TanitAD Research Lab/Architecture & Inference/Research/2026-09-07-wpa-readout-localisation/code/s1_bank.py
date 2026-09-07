# -*- coding: utf-8 -*-
"""WP-A step 1 -- bank encoder tokens + raw-patch pixels + GT BEV rasters.

ONE feature per labelled episode row.  No windowing, so NO frame is duplicated
(the dense-windowed-tensor trap, CLAUDE.md).

Banked, all on LOCAL disk:
  tok.f16     [N, 640, 128]  v6 encoder PATCH TOKENS at the 16x40 grid
  pix.f16     [N, 640, 9]    per-patch MEAN of the raw 9-channel input (pixel floor)
  y.u8        [N, 7680]      GT BEV occupancy raster, GRID_DEFAULT (120x64, 0.5 m)
  idx.json                   clip ids, per-row (clip_i, row) and split assignment
"""
import argparse
import json
import lzma
import os
import sys
import time

sys.path.insert(0, r"C:\Users\Admin\wpa-readout\stack_snapshot")
sys.path.insert(0, r"C:\Users\Admin\wpa-readout\stack_snapshot\scripts")

import numpy as np
import torch

from tanitad.data._contract import to_float_frames
from tanitad.data.v2_dataset import decode_full_episode
from tanitad.data.bev_raster import GRID_DEFAULT, agents_to_array, rasterize
from tanitad.eval.v6_probe_trunk import load_trunk_auto

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=r"C:\Users\Admin\k8-pull\ckpt.pt")
ap.add_argument("--out", default=r"C:\Users\Admin\wpa-readout\bank\k8")
ap.add_argument("--stride", type=int, default=2, help="row subsample")
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

EPS = r"C:\Users\Admin\tanitad-data\refav1-eval141\eps"
JOIN = r"C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz"
os.makedirs(a.out, exist_ok=True)

# ---- 1. join -> {clip: {row: agents}} --------------------------------------
t0 = time.time()
per_clip = {}
n_lines = 0
with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        n_lines += 1
        # `frame` is RAW v2ep index; `frame_idx` is the post-n_stack-trim row.
        # Assert the relation instead of trusting one of them.
        assert r["frame_idx"] == r["frame"] - 2, (r["frame"], r["frame_idx"])
        per_clip.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r["agents"]
print(f"[join] {n_lines} lines, {len(per_clip)} clips, {time.time()-t0:.1f}s",
      flush=True)

have = {f.split(".v2ep")[0] for f in os.listdir(EPS) if f.endswith(".v2ep.pt")}
clips = sorted(set(per_clip) & have)
print(f"[corpus] join {len(per_clip)} x eps {len(have)} -> {len(clips)} clips",
      flush=True)

# ---- 2. trunk --------------------------------------------------------------
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
trunk, _g, step = load_trunk_auto(ck, dev, ckpt_path=a.ckpt)
st = trunk.stack
TG = tuple(trunk.token_grid)                       # (16, 40)
NT, DM = TG[0] * TG[1], int(st.cfg.encoder.d_model)
CIN = int(trunk.in_channels)
print(f"[trunk] step {step} token_grid {TG} d_model {DM} readout "
      f"{trunk.grid_shape} d_readout {trunk.d_readout}", flush=True)

# ---- 3. pass 1: count rows -------------------------------------------------
plan = []          # (clip_i, row)
for ci, c in enumerate(clips):
    rows = sorted(per_clip[c])
    for r in rows[::a.stride]:
        plan.append((ci, r))
N = len(plan)
NCELL = GRID_DEFAULT.shape[0] * GRID_DEFAULT.shape[1]
print(f"[plan] N={N} rows  stride={a.stride}  cells={NCELL}", flush=True)

tok_mm = np.lib.format.open_memmap(os.path.join(a.out, "tok.npy"), mode="w+",
                                   dtype=np.float16, shape=(N, NT, DM))
pix_mm = np.lib.format.open_memmap(os.path.join(a.out, "pix.npy"), mode="w+",
                                   dtype=np.float16, shape=(N, NT, CIN))
y_mm = np.lib.format.open_memmap(os.path.join(a.out, "y.npy"), mode="w+",
                                 dtype=np.uint8, shape=(N, NCELL))

# ---- 4. pass 2: decode + encode -------------------------------------------
ph, pw = int(st.cfg.encoder.patch_size), int(st.cfg.encoder.patch_size)
w = 0
t0 = time.time()
for ci, c in enumerate(clips):
    ep = decode_full_episode(os.path.join(EPS, c + ".v2ep.pt"))
    rows = [r for (cc, r) in plan if cc == ci]
    fr = ep.frames                                    # [T, 9, 256, 640] u8
    assert fr.shape[1] == CIN and fr.shape[2:] == (256, 640), fr.shape
    rows = [r for r in rows if r < fr.shape[0]]
    for b0 in range(0, len(rows), a.batch):
        sel = rows[b0:b0 + a.batch]
        f = to_float_frames(fr[sel]).to(dev)          # [b, 9, 256, 640]
        with torch.no_grad():
            _z, tk = st.encode_window(f[None], return_tokens=True)
        tk = tk[0]                                    # [b, 640, 128]
        # per-patch mean of the RAW input -> the pixel floor at the same grid
        pm = torch.nn.functional.avg_pool2d(f, (ph, pw))          # [b, 9, 16, 40]
        pm = pm.flatten(2).transpose(1, 2)                        # [b, 640, 9]
        n = len(sel)
        tok_mm[w:w + n] = tk.to(torch.float16).cpu().numpy()
        pix_mm[w:w + n] = pm.to(torch.float16).cpu().numpy()
        for j, r in enumerate(sel):
            arr = agents_to_array(per_clip[c][r])
            y_mm[w + j] = rasterize(arr, GRID_DEFAULT).astype(np.uint8).reshape(-1)
        w += n
    if ci % 20 == 0:
        el = time.time() - t0
        print(f"[bank] clip {ci}/{len(clips)} rows {w}/{N} {el:.0f}s "
              f"eta {el / max(w, 1) * (N - w):.0f}s", flush=True)

print(f"[bank] wrote {w} rows in {time.time()-t0:.0f}s", flush=True)
assert w == N or w > 0
tok_mm.flush(); pix_mm.flush(); y_mm.flush()

# ⛔ CONTENT assertion -- a memmap that a failed decode left as zeros looks
# exactly like a finished bank (CLAUDE.md, the E-DETECT-1 all-zero floor).
tk_abs = float(np.abs(np.asarray(tok_mm[:w:max(1, w // 200)], dtype=np.float32)).mean())
px_abs = float(np.abs(np.asarray(pix_mm[:w:max(1, w // 200)], dtype=np.float32)).mean())
y_rate = float(np.asarray(y_mm[:w], dtype=np.float32).mean())
n_nonzero_rows = int((np.asarray(y_mm[:w]).sum(1) > 0).sum())
print(f"[assert] mean|tok| {tk_abs:.5f}  mean|pix| {px_abs:.5f}  "
      f"occ_rate {y_rate:.6f}  rows_with_any_agent {n_nonzero_rows}/{w}")
assert tk_abs > 1e-4 and px_abs > 1e-4 and y_rate > 0, "BANK IS DEGENERATE"

json.dump({"clips": clips, "plan": plan[:w], "N": w, "stride": a.stride,
           "token_grid": list(TG), "d_model": DM, "in_channels": CIN,
           "grid": list(GRID_DEFAULT.shape), "cell_m": GRID_DEFAULT.cell_m,
           "x_fwd_m": GRID_DEFAULT.x_fwd_m, "y_half_m": GRID_DEFAULT.y_half_m,
           "ckpt": a.ckpt, "step": int(step),
           "readout_grid": list(trunk.grid_shape),
           "d_readout": int(trunk.d_readout),
           "assert": {"mean_abs_tok": tk_abs, "mean_abs_pix": px_abs,
                      "occ_rate": y_rate, "rows_with_agent": n_nonzero_rows}},
          open(os.path.join(a.out, "idx.json"), "w", encoding="utf-8"), indent=1)
print("DONE", a.out)
