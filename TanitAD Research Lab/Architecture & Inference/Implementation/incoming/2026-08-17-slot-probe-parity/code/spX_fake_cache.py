"""SPX — a RANDOM-LATENT cache with the REAL targets.

⭐ THIS IS A PIPELINE NULL CONTROL, not a result. It builds the same cache shape
`sp1` writes but fills the memory with noise. A probe trained on noise MUST fail
K1 (it cannot beat the corpus-median constant). If it "passes", the defect is in
MY scoring code — a leak from target to prediction — and any real number would
be uninterpretable. Running it before the real cache exists is the cheap way to
find that.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

STACK = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack")
for p in (str(STACK), str(STACK / "scripts")):
    sys.path.insert(0, p)
from train_p8_occupancy import JoinFileReader, episode_uid_of_clip  # noqa: E402
from tanitad.models.agent_slots import track_rates_from_join  # noqa: E402
from tanitad.data.bev_raster import GRID_DEFAULT as GRID  # noqa: E402

join = sys.argv[1]
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2
N_CELLS, D_R = 16, 128

src = JoinFileReader(join)
recs: dict[tuple[str, int], dict] = {}
with open(join, encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            r = json.loads(line)
            recs[(str(r["clip_id"]), int(r["frame_idx"]))] = r

g = torch.Generator().manual_seed(0)
rows = []
keys = sorted(recs)
for n, (cid, fi) in enumerate(keys):
    if n % stride:
        continue
    ag = np.asarray(src.lookup(episode_uid_of_clip(cid), fi),
                    dtype=np.float64).reshape(-1, 6)
    cls = src.lookup_classes(episode_uid_of_clip(cid), fi)
    rates, rmask = track_rates_from_join(recs.get((cid, fi - 1)),
                                         recs[(cid, fi)],
                                         recs.get((cid, fi + 1)))
    keep = ((ag[:, 0] > 0) & (ag[:, 0] <= GRID.x_fwd_m)
            & (np.abs(ag[:, 1]) <= GRID.y_half_m)) if len(ag) \
        else np.zeros(0, bool)
    rows.append({
        "episode_uid": episode_uid_of_clip(cid), "clip_id": cid,
        "frame_idx": fi,
        "cells": torch.randn(N_CELLS, D_R, generator=g).to(torch.float16),
        "tokens": None,
        "agents": torch.from_numpy(ag[keep]).float(),
        "classes": (np.asarray(cls, dtype=object)[keep].tolist()
                    if cls is not None and len(ag) else None),
        "rates": torch.from_numpy(rates[keep] if len(ag) else rates).float(),
        "rates_mask": torch.from_numpy(rmask[keep] if len(ag) else rmask),
        "n_out_of_grid": int(len(ag) - keep.sum())})

meta = {"run_stamp": "RANDOM-LATENT-NULL@na", "step": -1,
        "step_source": "n/a — synthetic", "n_frames": len(rows),
        "stride": stride, "n_cells": N_CELLS, "d_readout": D_R,
        "token_grid": None, "cuda_max_mem_gb": None,
        "_evidence_class": "SYNTHETIC (pipeline null control, not a result)"}
torch.save({"rows": rows, "meta": meta}, out / "latents.pt")
print(json.dumps({"n_frames": len(rows),
                  "n_clips": len({r['clip_id'] for r in rows})}))
