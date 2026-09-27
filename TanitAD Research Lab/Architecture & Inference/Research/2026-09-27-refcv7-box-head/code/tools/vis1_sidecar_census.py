#!/usr/bin/env python3
"""VIS-1 census of the sidecar: POSITIVES / IGNORE per labelled NOW frame, per split (the R4 query-budget basis
after VIS-1, A10 §15.4), with the trainer's OWN target filter applied to the stored float32 centres.

usage: python vis1_sidecar_census.py <sidecar.npz> <out.json>     (sha12 only; reads nothing else)
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else ".")
from tanitad.data import vis1 as V                          # noqa: E402
from tanitad.refs.refc_agents import visible_target_filter  # noqa: E402

sc = V.VIS1Sidecar(sys.argv[1])
a = sc.a
cx, cy = torch.from_numpy(a["cx"]), torch.from_numpy(a["cy"])
t = {"box": torch.stack([cx, cy, torch.ones_like(cx), torch.ones_like(cx)], -1)[None],
     "valid": torch.ones(1, len(cx), dtype=torch.bool)}
in_f = visible_target_filter(t)["valid"][0].numpy()
nf, nv = a["n_full"].astype(np.float64), a["n_vis"].astype(np.int64)
with np.errstate(divide="ignore", invalid="ignore"):
    vf = np.where(nf > 0, nv / np.maximum(nf, 1), np.nan)
pos = in_f & np.isfinite(vf) & (vf >= V.VIS1_T_POS) & (nv >= V.VIS1_PX_MIN)
ign = in_f & ~pos
hidden = in_f & np.isfinite(vf) & (vf < 0.05)
fp = a["frame_ptr"]
row_frame = np.repeat(np.arange(len(fp) - 1), np.diff(fp))
clip_of_frame = a["frame_clip"]
split_of_clip = a["clip_split"]
out = {"sidecar_sha256": sc.sha256, "rule": V.vis1_rule_dict(), "splits": {}}
for split in ("train", "eval"):
    fmask = split_of_clip[clip_of_frame] == split
    n_frames = int(fmask.sum())
    def per_frame(m):
        c = np.bincount(row_frame[m], minlength=len(fp) - 1)
        return c[fmask]
    P, I, T, H = per_frame(pos), per_frame(ign), per_frame(in_f), per_frame(hidden)
    def q(x):
        return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99)),
                "max": int(x.max())}
    out["splits"][split] = {"n_clips": int((split_of_clip == split).sum()), "n_frames": n_frames,
                            "targets_in_filter_per_frame": q(T), "vis1_positives_per_frame": q(P),
                            "vis1_ignore_per_frame": q(I), "vis_below_0p05_per_frame": q(H),
                            "totals": {"in_filter": int(T.sum()), "positive": int(P.sum()), "ignore": int(I.sum()),
                                       "hidden_lt_0p05": int(H.sum())},
                            "frac_in_filter_positive": float(P.sum() / max(T.sum(), 1))}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out["splits"], indent=1))
