#!/usr/bin/env python3
"""crosscheck_feed.py -- is geodac's sampler the SAME instrument as the repo's H2 feed (refe/aux_drivable_head.py:248 sample_pred_sdf / :262 feed_scalars)?

Compares, on random 40 x 32 predicted-like SDF rasters (2 m cells) and random footprint corner points: (1) the bilinear lookup, (2) the unknown
(out-of-extent) value, (3) min SDF / share < 0 of a trajectory set vs feed_scalars (which EXCLUDES t = 0, uses 20-pose 5 Hz trajectories mapped to NAVSIM's grid).
Writes CROSSCHECK_FEED.json.  A DIFFERENCE here is a finding, not a failure: it says which numbers are not interchangeable.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(PK / "refe"))
import torch
import geodac as G
import aux_drivable_head as H

rng = np.random.default_rng(3)
out = {}
B = 6
sdf = rng.normal(0, 3, (B, 40, 32)).clip(-6, 6).astype(np.float32)
pts = rng.uniform([-12, -36], [76, 36], (B, 500, 2))                        # includes points OUTSIDE the window on purpose
s_ref, inb = H.sample_pred_sdf(torch.tensor(sdf), torch.tensor(pts, dtype=torch.float32))
s_ref, inb = s_ref.numpy(), inb.numpy()
mine = np.stack([G.sample_field(sdf[b].astype(np.float64), G.PRED_GEOM, pts[b], "bilinear") for b in range(B)])
known = ~np.isnan(mine)
out["sampler"] = {"in_extent_equal": bool((known == inb).all()), "max_abs_diff_in_extent_m": float(np.abs(mine[known] - s_ref[inb]).max()),
                  "repo_unknown_value_m": float(np.unique(s_ref[~inb])[0]) if (~inb).any() else None, "geodac_unknown": "NaN, then policy (drivable = benefit of the doubt)"}
# trajectory level: 20-pose 5 Hz REFe trajectories
import field_loss as FL
tr = np.zeros((B, 8, 20, 3), np.float32)
t = np.arange(1, 21) * 0.2
for b in range(B):
    for k in range(8):
        v, curv = rng.uniform(2, 14), rng.uniform(-0.04, 0.04)
        th = curv * v * t
        tr[b, k, :, 0] = np.cumsum(v * np.cos(th)) * 0.2
        tr[b, k, :, 1] = np.cumsum(v * np.sin(th)) * 0.2
        tr[b, k, :, 2] = th
feats, oob = H.feed_scalars(torch.tensor(sdf), torch.tensor(tr))
feats = feats.numpy()
import navsim_dac as ND
dmin, dshare, dmin_t0 = [], [], []
for b in range(B):
    for k in range(8):
        p8 = ND.to_navsim(tr[b, k].astype(np.float64))
        r = G.geodac(sdf[b].astype(np.float64), G.PRED_GEOM, p8, mode="bilinear", unknown="drivable")
        # repo: steps 1..40 only, unknown corners count as +6 m
        dense = G.densify(p8)[1:]
        cs = G.corners(dense)
        v = G.sample_field(sdf[b].astype(np.float64), G.PRED_GEOM, cs, "bilinear")
        v = np.where(np.isnan(v), 6.0, v)
        dmin.append(float(v.min()) - float(feats[b, k, 0])); dshare.append(float((v < 0).mean()) - float(feats[b, k, 2]))
        dmin_t0.append(float(r.min_margin_m) - float(feats[b, k, 0]))
out["trajectory_feed_vs_geodac_steps_1_to_40"] = {"max_abs_diff_min_sdf_m": float(np.abs(dmin).max()), "max_abs_diff_share_below_0": float(np.abs(dshare).max()),
                                                  "note": "repo feed excludes t=0 and runs 20-pose 5 Hz trajectories through FL.poses_10hz_navsim; geodac uses the 8-pose NAVSIM grid; both reach the same 0.1 s grid"}
out["geodac_including_t0_vs_repo_feed_min_sdf_max_abs_diff_m"] = float(np.abs(dmin_t0).max())
json.dump(out, open(HERE / "CROSSCHECK_FEED.json", "w"), indent=1)
print(json.dumps(out, indent=1))
