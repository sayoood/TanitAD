"""Direct inversion of the pose-index -> clip-time map, no optimiser.

`build_episode` interpolates egomotion at t_query = linspace(A, B, N), so the
map is EXACTLY affine in the pose index. Recover it by nearest-neighbour
matching each pose to the dense 100 Hz egomotion, then a robust (Theil-Sen)
affine fit over the matches. Stationary stretches are the only ambiguity and
they show up as slope outliers, which the median absorbs.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

amap = {int(r["ep_index"]): r for r in
        json.load(open("/workspace/_t3/val40_alias_map.json"))}


def dense_ego(ego):
    et = ego["timestamp"].to_numpy(np.float64)
    g = np.nonzero(np.diff(et) > 5e5)[0]
    hi = int(g[0]) + 1 if len(g) else len(et)
    return ego.iloc[:hi]


for ep in (0, 1, 2, 17, 33):
    rec = amap[ep]
    poses = torch.load(f"/root/valdata/physicalai-val-0c5f7dac3b11/{rec['file']}",
                       map_location="cpu", weights_only=False)["poses"].numpy()
    ego = pd.read_parquet(
        f"/workspace/_t3/pai_val40/{rec['alias']}.egomotion.parquet"
    ).sort_values("timestamp").reset_index(drop=True)
    d = dense_ego(ego)
    et = d["timestamp"].to_numpy(np.float64)
    ex = d["x"].to_numpy(np.float64)
    ey = d["y"].to_numpy(np.float64)
    ev = np.hypot(d["vx"].to_numpy(np.float64), d["vy"].to_numpy(np.float64))
    T = poses.shape[0]
    dist = ((poses[:, 0][:, None] - ex[None]) ** 2
            + (poses[:, 1][:, None] - ey[None]) ** 2
            + (poses[:, 3][:, None] - ev[None]) ** 2)
    j = dist.argmin(1)
    tm = et[j]
    resid_m = np.sqrt(dist[np.arange(T), j]).max()
    i = np.arange(T, dtype=np.float64)
    # Theil-Sen slope from a stride-20 subsample of pairs
    sl = []
    for a in range(0, T - 20, 7):
        for b in range(a + 20, T, 37):
            sl.append((tm[b] - tm[a]) / (b - a))
    slope = float(np.median(sl))
    inter = float(np.median(tm - slope * i))
    fit = inter + slope * i
    print(f"ep {ep} {rec['alias']} T={T} dense_rows={len(et)} "
          f"dense_span=[{et.min()/1e6:.3f},{et.max()/1e6:.3f}]s")
    print(f"   NN max resid (xy+v) = {resid_m:.4f}   slope={slope/1e3:.4f} ms "
          f"t0={inter/1e6:.4f}s  t_end={(inter+slope*(T-1))/1e6:.4f}s")
    print(f"   |NN - affine| max = {np.abs(tm - fit).max()/1e3:.2f} ms, "
          f"median = {np.median(np.abs(tm - fit))/1e3:.2f} ms")
    xy = np.stack([np.interp(fit, et, ex), np.interp(fit, et, ey)], 1)
    print(f"   affine-map xy rms = "
          f"{np.sqrt(((xy - poses[:, :2])**2).sum(1).mean()):.5f} m")
