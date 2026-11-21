"""HP-2's "n >= 200 PER STRATUM" bar, MEASURED on the 600-episode val.

Uses the SHIPPED code, not a re-implementation:
  * window starts        taniteval.rollout.collect  (range(0, T-W-K, stride))
  * head_deg per window  driving_diagnostic.net_heading_change_deg  (the exact
                         call rollout.collect:161 makes)
  * speed per window     poses[last, 3]             (rollout.collect:160)
  * the strata           taniteval.corridor.strata  (E1a's 3-way + overall,
                         junction = |net heading change over 2 s| >= 10 deg)

Poses-only: torch.load(..., mmap=True) faults in ~3 KB per episode, not 117 MB,
so all 600 episodes cost seconds and the 66 GB cache is only READ.

The resampling unit is the EPISODE, so `n_episode_clusters` is the number that
meets or fails the bar. `n_windows` is reported beside it and never instead.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from driving_diagnostic import net_heading_change_deg            # noqa: E402
from taniteval import corridor                                   # noqa: E402

SRC = sys.argv[1]
OUT = sys.argv[2]
W = 8
STRIDE = 8
KS = [20, 60, 90, 120, 150, 185]
HD_HORIZON = 20            # head_deg is the 2 s net heading change, FIXED
                           # across K so strata stay comparable (corridor.py:160)

files = sorted(Path(SRC).glob("ep_*.pt"))
print(f"[strata] {SRC}: {len(files)} episodes", flush=True)

rows = {K: {"eid": [], "hd": [], "spd": []} for K in KS}
for i, f in enumerate(files):
    d = torch.load(f, weights_only=False, map_location="cpu", mmap=True)
    po = torch.as_tensor(d["poses"]).clone().float()
    T = po.shape[0]
    for K in KS:
        starts = list(range(0, T - W - K, STRIDE))
        if not starts:
            continue
        # `last` must leave HD_HORIZON frames of future for the 2 s heading
        # change; at K >= 20 that is guaranteed because K >= HD_HORIZON.
        last = torch.tensor([t + W - 1 for t in starts])
        hd = net_heading_change_deg(po, last, horizon=HD_HORIZON)
        rows[K]["eid"].extend([f.name] * len(starts))
        rows[K]["hd"].append(hd)
        rows[K]["spd"].append(po[last, 3])
    if i and i % 200 == 0:
        print(f"  {i}/{len(files)}", flush=True)

out = {"source_dir": SRC, "n_episodes": len(files), "window": W,
       "stride": STRIDE, "junction_deg": corridor.JUNCTION_DEG,
       "head_deg_horizon_steps": HD_HORIZON,
       "method": ("strata = taniteval.corridor.strata (E1a 3-way); "
                  "head_deg = driving_diagnostic.net_heading_change_deg over "
                  "the first 2 s, held FIXED across K; speed = poses[last,3]. "
                  "Unit of power = EPISODE CLUSTER."),
       "by_K": {}}

for K in KS:
    if not rows[K]["hd"]:
        out["by_K"][str(K)] = {"n_windows": 0, "strata": {}}
        continue
    hd = torch.cat(rows[K]["hd"]).numpy()
    spd = torch.cat(rows[K]["spd"]).numpy()
    eid = np.array(rows[K]["eid"])
    st = corridor.strata(hd, spd)
    blk = {"K": K, "horizon_s": round(K * 0.1, 2),
           "n_windows": int(len(hd)),
           "n_episode_clusters": int(len(set(eid.tolist()))),
           "speed_median_mps": round(float(np.median(spd)), 4),
           "strata": {}}
    for name, mask in st.items():
        m = np.asarray(mask, dtype=bool)
        e = set(eid[m].tolist())
        blk["strata"][name] = {
            "n_windows": int(m.sum()),
            "n_episode_clusters": len(e),
            "meets_single_arm_bar_40": len(e) >= 40,
            "meets_two_arm_bar_200": len(e) >= 200,
            "windows_per_cluster": round(float(m.sum()) / max(1, len(e)), 3),
        }
    out["by_K"][str(K)] = blk
    print(f"  K={K:>3} ({K*0.1:>4.1f}s) win={blk['n_windows']:>6} "
          f"clusters={blk['n_episode_clusters']:>3} | " +
          " ".join(f"{n}:{v['n_episode_clusters']}"
                   f"{'OK' if v['meets_two_arm_bar_200'] else 'LOW'}"
                   for n, v in blk["strata"].items()), flush=True)

Path(OUT).write_text(json.dumps(out, indent=1))
print(f"-> {OUT}")
