import sys, json
sys.path[:0] = ["/root/dscale10k/tree/stack", "/root/dscale10k/tree", "/root/dscale10k/study"]
import numpy as np, torch
from tanitad.data.clip_clock import pose_dt
import dscale10k_merge as M
man = [json.loads(l) for l in open("/root/dscale10k/merged/manifest.jsonl")][:6] + [json.loads(l) for l in open("/root/dscale10k/merged/manifest.jsonl")][9000:9004]
z = np.load("/root/dscale10k/merged/v9_labels_union.npz", allow_pickle=True)
sid = z["clip__sid"]; dt = z["clip__dt_s"]; g0 = z["clip__grid_start_s"]; cs = z["clip__clock_src"]
ix = {int(s): i for i, s in enumerate(sid)}
for r in man:
    o = torch.load("/root/data/cache_union_dscale/train/clips/%s.pt" % r["clip"], map_location="cpu", weights_only=True, mmap=True)
    pg = o["pose_grid"]
    s = M.stable_episode_id(r["clip"])
    i = ix.get(s)
    print(r["clip"][:8], tuple(pg.shape), pg.dtype, "pose_dt", pose_dt(pg.double()), "rel dt", dt[i] if i is not None else None, "g0", g0[i] if i is not None else None, "src", cs[i] if i is not None else None, "v range", float(pg[:, 3].min()), float(pg[:, 3].max()))
