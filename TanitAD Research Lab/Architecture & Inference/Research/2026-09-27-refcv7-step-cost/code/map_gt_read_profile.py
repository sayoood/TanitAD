#!/usr/bin/env python3
"""cProfile of ONE refcv7 loader cost: the 10 cm map GT read (`FineMapGTStore.frames_for_windows`)
on the dev-box eval139 map GT, at the launch extent, cold LRU per call (the training regime: shuffled
windows over 4,369 clips vs an LRU of 4 per worker => ~0 hit rate). Clip ids are read from the cache
manifest and never printed."""
import cProfile, io, json, pstats, random, sys, time
from pathlib import Path
tree, out = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, f"{tree}/stack")
import torch
from tanitad.data import semantic_map_gt_fine as SF
from tanitad.data.semantic_map_gt import sha12
from tanitad.models import map_head_hires as MH
m = torch.load("D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt", weights_only=False)
root = Path("D:/refcv6_eval_kit/data/sam3_gt_v3_eval")
have = {p.name.split(".")[0] for p in root.glob("*.npz")}
cids = [str(c) for c in m["clip_id"] if sha12(str(c)) in have]
ext = SF.MapExtent(100.0, 30.0)
rng = random.Random(0)
calls = [(rng.choice(cids), rng.randrange(0, 180)) for _ in range(12)]
times = []
pr = cProfile.Profile()
for cid, w in calls:
    st = SF.FineMapGTStore(root, max_open=4, extent=ext)        # cold: a fresh store per call
    t = time.perf_counter()
    pr.enable()
    fm = st.frames_for_windows(cid, [w], n_stack=3)
    pr.disable()
    times.append(time.perf_counter() - t)
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28)
res = {"n": len(times), "median_s": sorted(times)[len(times) // 2], "times_s": [round(x, 3) for x in times]}
out.mkdir(parents=True, exist_ok=True)
(out / "map_gt_read_profile.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
(out / "map_gt_read_cprofile.txt").write_text(s.getvalue(), encoding="utf-8")
print(json.dumps(res))
print(s.getvalue()[:6000])
