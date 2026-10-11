#!/usr/bin/env python3
"""Extra disjoint cached-mode samples cached_r4..r6 (240 sets each) from the SAME seeded shuffle as mlab_plan.py -- written to NEW
files only, so a running chain never sees a file being rewritten.  python mlab_plan_extra.py"""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlab_plan as MP
HERE = MP.HERE
idx = json.load(open(os.path.join(HERE, "index.json"), encoding="utf-8"))
P = np.load(MP.PROPS, allow_pickle=True)
pidx = {str(t): i for i, t in enumerate(P["token"])}
census = {}
for line in open(MP.CENSUS, encoding="utf-8"):
    r = json.loads(line)
    if "error" not in r:
        census[r["token"]] = np.asarray(r["pdms"][1:65], np.float64)
toks = sorted(t for t in idx if t in pidx and t in census)
rng = random.Random(MP.SEED)
pool = toks[:]
rng.shuffle(pool)
for r in (4, 5, 6):
    name = f"cached_r{r}"
    if os.path.exists(os.path.join(HERE, "plan", name + ".json")):
        continue
    tl = pool[(r - 1) * 240: r * 240]
    ptk = np.asarray([pidx[t] for t in tl])
    np.savez(os.path.join(HERE, "plan", name + ".npz"), token=np.asarray(tl), proposals=P["proposals"][ptk].astype(np.float32),
             census_pdms=np.stack([census[t] for t in tl]))
    json.dump({"name": name, "n": len(tl), "tokens": tl, "logs": sorted({idx[t]["log"] for t in tl}),
               "paths": {t: idx[t]["path"] for t in tl}}, open(os.path.join(HERE, "plan", name + ".json"), "w"), indent=0)
    print(name, len(tl))
