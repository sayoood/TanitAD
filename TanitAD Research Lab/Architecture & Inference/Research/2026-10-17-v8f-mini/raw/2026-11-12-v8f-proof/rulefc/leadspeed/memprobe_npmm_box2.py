"""memprobe_npmm_box2.py -- card ADD149: commit per clip of read-only np.memmap token views (tokstore.py), kept open, every page touched;
bytes compared with torch.load for every clip; a planted wrong offset must differ. Memory only; no label or outcome is read.

  python memprobe_npmm_box2.py <n>
"""
import gc
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memstat  # noqa: E402
import tokstore  # noqa: E402
import torch  # noqa: E402

CORPUS = "D:/tanitad/cache_full/train"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
man = [json.loads(x) for x in open(os.path.join(CORPUS, "manifest.jsonl"), encoding="utf-8") if x.strip()]
recs = man[3 * N:4 * N]
shape = (101, 40, 1024)
gc.collect()
o0, a0, t0 = memstat.own_commit_gb(), memstat.status()["commit_avail_GB"], time.time()
keep, s = [], 0
for r in recs:
    mm = tokstore.open_tokens(os.path.join(CORPUS, r["file"]), r.get("shapes", {}).get("pool4x10", shape))
    s += int(mm.sum(dtype=np.uint64) % 997)                 # touch every byte
    keep.append(mm)
gc.collect()
o1, a1, t1 = memstat.own_commit_gb(), memstat.status()["commit_avail_GB"], time.time()
eq = 0
for r, mm in zip(recs, keep):                                 # equality with torch.load, every clip
    d = torch.load(os.path.join(CORPUS, r["file"]), map_location="cpu", weights_only=False, mmap=True)
    eq += int(np.array_equal(np.asarray(mm), d["pool4x10"].view(torch.uint8).numpy()))
    del d
gc.collect()
r0 = recs[0]
p0 = os.path.join(CORPUS, r0["file"])
off, nb = tokstore.member(p0, shape)
wrong = np.memmap(p0, dtype=np.uint8, mode="r", offset=off - 64, shape=shape)
d = torch.load(p0, map_location="cpu", weights_only=False, mmap=True)
planted_differs = not np.array_equal(np.asarray(wrong), d["pool4x10"].view(torch.uint8).numpy())
del d, wrong
print(json.dumps({"n": N, "own_commit_MB_per_clip": round((o1 - o0) * 2 ** 30 / 1e6 / N, 3), "commit_avail_drop_MB_per_clip": round((a0 - a1) * 2 ** 30 / 1e6 / N, 3),
                  "open_touch_seconds": round(t1 - t0, 1), "equal_to_torch_load": eq, "of": N, "planted_offset_differs": bool(planted_differs), "phys_avail_GB": round(memstat.status()["phys_avail_GB"], 2)}))
