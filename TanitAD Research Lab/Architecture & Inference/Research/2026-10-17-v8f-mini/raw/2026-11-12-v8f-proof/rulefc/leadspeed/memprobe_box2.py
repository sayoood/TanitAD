"""memprobe_box2.py -- card ADD149: the commit charge per loaded clip on box 2. Memory only; no label or outcome is read.

  python memprobe_box2.py <n> <mode>     mode: plain | mmap (load, clone the 4x10 tokens) | view (torch.load(mmap=True), keep the 4x10 view, no copy, and touch every byte)
One mode per process (two modes in one process are order-confounded: the second reuses heap the first committed).
"""
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memstat  # noqa: E402
import torch  # noqa: E402

CORPUS = "D:/tanitad/cache_full/train"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MODE = sys.argv[2] if len(sys.argv) > 2 else "plain"
OFF = {"plain": 0, "mmap": 1, "view": 2}[MODE]
man = [json.loads(x) for x in open(os.path.join(CORPUS, "manifest.jsonl"), encoding="utf-8") if x.strip()]
recs = man[OFF * N:(OFF + 1) * N]
gc.collect()
o0, a0, t0 = memstat.own_commit_gb(), memstat.status()["commit_avail_GB"], time.time()
keep = []
chk = 0
for r in recs:
    p = os.path.join(CORPUS, r["file"])
    d = torch.load(p, map_location="cpu", weights_only=False, mmap=(MODE != "plain"))
    if MODE == "view":
        t = d["pool4x10"].view(torch.uint8)
        chk += int(t[::7, ::3, ::64].sum())                       # touch pages across the tensor
        keep.append((t, d["v"]))
    else:
        keep.append((d["pool4x10"].view(torch.uint8).clone(), d["v"].clone()))
    del d
gc.collect()
o1, a1 = memstat.own_commit_gb(), memstat.status()["commit_avail_GB"]
t1 = time.time()
if MODE == "view":                                                   # a batch-like random read over the kept views
    import random
    random.seed(0)
    for _ in range(200):
        c = random.randrange(len(keep)); f = random.randrange(10, 101)
        chk += int(keep[c][0][f - 10:f].float().sum() > 0)
tok_mb = sum(k[0].numel() for k in keep) / 1e6 / N
o2, a2 = memstat.own_commit_gb(), memstat.status()["commit_avail_GB"]
print(json.dumps({"n": N, "mode": MODE, "torch": torch.__version__, "own_commit_MB_per_clip": round((o1 - o0) * 2 ** 30 / 1e6 / N, 3),
                  "commit_avail_drop_MB_per_clip": round((a0 - a1) * 2 ** 30 / 1e6 / N, 3), "own_commit_after_reads_MB_per_clip": round((o2 - o0) * 2 ** 30 / 1e6 / N, 3),
                  "token_MB_per_clip": round(tok_mb, 3), "load_seconds": round(t1 - t0, 1), "read200_seconds": round(time.time() - t1, 2), "chk": chk % 1000}))
