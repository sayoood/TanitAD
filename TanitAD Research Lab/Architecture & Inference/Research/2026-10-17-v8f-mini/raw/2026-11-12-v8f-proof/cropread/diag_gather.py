# diag_gather.py -- lead's timing probe for card ADD99 (no label, no head, no value): how long one batch of 256 five-frame samples takes to read from the token files.
import os
import sys
import time

import numpy as np

root = sys.argv[1]
rows = [ln.split() for ln in open(os.path.join(root, "lists", "train_final.txt"), encoding="utf-8") if ln.strip()]
clips = [r[0] for r in rows if r[1] == "fit"]
rng = np.random.default_rng(0)
maps = {}
t0 = time.time()
for rep in range(3):
    t = time.time()
    out = np.empty((256, 5, 130, 1024), np.float16)
    for k in range(256):
        c = clips[int(rng.integers(0, len(clips)))]
        a = maps.get(c)
        if a is None:
            a = maps[c] = np.load(os.path.join(root, "tok", "D", c + ".npy"), mmap_mode="r")
        f = int(rng.integers(4, a.shape[0]))
        out[k] = np.asarray(a[f - 4:f + 1]).reshape(5, 130, 1024)
    print("batch %d gather %.2f s" % (rep, time.time() - t))
t = time.time()
x = out.astype(np.float32)
print("to float32 on cpu %.2f s; total %.1f s" % (time.time() - t, time.time() - t0))
