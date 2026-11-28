#!/usr/bin/env python3
"""extract_v.py -- step 2: the frozen DINOv3 pool4x10 features of the CURRENT frame (t) for the sampled windows of both populations -> fp16 memmaps under scratch/.
Also verifies, per clip, that the cache's pose_grid equals the native track's even frames (the same tie crawl_diag/ClipStore rely on) and that its length equals the manifest's n_frames.
CUDA_VISIBLE_DEVICES=-1  python -I extract_v.py"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np
import torch

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW, SCR = f"{HERE}/raw", f"{HERE}/scratch"
CACHE = "F:/TanitAD-scratch/speedread/cache/train/clips"
MANIFEST_TRAIN = "C:/Users/Admin/box2_ship/logs/v8f_dscale/manifest_train.jsonl"
TRACKS = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
D_V = 40 * 1024


def main():
    os.makedirs(SCR, exist_ok=True)
    sys.path[:0] = [f"{TREE}/stack", f"{TREE}/stack/scripts"]
    from tanitad.data import v8f_mini_data as D
    A = np.load(f"{RAW}/windows_train.npz", allow_pickle=False)
    clips = [str(c) for c in A["clips"]]
    ents = {}
    for ln in open(MANIFEST_TRAIN, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            ents[r["clip"]] = int(r["n_frames"])
    tracks = D.load_native_tracks(TRACKS)
    pops = {}
    for nm in ("pstop", "pmove"):
        idx = np.load(f"{RAW}/sample_{nm}.npz")["idx"]
        mm = np.lib.format.open_memmap(f"{SCR}/V_{nm}.npy", mode="w+", dtype=np.float16, shape=(len(idx), D_V))
        pops[nm] = (idx, mm)
    # clip -> [(pop, row, t)]
    by = {}
    for nm, (idx, mm) in pops.items():
        for row, w in enumerate(idx):
            by.setdefault(int(A["clip_i"][w]), []).append((nm, row, int(A["t"][w])))
    t0 = time.time()
    n_ok, worst_grid = 0, 0.0
    for j, (ci, items) in enumerate(sorted(by.items())):
        c = clips[ci]
        o = torch.load(f"{CACHE}/{c}.pt", map_location="cpu", weights_only=True, mmap=True)
        f4 = o["pool4x10"]
        T = f4.shape[0]
        assert T == ents[c] == len(tracks[c][0::2]), (c, T, ents[c])
        pg = o["pose_grid"].numpy().astype(np.float64)
        d = float(np.abs(pg - tracks[c][0::2][:, :4].astype(np.float64)).max()) if pg.shape == tracks[c][0::2][:, :4].shape else float("inf")
        worst_grid = max(worst_grid, d)
        ts = np.array([t for _, _, t in items])
        fr = f4[torch.as_tensor(ts)].to(torch.float16).reshape(len(ts), -1).numpy()
        for (nm, row, _), x in zip(items, fr):
            pops[nm][1][row] = x
        n_ok += 1
        if j % 300 == 0:
            print(j, len(by), f"{time.time() - t0:.0f}s", flush=True)
    for nm, (idx, mm) in pops.items():
        mm.flush()
    meta = {"clips_read": n_ok, "max_abs_pose_grid_minus_track": worst_grid, "cache_dir": CACHE, "seconds": time.time() - t0,
            "rows": {nm: int(len(p[0])) for nm, p in pops.items()}}
    json.dump(meta, open(f"{RAW}/extract_v_meta.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
