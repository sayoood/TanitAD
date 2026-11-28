#!/usr/bin/env python3
"""extract_g8.py -- deliverable 2.  The cached pool8x20 [T,160,1024] of the CURRENT frame (cache step t) of the 24,000 TRAIN windows -> fp16 memmaps F:/TanitAD-scratch/trigger_probe3_scratch/G8_<pop>.npy [12000, 163840]
(row-major 8x20 tokens), then the gates of PREREG 2:
  * every row filled and non-zero;
  * C4: g4_from_g8 == the independent loop implementation BIT-FOR-BIT on a spread of rows (max abs diff must be exactly 0.0);
  * alignment/continuity: relative L2 of g4_from_g8(G8 row) vs probe 1's cached pool4x10 row (V_<pop>.npy) on ALL 24,000 rows; median <= 0.05, p99.9 <= 0.15, max <= 0.5 or REFUSE.
Writes raw/extract_g8_meta.json, raw/align_<pop>.npz.   CUDA_VISIBLE_DEVICES=-1  python -I extract_g8.py [--gate-only]"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import g_lib as G  # noqa: E402
import numpy as np
import torch

torch.set_num_threads(4)
L = G.L


def fill():
    A = G.load_windows()
    clips = [str(c) for c in A["clips"]]
    ents = {}
    for ln in open(G.MANIFEST_TRAIN, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            ents[r["clip"]] = int(r["n_frames"])
    os.makedirs(G.SCR, exist_ok=True)
    pops = {}
    for nm in G.POPS:
        idx = np.load(f"{G.OLDRAW}/sample_{nm}.npz")["idx"]
        mm = np.lib.format.open_memmap(f"{G.SCR}/G8_{nm}.npy", mode="w+", dtype=np.float16, shape=(len(idx), 160 * 1024))
        pops[nm] = (idx, mm, np.zeros(len(idx), bool))
    by = {}
    for nm, (idx, mm, _) in pops.items():
        for row, w in enumerate(idx):
            by.setdefault(int(A["clip_i"][w]), []).append((nm, row, int(A["t"][w])))
    t0 = time.time()
    for j, (ci, items) in enumerate(sorted(by.items())):
        c = clips[ci]
        o = torch.load(f"{G.CACHE_CLIPS}/{c}.pt", map_location="cpu", weights_only=True, mmap=True)
        f8 = o["pool8x20"]
        T = f8.shape[0]
        assert f8.shape[1:] == (160, 1024), (c, tuple(f8.shape))
        assert T == ents[c], (c, T, ents[c])
        ts = np.array([t for _, _, t in items])
        assert ts.max() < T, (c, ts.max(), T)
        fr = f8[torch.as_tensor(ts)].to(torch.float16).reshape(len(ts), -1).numpy()
        for (nm, row, _), x in zip(items, fr):
            pops[nm][1][row] = x
            pops[nm][2][row] = True
        if j % 300 == 0:
            print(j, len(by), f"{time.time() - t0:.0f}s freeMB {G.free_mb()}", flush=True)
    for nm, (idx, mm, done) in pops.items():
        mm.flush()
        assert done.all(), (nm, int((~done).sum()))
    return {"clips_read": len(by), "fill_seconds": time.time() - t0}


def gates():
    out = {}
    for nm in G.POPS:
        G8 = np.load(f"{G.SCR}/G8_{nm}.npy", mmap_mode="r")
        V = np.load(f"{G.OLDSCR}/V_{nm}.npy", mmap_mode="r")
        assert G8.shape == (12000, 163840) and V.shape == (12000, 40960), (G8.shape, V.shape)
        rel = np.zeros(len(G8))
        nrm = np.zeros(len(G8))
        c4 = {"rows_checked": 0, "max_abs_diff": 0.0}
        for a in range(0, len(G8), 250):
            x = np.asarray(G8[a:a + 250], np.float32).reshape(-1, 160, 1024)
            g4 = G.g4_from_g8(x)
            v = np.asarray(V[a:a + 250], np.float32).reshape(-1, 40, 1024)
            rel[a:a + 250] = np.linalg.norm((g4 - v).reshape(len(x), -1), axis=1) / np.maximum(np.linalg.norm(v.reshape(len(x), -1), axis=1), 1e-12)
            nrm[a:a + 250] = np.linalg.norm(x.reshape(len(x), -1), axis=1)
            for i in (0, 77, 163, 249):                                   # C4: bit-for-bit vs the loop implementation (16 rows per 1,000)
                if i < len(x) and (a // 250) % 6 == 0:
                    ref = G.g4_reference(x[i])
                    c4["max_abs_diff"] = max(c4["max_abs_diff"], float(np.abs(ref - g4[i]).max()))
                    c4["rows_checked"] += 1
        q = {k: float(np.percentile(rel, p)) for k, p in (("median", 50), ("p99", 99), ("p99_9", 99.9))}
        q.update({"mean": float(rel.mean()), "max": float(rel.max()), "n_rows_gt_0_15": int((rel > 0.15).sum()), "min_row_norm": float(nrm.min())})
        ok = q["median"] <= 0.05 and q["p99_9"] <= 0.15 and q["max"] <= 0.5 and q["min_row_norm"] > 0 and c4["max_abs_diff"] == 0.0
        out[nm] = {"rel_l2_g4_vs_cached_pool4x10": q, "C4_g4_vs_loop_reference": c4, "gate_pass": bool(ok)}
        np.savez(f"{G.RAW}/align_{nm}.npz", rel=rel, norm=nrm)
        print(nm, json.dumps(out[nm]), flush=True)
        if not ok:
            raise SystemExit(f"GATE FAILED for {nm}: {out[nm]}")
    return out


if __name__ == "__main__":
    G.require_ram(3000, "extract_g8")
    meta = {"cache_dir": G.CACHE_CLIPS, "memmap_dir": G.SCR}
    if "--gate-only" not in sys.argv:
        meta.update(fill())
    meta["gates"] = gates()
    meta["sha256_windows_train"] = G.sha256_file(f"{G.OLDRAW}/windows_train.npz")
    assert meta["sha256_windows_train"] == G.WINDOWS_SHA, "windows_train.npz sha256 differs from the pinned value"
    json.dump(meta, open(f"{G.RAW}/extract_g8_meta.json", "w"), indent=1)
    print("done", json.dumps({k: v for k, v in meta.items() if k != "gates"}))
