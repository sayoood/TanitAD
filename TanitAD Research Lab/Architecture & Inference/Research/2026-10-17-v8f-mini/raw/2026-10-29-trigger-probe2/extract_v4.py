#!/usr/bin/env python3
"""extract_v4.py -- ARM 3 (PREREG_FROZEN.md 6, 8): the frozen DINOv3 pool4x10 features of frames t-1, t-2, t-3 (cache steps; hist = 4 so the model's own 4 frames are t-3..t) of the SAME sampled TRAIN windows
-> fp16 memmaps scratch/Vm{1,2,3}_<pop>.npy  [12000, 40960] (fp8 -> fp16 is exact).  Frame t is the previous package's scratch/V_<pop>.npy (control V1 checks the equality on a subset).
Resumable by clip, time-boxed.   CUDA_VISIBLE_DEVICES=-1  python -I extract_v4.py --stage run [--budget-s 200] | --stage check"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import torch

import common2 as C  # noqa: E402

torch.set_num_threads(4)
D_V = 40 * 1024
LAGS = (1, 2, 3)


def gather_lag(f4, ts, lag):
    """Frames t - lag of a clip's pool4x10 tensor f4 [T, 40, 1024] at the cache steps ts -> fp16 numpy [len(ts), 40960]."""
    return f4[torch.as_tensor(np.asarray(ts) - lag)].to(torch.float16).reshape(len(ts), -1).numpy()


def by_clip():
    A = C.load_windows()
    by = {}
    for pop in C.POPS:
        idx = np.load(f"{C.OLDRAW}/sample_{pop}.npz")["idx"]
        for row, w in enumerate(idx):
            by.setdefault(int(A["clip_i"][w]), []).append((pop, row, int(A["t"][w])))
    return A, by


def stage_run(budget_s, lo=0, hi=None):
    t0 = time.time()
    C.require_ram(2500, "extract_v4")
    A, by = by_clip()
    clips = [str(c) for c in A["clips"]]
    sharded = (lo != 0 or hi is not None)                                        # a worker on order[lo:hi] (the memmaps already exist; rows are disjoint per clip, so concurrent writers are safe)
    prog = f"{C.SCR}/v4_progress{'_' + str(lo) + '_' + str(hi) if sharded else ''}.json"
    j0 = json.load(open(prog))["j_done"] if os.path.exists(prog) else lo
    mode = "r+" if (j0 > 0 or sharded) else "w+"
    mm = {(lag, pop): np.lib.format.open_memmap(f"{C.SCR}/Vm{lag}_{pop}.npy", mode=mode, dtype=np.float16, shape=(12000, D_V)) for lag in LAGS for pop in C.POPS}
    order = sorted(by.items())
    ents = {}
    for ln in open(C.MANIFEST_TRAIN, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            ents[r["clip"]] = int(r["n_frames"])
    j = j0
    hi = len(order) if hi is None else hi
    while j < hi:
        if time.time() - t0 > budget_s:
            break
        ci, items = order[j]
        c = clips[ci]
        o = torch.load(f"{C.CACHE_CLIPS}/{c}.pt", map_location="cpu", weights_only=True, mmap=True)
        f4 = o["pool4x10"]
        assert f4.shape[0] == ents[c], (c, f4.shape, ents[c])
        ts = np.array([t for _, _, t in items])
        assert ts.min() >= 3, "a window with t < 3 has fewer than 4 real frames"
        for lag in LAGS:
            fr = gather_lag(f4, ts, lag)
            for (pop, row, _), x in zip(items, fr):
                mm[(lag, pop)][row] = x
        j += 1
        if j % 300 == 0:
            print(j, len(order), f"{time.time() - t0:.0f}s", flush=True)
    for m_ in mm.values():
        m_.flush()
    json.dump({"j_done": j, "n_clips": len(order)}, open(prog, "w"))
    rep = {"lo": lo, "hi": hi, "j_done": j, "n_clips": len(order), "complete": j >= hi, "seconds_last_call": time.time() - t0, "cache_dir": C.CACHE_CLIPS, "lags": list(LAGS)}
    json.dump(rep, open(f"{C.RAW}/extract_v4_status{'_' + str(lo) + '_' + str(hi) if sharded else ''}.json", "w"), indent=1)
    print(json.dumps(rep), flush=True)


def missing_clips():
    """Clip positions j (in sorted order) with at least one UNFILLED row (the first 4096 values of a lag frame are all zero: a real fp8 feature row is never all zero)."""
    A, by = by_clip()
    order = sorted(by.items())
    bad = set()
    for pop in C.POPS:
        for lag in LAGS:
            X = np.load(f"{C.SCR}/Vm{lag}_{pop}.npy", mmap_mode="r")
            empty = ~np.asarray(X[:, :4096], np.float32).any(1)
            for j, (ci, items) in enumerate(order):
                if any(empty[row] for p_, row, _ in items if p_ == pop):
                    bad.add(j)
    return sorted(bad)


def stage_fixup(budget_s):
    miss = missing_clips()
    print("unfilled clips:", len(miss), flush=True)
    A, by = by_clip()
    clips = [str(c) for c in A["clips"]]
    order = sorted(by.items())
    mm = {(lag, pop): np.lib.format.open_memmap(f"{C.SCR}/Vm{lag}_{pop}.npy", mode="r+", dtype=np.float16, shape=(12000, D_V)) for lag in LAGS for pop in C.POPS}
    t0 = time.time()
    for j in miss:
        if time.time() - t0 > budget_s:
            break
        ci, items = order[j]
        o = torch.load(f"{C.CACHE_CLIPS}/{clips[ci]}.pt", map_location="cpu", weights_only=True, mmap=True)
        ts = np.array([t for _, _, t in items])
        for lag in LAGS:
            fr = gather_lag(o["pool4x10"], ts, lag)
            for (pop, row, _), x in zip(items, fr):
                mm[(lag, pop)][row] = x
    for m_ in mm.values():
        m_.flush()
    left = missing_clips()
    json.dump({"unfilled_before": len(miss), "unfilled_after": len(left), "complete": len(left) == 0}, open(f"{C.RAW}/extract_v4_fill.json", "w"))
    print("unfilled after:", len(left), flush=True)


def stage_check(n=200):
    """V1: frame t of the old extraction == the cache's frame t; frame order t-1..t-3 distinct from t (>= 99 % of rows differ); the memmap rows are not all-zero."""
    A, by = by_clip()
    clips = [str(c) for c in A["clips"]]
    out = {"n_per_pop": n, "pops": {}}
    for pop in C.POPS:
        V0 = np.load(f"{C.OLDSCR}/V_{pop}.npy", mmap_mode="r")
        Vm = {lag: np.load(f"{C.SCR}/Vm{lag}_{pop}.npy", mmap_mode="r") for lag in LAGS}
        idx = np.load(f"{C.OLDRAW}/sample_{pop}.npz")["idx"]
        rows = np.linspace(0, 11999, n).astype(int)
        mx_t, differs, zero_rows = 0.0, {lag: 0 for lag in LAGS}, 0
        for r in rows:
            w = idx[r]
            c, t = clips[int(A["clip_i"][w])], int(A["t"][w])
            o = torch.load(f"{C.CACHE_CLIPS}/{c}.pt", map_location="cpu", weights_only=True, mmap=True)
            f4 = o["pool4x10"]
            ft = f4[t].to(torch.float16).reshape(-1).numpy()
            mx_t = max(mx_t, float(np.abs(ft.astype(np.float32) - np.asarray(V0[r], np.float32)).max()))
            for lag in LAGS:
                fl = f4[t - lag].to(torch.float16).reshape(-1).numpy()
                mx = float(np.abs(fl.astype(np.float32) - np.asarray(Vm[lag][r], np.float32)).max())
                assert mx == 0.0, (pop, r, lag, mx)                                    # the stored lag frame equals the cache's frame t - lag
                differs[lag] += int(np.abs(ft.astype(np.float32) - fl.astype(np.float32)).max() > 0)
            zero_rows += int(not np.asarray(Vm[3][r], np.float32).any())
        out["pops"][pop] = {"max_abs_frame_t_old_vs_cache": mx_t, "share_rows_where_lag_frame_differs_from_t": {str(l): differs[l] / len(rows) for l in LAGS}, "all_zero_lag3_rows": zero_rows, "stored_lag_frames_equal_cache": True}
    json.dump(out, open(f"{C.RAW}/control_v4.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("run", "check", "fixup"))
    ap.add_argument("--budget-s", type=float, default=200.0)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=None)
    a = ap.parse_args()
    {"run": lambda: stage_run(a.budget_s, a.lo, a.hi), "check": stage_check, "fixup": lambda: stage_fixup(a.budget_s)}[a.stage]()
