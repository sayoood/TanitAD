"""SMOKE equivalence check (not a result): the chain's decisive-rollout command, FULL SIZE, on the T1-HEADS causal-s0 checkpoint (M2b_s0_hacc.pt, sha256 71de3c61...) vs the banked
T1-HEADS output of the same checkpoint (C:/Users/Admin/t1hd_run).  If the chain's command (flags, tool tree, inputs, threads) is the T1-HEADS one, every array must agree."""
import json
import sys

import numpy as np

A = "E:/Projects/TanitAD-artifacts/orct1/smoke_equiv"                 # this chain's command
B = "C:/Users/Admin/t1hd_run"                                         # T1-HEADS banked
ta, tb = sys.argv[1], sys.argv[2]                                     # tags
za, zb = np.load(f"{A}/ticks/ticks_{ta}.npz", allow_pickle=True), np.load(f"{B}/ticks/ticks_{tb}.npz", allow_pickle=True)
ka, kb = set(za.files), set(zb.files)
print("keys equal:", ka == kb, "| only here:", sorted(ka - kb), "| only banked:", sorted(kb - ka))
bad, same = [], 0
for k in sorted(ka & kb):
    a, b = za[k], zb[k]
    if a.dtype.kind in "OUS":
        ok = (a.shape == b.shape) and (a.tolist() == b.tolist() if k not in ("tag", "ckpt") else True)
        d = "n/a" if ok else "differs"
    elif a.shape != b.shape:
        ok, d = False, f"shape {a.shape} vs {b.shape}"
    else:
        ok = bool(np.array_equal(a, b, equal_nan=True))          # NaNs (unscored GT) count as equal where both have them
        d = 0.0 if ok else float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    same += bool(ok)
    if not ok:
        bad.append((k, d))
print(f"arrays bit-identical: {same}/{len(ka & kb)}; not identical: {bad}")
ja, jb = json.load(open(f"{A}/t1v8f_{ta}.json")), json.load(open(f"{B}/t1v8f_{tb}.json"))


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in ("tag", "ckpt", "ticks_file", "wall_s", "seconds")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


sa, sb = strip(ja), strip(jb)
print("result json equal (minus tag / ckpt / ticks_file / wall):", sa == sb)
if sa != sb:
    for k in sa:
        if sa[k] != sb.get(k):
            print("  differs at top-level key:", k)
print("header here:", {k: ja["t1v8f"][k] for k in ("tag", "limit", "replan_steps", "schedule", "scene", "nav", "n_windows")})
print("header T1-HEADS:", {k: jb["t1v8f"][k] for k in ("tag", "limit", "replan_steps", "schedule", "scene", "nav", "n_windows")})
