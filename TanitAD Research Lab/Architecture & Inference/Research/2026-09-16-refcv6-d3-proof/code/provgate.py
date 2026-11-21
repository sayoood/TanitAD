#!/usr/bin/env python3
"""PROVENANCE GATE: does C:/Users/Admin/refcv5cmp/repo still reproduce the BANKED
refcv5-v2 dump, and what does dropping the nav conditionings (batch 3 -> 1) cost?

Reads two dumps' ep000/ep001 and prints, per arm, max |delta| and the exact-equal
fraction. The model-free arms (ha / ha0 / ha0_ext) are CPU-side and batch-independent:
they MUST be BIT-IDENTICAL, and that is the control on this gate.
"""
import sys
import numpy as np

A = sys.argv[1]      # banked
B = sys.argv[2]      # new
EPS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1"])]

print(f"A(banked) = {A}\nB(new)    = {B}\n")
for ep in EPS:
    da = np.load(f"{A}/ep{ep:03d}.npz")
    db = np.load(f"{B}/ep{ep:03d}.npz")
    assert np.array_equal(da["ws"], db["ws"]), f"ep{ep}: window grids differ"
    print(f"ep{ep:03d}  n={len(da['ws'])}  ws identical={np.array_equal(da['ws'], db['ws'])}"
          f"  v0 max|d|={np.abs(da['v0'] - db['v0']).max():.3e}")
    for k in ("g", "os", "ha", "ha0", "ha0_ext"):
        if k not in da.files or k not in db.files:
            continue
        x, y = da[k].astype(np.float64), db[k].astype(np.float64)
        d = np.abs(x - y)
        print(f"    {k:9s} max|d|={d.max():.6e}  mean|d|={d.mean():.6e}  "
              f"exact_equal={float((x == y).mean()):.6f}")
    print()
