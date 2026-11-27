#!/usr/bin/env python3
"""Check DATA_CONTRACT.md keys/shapes/dtypes/ranges of cache npz files (read only).  check_contract.py <npz>..."""
import json
import sys

import numpy as np

SHAPES = {"props": (64, 20, 3), "props8": (64, 8, 3), "s": (64, 256), "logits": (64, 6)}
TR = ["tr_nc", "tr_dac", "tr_ddc", "tr_ttc", "tr_ep", "tr_c", "tr_pdms"]
ok_all = True
for p in sys.argv[1:]:
    d = np.load(p)
    keys = set(d.files)
    need = {"token", "log", "split", "props", "props8", "s", "logits", "pick", "meta"} | set(TR)
    errs = [f"missing {sorted(need - keys)}"] if need - keys else []
    n = len(d["token"]) if "token" in keys else 0
    for k, sh in SHAPES.items():
        if k in keys and (d[k].shape != (n,) + sh or d[k].dtype != np.float32 or not np.isfinite(d[k]).all()):
            errs.append(f"{k} {d[k].shape} {d[k].dtype}")
    for k in TR:
        if k in keys and (d[k].shape != (n, 64) or d[k].dtype != np.float32 or not np.isfinite(d[k]).all() or d[k].min() < 0 or d[k].max() > 1):
            errs.append(f"{k} bad")
    if "pick" in keys and (d["pick"].dtype != np.int64 or d["pick"].shape != (n,) or d["pick"].min() < 0 or d["pick"].max() > 63):
        errs.append("pick bad")
    ok = not errs and n > 0
    ok_all &= ok
    print("ZZCONTRACT", "PASS" if ok else "FAIL", p, "n_tokens", n, "split", str(d["split"]) if "split" in keys else None, errs)
sys.exit(0 if ok_all else 1)
