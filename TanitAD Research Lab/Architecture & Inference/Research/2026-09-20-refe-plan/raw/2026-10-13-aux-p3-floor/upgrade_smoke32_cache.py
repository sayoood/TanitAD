#!/usr/bin/env python3
"""ADDITIVE: give the old-layout 32-token smoke cache the per-token `logits_<token>.npy` files the current forward tool's `cache_complete` requires.

`D:/Projects/TanitAD-artifacts/aux_navtest_forward/smoke32_ctx/` holds 32 `ctx_<token>.npy` and ONE `scorer_logits.npz` (token [32], logits [32, 64, 7]) from the first smoke run;
`aux_navtest_forward.cache_complete` wants `ctx_<token>.npy` AND `logits_<token>.npy`, so without this the head-only pass would find no complete entry and try to run the trunk.
It writes `logits_<token>.npy` ([64, 7] float32) from the npz and touches nothing else.  Whether those logits are the right ones is NOT assumed: the forward tool's own identity check
compares them with the banked C3 table (bit-identical bar) on every head-only run.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe upgrade_smoke32_cache.py [--cache <dir>]
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="D:/Projects/TanitAD-artifacts/aux_navtest_forward/smoke32_ctx")
a = ap.parse_args()
z = np.load(os.path.join(a.cache, "scorer_logits.npz"), allow_pickle=False)
toks, lg = [str(t) for t in z["token"]], np.asarray(z["logits"], np.float32)
assert lg.ndim == 3 and lg.shape[0] == len(toks) == len(set(toks))
ctx = {f[len("ctx_"):-len(".npy")] for f in os.listdir(a.cache) if f.startswith("ctx_") and f.endswith(".npy") and ".tmp" not in f}
assert ctx == set(toks), f"ctx files ({len(ctx)}) and scorer_logits.npz tokens ({len(toks)}) are not the same set"
n = 0
for t, row in zip(toks, lg):
    p = os.path.join(a.cache, f"logits_{t}.npy")
    if os.path.exists(p):
        assert np.array_equal(np.load(p), row), f"{p} exists and differs"
        continue
    np.save(p, row)
    n += 1
print(f"ZZUPGRADE_OK wrote {n} logits files for {len(toks)} tokens in {a.cache}")
