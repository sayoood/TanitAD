"""32-window timing run (reads eval windows; prints wall time only and plan-shape sanity, no statistic)."""
import sys, time, json
sys.dont_write_bytecode = True
import numpy as np, torch
import plantest_real as R
ctx = R.prepare("M2b_s0")
print("n", ctx.n, "n_total", ctx.n_total, "v0 nan", ctx.v0_nan, flush=True)
idx = np.arange(32)
out = {}
for kind in ("td3", "flat", "flat1"):
    r = R.unit(ctx, kind, "true", idx=idx)
    out[kind] = float(r["wall_s"]); print(kind, "wall_s for 32 windows (batch 32)", out[kind], "iters", {k: float(v.mean()) for k, v in r.items() if k.startswith("iters")}, flush=True)
json.dump(out, open("timing_32_1thr.json", "w"))
