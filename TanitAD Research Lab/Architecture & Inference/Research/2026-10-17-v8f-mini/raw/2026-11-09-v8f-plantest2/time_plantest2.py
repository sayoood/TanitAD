"""Timing of one job: TD3 and FLAT1 at lambda 0.1 on the first 64 TUNE windows (one batch), 1 thread. Prints wall time only (no statistic)."""
import sys, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/plantest2")
import plantest2_real as R2
t0 = time.time()
ctx = R2.prepare2("M2b_s0")
out = dict(prepare_s=time.time() - t0, n_tune=len(ctx.tune_idx), n_score=len(ctx.score_idx), v0_nan=ctx.v0_nan)
idx = ctx.tune_idx[:64]
for kind in ("td3", "flat1"):
    for lam in (0.0, 0.1):
        r = R2.unit2(ctx, kind, "true", idx, lam, collect=False)
        out[f"{kind}_L{lam}_wall_64"] = float(r["wall_s"])
        out[f"{kind}_L{lam}_iters"] = {k: float(v.mean()) for k, v in r.items() if k.startswith("iters")}
        print(kind, lam, out[f"{kind}_L{lam}_wall_64"], out[f"{kind}_L{lam}_iters"], flush=True)
json.dump(out, open("E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/plantest2/timing_64.json", "w"), indent=1)
