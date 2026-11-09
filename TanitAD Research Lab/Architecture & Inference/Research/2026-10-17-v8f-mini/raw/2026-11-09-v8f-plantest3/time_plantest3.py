"""Timing of one job: FLAT1 DEC-END plain / DEC-PATH shrink, TD3 DEC-END plain, lambda 0.1, G-TRUE, on the first 64 TUNE windows (one batch), 1 thread. Prints wall time only (no statistic)."""
import sys, time, json
sys.dont_write_bytecode = True
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/plantest3")
import plantest3_real as R3
t0 = time.time()
ctx = R3.prepare3("M2b_s0")
dec = R3.get_dec(ctx)
out = dict(prepare_and_fit_s=time.time() - t0, n_tune=len(ctx.tune_idx), n_score=len(ctx.score_idx), v0_nan=ctx.v0_nan, decoder_sha256=dec.digest())
idx = ctx.tune_idx[:64]
for kind, arm, form in (("flat1", "end", "plain"), ("flat1", "path", "shrink"), ("td3", "end", "plain")):
    r = R3.unit3(ctx, kind, "true", idx, 0.1, arm, form, collect=False)
    k = f"{kind}_{arm}_{form}"
    out[f"{k}_wall_64"] = float(r["wall_s"])
    out[f"{k}_iters"] = {kk: float(v.mean()) for kk, v in r.items() if kk.startswith("iters")}
    print(k, out[f"{k}_wall_64"], out[f"{k}_iters"], flush=True)
json.dump(out, open("E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/plantest3/timing_64.json", "w"), indent=1)
