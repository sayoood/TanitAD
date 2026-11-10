"""Smoke of run_lp_speed.work on 3 windows per form (E-NOW_s0): imports, paths, the speed context, one tune unit and one p1 unit per form. Writes to a smoke folder, never to plan_runs."""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import run_lp_speed as RS
import plantest_lib as P
RS.RUNS = "F:/TanitAD-scratch/egowm/plan_smoke"
ctx = RS.S.prepare_arm_speed("E-NOW_s0", "C:/Users/Admin/t2v8f_pulls/egowm_E-NOW_s0", expect_mode="now")
print("mode", ctx.speed_mode, "n", ctx.n, "gap", ctx.speed_source_gap)
for f in ("c1", "c3"):
    sel = RS.S.LP.sel_arrays(ctx, f)
    print(f, {k: len(v) for k, v in sel.items()})
    t = time.time()
    r = RS.S.unit_speed(ctx, "flat1", "true", f, sel["tune"][:3], 0.1, seed_mode="hold", collect=False)
    m = P.window_metrics(r["plan_raw"], ctx.A[sel["tune"][:3]], ctx.v0[sel["tune"][:3]])
    print(f, "tune ok", r["plan_raw"].shape, bool(np.isfinite(r["plan_raw"]).all()), float(m["acc_mae"].mean()), round(time.time() - t, 1), "s")
print("SMOKE OK")
