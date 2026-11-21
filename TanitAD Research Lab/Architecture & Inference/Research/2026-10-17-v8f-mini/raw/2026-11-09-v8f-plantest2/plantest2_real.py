"""PREREG_V8F_PLANTEST2: the real-data side. Reuses plantest/plantest_real.prepare (same models, windows, TRAIN prior, v0, donors); adds the TUNE / SCORE split, HOLD_c, one-unit planning with SEED / REG.
The logged actions reach the planner ONLY as a0 (hold_scaled reads A[:, :1]); the metric reads all 12."""
import os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import plantest2_lib as Q
import plantest_lib as P
import plantest_real as R

RUNS = f"{HERE}/runs"


def prepare2(tag):
    ctx = R.prepare(tag)
    ctx.tune_idx, ctx.score_idx = Q.split_indices(ctx.eid)
    ctx.std_raw = Q.std_raw_of(ctx.st, ctx.ch1)
    return ctx


def hold_c_of(ctx):
    """HOLD_c from the CURRENT ctx.A (first action of every window only), in the planner's scaled units, clipped to the level-1 box."""
    return Q.hold_scaled(ctx.wm, ctx.A[:, :1], ctx.st.prior_lo_1, ctx.st.prior_hi_1)


def hold_raw(ctx, idx):
    return P.hold_plan(ctx.A[idx])


def unit2(ctx, kind, gname, idx, lam, seed=0, collect=True, seed_mode="hold"):
    t = time.time()
    r = Q.plan_all2(ctx.st, kind, gname, ctx.z_now, ctx.z_fut, ctx.donor, hold_c_of(ctx), idx=idx, seed=seed, collect=collect, lam=lam, seed_mode=seed_mode)
    r["plan_raw"] = R.to_raw(ctx, r.pop("actions"))
    r["idx"] = np.asarray(idx)
    r["wall_s"] = np.array(time.time() - t)
    return r


def tune_lambdas(ctx, kind, lams, tune_idx, seed=0):
    """The TUNE grid under G-TRUE. Reads only the windows in tune_idx (planner inputs and metric inputs). Returns {lam: {acc_mae, yaw_mae}} (means over the TUNE windows) and the plans."""
    out, plans = {}, {}
    for lam in lams:
        r = unit2(ctx, kind, "true", tune_idx, lam, seed=seed, collect=False)
        m = P.window_metrics(r["plan_raw"], ctx.A[tune_idx], ctx.v0[tune_idx])
        out[lam] = dict(acc_mae=float(m["acc_mae"].mean()), yaw_mae=float(m["yaw_mae"].mean()))
        plans[lam] = r["plan_raw"]
    return out, plans


def choose_all(tab):
    """tab {lam: {acc_mae, yaw_mae}} -> {family: lam}."""
    return {fam: Q.choose_lambda({l: v[q] for l, v in tab.items()}) for fam, q in Q.FAMS.items()}
