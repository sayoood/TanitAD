#!/usr/bin/env python3
"""precheck_feas3.py -- the pre-check of PREREG_FEAS3_FROZEN.md, on SEEN data (KAPPA-T1's four fresh S runs, `SREP_*`, 7,197 ticks each; `plan` = the plan after S, `P` = the P10 prior,
`gt` = the logged future in the ego frame, `v0` = the tick's measured ego speed).  A per-tick counterfactual on issued S plans, NOT a rollout; it informs the predictions and decides no arm,
row or bar.  Uses the FEAS3 tool's own operators (feasible_plan: kbox, kbox2, lonfloor, kbox2lf), executed knots 1..10 (replan 10).
    python -I code/precheck_feas3.py  ->  raw/PRECHECK_FEAS3.json
"""
import importlib.util
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
TOOL = "F:/TanitAD-scratch/feas2_t1/tree_dag/stack/scripts/v8f_t1_eval_feas3.py"
OUT_Y = "F:/TanitAD-scratch/kappa_t1/out"
M = 10
spec = importlib.util.spec_from_file_location("v8f_t1_eval_feas3", TOOL)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
DT = T.DT


def steps(plan):
    d = np.diff(np.concatenate([np.zeros_like(plan[:, :1]), plan], 1), axis=1)
    return np.linalg.norm(d, axis=-1) / DT


def q(x):
    x = np.asarray(x, np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p99": float(np.percentile(x, 99)), "max": float(x.max()), "n": int(x.size)} if x.size else {"n": 0}


res = {"tier": "T1-kin plan-level proxy (per-tick counterfactual on the issued S plans of KAPPA-T1's fresh S runs; not a rollout)", "evidence_class": "MEASURED (ours)", "executed_knots": M, "runs": {}}
for tag in ("SREP_DAGOACC_s0", "SREP_DAGOACC_s1", "SREP_SOUP_s0", "SREP_SOUP_s1"):
    z = np.load(f"{OUT_Y}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
    plan, P, gt, ok, v0 = z["plan"], z["P"], z["gt"], z["gt_ok"], z["v0"]
    band = (v0 >= 3.0) & (v0 <= 8.0)
    sx, sp, sg = steps(plan)[:, :M], steps(P)[:, :M], steps(gt)[:, :M]
    floor = np.maximum(sp - 2.0, 0.0)
    act = sx < floor
    out = {"n_ticks": int(len(v0)), "n_band_3_8": int(band.sum())}
    out["lonfloor_active"] = {"step_share_all": float(act.mean()), "step_share_v0_3_8": float(act[band].mean()), "tick_share_any_step": float(act.any(1).mean()), "tick_share_any_step_v0_3_8": float(act[band].any(1).mean()),
                              "raise_mps_on_active_steps": q((floor - sx)[act]), "n_active_steps": int(act.sum())}
    lf = T.feasible_plan(plan, "lonfloor", prior=P)
    k2lf = T.feasible_plan(plan, "kbox2lf", prior=P)
    k1, k2 = T.feasible_plan(plan, "kbox"), T.feasible_plan(plan, "kbox2")
    mv = lambda a, b: np.abs(a[:, :M] - b[:, :M]).max(axis=(1, 2))      # per-tick max abs displacement of the executed knots
    out["executed_knot_displacement_m"] = {n: {"tick_share_gt_1mm": float((mv(a, plan) > 1e-3).mean()), "p99_max_per_tick": float(np.percentile(mv(a, plan), 99)), "max": float(mv(a, plan).max())}
                                           for n, a in (("kbox", k1), ("kbox2", k2), ("lonfloor", lf), ("kbox2lf", k2lf))}
    out["kbox2_vs_kbox"] = {"tick_share_executed_knots_differ_gt_1mm": float((mv(k2, k1) > 1e-3).mean()), "max_m": float(mv(k2, k1).max())}
    sel = ok[:, None] & band[:, None] & np.ones((1, M), bool)
    crawl = lambda s_: float(((s_ < sg - 2.0) & sel).sum() / max(sel.sum(), 1))
    out["crawl_proxy_step_share_v0_3_8"] = {"plan_s": crawl(sx), "P10_prior_P": crawl(sp), "lonfloor": crawl(steps(lf)[:, :M]), "kbox2lf": crawl(steps(k2lf)[:, :M]), "kbox2": crawl(steps(k2)[:, :M])}
    # M1 as the 19-row definition (GT-fast steps: GT step speed >= 5.2; slow: the step's own speed < 5.2) and its margin family (slow_d: own speed < GT step speed - d), at plan level, all ok ticks
    fastm = ok[:, None] & (sg >= 5.2)
    m1 = lambda s_, d=None: float(((s_ < (5.2 if d is None else sg - d)) & fastm).sum() / max(fastm.sum(), 1))
    out["m1_plan_level"] = {n: {"thr_5.2": m1(s_), "margin_1": m1(s_, 1.0), "margin_2": m1(s_, 2.0), "margin_3": m1(s_, 3.0)}
                            for n, s_ in (("plan_s", sx), ("P10_prior_P", sp), ("lonfloor", steps(lf)[:, :M]), ("kbox2lf", steps(k2lf)[:, :M]))}
    # step-speed error against the logged future, MAE over ok ticks in the band: does the floor move the plan toward or away from the GT speed?
    e = lambda s_: float(np.abs(s_ - sg)[sel].mean())
    out["step_speed_mae_mps_v0_3_8"] = {"plan_s": e(sx), "lonfloor": e(steps(lf)[:, :M]), "P10_prior_P": e(sp)}
    # hard-braking ticks: the top 5 % of GT step deceleration over knots 1..10 (the floor must not block these)
    dec = -np.diff(np.concatenate([np.zeros((len(sg), 1)), sg], 1), axis=1).min(axis=1) / DT
    hb = ok & (dec >= np.nanpercentile(dec[ok], 95))
    out["hard_braking_ticks"] = {"n": int(hb.sum()), "lonfloor_active_step_share": float(act[hb].mean()), "lonfloor_tick_share_any_step": float(act[hb].any(1).mean()),
                                 "step_speed_mae_mps": {"plan_s": float(np.abs(sx - sg)[hb].mean()), "lonfloor": float(np.abs(steps(lf)[:, :M] - sg)[hb].mean())}}
    res["runs"][tag] = out
    print(tag, json.dumps({k: out[k] for k in ("m1_plan_level", "lonfloor_active", "kbox2_vs_kbox", "crawl_proxy_step_share_v0_3_8", "step_speed_mae_mps_v0_3_8", "hard_braking_ticks")}, default=float)[:1500], flush=True)
json.dump(res, open(f"{PK}/raw/PRECHECK_FEAS3.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
print("PRECHECK_FEAS3.json written")
