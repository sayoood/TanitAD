"""Z6 (EXPLORATORY, post-seal; ORACLE-TUNED on eval = an UPPER BOUND on a downstream plan-speed lever, never a result).
Where in dv = |P6-P5| - v0 does the stage's false-decel sit, and what FD/MD frontier does a pure end-speed shift of the stage plan trace?"""
import sys, json
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import numpy as np
import l2_common as C
out = {}
for seed in (0, 1):
    E = C.load_eval_pf(seed)
    stg = C.apply_stage(E["plan8"], E["prior8"]); comp = E["compliant"]
    gt = C.plan_class(E["g"], E["v0"], E["dt"])
    def dv(P): return np.linalg.norm(P[:, -1].astype(np.float64) - P[:, -2].astype(np.float64), axis=-1) / E["dt"] - E["v0"]
    d_s, d_g, d_7 = dv(stg), dv(E["g"]), dv(E["r7"])
    cm, dm = comp & (gt == "cruise"), comp & (gt == "decel")
    fdm = cm & (d_s <= -1.5)
    r = {"n_gt_cruise": int(cm.sum()), "n_false_decel": int(fdm.sum())}
    r["dv_false_decel_quantiles(10,25,50,75,90)"] = [float(x) for x in np.quantile(d_s[fdm], [.1, .25, .5, .75, .9])]
    r["share_false_decel_within_1.0_of_edge(-2.5,-1.5]"] = float(((d_s[fdm] > -2.5)).mean())
    r["share_false_decel_within_0.5_of_edge(-2.0,-1.5]"] = float(((d_s[fdm] > -2.0)).mean())
    r["mean_dv_plan_minus_gt_in_gt_cruise"] = float((d_s[cm] - d_g[cm]).mean())
    r["median_dv_plan_minus_gt_in_gt_cruise"] = float(np.median(d_s[cm] - d_g[cm]))
    r["sd_dv_plan_minus_gt_in_gt_cruise"] = float((d_s[cm] - d_g[cm]).std())
    fr = []
    for sh in np.arange(0.0, 3.01, 0.25):
        fd = float((d_s[cm] + sh <= -1.5).mean()); md = float((d_s[dm] + sh > -1.5).mean())
        ac = float((d_s[comp & (gt == "accel")] + sh < 1.5).mean())   # accel recall loss proxy: share of GT-accel no longer >= +1.5
        fr.append({"shift_mps": round(float(sh), 2), "fd": fd, "md": md, "gt_accel_lost": ac})
    r["frontier_oracle_tuned"] = fr
    # bound: smallest shift reaching FD <= 0.10, and the MD it costs
    ok = [x for x in fr if x["fd"] <= 0.10]
    r["first_shift_with_fd_le_0.10"] = ok[0] if ok else None
    out[str(seed)] = r
    print("seed", seed, {k: v for k, v in r.items() if k != "frontier_oracle_tuned"})
    print("  frontier", [(x["shift_mps"], round(x["fd"], 3), round(x["md"], 3)) for x in fr])
json.dump(out, open("C:/Users/Admin/refcv8_pf/refcv9_l2/result/Z6_speed_shift_EXPLORATORY.json", "w"), indent=1)
