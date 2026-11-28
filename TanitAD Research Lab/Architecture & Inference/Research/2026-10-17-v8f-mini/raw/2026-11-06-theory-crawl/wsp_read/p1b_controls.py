"""P1 descriptives that are NOT verdict-bearing (they were not in SPEC_P1.md): robust medians, runaway counts, the arm-minus-human paired band contrast,
and the T1-kin tick dumps as a larger-n second tier (never pooled with T2). Writes P1B_RESULT.json. Reuses p1_return_map (same bins, same bootstrap)."""
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

import p1_return_map as P

HERE = P.HERE
DUMPS = {"C0C_s0": "C:/Users/Admin/rr_scratch/out/C0C_s0/ticks/ticks_C0C_s0.npz", "C0C_s1": "C:/Users/Admin/rr_scratch/out/C0C_s1/ticks/ticks_C0C_s1.npz",
         "C0_s0": "C:/Users/Admin/rr_scratch/out/C0_s0/ticks/ticks_C0_s0.npz", "C0_s1": "C:/Users/Admin/rr_scratch/out/C0_s1/ticks/ticks_C0_s1.npz"}


def band_rep(v, d, cl, B=P.B_BOOT, stat="mean"):
    """band stat of d over v in [3,8) with a cluster bootstrap (cl = cluster id array). -> point, ci, n, replicate array."""
    inb = (v >= 3) & (v < 8)
    u = np.unique(cl)
    si = np.searchsorted(u, cl)
    per = [d[inb & (si == i)] for i in range(len(u))]
    rng = np.random.default_rng(P.SEED)
    idx = rng.integers(0, len(u), (B, len(u)))
    f = np.mean if stat == "mean" else np.median
    pt = float(f(d[inb])) if inb.any() else float("nan")
    rep = np.array([f(np.concatenate([per[j] for j in row])) if sum(len(per[j]) for j in row) else np.nan for row in idx])
    return pt, P.pct(rep), int(inb.sum()), rep


def main():
    out = {"note": "descriptives, not in SPEC_P1.md; no verdict leg is changed by anything here", "t2": {}, "t1kin": {}}
    for arm in P.ARMS:
        D, _ = P.load_arm(arm)
        e = {}
        for yk, nm in (("gp", "g_plan"), ("ge", "g_exec")):
            d = D[yk] - D["v"]
            mean_pt, mean_ci, n, rep_mean = band_rep(D["v"], d, D["scene"], stat="mean")
            med_pt, med_ci, _, _ = band_rep(D["v"], d, D["scene"], stat="median")
            # human control on the same scenes
            hv_d = D["hn"] - D["hv"]
            hm_pt, hm_ci, hn_, rep_h = band_rep(D["hv"], hv_d, D["scene"], stat="mean")
            hmed_pt, hmed_ci, _, _ = band_rep(D["hv"], hv_d, D["scene"], stat="median")
            e[nm] = {"band_mean": mean_pt, "band_mean_ci": mean_ci, "band_median": med_pt, "band_median_ci": med_ci, "n_band": n,
                     "human_band_mean": hm_pt, "human_band_mean_ci": hm_ci, "human_band_median": hmed_pt, "human_band_median_ci": hmed_ci, "n_band_human": hn_,
                     "arm_minus_human_mean": mean_pt - hm_pt, "arm_minus_human_mean_ci": P.pct(rep_mean - rep_h)}
            e[nm]["per_tick_band_mean"] = {}
            for k in range(3):
                m = D["k"] == k
                inb = m & (D["v"] >= 3) & (D["v"] < 8)
                e[nm]["per_tick_band_mean"][str(k)] = {"n": int(inb.sum()), "mean": float(d[inb].mean()) if inb.any() else None, "median": float(np.median(d[inb])) if inb.any() else None}
        # runaway counts: plan speed at +2 s more than 10 m/s above the executed speed, or above 40 m/s
        run = (D["gp"] - D["v"] > 10.0) | (D["gp"] > 40.0)
        e["runaway_ticks"] = {"n": int(run.sum()), "of": int(len(run)), "by_tick": [int(run[D["k"] == k].sum()) for k in range(3)], "scenes": sorted(set(int(s) for s in D["scene"][run]))}
        e["speed_pct_exec"] = [float(x) for x in np.percentile(D["v"], [10, 25, 50, 75, 90])]
        out["t2"][arm] = e
    # T1-kin dumps
    for arm, path in DUMPS.items():
        if not os.path.isfile(path):
            out["t1kin"][arm] = {"missing": path}
            continue
        z = np.load(path, allow_pickle=True)
        plan, gt, v0, tick, ci = z["plan"], z["gt"], z["v0"], z["tick"], z["ci"]
        sec = lambda p: np.hypot(p[:, 11, 0] - p[:, 7, 0], p[:, 11, 1] - p[:, 7, 1]) / 0.8
        gp, gh = sec(plan), sec(gt)
        ok = z["gt_ok"].astype(bool)
        d = gp - v0
        pt, cic, n, rep = band_rep(v0[ok], d[ok], ci[ok])
        med, medci, _, _ = band_rep(v0[ok], d[ok], ci[ok], stat="median")
        # human map in the same windows: logged speed now is v0 (the dump's v0 is the logged/executed speed at the tick); logged speed at +2 s from gt
        hd = gh - v0
        hpt, hci, hn, hrep = band_rep(v0[ok], hd[ok], ci[ok])
        a, b, vs = P.fit_from_M(np.array([[ok.sum(), v0[ok].sum(), (v0[ok] ** 2).sum(), gp[ok].sum(), (v0[ok] * gp[ok]).sum()]]))
        ah, bh, vsh = P.fit_from_M(np.array([[ok.sum(), v0[ok].sum(), (v0[ok] ** 2).sum(), gh[ok].sum(), (v0[ok] * gh[ok]).sum()]]))
        out["t1kin"][arm] = {"tier": "T1-kin (dump), never pooled with T2", "rows": int(len(v0)), "rows_ok": int(ok.sum()), "n_band": n,
                             "plan_band_mean_g_minus_v": pt, "ci": cic, "plan_band_median": med, "median_ci": medci,
                             "logged_band_mean_g_minus_v_same_windows": hpt, "logged_ci": hci,
                             "plan_minus_logged_mean": pt - hpt, "plan_minus_logged_ci": P.pct(rep - hrep),
                             "fit_plan": {"a": float(a[0]), "b": float(b[0]), "v_star": None if not np.isfinite(vs[0]) else float(vs[0])},
                             "fit_logged": {"a": float(ah[0]), "b": float(bh[0]), "v_star": None if not np.isfinite(vsh[0]) else float(vsh[0])},
                             "per_tick_band_mean": {str(k): {"n": int(((tick == k) & ok & (v0 >= 3) & (v0 < 8)).sum()), "plan": float(d[(tick == k) & ok & (v0 >= 3) & (v0 < 8)].mean()), "logged": float(hd[(tick == k) & ok & (v0 >= 3) & (v0 < 8)].mean())} for k in range(3)},
                             "v0_median": float(np.median(v0[ok]))}
    json.dump(out, open(f"{HERE}/P1B_RESULT.json", "w"), indent=1)
    for arm, e in out["t2"].items():
        g = e["g_plan"]
        print(f"T2 {arm:7s} plan band mean {g['band_mean']:+.2f} {np.round(g['band_mean_ci'],2)} median {g['band_median']:+.2f} {np.round(g['band_median_ci'],2)} n={g['n_band']} | human {g['human_band_mean']:+.2f} (n={g['n_band_human']}) | arm-human {g['arm_minus_human_mean']:+.2f} {np.round(g['arm_minus_human_mean_ci'],2)} | runaway {e['runaway_ticks']['n']}/{e['runaway_ticks']['of']} {e['runaway_ticks']['by_tick']} | per-tick {({k: (v['n'], None if v['mean'] is None else round(v['mean'],2)) for k, v in g['per_tick_band_mean'].items()})}")
    for arm, e in out["t1kin"].items():
        if "missing" in e:
            print("T1kin", arm, "missing"); continue
        print(f"T1kin {arm:7s} plan {e['plan_band_mean_g_minus_v']:+.3f} {np.round(e['ci'],3)} med {e['plan_band_median']:+.3f} | logged {e['logged_band_mean_g_minus_v_same_windows']:+.3f} {np.round(e['logged_ci'],3)} | plan-logged {e['plan_minus_logged_mean']:+.3f} {np.round(e['plan_minus_logged_ci'],3)} n={e['n_band']} fit plan {e['fit_plan']} fit logged {e['fit_logged']} v0med {e['v0_median']:.2f}")


if __name__ == "__main__":
    main()
