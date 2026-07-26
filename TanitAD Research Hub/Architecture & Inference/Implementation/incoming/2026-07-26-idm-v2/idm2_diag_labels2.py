"""IDM-v2 DIAGNOSIS (b), part 2 — nail the two anomalies found in part 1.

A. comma2k19 `yaw_rate`: std 0.429 rad/s with MAD 0.011 rad/s and a
   smooth-fit R2 ceiling of only 0.352.  Which episodes?  How many frames are
   PHYSICALLY IMPOSSIBLE?  What does the raw yaw series look like there?

B. `long_accel`: the label correlates only r=0.60 (pai) / 0.67 (cm) with the
   pose-derived dv/dt even though BOTH signals are smooth.  Is the residual a
   slowly-varying term (road grade / accelerometer tilt -> partially visible)
   or white noise (unlearnable)?  What is the R2 CEILING of the best possible
   video-derived predictor against THIS label?

C. Variance decomposition BETWEEN vs WITHIN clip for every channel -- needed to
   read the per-clip oracle recalibration honestly (a per-clip affine can look
   spectacular purely by supplying the clip mean).

Writes /root/idm2/out/labels2.json
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/root/idm2")
import idm2_lib as L                      # noqa: E402
from idm2_diag_labels import label_series, savgol_center  # noqa: E402


def main():
    tags = L.all_tags()
    out = {}

    # ---------------- A. per-episode yaw_rate audit --------------------- #
    rows = []
    for tag in tags:
        d = L.load_ep(tag)
        s = label_series(d)
        yr = s["yaw_rate"][np.isfinite(s["yaw_rate"])]
        v = s["speed"]
        rows.append({"tag": tag, "dom": tag.split("_")[0],
                     "eid": int(d.get("episode_id", -1)),
                     "yr_std": float(yr.std()), "yr_mad": float(np.median(np.abs(yr - np.median(yr)))),
                     "yr_absmax": float(np.abs(yr).max()),
                     "frac_yr_gt_1p5": float((np.abs(yr) > 1.5).mean()),
                     "frac_yr_gt_0p6": float((np.abs(yr) > 0.6).mean()),
                     "v_mean": float(v.mean()), "v_min": float(v.min())})
    rows.sort(key=lambda r: -r["yr_absmax"])
    out["yaw_audit_worst15"] = rows[:15]
    for dom in ("pai", "cm"):
        rr = [r for r in rows if r["dom"] == dom]
        yrmax = np.array([r["yr_absmax"] for r in rr])
        out[f"yaw_audit_{dom}"] = {
            "n_eps": len(rr),
            "n_eps_with_|yr|>1.5_rad_s": int((yrmax > 1.5).sum()),
            "n_eps_with_|yr|>5_rad_s": int((yrmax > 5.0).sum()),
            "frac_frames_|yr|>1.5": float(np.mean([r["frac_yr_gt_1p5"] for r in rr])),
            "median_ep_yr_std": float(np.median([r["yr_std"] for r in rr])),
            "max_ep_yr_std": float(max(r["yr_std"] for r in rr))}
    # raw yaw around the worst frame of the worst comma clip
    worst = [r for r in rows if r["dom"] == "cm"][0]
    d = L.load_ep(worst["tag"])
    s = label_series(d)
    j = int(np.nanargmax(np.abs(s["yaw_rate"])))
    out["worst_cm_clip"] = {
        "tag": worst["tag"], "eid": worst["eid"], "frame": j,
        "yaw_raw_j-6..j+6": [float(x) for x in s["_yaw"][max(0, j - 6):j + 7]],
        "speed_j-6..j+6": [float(x) for x in s["speed"][max(0, j - 6):j + 7]],
        "yaw_rate_j-6..j+6": [float(x) for x in s["yaw_rate"][max(0, j - 6):j + 7]]}

    # effect of winsorising yaw_rate at a physical limit
    allyr = {dom: np.concatenate([label_series(L.load_ep(t))["yaw_rate"]
                                  [np.isfinite(label_series(L.load_ep(t))["yaw_rate"])]
                                  for t in tags if t.startswith(dom)])
             for dom in ("pai", "cm")}
    out["winsorise_effect"] = {}
    for dom, x in allyr.items():
        e = {"std_raw": float(x.std())}
        for lim in (0.6, 1.0, 1.5):
            e[f"std_clipped_{lim}"] = float(np.clip(x, -lim, lim).std())
            e[f"frac_clipped_{lim}"] = float((np.abs(x) > lim).mean())
        out["winsorise_effect"][dom] = e

    # ---------------- B. long_accel provenance -------------------------- #
    k9 = savgol_center(9)
    acc = {}
    for dom in ("pai", "cm"):
        A, DV, RES = [], [], []
        lagc = np.zeros(11)
        nlag = 0
        for tag in [t for t in tags if t.startswith(dom)]:
            s = label_series(L.load_ep(tag))
            v, a = s["speed"], s["long_accel"]
            T = len(v)
            dv = np.full(T, np.nan)
            dv[1:-1] = (v[2:] - v[:-2]) / (2 * L.DT)
            m = np.isfinite(dv)
            aa, dd = a[m], dv[m]
            A.append(aa); DV.append(dd); RES.append(aa - dd)
            for i, lg in enumerate(range(-5, 6)):
                x = np.roll(dd, lg)
                sl = slice(6, len(dd) - 6)
                lagc[i] += np.corrcoef(aa[sl], x[sl])[0, 1]
            nlag += 1
        A = np.concatenate(A); DV = np.concatenate(DV); RES = np.concatenate(RES)
        r = float(np.corrcoef(A, DV)[0, 1])
        # best affine predictor a ~ c1*dv + c0  (ceiling for a perfect
        # video-derived kinematic accel estimator against THIS label)
        C = np.stack([DV, np.ones_like(DV)], 1)
        coef, *_ = np.linalg.lstsq(C, A, rcond=None)
        pred = C @ coef
        rc = 1 - ((A - pred) ** 2).sum() / ((A - A.mean()) ** 2).sum()
        # is the residual slow (grade/tilt) or white?
        rz = RES - RES.mean()
        racf = [float((rz[:-k] * rz[k:]).mean() / (rz * rz).mean())
                for k in (1, 2, 5, 10, 20)]
        acc[dom] = {
            "corr_label_vs_dv_dt": r,
            "r2_ceiling_of_best_affine_in_dv_dt": float(rc),
            "lag_corr_-5..5": [float(x / nlag) for x in lagc],
            "best_lag_frames": int(np.argmax(lagc) - 5),
            "residual_std": float(RES.std()), "label_std": float(A.std()),
            "dv_dt_std": float(DV.std()),
            "residual_autocorr_lag_1_2_5_10_20": racf,
            "label_quantisation_min_gap": float(np.min(np.diff(np.unique(np.round(A, 6))))
                                                if np.unique(np.round(A, 6)).size > 1 else 0.0),
        }
    out["long_accel"] = acc

    # ---------------- C. between- vs within-clip variance ---------------- #
    dec = {}
    for c in L.SCALARS:
        for dom in ("pai", "cm"):
            mus, xs = [], []
            for tag in [t for t in tags if t.startswith(dom)]:
                x = label_series(L.load_ep(tag))[c]
                x = x[np.isfinite(x)]
                mus.append(x.mean()); xs.append(x)
            allx = np.concatenate(xs)
            n = np.array([len(x) for x in xs], dtype=np.float64)
            gm = allx.mean()
            between = float((n * (np.array(mus) - gm) ** 2).sum() / n.sum())
            total = float(((allx - gm) ** 2).mean())
            dec[f"{c}|{dom}"] = {"total_var": total, "between_clip_var": between,
                                 "between_frac": between / max(total, 1e-12)}
    out["variance_decomposition"] = dec

    L.jdump(out, "/root/idm2/out/labels2.json")
    print("\n=== A. yaw audit ===")
    for dom in ("pai", "cm"):
        print(dom, out[f"yaw_audit_{dom}"])
    print("worst comma clip:", {k: v for k, v in out["worst_cm_clip"].items()
                                if not isinstance(v, list)})
    print("  yaw raw   :", [round(x, 3) for x in out["worst_cm_clip"]["yaw_raw_j-6..j+6"]])
    print("  yaw_rate  :", [round(x, 3) for x in out["worst_cm_clip"]["yaw_rate_j-6..j+6"]])
    print("  speed     :", [round(x, 2) for x in out["worst_cm_clip"]["speed_j-6..j+6"]])
    print("winsorise:", out["winsorise_effect"])
    print("\n=== B. long_accel ===")
    for dom, v in acc.items():
        print(dom, {k: (round(x, 4) if isinstance(x, float) else x)
                    for k, x in v.items() if k != "lag_corr_-5..5"})
        print("   lag corr -5..5:", [round(x, 3) for x in v["lag_corr_-5..5"]])
    print("\n=== C. between-clip variance fraction ===")
    for k, v in dec.items():
        print(f"  {k:<18} between {v['between_frac']:.3f}")


if __name__ == "__main__":
    main()
