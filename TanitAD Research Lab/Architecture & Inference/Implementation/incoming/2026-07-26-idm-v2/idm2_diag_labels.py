"""IDM-v2 DIAGNOSIS (b): is the CEILING the TARGET'S OWN NOISE?

For each of the 4 channels we measure, per domain and pooled:
  * the label distribution (std / MAD / tails / physically impossible count)
  * lag-k autocorrelation of the label INSIDE an episode
  * the NOISE-FLOOR R2 ceiling: R2 of the label against the best degree-2
    polynomial fit over the SAME 9-frame (and 17-frame) window the head sees.
    The encoder observes images, which are near-noiseless observations of the
    physical state; therefore no head can predict the part of the label that is
    not a smooth function of time over its own receptive field.  That fit is
    the ceiling.
  * the provenance of `long_accel` (is it a discrete 2nd derivative of pose?)
  * the redundancy of `steer` (bicycle model: steer ~ L * yaw_rate / v)
  * the arithmetic that reconciles idm_head_v1's recorded TRAIN standardiser
    std with the label std we measure here (the outlier-contamination check).

Writes /root/idm2/out/labels.json
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "/root/idm2")
import idm2_lib as L  # noqa: E402

CARD_STD = {"speed": 10.929014205932617, "yaw_rate": 0.7903658151626587,
            "steer": 0.05949832871556282, "long_accel": 0.7485366463661194}
CARD_MEAN = {"speed": 14.89643669128418, "yaw_rate": -0.011011586524546146,
             "steer": 0.002091496717184782, "long_accel": -0.09161043912172318}


def savgol_center(width: int, order: int = 2):
    """Center tap of a Savitzky-Golay smoother = the value at t of the best
    degree-`order` polynomial fit over [t-w..t+w].  Returned as a conv kernel."""
    h = width // 2
    t = np.arange(-h, h + 1, dtype=np.float64)
    A = np.vander(t, order + 1, increasing=True)          # [w, order+1]
    pinv = np.linalg.pinv(A)                              # [order+1, w]
    return pinv[0]                                        # value at t=0


def label_series(d):
    """Per-frame label series for one episode -> dict name -> [T] (nan-padded
    at the ends where the centred difference is undefined)."""
    po = d["poses"].float().numpy().astype(np.float64)
    ac = d["actions"].float().numpy().astype(np.float64)
    T = po.shape[0]
    yaw = po[:, 2]
    yr = np.full(T, np.nan)
    dif = yaw[2:] - yaw[:-2]
    dif = dif - 2 * np.pi * np.floor((dif + np.pi) / (2 * np.pi))
    yr[1:-1] = dif / (2 * L.DT)
    return {"speed": po[:, 3], "yaw_rate": yr, "steer": ac[:, 0],
            "long_accel": ac[:, 1], "_yaw": yaw, "_T": T}


def autocorr(x, kmax=6):
    x = x[np.isfinite(x)]
    x = x - x.mean()
    v = float((x * x).mean())
    if v <= 0:
        return [float("nan")] * kmax
    return [float((x[:-k] * x[k:]).mean() / v) for k in range(1, kmax + 1)]


def main():
    tags = L.all_tags()
    per_dom = {"pai": {}, "cm": {}, "all": {}}
    series = {d: {c: [] for c in L.SCALARS} for d in ("pai", "cm")}
    smooth_r2 = {d: {c: {} for c in L.SCALARS} for d in ("pai", "cm")}
    ac_stack = {d: {c: [] for c in L.SCALARS} for d in ("pai", "cm")}
    accel_prov = {d: [] for d in ("pai", "cm")}
    steer_bike = {d: [] for d in ("pai", "cm")}
    ker = {w: savgol_center(w) for w in (5, 9, 17)}
    resid = {d: {c: {w: [[], []] for w in ker} for c in L.SCALARS}
             for d in ("pai", "cm")}

    for tag in tags:
        d = L.load_ep(tag)
        dom = d.get("domain", tag.split("_")[0])
        s = label_series(d)
        T = s["_T"]
        for c in L.SCALARS:
            x = s[c]
            series[dom][c].append(x[np.isfinite(x)])
            ac_stack[dom][c].append(autocorr(x))
            # noise floor: best degree-2 poly over the head's own window
            for w, kk in ker.items():
                h = w // 2
                xs = np.copy(x)
                if not np.isfinite(xs).all():
                    idx = np.arange(T)
                    ok = np.isfinite(xs)
                    xs = np.interp(idx, idx[ok], xs[ok])
                sm = np.convolve(xs, kk[::-1], mode="valid")     # [T-2h]
                tgt = xs[h:T - h]
                resid[dom][c][w][0].append(tgt)
                resid[dom][c][w][1].append(sm)
        # long_accel provenance: is actions[:,1] == d(speed)/dt ?
        v = s["speed"]
        dv = np.full(T, np.nan)
        dv[1:-1] = (v[2:] - v[:-2]) / (2 * L.DT)
        dv1 = np.full(T, np.nan)
        dv1[1:] = (v[1:] - v[:-1]) / L.DT
        a = s["long_accel"]
        m = np.isfinite(dv) & np.isfinite(a)
        accel_prov[dom].append((np.corrcoef(a[m], dv[m])[0, 1],
                                np.corrcoef(a[np.isfinite(dv1)],
                                            dv1[np.isfinite(dv1)])[0, 1],
                                float(np.abs(a[m] - dv[m]).mean()),
                                float(np.abs(a[np.isfinite(dv1)] -
                                             dv1[np.isfinite(dv1)]).mean())))
        # steer redundancy: bicycle model steer ~ L * yaw_rate / v
        yr = s["yaw_rate"]; st = s["steer"]
        m2 = np.isfinite(yr) & (v > 2.0)
        if m2.sum() > 20:
            kappa = yr[m2] / v[m2]
            steer_bike[dom].append((float(np.corrcoef(st[m2], kappa)[0, 1]),
                                    float(np.corrcoef(st[m2], yr[m2])[0, 1]),
                                    int(m2.sum())))

    out = {"n_episodes": len(tags), "channels": {}}
    for c in L.SCALARS:
        rec = {}
        for dom in ("pai", "cm"):
            x = np.concatenate(series[dom][c])
            mad = float(np.median(np.abs(x - np.median(x))))
            ceil = {}
            for w in ker:
                tgt = np.concatenate(resid[dom][c][w][0])
                sm = np.concatenate(resid[dom][c][w][1])
                ssr = float(((tgt - sm) ** 2).sum())
                sst = float(((tgt - tgt.mean()) ** 2).sum())
                ceil[f"r2_ceiling_w{w}"] = 1.0 - ssr / max(sst, 1e-12)
                ceil[f"noise_rms_w{w}"] = float(np.sqrt(((tgt - sm) ** 2).mean()))
            acs = np.array(ac_stack[dom][c], dtype=np.float64)
            acm = np.nanmean(acs, axis=0)
            rec[dom] = {
                "n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
                "mad": mad, "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
                "p01": float(np.percentile(x, 1)), "p50": float(np.median(x)),
                "p99": float(np.percentile(x, 99)),
                "p999": float(np.percentile(x, 99.9)),
                "absmax": float(np.abs(x).max()),
                "frac_beyond_phys_limit": float((np.abs(x) > L.PHYS_LIMIT[c]).mean()),
                "autocorr_lag1_6": [float(v) for v in acm],
                **ceil,
            }
        # what fraction of windows at |x|=OUT would explain the card's train std?
        obs = max(rec["pai"]["std"], rec["cm"]["std"])
        card = CARD_STD[c]
        rec["card_train_std"] = card
        rec["card_over_measured_std"] = card / obs
        if card > obs:
            # contamination model: (1-f) clean var + f * OUT^2 = card^2
            rec["contamination_model"] = {
                f"f_at_|x|={OUT}": float(max(0.0, (card ** 2 - obs ** 2) / OUT ** 2))
                for OUT in (2.0, 5.0, 8.0, 20.0)}
        out["channels"][c] = rec

    out["long_accel_provenance"] = {
        dom: {"corr_vs_centred_dv_dt": float(np.mean([r[0] for r in v])),
              "corr_vs_backward_dv_dt": float(np.mean([r[1] for r in v])),
              "mae_vs_centred_dv_dt": float(np.mean([r[2] for r in v])),
              "mae_vs_backward_dv_dt": float(np.mean([r[3] for r in v]))}
        for dom, v in accel_prov.items()}
    out["steer_redundancy"] = {
        dom: {"corr_steer_vs_curvature(yr/v)": float(np.mean([r[0] for r in v])),
              "corr_steer_vs_yawrate": float(np.mean([r[1] for r in v])),
              "n_clips": len(v)} for dom, v in steer_bike.items()}
    out["card_standardizer"] = {"mean": CARD_MEAN, "std": CARD_STD,
                                "source": "idm_head_v1_card.json "
                                          "config.target_normalisation (MEASURED artifact)"}
    L.jdump(out, "/root/idm2/out/labels.json")

    # human-readable
    print("\n=== label noise floor (R2 ceiling for ANY smooth predictor) ===")
    print(f"{'channel':<12}{'dom':<5}{'std':>9}{'mad':>9}{'rho1':>8}"
          f"{'ceil_w9':>9}{'ceil_w17':>9}{'noise_rms_w9':>14}")
    for c in L.SCALARS:
        for dom in ("pai", "cm"):
            r = out["channels"][c][dom]
            print(f"{c:<12}{dom:<5}{r['std']:>9.4f}{r['mad']:>9.4f}"
                  f"{r['autocorr_lag1_6'][0]:>8.3f}{r['r2_ceiling_w9']:>9.4f}"
                  f"{r['r2_ceiling_w17']:>9.4f}{r['noise_rms_w9']:>14.4f}")
    print("\n=== card train std vs measured ===")
    for c in L.SCALARS:
        r = out["channels"][c]
        print(f"{c:<12} card {r['card_train_std']:>8.4f}  "
              f"x measured = {r['card_over_measured_std']:.2f}")
    print("\n", out["long_accel_provenance"])
    print("\n", out["steer_redundancy"])


if __name__ == "__main__":
    main()
