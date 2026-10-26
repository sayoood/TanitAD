"""PROBE 3 (DESCRIPTIVE, eval dumps, read-only): the GT-fast pairs on which the ARM's own executed speed is < 5.2 m/s (they carry most of L2's excess, probe 2).
 (a) where they are (plan segment, q), which windows (v0, GT speed profile), how slow, and how P10 moves on the same pairs;
 (b) the lateral wiggle on them in METRES: |dk| x ds_X^2 (the second-difference lateral error the curvature error corresponds to on the arm's own chords),
     against the same quantity for the GT itself perturbed by i.i.d. 5 mm noise (the instrument's noise floor at that chord length);
 (c) whether the post-map's alternating cross weights (w_cross knots 1-4 = 1, 0, 1, 0) are where those pairs' curvature comes from: share of slow-pair excess at q = 0..3.
python probe3_slowpairs.py > probe3.log"""
import json
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402

out = {}
rng = np.random.default_rng(0)
for cfg in ("causal", "plainB_logged"):
    out[cfg] = {}
    for s in (0, 1):
        d = C.load(cfg, s)
        G, X, F, eid, v0 = d["g"], d["cl"], d["fl"], d["eid"], d["v0"]
        pa, pf = C.pair_err(X, G), C.pair_err(F, G)
        ok = (pa["c"] > 0) & (pf["c"] > 0)
        slow = pa["fast"] & (pa["vX"] < C.VMIN) & ok[:, None]
        w_slow = slow.any(1)
        r = {"n_slow_pairs": int(slow.sum()), "n_windows_with_slow_pair": int(w_slow.sum()), "n_episodes": int(len(np.unique(eid[w_slow])))}
        pidx = np.nonzero(slow)[1]
        r["by_segment"] = {"plan0": int((pidx < 9).sum()), "plan1": int(((pidx >= 10) & (pidx < 19)).sum()), "plan2": int((pidx >= 20).sum()),
                           "seams": int(np.isin(pidx, [9, 19]).sum())}
        r["by_q"] = [int((pidx % 10 == q).sum()) for q in range(10)]
        r["v0_median_slow_windows"] = float(np.median(v0[w_slow]))
        r["v0_median_all"] = float(np.median(v0[ok]))
        r["arm_speed_on_slow_pairs_median"] = float(np.median(pa["vX"][slow]))
        r["gt_speed_on_slow_pairs_median"] = float(np.median(pa["vG"][slow]))
        r["p10_speed_on_same_pairs_median"] = float(np.median(pf["vX"][slow]))
        r["p10_slow_on_same_pairs_frac"] = float((pf["vX"][slow] < C.VMIN).mean())
        # GT speed at t0 vs at the pair: accelerating windows?
        r["gt_accel_windows_frac"] = float(((pa["vG"][w_slow][:, -1] - v0[w_slow]) > 1.0).mean())
        # (b) metres: |dk| * ds^2 on the arm's own chords
        dsX = pa["vX"] * C.DT
        lat_m = pa["e"] * dsX ** 2
        r["lat_m_on_slow_pairs_median_p90"] = [float(np.median(lat_m[slow])), float(np.percentile(lat_m[slow], 90))]
        fastown = pa["fast"] & (pa["vX"] >= C.VMIN) & ok[:, None]
        r["lat_m_on_ownfast_pairs_median_p90"] = [float(np.median(lat_m[fastown])), float(np.percentile(lat_m[fastown], 90))]
        r["dk_on_slow_pairs_median_mean"] = [float(np.median(pa["e"][slow])), float(pa["e"][slow].mean())]
        # noise floor at the arm's own chord length: perturb a straight track at the arm's own speed with 5 mm i.i.d. noise
        vv = np.clip(pa["vX"][slow], 0.3, None)
        n = len(vv)
        seg = np.cumsum(np.repeat(vv[:, None] * C.DT, 3, 1), 1)                 # 3 knots on a straight line
        trk = np.stack([seg, np.zeros_like(seg)], -1) + rng.normal(0, 0.005, (n, 3, 2))
        p = np.concatenate([np.zeros((n, 1, 2)), trk], 1)                        # origin + 3 knots -> 3 chords -> 2 pairs; use pair 1 (no origin)
        ch = np.diff(p, axis=1)
        hd = np.arctan2(ch[..., 1], ch[..., 0])
        dsn = np.linalg.norm(ch, axis=-1)
        kn = np.abs((hd[:, 2] - hd[:, 1] + np.pi) % (2 * np.pi) - np.pi) / (0.5 * (dsn[:, 2] + dsn[:, 1]))
        r["noise5mm_dk_at_arm_speed_median_mean"] = [float(np.median(kn)), float(kn.mean())]
        # (c) share of the slow-pair arm curvature at q = 0..3
        shs = np.where(slow, pa["sh"], 0.0)
        tot = shs.sum()
        r["slow_share_q0_3"] = float(shs[:, [p_ for p_ in range(29) if p_ % 10 in (0, 1, 2, 3)]].sum() / tot)
        r["slow_share_seams"] = float(shs[:, [9, 19]].sum() / tot)
        r["slow_share_plan2"] = float(shs[:, 20:29].sum() / tot)
        out[cfg][f"s{s}"] = r
        print(f"== {cfg} s{s}"); [print("  ", k, v) for k, v in r.items()]
json.dump(out, open("C:/Users/Admin/lcdiag/probe3.json", "w"), indent=1)
