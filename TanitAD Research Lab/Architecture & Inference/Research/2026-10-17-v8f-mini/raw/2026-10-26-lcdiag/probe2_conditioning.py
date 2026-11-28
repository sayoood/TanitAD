"""PROBE 2 (DESCRIPTIVE, eval dumps, read-only): what conditions the plan-2 L2 excess.
 (a) absolute L2-cell levels per plan segment (pairs 0-8 / 10-18 / 20-28; seams 9, 19 apart) for HACC, P10, plain B;
 (b) OWN-speed conditioning: L2's mask is GT pair speed >= 5.2 only; an arm crawling where the human is fast computes curvature on short chords.
     Re-read with the JOINT mask (GT, HACC's own and P10's own pair speed all >= 5.2: the SAME pairs for both arms), and the share of the excess
     carried by pairs where the arm's own speed is < 5.2;
 (c) ARC-LENGTH alignment: compare each executed pair's curvature with the GT curvature at the SAME distance travelled (linear interpolation of the
     GT pair curvature over the GT's arc length at the shared knot), mask = GT speed at that arc >= 5.2 and the pair valid: the path-shape error
     with the timing removed;
 (d) per-window plan-2 excess vs HACC's and P10's along-track offset at t = 4 s.
python probe2_conditioning.py > probe2.log"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402

SEG = {"plan0": list(range(0, 9)), "plan1": list(range(10, 19)), "plan2": list(range(20, 29)), "seams": [9, 19]}


def arc_aligned(X, G, vmin=C.VMIN):
    """Per pair: |kappa_X(p) - kappa_G(s_X(p))|, s_X(p) = X's arc length to the pair's shared knot (p+1-th knot); GT curvature and GT pair speed
    interpolated on the GT's own arc length at its shared knots. Masked where X's pair is invalid, s_X is beyond the GT's covered arc, or GT speed < vmin."""
    px, pg = C.pair_err(X, X), C.pair_err(G, G)                       # (geometry only; e is zero here)
    def arc(T):
        p = np.concatenate([np.zeros((len(T), 1, 2)), np.asarray(T, np.float64)], 1)
        return np.concatenate([np.zeros((len(T), 1)), np.cumsum(np.linalg.norm(np.diff(p, axis=1), axis=-1), 1)], 1)   # [n, 31] at knots 0..30
    sX, sG = arc(X)[:, 1:30], arc(G)[:, 1:30]                         # arc at the shared knot of pair p (knot p+1), [n, 29]
    import torch
    PX = C.FF._seq_geometry(torch.as_tensor(np.asarray(X, np.float64)), C.DT)
    PG = C.FF._seq_geometry(torch.as_tensor(np.asarray(G, np.float64)), C.DT)
    vX_ok = PX["pair_valid"].numpy()
    gk, gv, gok = PG["curvature"].numpy(), (0.5 * (PG["speed"][:, 1:] + PG["speed"][:, :-1])).numpy(), PG["pair_valid"].numpy()
    kX = PX["curvature"].numpy()
    e = np.full(kX.shape, np.nan)
    m = np.zeros(kX.shape, bool)
    for i in range(len(X)):
        ok = gok[i]
        if ok.sum() < 2:
            continue
        sg = sX_g = sG[i][ok]
        if not np.all(np.diff(sg) > 0):
            continue
        kg = np.interp(sX[i], sg, gk[i][ok])
        vg = np.interp(sX[i], sg, gv[i][ok])
        inside = (sX[i] >= sg[0]) & (sX[i] <= sg[-1])
        mm = vX_ok[i] & inside & (vg >= vmin)
        e[i] = np.abs(kX[i] - kg)
        m[i] = mm
    c = m.sum(1)
    sh = np.where(m, np.nan_to_num(e), 0.0) / np.maximum(c, 1)[:, None]
    return sh, c, m


out = {}
for cfg in ("causal", "logged", "plainB_logged"):
    out[cfg] = {}
    for s in (0, 1):
        d = C.load(cfg, s)
        G, X, F, eid = d["g"], d["cl"], d["fl"], d["eid"]
        pa, pf = C.pair_err(X, G), C.pair_err(F, G)
        ok = (pa["c"] > 0) & (pf["c"] > 0)
        r = {}
        # (a) absolute per-segment levels on the L2 mask: mean |dk| over fast pairs in that segment (pooled over windows)
        lv = {}
        for g, ps in SEG.items():
            ma, mf = pa["fast"][:, ps], pf["fast"][:, ps]
            lv[g] = {"arm": float(pa["e"][:, ps][ma].mean()), "p10": float(pf["e"][:, ps][mf].mean()), "n_pairs": int(ma.sum())}
        r["a_levels_pooled_pairs"] = lv
        # (b) joint own-speed mask
        jm = pa["fast"] & pf["fast"] & (pa["vX"] >= C.VMIN) & (pf["vX"] >= C.VMIN)
        cj = jm.sum(1)
        okj = cj > 0
        sa = np.where(jm, pa["e"], 0).sum(1) / np.maximum(cj, 1)
        sf = np.where(jm, pf["e"], 0).sum(1) / np.maximum(cj, 1)
        r["b_joint_mask_delta"] = C.boot(sa[okj], sf[okj], eid[okj])
        r["b_joint_mask_frac_of_fast_pairs_kept"] = float(jm.sum() / pa["fast"].sum())
        slow_a = pa["fast"] & (pa["vX"] < C.VMIN)
        slow_f = pf["fast"] & (pf["vX"] < C.VMIN)
        exc = (pa["sh"] - pf["sh"])[ok]
        r["b_share_of_excess_on_pairs_where_arm_own_speed_lt_vmin"] = float(np.where(slow_a[ok], pa["sh"][ok], 0).sum(1).mean()
                                                                           - np.where(slow_f[ok], pf["sh"][ok], 0).sum(1).mean())
        r["b_frac_fast_pairs_arm_slow"] = float(slow_a.sum() / pa["fast"].sum())
        r["b_frac_fast_pairs_p10_slow"] = float(slow_f.sum() / pf["fast"].sum())
        r["L2_delta"] = float(exc.sum(1).mean())
        # (b2) the excess by the ARM's own pair speed bin (GT-fast pairs only)
        bins = [0, 2, 5.2, 8.6, 11.8, 99]
        r["b2_excess_by_arm_own_speed"] = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            ma = (pa["vX"] >= lo) & (pa["vX"] < hi)
            r["b2_excess_by_arm_own_speed"][f"{lo}-{hi}"] = {"arm_share": float(np.where(ma[ok], pa["sh"][ok], 0).sum(1).mean()),
                                                              "n_pairs_arm": int((ma & pa["fast"]).sum())}
            mf = (pf["vX"] >= lo) & (pf["vX"] < hi)
            r["b2_excess_by_arm_own_speed"][f"{lo}-{hi}"]["p10_share"] = float(np.where(mf[ok], pf["sh"][ok], 0).sum(1).mean())
            r["b2_excess_by_arm_own_speed"][f"{lo}-{hi}"]["n_pairs_p10"] = int((mf & pf["fast"]).sum())
        # (c) arc-length aligned
        sha, ca, ma = arc_aligned(X, G)
        shf, cf, mf = arc_aligned(F, G)
        okc = (ca > 0) & (cf > 0)
        r["c_arc_aligned_delta"] = C.boot(sha.sum(1)[okc], shf.sum(1)[okc], eid[okc])
        r["c_arc_aligned_by_segment"] = {g: C.boot(sha[okc][:, ps].sum(1), shf[okc][:, ps].sum(1), eid[okc]) for g, ps in SEG.items()}
        # (d) along-track offset at t = 4 s (knot index 19) in the GT tangent frame
        def along_off(T):
            tg = G[:, 19] - G[:, 18]
            tg = tg / np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            return ((T[:, 19] - G[:, 19]) * tg).sum(1)
        exc2 = (pa["sh"] - pf["sh"])[:, SEG["plan2"]].sum(1)
        okd = ok & (pa["fast"][:, SEG["plan2"]].sum(1) > 0)
        aH, aF = along_off(X), along_off(F)
        r["d_spearman_plan2excess_vs_absalongH"] = float(spearmanr(exc2[okd], np.abs(aH[okd]))[0])
        r["d_spearman_plan2excess_vs_absalongF"] = float(spearmanr(exc2[okd], np.abs(aF[okd]))[0])
        r["d_median_along_off_4s_H_F"] = [float(np.median(aH[okd])), float(np.median(aF[okd]))]
        out[cfg][f"s{s}"] = r
        print(f"== {cfg} s{s}  L2 delta {r['L2_delta']:+.4f}")
        print("  (a) levels arm/p10:", {g: (round(v['arm'], 4), round(v['p10'], 4), v['n_pairs']) for g, v in lv.items()})
        q = r["b_joint_mask_delta"]
        print(f"  (b) joint own-speed mask: {q['delta']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}] sep {q['separated']} (kept {r['b_joint_mask_frac_of_fast_pairs_kept']:.3f} of fast pairs)")
        print(f"      excess on arm-slow pairs {r['b_share_of_excess_on_pairs_where_arm_own_speed_lt_vmin']:+.4f}; frac fast pairs arm slow {r['b_frac_fast_pairs_arm_slow']:.4f}, p10 slow {r['b_frac_fast_pairs_p10_slow']:.4f}")
        print("      by own speed (arm share, p10 share):", {k: (round(v['arm_share'], 4), round(v['p10_share'], 4), v['n_pairs_arm'], v['n_pairs_p10']) for k, v in r["b2_excess_by_arm_own_speed"].items()})
        q = r["c_arc_aligned_delta"]
        print(f"  (c) arc-aligned: {q['delta']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}] sep {q['separated']} n {q['n_windows']}; by seg",
              {g: (v['delta'], v['separated']) for g, v in r["c_arc_aligned_by_segment"].items()})
        print(f"  (d) spearman plan2 excess vs |along H| {r['d_spearman_plan2excess_vs_absalongH']:+.3f}, vs |along F| {r['d_spearman_plan2excess_vs_absalongF']:+.3f}; median along H/F {r['d_median_along_off_4s_H_F']}")
json.dump(out, open("C:/Users/Admin/lcdiag/probe2.json", "w"), indent=1)
