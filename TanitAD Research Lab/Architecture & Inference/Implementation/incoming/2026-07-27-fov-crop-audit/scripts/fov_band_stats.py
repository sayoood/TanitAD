"""FOV crop audit — PART 1 analysis: the per-situation loss, with intervals, never pooled.

Consumes `fov_bands.npz` (per-agent-sample bands) + the situation classifier's own label bundle
and emits, PER SITUATION and PER POPULATION:

  (a) CONTENT SHARE   — of the agent-samples visible at frames of a given kind, what fraction sits
                        IN_CROP / CROPPED_AWAY / OFF_FRONT.
  (b) PRESENCE LIFT   — P(>=1 agent of population P in band B at frame t | frame kind) divided by
                        the MATCHED baseline on the SAME clips. A band that always contains
                        something proves nothing; only the lift over the matched baseline is
                        evidence (the sibling stream's any_off_front 1.009 [0.970, 1.045] is the
                        worked example of this trap).
  (c) RECOVERY CURVE  — for each candidate crop half-angle, the fraction of the currently-missed
                        decision-relevant agent-samples a WIDER CROP ALONE would recover
                        (i.e. excluding everything that is off the front sensor entirely).

Frame kinds, all against the same BASELINE:
  ANTICIPATION  y_S = 1 and valid_S   — the frames a sensor-activation policy must fire on
  ONGOING       ongoing_S             — the frames the manoeuvre is actually happening in
  BASELINE      y_S = 0 and valid_S and not ongoing_S

Estimator: clip-cluster bootstrap (`taniteval.ci._draws`, B = 2000) — the clip is the independent
unit, exactly as in the situation classifier. Both subsets are recomputed inside each draw, so the
ratio's interval is PAIRED by construction.  ⛔ `overlapping_holdout_se` is not used.

usage:  python fov_band_stats.py <bands_dir> <bundle_dir> <out_json> [side]
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get(
    "TANITEVAL_DIR",
    r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\taniteval"))
from taniteval.ci import _draws, episode_index   # noqa: E402

B_BOOT = 2000
SEED = 0
SITS = ["lane_change", "intersection", "roundabout"]
POPS = ["ALL", "NEAR", "CROSS"]
BANDS = ["IN_CROP", "CROPPED_AWAY", "OFF_FRONT"]
NEAR_M = 40.0
F_REF, SIZE = 266.0, 256
CANON_HALF_DEG = math.degrees(math.atan((SIZE / 2) / F_REF))
HALF_GRID_DEG = [CANON_HALF_DEG, 28.0, 30.0, 32.5, 35.0, 37.5, 40.0, 45.0, 50.0, 55.0, 60.25]
MIN_CLUSTERS = 40          # the situation classifier's own C-POW bar; below it -> no verdict


def poly_r(poly, th):
    r = np.zeros_like(np.asarray(th, dtype=float))
    for c in reversed(poly):
        r = r * th + c
    return r


def _ratio_ci(num_mask, den_mask, val, eid, n_boot=B_BOOT, seed=SEED):
    """Point + percentile CI for mean(val[num_mask]) / mean(val[den_mask]), clip-cluster paired.

    Returns None when either subset is empty (a rate with no denominator is refused, not NaN'd).
    """
    if not num_mask.any() or not den_mask.any():
        return None
    uniq, idx_by_ep = episode_index(eid)
    a0, b0 = float(val[num_mask].mean()), float(val[den_mask].mean())
    ratio = a0 / b0 if b0 > 0 else float("inf")
    rs, ds = [], []
    for sel in _draws(uniq, idx_by_ep, n_boot, seed):
        nm, dm = num_mask[sel], den_mask[sel]
        if not nm.any() or not dm.any():
            continue
        a, b = float(val[sel][nm].mean()), float(val[sel][dm].mean())
        ds.append(a - b)
        rs.append(a / b if b > 0 else np.nan)
    rs = np.asarray(rs, float)
    rs = rs[np.isfinite(rs)]
    ds = np.asarray(ds, float)
    if rs.size < n_boot // 4:
        return None
    lo, hi = np.percentile(rs, [2.5, 97.5])
    dlo, dhi = np.percentile(ds, [2.5, 97.5])
    return {"rate_in": round(a0, 5), "rate_base": round(b0, 5),
            "lift": round(ratio, 4), "lift_lo": round(float(lo), 4),
            "lift_hi": round(float(hi), 4),
            "separated_from_1": bool(lo > 1.0 or hi < 1.0),
            "delta": round(a0 - b0, 5), "delta_lo": round(float(dlo), 5),
            "delta_hi": round(float(dhi), 5),
            "delta_separated": bool(dlo > 0 or dhi < 0),
            "n_in": int(num_mask.sum()), "n_base": int(den_mask.sum()),
            "n_clusters": int(len(uniq)), "n_boot": int(n_boot),
            "estimator": "paired_clip_cluster_bootstrap"}


def main():
    bands_dir, bundle, out_json = sys.argv[1:4]
    side = sys.argv[4] if len(sys.argv) > 4 else "HELDOUT"
    Z = np.load(os.path.join(bands_dir, "fov_bands.npz"))
    L = np.load(os.path.join(bundle, "sc_labels.npz"))
    D = pd.read_parquet(os.path.join(bands_dir, "fov_bands_index.parquet"))
    # `ALL` = no split filter. Legitimate here and ONLY here: Part 1 fits NOTHING — it is pure
    # geometry against a privileged label — so there is no estimator to leak into. Reported as a
    # declared power-secondary; HELDOUT stays the pre-registered primary.
    D = D[(D.n_samples > 0) & (True if side == "ALL" else D.side == side)].reset_index(drop=True)

    # ---------------------------------------------------------------- assemble the flat arrays
    fr = {s: {} for s in SITS}                    # per-frame kind masks
    fr_eid, fr_flags = [], {f"{p}_{b}": [] for p in POPS for b in BANDS}
    smp = {c: [] for c in ("k", "t", "req_half", "in_front", "rng", "cross", "c_half")}
    kinds = {s: {"ANT": [], "ONG": [], "BASE": []} for s in SITS}
    for row in D.itertuples(index=False):
        k, T = int(row.k), int(row.T)
        rec = {n: Z[f"c{k}_{n}"] for n in
               ("t", "req_half", "in_front", "in_crop", "vis_other", "rng", "cross")}
        pops = {"ALL": np.ones(len(rec["t"]), bool), "NEAR": rec["rng"] <= NEAR_M,
                "CROSS": rec["cross"]}
        bands = {"IN_CROP": rec["in_crop"],
                 "CROPPED_AWAY": rec["in_front"] & ~rec["in_crop"],
                 "OFF_FRONT": (~rec["in_front"]) & rec["vis_other"]}
        for p in POPS:
            for b in BANDS:
                a = np.zeros(T, bool)
                m = pops[p] & bands[b]
                if m.any():
                    np.logical_or.at(a, rec["t"][m], True)
                fr_flags[f"{p}_{b}"].append(a)
        fr_eid.append(np.full(T, k))
        for s in SITS:
            y, val, ong = (L[f"c{k}_y_{s}"].astype(bool), L[f"c{k}_valid_{s}"].astype(bool),
                           L[f"c{k}_ongoing_{s}"].astype(bool))
            kinds[s]["ANT"].append(y & val)
            kinds[s]["ONG"].append(ong)
            kinds[s]["BASE"].append((~y) & val & (~ong))
        for c, v in (("k", np.full(len(rec["t"]), k)), ("t", rec["t"]),
                     ("req_half", rec["req_half"]), ("in_front", rec["in_front"]),
                     ("rng", rec["rng"]), ("cross", rec["cross"]),
                     ("c_half", np.full(len(rec["t"]), float(row.c_half)))):
            smp[c].append(v)
    fr_eid = np.concatenate(fr_eid)
    for kk in fr_flags:
        fr_flags[kk] = np.concatenate(fr_flags[kk])
    for s in SITS:
        for kk in kinds[s]:
            kinds[s][kk] = np.concatenate(kinds[s][kk])
    for c in smp:
        smp[c] = np.concatenate(smp[c])
    smp_eid = smp["k"]

    # per-sample frame-kind membership (join the sample onto its frame's label)
    off = {}
    cursor = 0
    for row in D.itertuples(index=False):
        off[int(row.k)] = cursor
        cursor += int(row.T)
    lin = np.array([off[int(kk)] for kk in smp["k"]]) + smp["t"].astype(np.int64)

    res = {"side": side, "n_clips": int(len(D)),
           "n_frames": int(len(fr_eid)), "n_samples": int(len(smp["k"])),
           "canonical_half_deg": round(CANON_HALF_DEG, 4),
           "estimator": "clip-cluster bootstrap (taniteval.ci._draws), B=%d" % B_BOOT,
           "MIN_CLUSTERS": MIN_CLUSTERS,
           "populations": {}, "presence_lift": {}, "content_share": {}, "recovery": {},
           "power": {}}

    # ------------------------------------------------------------------------- power, per situation
    for s in SITS:
        ant = kinds[s]["ANT"]
        clusters = len(np.unique(fr_eid[ant])) if ant.any() else 0
        res["power"][s] = {"n_ant_frames": int(ant.sum()),
                           "n_ongoing_frames": int(kinds[s]["ONG"].sum()),
                           "n_base_frames": int(kinds[s]["BASE"].sum()),
                           "n_positive_clip_clusters": int(clusters),
                           "base_rate": round(float(ant.sum()) /
                                              max(1, int(ant.sum() + kinds[s]["BASE"].sum())), 5),
                           "C_POW": "OK" if clusters >= MIN_CLUSTERS else "UNDERPOWERED"}

    # -------------------------------------------------------------------------- (b) presence lift
    for s in SITS:
        base = kinds[s]["BASE"]
        res["presence_lift"][s] = {}
        for kind in ("ANT", "ONG"):
            sel = kinds[s][kind]
            res["presence_lift"][s][kind] = {}
            for p in POPS:
                for b in BANDS:
                    v = fr_flags[f"{p}_{b}"].astype(np.float64)
                    r = _ratio_ci(sel, base, v, fr_eid)
                    if r is not None:
                        res["presence_lift"][s][kind][f"{p}|{b}"] = r

    # ------------------------------------------------------------------------- (a) content share
    for s in SITS:
        res["content_share"][s] = {}
        for kind, sel in (("ANT", kinds[s]["ANT"]), ("ONG", kinds[s]["ONG"]),
                          ("BASE", kinds[s]["BASE"])):
            m_smp = sel[lin]
            res["content_share"][s][kind] = {}
            for p in POPS:
                pm = {"ALL": np.ones(len(lin), bool), "NEAR": smp["rng"] <= NEAR_M,
                      "CROSS": smp["cross"].astype(bool)}[p]
                sub = m_smp & pm
                n = int(sub.sum())
                if n == 0:
                    continue
                in_crop = smp["in_front"] & (smp["req_half"] <= smp["c_half"])
                bm = {"IN_CROP": in_crop,
                      "CROPPED_AWAY": smp["in_front"] & ~in_crop,
                      "OFF_FRONT": ~smp["in_front"]}
                d = {"n_samples": n}
                for b in BANDS:
                    share = np.zeros(len(lin))
                    share[sub & bm[b]] = 1.0
                    uniq, idx_by_ep = episode_index(smp_eid[sub])
                    pt = float(share[sub].mean())
                    boots = np.array([float(share[sub][x].mean()) for x in
                                      _draws(uniq, idx_by_ep, 400, SEED)])
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    d[b] = {"share": round(pt, 5), "lo": round(float(lo), 5),
                            "hi": round(float(hi), 5), "n_boot": 400}
                res["content_share"][s][kind][p] = d

    # ------------------------------------------------------------------------- (c) recovery curve
    poly = (0.0, 927.5032, 23.1353, -58.5012, 16.5067)
    # the per-clip crop half-side scales with the clip's own focal; approximate a candidate
    # half-angle theta for clip i as c_half_i * r_med(theta)/r_med(canon) -- exact to the
    # per-clip focal sigma of 0.47 % (MEASURED, calib.py:150-152)
    r_med = {h: float(poly_r(poly, math.radians(h))) for h in HALF_GRID_DEG}
    r_canon = r_med[HALF_GRID_DEG[0]]
    for s in SITS:
        res["recovery"][s] = {}
        for kind, sel in (("ANT", kinds[s]["ANT"]), ("ONG", kinds[s]["ONG"]),
                          ("BASE", kinds[s]["BASE"])):
            m_smp = sel[lin]
            res["recovery"][s][kind] = {}
            for p in POPS:
                pm = {"ALL": np.ones(len(lin), bool), "NEAR": smp["rng"] <= NEAR_M,
                      "CROSS": smp["cross"].astype(bool)}[p]
                sub = m_smp & pm
                if not sub.any():
                    continue
                in_crop0 = smp["in_front"] & (smp["req_half"] <= smp["c_half"])
                missed = sub & ~in_crop0
                n_missed = int(missed.sum())
                if n_missed == 0:
                    continue
                curve = {}
                for h in HALF_GRID_DEG:
                    thr = smp["c_half"] * (r_med[h] / r_canon)
                    rec = missed & smp["in_front"] & (smp["req_half"] <= thr)
                    curve[f"{h:.3f}"] = {
                        "full_hfov_deg": round(2 * h, 2),
                        "f_eff": round((SIZE / 2) / math.tan(math.radians(h)), 1),
                        "recovered_frac_of_missed": round(float(rec.sum()) / n_missed, 5),
                        "captured_frac_of_pop": round(
                            float((sub & smp["in_front"] &
                                   (smp["req_half"] <= thr)).sum()) / int(sub.sum()), 5)}
                res["recovery"][s][kind][p] = {
                    "n_pop": int(sub.sum()), "n_missed_at_canon": n_missed,
                    "frac_missed_at_canon": round(n_missed / int(sub.sum()), 5),
                    "ceiling_recoverable_by_crop": round(
                        float((missed & smp["in_front"]).sum()) / n_missed, 5),
                    "curve": curve}

    json.dump(res, open(out_json, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("side", "n_clips", "n_frames", "n_samples", "power")},
                     indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
