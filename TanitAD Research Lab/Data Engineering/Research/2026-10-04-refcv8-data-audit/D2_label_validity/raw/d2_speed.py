"""D2 deliverable 4: validity of the fed per-clip max-speed ceiling.

For every training/eval WINDOW (NOW row = t + w - 1, w = 8, rows 7 .. T-22 -- V3Dataset index
range(T - w - 20)) compare
  fed      = the clip's sidecar v_hi_ms (+ its 4-way bin {30,50,100,120} km/h), constant per clip
  own_max  = max speed over [NOW+2, NOW+6] s, read from the 100 Hz egomotion LOG (independent of
             the cache; the builder's own 10 Hz grid sampling is used, so it is like-for-like)
  v_now    = speed at NOW (cache pose)
usage: d2_speed.py <split> <manifest.pt> <sidecar.jsonl> <labels.jsonl.gz> <out.json>
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import d2_lib as L

split, man_p, spd_p, lab_p, out_p = sys.argv[1:6]
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"
side = L.load_clock_sidecar("D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
trs = L.tracks_from_manifest(man_p, side)
spd = {}
for l in open(spd_p, encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        spd[int(r["sid"])] = r
labs = {L.stable_episode_id(r["clip_id"]): r for r in L.read_jsonl_gz(lab_p)}
LADDER_KMH = [(30, 0), (50, 1), (100, 2), (120, 3)]


def my_bin(v_ms):
    k = v_ms * 3.6
    for lim, b in LADDER_KMH:
        if k <= lim + 1e-9:
            return b, lim
    return 3, 120


R = {"split": split, "n_clips_tracks": len(trs)}
rows = []            # per-window arrays
clip_rows = []
n_noego = 0
bin_mismatch = 0
for tr in trs:
    s = spd.get(tr.sid)
    if s is None:
        continue
    lab = labs.get(tr.sid)
    b_mine, lim_mine = my_bin(float(s["v_hi_ms"]))
    if b_mine != int(s["bin"]):
        bin_mismatch += 1
    try:
        df = pd.read_parquet(EGO.format(tr.clip_id), columns=["timestamp", "vx", "vy", "vz"])
    except Exception:
        n_noego += 1
        continue
    ts = df.timestamp.to_numpy(dtype=np.float64) / 1e6
    ts = ts - ts[0]
    vlog = np.linalg.norm(df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64), axis=1)
    T = len(tr.t)
    rr = np.arange(7, T - 21)                    # NOW rows
    now = tr.t[rr]
    grid = np.arange(2.0, 6.0 + 1e-9, 0.1)
    tt = (now[:, None] + grid[None, :])
    v_fut = np.interp(tt.ravel(), ts, vlog).reshape(tt.shape)
    own_max = v_fut.max(axis=1)
    own_min = v_fut.min(axis=1)
    ok = tt[:, -1] <= ts[-1]
    t0 = float(lab["t0_s"]) if lab else 8.0
    v0 = float(np.interp(t0, ts, vlog))
    clip_rows.append({"sid": tr.sid, "sha12": L.sha12(tr.clip_id), "v_hi": float(s["v_hi_ms"]), "bin": int(s["bin"]), "limit": int(s["limit_kmh"]),
                      "v0": v0, "road": (lab.get("strata") or {}).get("road_class") if lab else None,
                      "dn": (lab.get("strata") or {}).get("daynight_clock") if lab else None,
                      "own_at_anchor": float(np.max(np.interp(t0 + grid, ts, vlog))),
                      "lat": lab["a_tac"]["lat"] if lab else None, "lon": lab["a_tac"]["lon"] if lab else None})
    for i in range(len(rr)):
        if ok[i]:
            rows.append((tr.sid, float(now[i] - t0), float(s["v_hi_ms"]), int(s["bin"]), int(s["limit_kmh"]),
                         float(own_max[i]), float(own_min[i]), float(tr.v[rr[i]])))
R["n_clips_with_sidecar_and_ego"] = len(clip_rows)
R["n_clips_without_ego_log"] = n_noego
R["sidecar_bin_vs_independent_ladder_mismatch"] = bin_mismatch
A = np.array(rows, dtype=np.float64)
R["n_windows"] = int(len(A))
off, fed, binn, lim, own, ownmin, vnow = A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5], A[:, 6], A[:, 7]
lim_ms = lim / 3.6


def q(x):
    return {k: round(float(np.percentile(x, k)), 3) for k in (5, 25, 50, 75, 95)}


def block(m, name):
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    e = own[m] - fed[m]
    d = {"n": n,
         "corr_fed_vs_ownmax": round(float(np.corrcoef(fed[m], own[m])[0, 1]), 4),
         "corr_vnow_vs_ownmax": round(float(np.corrcoef(vnow[m], own[m])[0, 1]), 4),
         "own_minus_fed_quantiles_ms": q(e),
         "mean_abs_err_ms": round(float(np.abs(e).mean()), 3),
         "frac_own_exceeds_fed_by_gt_1ms": round(float((e > 1.0).mean()), 4),
         "frac_own_exceeds_fed_by_gt_3ms": round(float((e > 3.0).mean()), 4),
         "frac_fed_exceeds_own_by_gt_3ms": round(float((e < -3.0).mean()), 4),
         "frac_fed_exceeds_own_by_gt_5ms": round(float((e < -5.0).mean()), 4),
         "frac_own_exceeds_BIN_limit": round(float((own[m] > lim_ms[m] + 1e-9).mean()), 4),
         "frac_bin_limit_gt_2x_own": round(float((lim_ms[m] > 2.0 * np.maximum(own[m], 1.0)).mean()), 4)}
    # R^2 of own_max explained by fed vs by v_now (simple linear fits, same windows)
    for nm, x in (("fed", fed[m]), ("vnow", vnow[m]), ("bin_limit", lim_ms[m])):
        c = np.polyfit(x, own[m], 1)
        res = own[m] - np.polyval(c, x)
        d[f"R2_ownmax_from_{nm}"] = round(float(1 - res.var() / own[m].var()), 4)
    return d


R["all_windows"] = block(np.ones(len(A), bool), "all")
R["supervised_range_|NOW-t0|<=2"] = block(np.abs(off) <= 2.0, "sup")
bins = [(-9, -6), (-6, -4), (-4, -2), (-2, 0), (0, 2), (2, 4), (4, 6), (6, 12)]
R["by_NOW_minus_t0"] = {f"[{a},{b})": block((off >= a) & (off < b), "b") for a, b in bins}
# is the fed ceiling EVER computed over the window the model is looking at?  |NOW - t0| tiny only.
R["fed_ceiling_refers_to_window_start_s_after_NOW"] = "t0+2 .. t0+6 raw; for window NOW the ceiling's source stretch starts at (t0 - NOW) + 2 s after NOW"
# ---- clip-level: intersection artefact -------------------------------------------------
C = clip_rows
v0 = np.array([c["v0"] for c in C])
binC = np.array([c["bin"] for c in C])
road = np.array([str(c["road"]) for c in C])
stopped = v0 < 1.0
R["clip_level"] = {
    "n": len(C),
    "bin_distribution_clips": dict(Counter(int(b) for b in binC)),
    "limit_kmh_distribution_clips": dict(Counter(c["limit"] for c in C)),
    "frac_clips_bin_le_30kmh": round(float((binC == 0).mean()), 4),
    "n_stopped_v0_lt_1": int(stopped.sum()),
    "stopped_frac_of_clips": round(float(stopped.mean()), 4),
    "stopped_and_bin_le_30": int((stopped & (binC == 0)).sum()),
    "stopped_frac_bin_le_30": round(float((binC[stopped] == 0).mean()), 4) if stopped.any() else None,
    "stopped_bin_distribution": dict(Counter(int(b) for b in binC[stopped])),
    "stopped_realised_own_max_quantiles_ms": q(np.array([c["own_at_anchor"] for c in C])[stopped]) if stopped.any() else None,
    "moving_frac_bin_le_30": round(float((binC[~stopped] == 0).mean()), 4),
}
for rc in ("intersection", "urban", "highway"):
    m = road == rc
    if m.any():
        R["clip_level"][f"road={rc}"] = {
            "n": int(m.sum()), "frac_bin_le_30": round(float((binC[m] == 0).mean()), 4),
            "frac_v0_lt_1": round(float((v0[m] < 1.0).mean()), 4),
            "frac_bin_le_30_given_v0_ge_1": round(float((binC[m & ~stopped] == 0).mean()), 4) if (m & ~stopped).any() else None,
            "median_v_hi_ms": round(float(np.median([c["v_hi"] for c, mm in zip(C, m) if mm])), 2)}
# bin granularity: how much real speed does each bin hide?
for b, name in ((0, "30"), (1, "50"), (2, "100"), (3, "120")):
    m = binC == b
    if m.any():
        vh = np.array([c["v_hi"] for c in C])[m]
        R["clip_level"][f"bin_{name}_kmh_vhi_range_ms"] = {"n": int(m.sum()), "min": round(float(vh.min()), 2), "median": round(float(np.median(vh)), 2), "max": round(float(vh.max()), 2)}
# 'what does the bin lose': R^2 of v_hi from bin limit alone, clip level
vh_all = np.array([c["v_hi"] for c in C])
lim_c = np.array([c["limit"] for c in C]) / 3.6
cc = np.polyfit(lim_c, vh_all, 1)
R["clip_level"]["R2_vhi_from_bin_limit_only"] = round(float(1 - (vh_all - np.polyval(cc, lim_c)).var() / vh_all.var()), 4)
cc = np.polyfit(v0, vh_all, 1)
R["clip_level"]["R2_vhi_from_v0_only"] = round(float(1 - (vh_all - np.polyval(cc, v0)).var() / vh_all.var()), 4)
# the 50 vs 100 km/h bin: width of the bin used by most clips
R["clip_level"]["frac_clips_in_50kmh_bin(13.9..27.8 m/s means 50..100 km/h are ONE input)"] = round(float((binC == 1).mean()), 4)
R["clip_level"]["frac_clips_in_100kmh_bin"] = round(float((binC == 2).mean()), 4)
json.dump(R, open(out_p, "w"), indent=1)
print(json.dumps(R, indent=1)[:9000])
