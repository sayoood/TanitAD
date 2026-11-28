"""D2 deliverables 2 & 3: independent-geometry agreement of a_tac.lat / a_tac.lon, plus the
lat_peak_m / within_m / SPEED_BAND recomputation checks, at (i) the anchor (NOW = t0) and
(ii) every window the trainer actually supervises (|NOW - t0| <= 2.0 s).

usage: d2_agreement.py <labels.jsonl.gz> <v2manifest.pt> <clock_sidecar.jsonl> <out.json> <tag>
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

import d2_lib as L

labels_p, man_p, side_p, out_p, tag = sys.argv[1:6]
recs = L.read_jsonl_gz(labels_p)
side = L.load_clock_sidecar(side_p)
trs = L.tracks_from_manifest(man_p, side)
by_sid = {t.sid: t for t in trs}

LAT_ROWS = ["LANE_KEEP", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R"]
LAT_COLS = ["LANE_KEEP", "BEND_L", "BEND_R", "NUDGE_L", "NUDGE_R", "SHIFT_L", "SHIFT_R", "TURN_L", "TURN_R"]
LON_ROWS = ["HOLD", "CREEP", "CRUISE", "ACCELERATE", "BRAKE_TO", "FOLLOW", "ADAPT_SPEED_FOR_CURVE"]
LON_COLS = ["HOLD", "CREEP", "CRUISE", "ACCEL", "BRAKE"]


def coarse_lat(c):
    if c.startswith("TURN"):
        return "turn"
    if c.startswith(("NUDGE", "SHIFT")):
        return "move"
    return "keep"      # LANE_KEEP and BEND (builder calls a road bend LANE_KEEP)


def side_of(c):
    return c[-1] if c[-1] in "LR" else None


R = {"tag": tag, "labels": labels_p, "labels_md5": L.md5_file(labels_p), "manifest": man_p,
     "n_records": len(recs), "n_tracks": len(trs)}
missing = [r for r in recs if L.stable_episode_id(r["clip_id"]) not in by_sid]
R["n_records_without_cache_track"] = len(missing)
R["clock_src"] = dict(Counter(t.clock_src for t in trs))

pairs = {k: [] for k in ["latA_26", "latB_26", "latA_06", "latA0_26", "lon_26", "lon_06"]}
per_rec = []
for r in recs:
    sid = L.stable_episode_id(r["clip_id"])
    tr = by_sid.get(sid)
    if tr is None:
        continue
    at = r["a_tac"]
    ta = float(r["t0_s"])
    f26 = L.lat_features(tr, ta, 2.0, 6.0)
    f06 = L.lat_features(tr, ta, 0.0, 6.0)
    f02 = L.lat_features(tr, ta, 0.0, 2.0)
    g26 = L.lon_features(tr, ta, 2.0, 6.0)
    g06 = L.lon_features(tr, ta, 0.0, 6.0)
    cA26, cB26, cA06 = L.classify_lat(f26, "A"), L.classify_lat(f26, "B"), L.classify_lat(f06, "A")
    cA0_26 = L.classify_lat(f26, "A0")
    l26, l06 = L.classify_lon(g26), L.classify_lon(g06)
    rec = {"sha12": L.sha12(r["clip_id"]), "b_lat": at["lat"], "b_lon": at["lon"],
           "truncated": at.get("truncated"), "latA_26": cA26, "latB_26": cB26, "latA_06": cA06, "latA0_26": cA0_26,
           "lon_26": l26, "lon_06": l06, "ok": bool(f26["ok"] and f06["ok"] and g26["ok"] and g06["ok"]),
           "eps26": f26["eps"], "dpsi_net26": f26["dpsi_net"], "dpsi_peak26": f26["dpsi_peak"],
           "dpsi_peak06": f06["dpsi_peak"], "dpsi_net06": f06["dpsi_net"],
           "eps06": f06["eps"], "eps02": f02["eps"], "lat_raw_peak02": f02["lat_raw_peak"], "dpsi_peak02": f02["dpsi_peak"], "latA_02": L.classify_lat(f02, "A"), "L26": f26["L"], "lat_raw_peak06": f06["lat_raw_peak"],
           "dv26": g26["dv"], "dv06": g06["dv"], "vmax26": g26["vmax"], "v0": g06["vstart"],
           "road_class": (r.get("strata") or {}).get("road_class"),
           "daynight": (r.get("strata") or {}).get("daynight_clock"),
           "g_goals": sorted((r.get("g_tac") or {}).get("goals", {}).keys()),
           "alpamayo_lat_agree": ((r.get("alpamayo") or {}).get("lateral") or {}).get("agree")}
    per_rec.append(rec)
    if rec["ok"] and not at.get("truncated"):
        pairs["latA_26"].append((at["lat"], cA26))
        pairs["latB_26"].append((at["lat"], cB26))
        pairs["latA_06"].append((at["lat"], cA06))
        pairs["latA0_26"].append((at["lat"], cA0_26))
        pairs["lon_26"].append((at["lon"], l26))
        pairs["lon_06"].append((at["lon"], l06))

R["n_scored"] = len(pairs["latA_26"])
R["n_not_ok_or_truncated"] = len(per_rec) - R["n_scored"]
R["n_truncated_by_builder"] = sum(1 for p in per_rec if p["truncated"])

out_mat = {}
txt = []


def lat_summary(name, prs):
    M = L.confusion(LAT_ROWS, LAT_COLS, prs)
    n = len(prs)
    strict = sum(1 for a, b in prs if b in L.LAT_OK[a])
    coarse = sum(1 for a, b in prs if coarse_lat(a) == coarse_lat(b) or (coarse_lat(a) == "move" and coarse_lat(b) == "move"))
    # side accuracy where both sides define a side (builder move/turn and mine move/turn)
    both = [(a, b) for a, b in prs if side_of(a) and side_of(b)]
    side_acc = sum(1 for a, b in both if side_of(a) == side_of(b))
    # row-wise recall and the worst off-diagonal cell
    offs = []
    for i, a in enumerate(LAT_ROWS):
        for j, b in enumerate(LAT_COLS):
            if b not in L.LAT_OK[a] and M[i, j] > 0:
                offs.append((int(M[i, j]), a, b, round(M[i, j] / max(1, M[i].sum()), 4)))
    offs.sort(reverse=True)
    d = {"n": n, "matrix": M.tolist(), "rows": LAT_ROWS, "cols": LAT_COLS,
         "strict_agree": strict, "strict_acc": round(strict / n, 4),
         "strict_ci95": [round(x, 4) for x in L.wilson(strict, n)],
         "coarse_agree": coarse, "coarse_acc": round(coarse / n, 4),
         "side_agree": side_acc, "side_n": len(both),
         "side_acc": round(side_acc / max(1, len(both)), 4),
         "top_off_diagonals": offs[:8],
         "per_builder_class_strict": {a: {"n": int(M[i].sum()), "agree": int(sum(M[i, j] for j, b in enumerate(LAT_COLS) if b in L.LAT_OK[a])),
                                         "acc": round(sum(M[i, j] for j, b in enumerate(LAT_COLS) if b in L.LAT_OK[a]) / max(1, M[i].sum()), 4)}
                                      for i, a in enumerate(LAT_ROWS)}}
    txt.append(L.fmt_matrix(M, LAT_ROWS, LAT_COLS, f"[{tag}] LATERAL {name}  n={n}  strict-agree {strict}/{n} = {strict / n:.3f}"
                            f" (95% Wilson {d['strict_ci95']}); side-agree {side_acc}/{len(both)}"))
    return d


def lon_summary(name, prs):
    M = L.confusion(LON_ROWS, LON_COLS, prs)
    n = len(prs)
    spec = [(a, b) for a, b in prs if a != "ADAPT_SPEED_FOR_CURVE"]
    agree = sum(1 for a, b in spec if b in L.LON_OK[a])
    agree_ex_follow = sum(1 for a, b in spec if a != "FOLLOW" and b in L.LON_OK[a])
    n_ex_follow = sum(1 for a, b in spec if a != "FOLLOW")
    offs = []
    for i, a in enumerate(LON_ROWS):
        if a == "ADAPT_SPEED_FOR_CURVE":
            continue
        for j, b in enumerate(LON_COLS):
            if b not in L.LON_OK[a] and M[i, j] > 0:
                offs.append((int(M[i, j]), a, b, round(M[i, j] / max(1, M[i].sum()), 4)))
    offs.sort(reverse=True)
    d = {"n": n, "matrix": M.tolist(), "rows": LON_ROWS, "cols": LON_COLS,
         "n_speed_defined": len(spec), "agree": agree, "acc": round(agree / max(1, len(spec)), 4),
         "ci95": [round(x, 4) for x in L.wilson(agree, len(spec))],
         "n_ex_follow": n_ex_follow, "agree_ex_follow": agree_ex_follow,
         "acc_ex_follow": round(agree_ex_follow / max(1, n_ex_follow), 4),
         "top_off_diagonals": offs[:8],
         "per_builder_class": {a: {"n": int(M[i].sum()), "agree": int(sum(M[i, j] for j, b in enumerate(LON_COLS) if b in L.LON_OK[a])),
                                  "acc": round(sum(M[i, j] for j, b in enumerate(LON_COLS) if b in L.LON_OK[a]) / max(1, M[i].sum()), 4)}
                               for i, a in enumerate(LON_ROWS) if a != "ADAPT_SPEED_FOR_CURVE"}}
    txt.append(L.fmt_matrix(M, LON_ROWS, LON_COLS, f"[{tag}] LONGITUDINAL {name}  n={n}  speed-defined classes agree {agree}/{len(spec)}"
                            f" = {agree / max(1, len(spec)):.3f} (excl. FOLLOW {agree_ex_follow}/{n_ex_follow})"))
    return d


out_mat["latA_band2_6"] = lat_summary("A: indep over the record's own band [t0+2,t0+6]", pairs["latA_26"])
out_mat["latB_band2_6"] = lat_summary("B(fast-turn 30deg/3s): [t0+2,t0+6]", pairs["latB_26"])
out_mat["latA_span0_6"] = lat_summary("A: indep over [t0,t0+6] (builder's actual plan span)", pairs["latA_06"])
out_mat["latA0_band2_6_FIRSTRUN_superseded"] = lat_summary("A0 FIRST-RUN rule (superseded): [t0+2,t0+6]", pairs["latA0_26"])
out_mat["lon_band2_6"] = lon_summary("indep over band [t0+2,t0+6]", pairs["lon_26"])
out_mat["lon_span0_6"] = lon_summary("indep over [t0,t0+6] (builder's actual plan span)", pairs["lon_06"])
R["matrices"] = out_mat

# ---- NUDGE audit: builder NUDGE vs curvature-detrended excursion ----------------------
nud = [p for p in per_rec if p["ok"] and p["b_lat"].startswith("NUDGE")]
R["nudge_audit"] = {
    "n_builder_nudge": len(nud),
    "indep_lane_keep_over_0_6": sum(1 for p in nud if p["latA_06"] == "LANE_KEEP"),
    "indep_nudge_over_0_6": sum(1 for p in nud if p["latA_06"].startswith("NUDGE")),
    "indep_shift_over_0_6": sum(1 for p in nud if p["latA_06"].startswith("SHIFT")),
    "indep_turn_over_0_6": sum(1 for p in nud if p["latA_06"].startswith("TURN")),
    "median_abs_netdyaw_deg_0_6": float(np.median([abs(p["dpsi_net06"]) for p in nud])) if nud else None,
    "median_raw_keyframe_lat_peak_m_0_6": float(np.median([abs(p["lat_raw_peak06"]) for p in nud])) if nud else None,
    "median_detrended_eps_m_0_6": float(np.median([p["eps06"] for p in nud])) if nud else None,
    "median_detrended_eps_m_0_2_operative_band": float(np.median([p["eps02"] for p in nud])) if nud else None,
    "median_detrended_eps_m_2_6_tactical_band": float(np.median([p["eps26"] for p in nud])) if nud else None,
    "frac_eps_2_6_lt_0.3": float(np.mean([p["eps26"] < 0.3 for p in nud])) if nud else None,
    "frac_eps_0_2_ge_0.3": float(np.mean([p["eps02"] >= 0.3 for p in nud])) if nud else None,
    "frac_eps_0_2_gt_eps_2_6": float(np.mean([p["eps02"] > p["eps26"] for p in nud])) if nud else None,
    "frac_detrended_eps_lt_0.3": float(np.mean([p["eps06"] < 0.3 for p in nud])) if nud else None,
}
lk = [p for p in per_rec if p["ok"] and p["b_lat"] == "LANE_KEEP"]
R["lane_keep_audit"] = {
    "n_builder_lane_keep": len(lk),
    "indep_nudge_or_shift_over_0_6": sum(1 for p in lk if p["latA_06"].startswith(("NUDGE", "SHIFT"))),
    "indep_turn_A_over_2_6": sum(1 for p in lk if p["latA_26"].startswith("TURN")),
    "indep_turn_B_over_2_6": sum(1 for p in lk if p["latB_26"].startswith("TURN")),
}

# ---- recomputation checks: lat_peak_m, within_m, SPEED_BAND, v_target ------------------
chk = defaultdict(list)
for r in recs:
    tr = by_sid.get(L.stable_episode_id(r["clip_id"]))
    if tr is None:
        continue
    at = r["a_tac"]
    ta = float(r["t0_s"])
    lp = (at.get("lat_args") or {}).get("lat_peak_m")
    if lp is not None:
        w = tr.t <= 20.0
        x, y, psi = tr.x[w], tr.y[w], tr.psi[w]
        c, s = np.cos(psi[0]), np.sin(psi[0])
        lat = -s * (x - x[0]) + c * (y - y[0])
        h1 = float(lat[int(np.argmax(np.abs(lat)))])        # H1: start frame, whole 20 s (documented)
        times = ta + np.arange(0.0, 6.0 + 1e-9, 0.1)
        xx, yy, pp, vv, ok = L.sample(tr, times)
        c2, s2 = np.cos(pp[0]), np.sin(pp[0])
        lat2 = -s2 * (xx - xx[0]) + c2 * (yy - yy[0])
        h2 = float(lat2[int(np.argmax(np.abs(lat2)))])       # H2: anchor (key) frame, [0,6] s
        times3 = ta + np.arange(2.0, 6.0 + 1e-9, 0.1)
        xx3, yy3, pp3, vv3, ok3 = L.sample(tr, times3)
        c3, s3 = np.cos(pp3[0]), np.sin(pp3[0])
        lat3 = -s3 * (xx3 - xx3[0]) + c3 * (yy3 - yy3[0])
        h3 = float(lat3[int(np.argmax(np.abs(lat3)))])       # H3: band [2,6] start frame
        chk["lat_peak"].append((lp, h1, h2, h3, at["lat"]))
    wm = (at.get("lat_args") or {}).get("within_m")
    if wm is not None:
        times = ta + np.arange(0.0, 6.0 + 1e-9, 0.1)
        xx, yy, pp, vv, ok = L.sample(tr, times)
        arc = float(np.sum(np.hypot(np.diff(xx), np.diff(yy))))
        chk["within_m"].append((wm, arc))
    sb = ((r.get("g_tac") or {}).get("goals") or {}).get("SPEED_BAND")
    if sb:
        times = ta + np.arange(2.0, 6.0 + 1e-9, 0.1)
        xx, yy, pp, vv, ok = L.sample(tr, times)
        chk["speed_band"].append((sb["v_lo_ms"], sb["v_hi_ms"], float(vv.min()), float(vv.max()), ok))
    lon_args = at.get("lon_args") or {}
    if "v_target_ms" in lon_args:
        times = ta + np.arange(0.0, 6.0 + 1e-9, 0.1)
        xx, yy, pp, vv, ok = L.sample(tr, times)
        chk["v_target"].append((lon_args["v_target_ms"], float(vv.min()), float(vv[-1]), float(vv[0]), at["lon"]))


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


C = {}
if chk["lat_peak"]:
    A = np.array([c[:4] for c in chk["lat_peak"]], dtype=float)
    d = {}
    for k, nm in [(1, "H1_startframe_whole20s"), (2, "H2_anchorframe_0to6s"), (3, "H3_bandstartframe_2to6s")]:
        e = np.abs(A[:, 0] - A[:, k])
        d[nm] = {"corr_signed": round(corr(A[:, 0], A[:, k]), 5), "median_abs_err_m": round(float(np.median(e)), 3),
                 "p90_abs_err_m": round(float(np.percentile(e, 90)), 3),
                 "frac_within_1m": round(float(np.mean(e <= 1.0)), 4),
                 "frac_within_0.5m_or_2pct": round(float(np.mean((e <= 0.5) | (e <= 0.02 * np.abs(A[:, 0])))), 4),
                 "frac_sign_equal": round(float(np.mean(np.sign(A[:, 0]) == np.sign(A[:, k]))), 4)}
    absl = np.abs(A[:, 0])
    d["n"] = int(len(A))
    d["recorded_abs_quantiles_m"] = {q: round(float(np.percentile(absl, q)), 2) for q in (10, 25, 50, 75, 90, 99, 100)}
    d["frac_recorded_gt_5m"] = round(float(np.mean(absl > 5.0)), 4)
    d["frac_recorded_lt_1m"] = round(float(np.mean(absl < 1.0)), 4)
    by = defaultdict(list)
    for c in chk["lat_peak"]:
        by[c[4]].append(abs(c[0]))
    d["median_abs_by_builder_lat"] = {k: round(float(np.median(v)), 2) for k, v in by.items()}
    d["n_by_builder_lat"] = {k: len(v) for k, v in by.items()}
    d["nudge_frac_below_1m"] = round(float(np.mean([v < 1.0 for k, vs in by.items() if k.startswith('NUDGE') for v in vs])), 4) if any(k.startswith('NUDGE') for k in by) else None
    d["lane_keep_frac_over_1m"] = round(float(np.mean([v >= 1.0 for v in by.get('LANE_KEEP', [])])), 4) if by.get('LANE_KEEP') else None
    C["lat_peak_m"] = d
if chk["within_m"]:
    A = np.array(chk["within_m"], dtype=float)
    e = np.abs(A[:, 0] - A[:, 1])
    C["within_m"] = {"n": int(len(A)), "corr": round(corr(A[:, 0], A[:, 1]), 5), "median_abs_err_m": round(float(np.median(e)), 3),
                     "p90_abs_err_m": round(float(np.percentile(e, 90)), 3), "frac_within_1m": round(float(np.mean(e <= 1.0)), 4)}
if chk["speed_band"]:
    A = np.array([c[:4] for c in chk["speed_band"]], dtype=float)
    ok = np.array([c[4] for c in chk["speed_band"]])
    e_hi = np.abs(A[:, 1] - A[:, 3])
    e_lo = np.abs(A[:, 0] - A[:, 2])
    C["speed_band_vs_cache"] = {"n": int(len(A)), "n_window_in_cache": int(ok.sum()),
                                "v_hi_median_abs_err": round(float(np.median(e_hi)), 3), "v_hi_p95_abs_err": round(float(np.percentile(e_hi, 95)), 3),
                                "v_hi_frac_within_0.3": round(float(np.mean(e_hi <= 0.3)), 4),
                                "v_lo_median_abs_err": round(float(np.median(e_lo)), 3), "corr_hi": round(corr(A[:, 1], A[:, 3]), 5)}
if chk["v_target"]:
    A = np.array([c[:4] for c in chk["v_target"]], dtype=float)
    lonc = [c[4] for c in chk["v_target"]]
    acc = np.array([x == "ACCELERATE" for x in lonc])
    C["v_target_ms"] = {"n": int(len(A)),
                        "corr_with_window_min": round(corr(A[:, 0], A[:, 1]), 5),
                        "median_abs_err_vs_min": round(float(np.median(np.abs(A[:, 0] - A[:, 1]))), 3),
                        "ACCELERATE_n": int(acc.sum()),
                        "ACCELERATE_median_abs_err_vs_min": round(float(np.median(np.abs(A[acc, 0] - A[acc, 1]))), 3) if acc.any() else None,
                        "ACCELERATE_median_abs_err_vs_end": round(float(np.median(np.abs(A[acc, 0] - A[acc, 2]))), 3) if acc.any() else None,
                        "ACCELERATE_median_end_minus_vtarget": round(float(np.median(A[acc, 2] - A[acc, 0])), 3) if acc.any() else None}
R["recompute_checks"] = C

# ---- trainer-as-applied: every supervised window (|NOW - t0| <= 2.0) -------------------
win = []
for r in recs:
    tr = by_sid.get(L.stable_episode_id(r["clip_id"]))
    if tr is None:
        continue
    at = r["a_tac"]
    if at.get("truncated"):
        continue
    t0 = float(r["t0_s"])
    lo_b, hi_b = r["bands"]["tactical_s"]
    half = (float(hi_b) - float(lo_b)) / 2.0
    T = len(tr.t)
    for rr in range(7, T - 21):           # V3Dataset index: t in range(T - w - 20), NOW row = t + w - 1, w = 8
        now = tr.t[rr]
        if abs(now - t0) > half:
            continue
        f = L.lat_features(tr, now, 2.0, 6.0)
        g = L.lon_features(tr, now, 2.0, 6.0)
        if not (f["ok"] and g["ok"]):
            continue
        win.append((now - t0, at["lat"], L.classify_lat(f, "A"), L.classify_lat(f, "B"), at["lon"], L.classify_lon(g)))
R["window_level"] = {"n_windows": len(win)}
if win:
    offs = np.array([w[0] for w in win])
    bins = [(-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.001)]
    rows = []
    for a, b in bins:
        sel = [w for w in win if a <= w[0] < b]
        n = len(sel)
        if n == 0:
            continue
        latA = sum(1 for w in sel if w[2] in L.LAT_OK[w[1]])
        latB = sum(1 for w in sel if w[3] in L.LAT_OK[w[1]])
        spec = [w for w in sel if w[4] != "ADAPT_SPEED_FOR_CURVE"]
        lon = sum(1 for w in spec if w[5] in L.LON_OK[w[4]])
        rows.append({"now_minus_t0_s": [a, b], "n": n, "lat_strict_acc_A": round(latA / n, 4), "lat_strict_acc_B": round(latB / n, 4),
                     "lon_acc_speed_defined": round(lon / max(1, len(spec)), 4), "n_lon": len(spec)})
    R["window_level"]["by_offset"] = rows
    M = L.confusion(LAT_ROWS, LAT_COLS, [(w[1], w[2]) for w in win])
    R["window_level"]["lat_matrix_A"] = M.tolist()
    n = len(win)
    R["window_level"]["lat_strict_acc_A"] = round(sum(1 for w in win if w[2] in L.LAT_OK[w[1]]) / n, 4)
    R["window_level"]["lat_strict_acc_B"] = round(sum(1 for w in win if w[3] in L.LAT_OK[w[1]]) / n, 4)
    spec = [w for w in win if w[4] != "ADAPT_SPEED_FOR_CURVE"]
    R["window_level"]["lon_acc_speed_defined"] = round(sum(1 for w in spec if w[5] in L.LON_OK[w[4]]) / len(spec), 4)
    R["window_level"]["n_lon"] = len(spec)
    txt.append(L.fmt_matrix(M, LAT_ROWS, LAT_COLS, f"[{tag}] LATERAL as-trained windows (|NOW-t0|<=2 s, indep A on each window's own [NOW+2,NOW+6]) n={n}"))

# ---- ADAPT_SPEED_FOR_CURVE: is the ego actually in a curve? ---------------------------
ad = [p for p in per_rec if p["ok"] and p["b_lon"] == "ADAPT_SPEED_FOR_CURVE"]
R["adapt_audit"] = {
    "n": len(ad),
    "frac_peak_dyaw_ge_20deg_over_0_6": float(np.mean([abs(p["dpsi_peak06"]) >= L.CURVE_DEG for p in ad])) if ad else None,
    "frac_peak_dyaw_ge_30deg_over_0_6": float(np.mean([abs(p["dpsi_peak06"]) >= 30 for p in ad])) if ad else None,
    "median_abs_peak_dyaw_deg_0_6": float(np.median([abs(p["dpsi_peak06"]) for p in ad])) if ad else None,
    "frac_builder_lat_is_turn": float(np.mean([p["b_lat"].startswith("TURN") for p in ad])) if ad else None,
    "frac_lat_lane_keep": float(np.mean([p["b_lat"] == "LANE_KEEP" for p in ad])) if ad else None,
    "median_abs_dv_0_6": float(np.median([abs(p["dv06"]) for p in ad])) if ad else None,
}

# ---- goals vs geometry ----------------------------------------------------------------
gg = {}
for tok, test in [("TURN_L", lambda p: p["latA_26"] == "TURN_L"), ("TURN_R", lambda p: p["latA_26"] == "TURN_R")]:
    for variant, key in (("A", "latA_26"), ("B", "latB_26")):
        pos = [p for p in per_rec if p["ok"] and (tok in p["g_goals"] or ("YIELD_FOR_" + tok) in p["g_goals"])]
        neg = [p for p in per_rec if p["ok"] and not (tok in p["g_goals"] or ("YIELD_FOR_" + tok) in p["g_goals"])]
        hit = lambda p: p[key] == tok
        gg[f"{tok}_variant{variant}"] = {
            "goal_positive_n": len(pos), "goal_positive_and_indep_turn": sum(1 for p in pos if hit(p)),
            "goal_negative_n": len(neg), "goal_negative_but_indep_turn": sum(1 for p in neg if hit(p))}
R["goal_vs_geometry"] = gg
R["goal_census"] = {t: sum(1 for p in per_rec if t in p["g_goals"]) for t in
                    sorted({g for p in per_rec for g in p["g_goals"]})}

R["per_record_n"] = len(per_rec)
json.dump(R, open(out_p, "w"), indent=1)
open(out_p.replace(".json", ".txt"), "w", encoding="utf-8").write("\n\n".join(txt) + "\n")
json.dump(per_rec, open(out_p.replace(".json", "_perrecord.json"), "w"))
print("\n\n".join(txt))
print(json.dumps({k: R[k] for k in ["n_records", "n_tracks", "n_records_without_cache_track", "n_scored", "n_truncated_by_builder",
                                    "nudge_audit", "lane_keep_audit", "adapt_audit", "recompute_checks", "goal_vs_geometry"]}, indent=1))
print(json.dumps(R["window_level"], indent=1)[:3000])
for k in ["latA_band2_6", "latB_band2_6", "latA_span0_6"]:
    m = R["matrices"][k]
    print(k, {x: m[x] for x in ["n", "strict_acc", "strict_ci95", "coarse_acc", "side_acc", "side_n"]}, m["top_off_diagonals"][:4])
for k in ["lon_band2_6", "lon_span0_6"]:
    m = R["matrices"][k]
    print(k, {x: m[x] for x in ["n", "n_speed_defined", "acc", "ci95", "acc_ex_follow"]}, m["top_off_diagonals"][:4])
