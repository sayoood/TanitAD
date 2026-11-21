"""D2 deliverable 3b: Alpamayo (VLM) vs our geometric label -- which side is right?

Independent geometry (d2_lib, literals) is evaluated on BOTH time bases:
  OUR   window  [t0, t0+6]  = raw [8.0, 14.0]
  ALPA  window  [5.1, 11.1]            (alpamayo_records.ALPAMAYO_T0_S = 5.1 + 6 s horizon)
usage: d2_alpamayo.py <out.json>   (reads train+eval labels and manifests from the paths below)
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

import d2_lib as L

OUT = sys.argv[1]
SPECS = [
    ("train", "D:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz",
     "D:/Projects/TanitAD-artifacts/refcv8_audit/D2/train_v2manifest.pt"),
    ("eval", "D:/refcv6_eval_kit/data/v8labels/labels/s2_labels_v8_eval.jsonl.gz",
     "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt"),
]
SIDECAR = L.load_clock_sidecar("D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
ALPA_T0 = 5.1
NEUTRAL = 1.0
LON_DV = {"Strong Deceleration": -5.88, "Gentle Deceleration": -1.73, "Maintain Speed": 0.03,
          "Gentle Acceleration": 1.93, "Strong Acceleration": 6.0}


def gside(cls):
    if cls.endswith("_L"):
        return "left"
    if cls.endswith("_R"):
        return "right"
    return "straight"


def sgn(x):
    return 0 if abs(x) <= NEUTRAL else (1 if x > 0 else -1)


rows = []
for split, lp, mp in SPECS:
    recs = L.read_jsonl_gz(lp)
    trs = {t.sid: t for t in L.tracks_from_manifest(mp, SIDECAR)}
    for r in recs:
        tr = trs.get(L.stable_episode_id(r["clip_id"]))
        if tr is None:
            continue
        al = r.get("alpamayo") or {}
        lat = al.get("lateral") or {}
        lon = al.get("longitudinal") or {}
        if lat.get("alpamayo_side") is None:
            continue
        meta = (r.get("cot_source") or {}).get("meta_action") or {}
        # independent geometry, both windows (class over a full 6 s span)
        fO = L.lat_features(tr, 8.0, 0.0, 6.0)
        fA = L.lat_features(tr, ALPA_T0, 0.0, 6.0)
        cO, cA = L.classify_lat(fO, "A"), L.classify_lat(fA, "A")
        gO = L.lon_features(tr, 8.0, 0.0, 6.0)
        gA = L.lon_features(tr, ALPA_T0, 0.0, 6.0)
        # agreement of the alpamayo side with geometry as a function of window start (timing test)
        offs = {}
        for s in range(-3, 7):                       # window start = 5.1 + s
            st = ALPA_T0 + s
            f = L.lat_features(tr, st, 0.0, 6.0)
            g = L.lon_features(tr, st, 0.0, 6.0)
            offs[s] = (gside(L.classify_lat(f, "A")), g["dv"], f["ok"] and g["ok"], g["vstart"])
        rows.append({
            "split": split, "sha12": L.sha12(r["clip_id"]), "our_lat": r["a_tac"]["lat"], "our_lon": r["a_tac"]["lon"],
            "alp_side": lat["alpamayo_side"], "lat_agree": lat.get("agree"), "concordance": lat.get("concordance"),
            "geomO_cls": cO, "geomA_cls": cA, "geomO_side": gside(cO), "geomA_side": gside(cA),
            "road": (r.get("strata") or {}).get("road_class"), "dn": (r.get("strata") or {}).get("daynight_clock"),
            "phrase": lon.get("phrase"), "lon_agree": lon.get("agree"), "lon_scored_on": lon.get("scored_on"),
            "dv_stored": lon.get("dv_measured_ms"), "dv_alpaW": gA["dv"], "dv_ourW": gO["dv"], "v0_alpa": gA["vstart"],
            "ok": bool(fO["ok"] and fA["ok"] and gO["ok"] and gA["ok"]),
            "offs": {str(k): [v[0], v[1], v[2], v[3]] for k, v in offs.items()},
            "meta_lateral": meta.get("lateral"), "meta_lane": meta.get("lane"),
        })

R = {"n_rows_with_alpamayo_side": len(rows), "n_ok_window": sum(r["ok"] for r in rows)}
ok = [r for r in rows if r["ok"]]


def rate(sel, f):
    n = len(sel)
    k = sum(1 for r in sel if f(r))
    lo, hi = L.wilson(k, n)
    return {"n": n, "k": k, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


# ---------------- lateral ----------------
scored = [r for r in ok if r["lat_agree"] is not None]
R["lateral"] = {"n_scored": len(scored), "flag_agree_rate": rate(scored, lambda r: r["lat_agree"]),
                "flag_false": sum(1 for r in scored if not r["lat_agree"])}
side_of_our = lambda r: ("left" if r["our_lat"].endswith("_L") else "right" if r["our_lat"].endswith("_R") else "straight")
# the flag re-derived from the label alone (sanity: must equal stored flag exactly)
R["lateral"]["flag_reproduced_from_labels"] = rate(scored, lambda r: (r["alp_side"] == side_of_our(r)) == bool(r["lat_agree"]))
R["lateral"]["alpamayo_side_distribution"] = dict(Counter(r["alp_side"] for r in scored))
R["lateral"]["our_side_distribution"] = dict(Counter(side_of_our(r) for r in scored))
# who is right, each on its OWN window against the independent geometry
R["lateral"]["alpamayo_side_matches_geometry_in_ALPA_window"] = rate(scored, lambda r: r["alp_side"] == r["geomA_side"])
R["lateral"]["alpamayo_side_matches_geometry_in_OUR_window"] = rate(scored, lambda r: r["alp_side"] == r["geomO_side"])
R["lateral"]["our_label_matches_geometry_in_OUR_window"] = rate(scored, lambda r: side_of_our(r) == r["geomO_side"])
R["lateral"]["our_label_matches_geometry_in_ALPA_window"] = rate(scored, lambda r: side_of_our(r) == r["geomA_side"])
# 'straight' share, to expose the base rate
R["lateral"]["geometry_straight_share_OUR_window"] = rate(scored, lambda r: r["geomO_side"] == "straight")
R["lateral"]["geometry_straight_share_ALPA_window"] = rate(scored, lambda r: r["geomA_side"] == "straight")
conf = [r for r in scored if not r["lat_agree"]]
R["lateral"]["conflicts"] = {
    "n": len(conf),
    "alpamayo_right_in_its_window": rate(conf, lambda r: r["alp_side"] == r["geomA_side"]),
    "ours_right_in_our_window": rate(conf, lambda r: side_of_our(r) == r["geomO_side"]),
    "alpamayo_right_in_OUR_window": rate(conf, lambda r: r["alp_side"] == r["geomO_side"]),
    "ours_right_in_ALPA_window": rate(conf, lambda r: side_of_our(r) == r["geomA_side"]),
}
quad = Counter()
for r in conf:
    a = r["alp_side"] == r["geomA_side"]
    o = side_of_our(r) == r["geomO_side"]
    quad[("VLM_right" if a else "VLM_wrong") + "/" + ("ours_right" if o else "ours_wrong")] += 1
R["lateral"]["conflict_quadrants_each_on_own_window"] = dict(quad)
quad2 = Counter()
for r in conf:
    a = r["alp_side"] == r["geomO_side"]
    o = side_of_our(r) == r["geomO_side"]
    quad2[("VLM_right" if a else "VLM_wrong") + "/" + ("ours_right" if o else "ours_wrong")] += 1
R["lateral"]["conflict_quadrants_both_on_OUR_window"] = dict(quad2)
R["lateral"]["conflicts_by_pair(alp_side,our_side)"] = {f"{a}|{b}": c for (a, b), c in Counter((r["alp_side"], side_of_our(r)) for r in conf).items()}


def strat(key, sel=scored):
    d = defaultdict(list)
    for r in sel:
        d[r[key]].append(r)
    out = {}
    for k, v in sorted(d.items(), key=lambda kv: str(kv[0])):
        c = [x for x in v if not x["lat_agree"]]
        out[str(k)] = {"n": len(v), "conflict_rate": round(len(c) / len(v), 4),
                       "alp_right_in_own_window_among_conflicts": round(sum(x["alp_side"] == x["geomA_side"] for x in c) / len(c), 4) if c else None,
                       "ours_right_in_own_window_among_conflicts": round(sum(side_of_our(x) == x["geomO_side"] for x in c) / len(c), 4) if c else None,
                       "n_conflicts": len(c)}
    return out


R["lateral"]["by_our_lat_class"] = strat("our_lat")
R["lateral"]["by_road_class"] = strat("road")
R["lateral"]["by_daynight"] = strat("dn")
R["lateral"]["by_alpamayo_side"] = strat("alp_side")
R["lateral"]["by_concordance"] = strat("concordance")
# timing test: P(alpamayo side == geometry side) as a function of window START (5.1 + s)
tim = {}
for s in range(-3, 7):
    sel = [r for r in scored if r["offs"][str(s)][2]]
    tim[f"start={ALPA_T0 + s:.1f}"] = rate(sel, lambda r: r["alp_side"] == r["offs"][str(s)][0])
R["lateral"]["timing_test_alpamayo_side_vs_geometry_by_window_start"] = tim
# same for OUR side, for symmetry
timo = {}
for s in range(-3, 7):
    sel = [r for r in scored if r["offs"][str(s)][2]]
    timo[f"start={ALPA_T0 + s:.1f}"] = rate(sel, lambda r: side_of_our(r) == r["offs"][str(s)][0])
R["lateral"]["timing_test_OUR_side_vs_geometry_by_window_start"] = timo

# ---------------- longitudinal ----------------
lg = [r for r in ok if r["phrase"] is not None and r["dv_stored"] is not None]
R["longitudinal"] = {"n": len(lg), "scored_on_counts": dict(Counter(r["lon_scored_on"] for r in lg))}
e = np.array([abs(r["dv_stored"] - r["dv_alpaW"]) for r in lg if r["lon_scored_on"] and r["lon_scored_on"].startswith("alpamayo 5.1")])
R["longitudinal"]["stored_dv_vs_independent_dv_in_ALPA_window"] = {
    "n": int(len(e)), "median_abs_err": round(float(np.median(e)), 4), "p95_abs_err": round(float(np.percentile(e, 95)), 4),
    "frac_within_0.3": round(float(np.mean(e <= 0.3)), 4)}
sc = [r for r in lg if r["lon_agree"] is not None]
R["longitudinal"]["flag_agree_rate"] = rate(sc, lambda r: r["lon_agree"])
dvexp = lambda r: LON_DV.get(r["phrase"])
dv_phr = [r for r in sc if dvexp(r) is not None]
R["longitudinal"]["phrase_dv_sign_agrees_with_independent_dv_ALPA_window"] = rate(dv_phr, lambda r: sgn(dvexp(r)) == sgn(r["dv_alpaW"]))
R["longitudinal"]["phrase_dv_sign_agrees_with_independent_dv_OUR_window"] = rate(dv_phr, lambda r: sgn(dvexp(r)) == sgn(r["dv_ourW"]))
by_ph = defaultdict(list)
for r in sc:
    by_ph[r["phrase"]].append(r)
R["longitudinal"]["by_phrase"] = {k: {"n": len(v), "agree": round(sum(bool(x["lon_agree"]) for x in v) / len(v), 4),
                                      "mean_dv_alpaW": round(float(np.mean([x["dv_alpaW"] for x in v])), 3)} for k, v in sorted(by_ph.items())}


def lstrat(key):
    d = defaultdict(list)
    for r in sc:
        d[r[key]].append(r)
    return {str(k): {"n": len(v), "conflict_rate": round(sum(not x["lon_agree"] for x in v) / len(v), 4)} for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}


R["longitudinal"]["by_our_lon_class"] = lstrat("our_lon")
R["longitudinal"]["by_road_class"] = lstrat("road")
R["longitudinal"]["by_daynight"] = lstrat("dn")
tl = {}
for s in range(-3, 7):
    sel = [r for r in dv_phr if r["offs"][str(s)][2]]
    tl[f"start={ALPA_T0 + s:.1f}"] = rate(sel, lambda r: sgn(dvexp(r)) == sgn(r["offs"][str(s)][1]))
R["longitudinal"]["timing_test_phrase_vs_dv_by_window_start"] = tl

# a short sample of lateral conflicts (sha12 only)
rng = np.random.default_rng(0)
samp = rng.choice(len(conf), size=min(24, len(conf)), replace=False)
R["lateral"]["conflict_sample_24"] = [{k: conf[i][k] for k in ["split", "sha12", "our_lat", "alp_side", "geomO_cls", "geomA_cls", "road", "dn", "concordance"]}
                                      for i in sorted(samp)]
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps({k: R["lateral"][k] for k in R["lateral"] if k not in ("conflict_sample_24",) and not k.startswith("timing") and not k.startswith("by_")}, indent=1)[:6000])
print(json.dumps(R["lateral"]["timing_test_alpamayo_side_vs_geometry_by_window_start"], indent=0)[:2500])
print(json.dumps(R["lateral"]["timing_test_OUR_side_vs_geometry_by_window_start"], indent=0)[:2500])
print(json.dumps({k: v for k, v in R["longitudinal"].items() if not k.startswith("timing")}, indent=1)[:3500])
print(json.dumps(R["longitudinal"]["timing_test_phrase_vs_dv_by_window_start"], indent=0)[:2500])
