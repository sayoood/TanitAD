"""D2 SEMANTICS support: goal-token provenance census, nav_command vs nav_30s consistency,
nav token temporal precision per window, turn_suppression, bands, strata.
usage: d2_census_nav.py <labels.jsonl.gz> <manifest.pt> <out.json> <tag>
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

import d2_lib as L

lab_p, man_p, out_p, tag = sys.argv[1:5]
recs = L.read_jsonl_gz(lab_p)
side = L.load_clock_sidecar("D:/refcv6_eval_kit/data/refcv6_clip_clock_sidecar.jsonl")
trs = {t.sid: t for t in L.tracks_from_manifest(man_p, side)}
R = {"tag": tag, "n_records": len(recs)}

# ---- bands / t0 uniformity ---------------------------------------------------------------
R["t0_s_values"] = dict(Counter(r.get("t0_s") for r in recs))
R["bands_values"] = dict(Counter(json.dumps({k: r["bands"].get(k) for k in ("operative_s", "tactical_s", "strategic_s")}) for r in recs))
R["schema_vocab_release"] = dict(Counter(str((r.get("schema_version"), r.get("vocab"), r.get("release"))) for r in recs))
R["truncated_a_tac"] = sum(1 for r in recs if (r.get("a_tac") or {}).get("truncated"))

# ---- goal token census -------------------------------------------------------------------
tok = defaultdict(lambda: {"n": 0, "prov": Counter(), "time_basis": Counter(), "disputed": 0, "grounded": 0,
                           "corroboration": Counter(), "has_t_nominal": 0})
for r in recs:
    for t, m in ((r.get("g_tac") or {}).get("goals") or {}).items():
        d = tok[t]
        d["n"] += 1
        if isinstance(m, dict):
            d["prov"][m.get("provenance", "<none: geometry by construction>")] += 1
            if m.get("time_basis"):
                d["time_basis"][m["time_basis"]] += 1
            d["disputed"] += bool(m.get("disputed"))
            d["grounded"] += bool(m.get("grounded"))
            d["corroboration"][m.get("corroboration", "-")] += 1
            d["has_t_nominal"] += m.get("t_nominal_s") is not None
R["goal_census"] = {t: {"n": d["n"], "prov": dict(d["prov"]), "time_basis": dict(d["time_basis"]), "disputed": d["disputed"],
                        "grounded": d["grounded"], "corroboration": dict(d["corroboration"]), "has_t_nominal": d["has_t_nominal"]}
                    for t, d in sorted(tok.items())}
R["goal_set_violations"] = sum(1 for r in recs if (r.get("g_tac") or {}).get("violations"))

# ---- turn suppression --------------------------------------------------------------------
sup = [r for r in recs if r.get("turn_suppression")]
R["turn_suppression"] = {"n_applied": len(sup), "by_side": dict(Counter(r["turn_suppression"].get("side") for r in sup)),
                         "a_tac_lat_among_suppressed": dict(Counter(r["a_tac"]["lat"] for r in sup)),
                         "nav_token_among_suppressed": dict(Counter(r["nav_command"]["token"] for r in sup)),
                         "keys": sorted({k for r in sup for k in r["turn_suppression"]})}

# ---- nav_command vs nav_30s ---------------------------------------------------------------
nc = Counter()
first_turn_mismatch = 0
for r in recs:
    tok_ = r["nav_command"]["token"]
    ents = (r.get("nav_30s") or {}).get("entries") or []
    turns = [e for e in ents if e["token"] != "NAV_FOLLOW_ROAD"]
    nc[(tok_, "nav30_has_turn" if turns else "nav30_no_turn")] += 1
    if tok_ != "NAV_FOLLOW_ROAD":
        if not turns or turns[0]["token"] != tok_ or abs(turns[0]["t_start_s"] - r["nav_command"]["args"].get("time_s", -9)) > 0.05:
            first_turn_mismatch += 1
R["nav_command_vs_nav30s"] = {f"{a}|{b}": c for (a, b), c in nc.items()}
R["nav_command_turn_not_equal_nav30_first_turn"] = first_turn_mismatch
tt = np.array([r["nav_command"]["args"]["time_s"] for r in recs if r["nav_command"]["token"] != "NAV_FOLLOW_ROAD"], dtype=float)
R["nav_turn_start_after_anchor_s"] = {"n": int(len(tt)), "median": round(float(np.median(tt)), 2),
                                      **{f"p{p}": round(float(np.percentile(tt, p)), 2) for p in (10, 25, 75, 90)},
                                      "frac_gt_6s": round(float((tt > 6).mean()), 4), "frac_gt_10s": round(float((tt > 10).mean()), 4)}
# turns in nav_30s that are NOT is_turn... and multi-turn clips
R["nav30_n_entries_dist"] = dict(Counter((r.get("nav_30s") or {}).get("n_entries") for r in recs))

# ---- nav token per WINDOW: does a turn of that side actually begin within the window's next 6 s?
# independent geometry (variant B: >= 30 deg inside any 3 s) on [NOW, NOW+6]; NOW over ALL windows
cnt = Counter()
cnt_ib = Counter()
by_off = defaultdict(Counter)
for r in recs:
    tr = trs.get(L.stable_episode_id(r["clip_id"]))
    if tr is None:
        continue
    navtok = r["nav_command"]["token"]
    nav = {"NAV_TURN_L": "L", "NAV_TURN_R": "R"}.get(navtok, "F")
    t0 = float(r["t0_s"])
    T = len(tr.t)
    for rr in range(7, T - 21):
        now = tr.t[rr]
        f = L.lat_features(tr, now, 0.0, 6.0)
        if not f["ok"]:
            continue
        a = f["dpsi_fast3"]
        geo = "L" if a >= 30 else "R" if a <= -30 else "F"
        key = (nav, geo)
        cnt[key] += 1
        ib = abs(now - t0) <= 2.0
        if ib:
            cnt_ib[key] += 1
        ob = int(np.floor((now - t0) / 2.0) * 2)
        by_off[ob][("hit" if (nav != "F" and nav == geo) else "miss") if nav != "F" else "follow_" + ("geo_turn" if geo != "F" else "geo_straight")] += 1


def mat(c):
    out = {}
    n = sum(c.values())
    for nav in "LRF":
        row = {g: c.get((nav, g), 0) for g in "LRF"}
        out[nav] = row
    return {"n_windows": n, "rows_nav_cols_geo": out}


R["nav_per_window_vs_independent_turn_next6s"] = {"all_windows": mat(cnt), "supervised_range": mat(cnt_ib)}
nL = sum(v for (a, g), v in cnt.items() if a in "LR")
hit = sum(v for (a, g), v in cnt.items() if a in "LR" and a == g)
R["nav_per_window_vs_independent_turn_next6s"]["turn_nav_windows"] = nL
R["nav_per_window_vs_independent_turn_next6s"]["turn_nav_windows_with_same_side_turn_in_next_6s"] = hit
R["nav_per_window_vs_independent_turn_next6s"]["turn_nav_precision"] = round(hit / max(1, nL), 4)
nF = sum(v for (a, g), v in cnt.items() if a == "F")
Fmiss = sum(v for (a, g), v in cnt.items() if a == "F" and g != "F")
R["nav_per_window_vs_independent_turn_next6s"]["follow_nav_windows"] = nF
R["nav_per_window_vs_independent_turn_next6s"]["follow_nav_windows_where_a_turn_starts_in_next_6s"] = Fmiss
turn_geo = sum(v for (a, g), v in cnt.items() if g != "F")
R["nav_per_window_vs_independent_turn_next6s"]["windows_with_independent_turn_next6s"] = turn_geo
R["nav_per_window_vs_independent_turn_next6s"]["recall_of_those_windows_by_nav_turn_same_side"] = round(hit / max(1, turn_geo), 4)
R["nav_per_window_by_offset_2s_bins"] = {str(k): dict(v) for k, v in sorted(by_off.items())}
json.dump(R, open(out_p, "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["t0_s_values", "bands_values", "schema_vocab_release", "truncated_a_tac", "goal_set_violations", "turn_suppression",
                                    "nav_command_vs_nav30s", "nav_command_turn_not_equal_nav30_first_turn", "nav_turn_start_after_anchor_s",
                                    "nav_per_window_vs_independent_turn_next6s"]}, indent=1, default=str)[:5000])
