#!/usr/bin/env python3
"""ESTIMATED navhard_two_stage cost, scaled linearly from MEASURED warmup per-scene costs.

Inputs (all banked in raw/): cache hooks (per-scene metric-cache seconds, split by stage),
scoring hooks + manifests (per-token wall between successive pdm_score calls, incl. scene load),
peak RSS per run, and the yaml-derived counts (raw/split_counts.json: 450 S1 + 5,462 S2, MEASURED).
Excluded: downloads, file extraction, copies, and the one-off import + map load (reported apart).
"""
import json
import statistics as st
import sys
from pathlib import Path

raw = Path(sys.argv[1])
counts = json.load(open(raw / "split_counts.json"))
nh = counts["navhard_two_stage"]
N1, N2 = nh["stage1_tokens_from_mapping"], nh["stage2_tokens_from_mapping"]
cache = json.load(open(raw / "cache" / "cache_hooks.json"))["cache_scenes"]
cman = json.load(open(raw / "cache" / "cache_manifest.json"))
c1 = [h["s"] for h in cache if not h["synthetic_by_len17"]]
c2 = [h["s"] for h in cache if h["synthetic_by_len17"]]
# first scene carries the one-off map load; report it apart
first = cache[0]["s"]
c1_steady = c1[1:] if not cache[0]["synthetic_by_len17"] else c1


def gaps(arm):
    p = raw / arm / f"{arm}_hooks.json"
    if not p.exists():
        return None
    calls = [c for c in json.load(open(p))["pdm_score_calls"] if "t_wall" in c]
    t = [c["t_wall"] for c in calls]
    g = [b - a for a, b in zip(t, t[1:])]
    s1 = [c for c in calls if "ORIGINAL" in c["scene_type"]]
    s2 = [c for c in calls if "SYNTHETIC" in c["scene_type"]]
    return {"n_calls": len(calls), "median_gap_s": st.median(g) if g else None, "mean_gap_s": st.mean(g) if g else None,
            "pdm_score_s_mean_s1": st.mean([c["pdm_score_s"] for c in s1]) if s1 else None,
            "pdm_score_s_mean_s2": st.mean([c["pdm_score_s"] for c in s2]) if s2 else None,
            "wall_s": json.load(open(raw / arm / f"{arm}_manifest.json"))["resources"]["wall_s"],
            "peak_rss_mb": json.load(open(raw / arm / f"{arm}_manifest.json"))["resources"]["peak_rss_mb"]}


a1 = gaps("A1")
res = {"_class": "ESTIMATED (linear scaling of MEASURED warmup per-scene costs; 1 process, sequential worker)",
       "navhard_counts_MEASURED_from_yaml": {"stage_one": N1, "stage_two": N2, "logs": nh["sf_log_names"]},
       "warmup_measured": {
           "cache_stage_one_s": {"n": len(c1), "mean": st.mean(c1_steady) if c1_steady else None, "median": st.median(c1) if c1 else None},
           "cache_stage_two_s": {"n": len(c2), "mean": st.mean(c2) if c2 else None, "median": st.median(c2) if c2 else None},
           "cache_first_scene_s_incl_map_load": first, "cache_wall_s": cman["resources"]["wall_s"],
           "cache_peak_rss_mb": cman["resources"]["peak_rss_mb"], "scoring_A1": a1}}
e_cache = N1 * res["warmup_measured"]["cache_stage_one_s"]["mean"] + N2 * res["warmup_measured"]["cache_stage_two_s"]["mean"]
res["estimate"] = {"cache_cpu_h": round(e_cache / 3600, 2)}
if a1 and a1["median_gap_s"]:
    e_score = (N1 + N2) * a1["mean_gap_s"]
    res["estimate"]["score_per_agent_h"] = round(e_score / 3600, 2)
    res["estimate"]["cache_plus_cv_plus_human_h_upper"] = round((e_cache + 2 * e_score) / 3600, 2)
    res["estimate"]["_human_note"] = ("upper bound: the human agent fails fast on stage 2 (no future frames), "
                                      "so its real cost is ~ N1/(N1+N2) of this")
res["estimate"]["one_off_not_scaled"] = "import (~20-30 s on C:) + map load per process; data copy to C: mirror"
json.dump(res, open(raw / "navhard_price.json", "w"), indent=1)
print(json.dumps(res["estimate"], indent=1))
