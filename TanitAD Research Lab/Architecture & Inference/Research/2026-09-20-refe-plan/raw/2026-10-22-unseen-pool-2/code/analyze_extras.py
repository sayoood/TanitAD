#!/usr/bin/env python3
"""UNSEEN-POOL-2 main analysis (CPU, local inputs only): intersects the OpenScene trainval / test / mini log lists with navtrain / navtest and writes
raw/pool_summary.json. Run with any python (3.14 ok): needs only the json outputs of the profile scripts."""
import sys, json, os, collections, statistics, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
B = "D:/TanitAstraResearch/data/e024_corpus_20261010/"
nt = parse_yaml_lists(B + "source_contracts/navtrain.yaml"); ns = parse_yaml_lists(B + "source_contracts/navtest.yaml")
NT, NS = set(nt["log_names"]), set(ns["log_names"])
NTS = {session_of(l) for l in NT}; NSS = {session_of(l) for l in NS}
NTVD = {vehicle_day_of(s) for s in NTS}; NSVD = {vehicle_day_of(s) for s in NSS}
vcity = json.load(open("raw/src/vehicle_city.json"))["vehicle_city"]
mem = json.load(open("raw/members_trainval_tgz.json"))["members"]
TV = sorted(m["name"].rsplit("/", 1)[-1][:-4] for m in mem if m["isfile"] and m["name"].endswith(".pkl"))
TEST = json.load(open("raw/test_release_logs.json"))["test_logs"]
MINI = json.load(open("raw/mini_logs.json"))["mini_logs"]
PT = json.load(open("raw/profile_logs.json"))["logs"]
PTE = json.load(open("raw/profile_test_logs.json"))["logs"]
DEN = json.load(open("raw/navtrain_density.json"))
ARCH = json.load(open("raw/openscene_trainval_extras_archives.json"))
ratio_route = sum(v["tokens"] for v in DEN.values()) / sum(v["anchors_route"] for v in DEN.values())
# per-session ratio spread (for an honest range)
bs = collections.defaultdict(lambda: [0, 0])
for l, v in DEN.items():
    bs[session_of(l)][0] += v["tokens"]; bs[session_of(l)][1] += v["anchors_route"]
sess_ratios = sorted(t / a for t, a in bs.values() if a >= 200)
q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
rng = {"p10": q(sess_ratios, .10), "median": q(sess_ratios, .5), "p90": q(sess_ratios, .90), "n_sessions": len(sess_ratios), "pooled": ratio_route}

def describe(logs, prof, label):
    logs = sorted(logs); ok = [l for l in logs if l in prof and "error" not in prof[l]]
    routeful = [l for l in ok if prof[l]["n_route_frames"] > 0]
    c = counts(logs)
    sess = sorted({session_of(l) for l in logs}); cities = sorted({vcity.get(vehicle_of(s), "UNKNOWN(" + vehicle_of(s) + ")") for s in sess})
    return {"label": label, **c, "cities": cities, "profiled": len(ok), "logs_with_any_route_frame": len(routeful), "logs_with_zero_route_frames": len(ok) - len(routeful),
            "frames": sum(prof[l]["n_frames"] for l in ok), "hours_2Hz": round(sum(prof[l]["duration_s"] for l in ok) / 3600, 2),
            "anchors_geom(4h+10f)": sum(prof[l]["anchors_geom"] for l in ok), "anchors_route": sum(prof[l]["anchors_route"] for l in routeful),
            "anchors_route_regular": sum(prof[l]["anchors_regular"] for l in routeful),
            "est_navsim_tokens_pooled_ratio": round(ratio_route * sum(prof[l]["anchors_route"] for l in routeful)),
            "est_navsim_tokens_range_p10_p90": [round(rng["p10"] * sum(prof[l]["anchors_route"] for l in routeful)), round(rng["p90"] * sum(prof[l]["anchors_route"] for l in routeful))]}

S = {}
tv = set(TV)
extra118 = sorted(tv - NT); seen60 = [l for l in extra118 if session_of(l) in NTS]; u58 = [l for l in extra118 if session_of(l) not in NTS]
S["navtrain"] = counts(NT); S["navtest"] = counts(NS)
S["trainval_release"] = {**counts(tv), "navtrain_in_trainval": len(NT & tv), "navtest_in_trainval": len(NS & tv)}
S["trainval_not_in_navtrain"] = describe(extra118, PT, "trainval \ navtrain logs")
S["trainval_session_seen_extras"] = describe(seen60, PT, "extras in navtrain sessions")
S["trainval_session_unseen_extras"] = describe(u58, PT, "extras in 9 sessions absent from navtrain")
for nm, lg in (("trainval_not_in_navtrain", extra118), ("trainval_session_seen_extras", seen60), ("trainval_session_unseen_extras", u58)):
    sv = {session_of(l) for l in lg}
    S[nm]["sessions_in_navtrain"] = len(sv & NTS); S[nm]["vehicle_days_in_navtrain"] = len({vehicle_day_of(s) for s in sv} & NTVD)
    S[nm]["sessions_in_navtest"] = len(sv & NSS); S[nm]["vehicle_days_in_navtest"] = len({vehicle_day_of(s) for s in sv} & NSVD)
S["test_release"] = {**counts(TEST), "navtest_in_test": len(NS & set(TEST))}
tx = sorted(set(TEST) - NS)
S["test_not_in_navtest"] = describe(tx, PTE, "test \ navtest logs")
xs = {session_of(l) for l in tx}
S["test_not_in_navtest"].update({"sessions_in_navtest": len(xs & NSS), "vehicle_days_in_navtest": len({vehicle_day_of(s) for s in xs} & NSVD), "sessions_in_navtrain": len(xs & NTS), "vehicle_days_in_navtrain": len({vehicle_day_of(s) for s in xs} & NTVD)})
S["mini_release"] = {**counts(MINI), "in_navtrain": len(set(MINI) & NT), "in_navtest": len(set(MINI) & NS), "in_neither": len(set(MINI) - NT - NS), "neither_logs_in_trainval_extras118": len((set(MINI) - NT - NS) & set(extra118)), "neither_logs": sorted(set(MINI) - NT - NS)}
S["token_density_ratio"] = rng
S["unseen58_sessions"] = {}
for s in sorted({session_of(l) for l in u58}):
    ls = [l for l in u58 if session_of(l) == s]
    vd = vehicle_day_of(s)
    S["unseen58_sessions"][s] = {"vehicle": vehicle_of(s), "city": vcity.get(vehicle_of(s)), "logs": len(ls), "frames": sum(PT[l]["n_frames"] for l in ls), "hours": round(sum(PT[l]["duration_s"] for l in ls) / 3600, 3),
                                 "logs_with_route": sum(1 for l in ls if PT[l]["n_route_frames"] > 0), "anchors_route": sum(PT[l]["anchors_route"] for l in ls),
                                 "navtrain_sibling_sessions_same_vehicle_day": sorted(x for x in NTS if vehicle_day_of(x) == vd)}
json.dump(S, open("raw/pool_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in S.items() if k != "unseen58_sessions"}, indent=1))
print(json.dumps(S["unseen58_sessions"], indent=1))
