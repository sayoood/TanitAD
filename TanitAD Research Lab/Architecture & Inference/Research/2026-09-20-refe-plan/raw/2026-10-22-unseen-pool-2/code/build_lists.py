#!/usr/bin/env python3
"""Consolidate the intersected lists (raw/pool_lists.json). Inputs are the raw/*.json files written by the other scripts; nothing is fetched."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
B = "D:/TanitAstraResearch/data/e024_corpus_20261010/"
NT = set(parse_yaml_lists(B + "source_contracts/navtrain.yaml")["log_names"]); NS = set(parse_yaml_lists(B + "source_contracts/navtest.yaml")["log_names"])
NTS = {session_of(l) for l in NT}; NSS = {session_of(l) for l in NS}; NTVD = {vehicle_day_of(s) for s in NTS}; NSVD = {vehicle_day_of(s) for s in NSS}
vc = json.load(open("raw/src/vehicle_city.json"))["vehicle_city"]
PT = json.load(open("raw/profile_logs.json"))["logs"]; PTE = json.load(open("raw/profile_test_logs.json"))["logs"]
AR = json.load(open("raw/openscene_trainval_extras_archives.json"))
mp = json.load(open(B + "source_contracts/openscene_sensor_trainval_0-199.json", encoding="utf-8"))
log2arch = {l: int(k.rsplit("_", 1)[1]) for k, v in mp.items() for l in v}
MS = {}
for part in ("u58", "test11"):
    for l, v in json.load(open("raw/mirror_log_sizes_%s.json" % part)).items():
        MS[l] = v
TV = {m["name"].rsplit("/", 1)[-1][:-4] for m in json.load(open("raw/members_trainval_tgz.json"))["members"] if m["isfile"] and m["name"].endswith(".pkl")}
TEST = set(json.load(open("raw/test_release_logs.json"))["test_logs"])
def row(l, prof, release):
    s = session_of(l); p = prof[l]
    r = {"log": l, "release": release, "session": s, "vehicle": vehicle_of(s), "city": vc.get(vehicle_of(s)), "vehicle_day": vehicle_day_of(s),
         "session_in_navtrain": s in NTS, "session_in_navtest": s in NSS, "vehicle_day_in_navtrain": vehicle_day_of(s) in NTVD, "vehicle_day_in_navtest": vehicle_day_of(s) in NSVD,
         "n_frames_2Hz": p["n_frames"], "hours": round(p["duration_s"] / 3600, 3), "route_frames": p["n_route_frames"], "anchors_geom": p["anchors_geom"], "anchors_route": p["anchors_route"],
         "scorable_has_route": p["n_route_frames"] > 0}
    if release == "trainval":
        r["openscene_camera_archive"] = log2arch[l]
    if l in MS:
        f = MS[l]["files"]; r["mirror_split"] = MS[l]["split"]
        r["mirror_cam_f0_bytes"] = f.get("camera.pcam_f0.arrow"); r["mirror_4cam_bytes"] = sum(f.get(c, 0) for c in ("camera.pcam_f0.arrow", "camera.pcam_l0.arrow", "camera.pcam_r0.arrow", "camera.pcam_b0.arrow"))
    return r
tv_out = [row(l, PT, "trainval") for l in sorted(TV - NT)]
te_out = [row(l, PTE, "test") for l in sorted(TEST - NS)]
sess = {}
for r in tv_out + te_out:
    sess.setdefault(r["session"], []).append(r)
sess_rows = [{"session": s, "release": rs[0]["release"], "vehicle": rs[0]["vehicle"], "city": rs[0]["city"], "vehicle_day": rs[0]["vehicle_day"], "logs_outside": len(rs),
              "session_in_navtrain": rs[0]["session_in_navtrain"], "session_in_navtest": rs[0]["session_in_navtest"], "vehicle_day_in_navtrain": rs[0]["vehicle_day_in_navtrain"], "vehicle_day_in_navtest": rs[0]["vehicle_day_in_navtest"],
              "logs_with_route": sum(1 for r in rs if r["scorable_has_route"]), "anchors_route": sum(r["anchors_route"] for r in rs), "hours": round(sum(r["hours"] for r in rs), 3)} for s, rs in sorted(sess.items())]
universe = TV | TEST
out = {"definitions": "session = log.rsplit('_',2)[0]; vehicle-day = log[:10]+'_'+veh-NN (draw_reserve.unit_of(session,'vehicle-day')); anchors_geom = N-13 (4 history frames incl. current, 10 future frames, 2 Hz); anchors_route additionally needs non-empty roadblock_ids at the current frame (NAVSIM has_route=true)",
       "universe_public_sensor_logs": {**counts(universe), "outside_navtrain_union_navtest_logs": len(universe - NT - NS), "outside_sessions": len({session_of(l) for l in universe - NT - NS} - NTS - NSS),
                                       "outside_vehicle_days": len({vehicle_day_of(session_of(l)) for l in universe - NT - NS} - NTVD - NSVD)},
       "trainval_not_in_navtrain": tv_out, "test_not_in_navtest": te_out, "sessions_outside": sess_rows}
json.dump(out, open("raw/pool_lists.json", "w"), indent=1)
print(json.dumps(out["universe_public_sensor_logs"]))
print(len(tv_out), len(te_out), len(sess_rows), "sessions;", "universe route-less logs outside:", sum(1 for r in tv_out + te_out if not r["scorable_has_route"]), "routed outside:", sum(1 for r in tv_out + te_out if r["scorable_has_route"]))
