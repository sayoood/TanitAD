#!/usr/bin/env python3
"""Cost matrix of the pool-B download, per log SUBSET x ROUTE. Inputs are the raw/*.json written by the other scripts (listings, local central directories, pkl table).
Subsets: A60 (all pool-B logs), R56 (route-bearing), K20 (>=1 route anchor whose NAVSIM command is LEFT/STRAIGHT/RIGHT: the only anchors P6 in command mode was trained for),
M13 (K20 logs with >=1 such anchor at ego speed >= 0.5 m/s, from TanitAstra compact_metadata: INHERITED, speed not in the pkl table).
Routes: M  = third-party mirror kesai-labs/nuplan, 4 camera Arrow files per log, whole files (EXACT, tree listing);
        O  = OpenScene first-party trainval camera tgz: whole archives (EXACT) and prefix cut (E: frame-count weights, order verified on 200/200 first members);
        D  = nuPlan DB range fetch from the S3 DB zips (EXACT: local central-directory cache; DBs already local are excluded);
        F  = frames-only floor: unique current frames x 4 cams x measured mean jpg bytes (E)."""
import os, json, gzip, collections
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "..", "raw")
J = lambda n: json.load(open(os.path.join(RAW, n)))
pk = J("poolB_pkl_summary.json")["per_log"]; B = {r["log"]: r for r in J("poolB_logs.json")}
mir = J("mirror_poolB_sizes.json"); dbr = {r["log"]: r for r in J("db_poolB_rows.json")}
arch = J("openscene_archives_poolb.json")["rows"]; jpg = J("jpg_sizes_local.json")
te = J("token_estimate.json")["per_log_poolB"]
TA = "D:/TanitAstraResearch/data/e024_corpus_20261010/"
mp = json.load(open(TA + "source_contracts/openscene_sensor_trainval_0-199.json", encoding="utf-8"))
arch_of = {l: int(k.rsplit("_", 1)[1]) for k, v in mp.items() for l in v}
arch_logs = {int(k.rsplit("_", 1)[1]): v for k, v in mp.items()}
tree = {int(i["path"].rsplit("_", 1)[1].split(".")[0]): i["size"] for i in J("hf_tree_openscene_trainval_camera_pinned.json")}
nfr = {}
CM = TA + "compact_metadata/"
for fn in os.listdir(CM):
    if fn.endswith(".json"): nfr[fn[:-5]] = len(json.load(open(CM + fn))["frames"])
for r in J("../../unseenpool/raw/pool_lists.json")["trainval_not_in_navtrain"] if False else json.load(open("C:/Users/Admin/par18/unseenpool/raw/pool_lists.json"))["trainval_not_in_navtrain"]:
    nfr.setdefault(r["log"], r["n_frames_2Hz"])
CAMF = ("camera.pcam_f0.arrow", "camera.pcam_l0.arrow", "camera.pcam_r0.arrow", "camera.pcam_b0.arrow")
logs_all = sorted(B)
R56 = [l for l in logs_all if pk[l]["anchors_route"] > 0]
K20 = [l for l in logs_all if pk[l]["anchors_known_cmd"] > 0]
M13 = [l for l in K20 if te[l]["F2"] > 0]
subsets = {"A60": logs_all, "R56": R56, "K20": K20, "M13": M13}
mean4 = jpg["all4"]["mean_per_frame_4cams"]
def mirror(ls): return sum(sum(mir[l]["files"][c]["size"] for c in CAMF) for l in ls)
def openscene(ls):
    by = collections.defaultdict(list)
    for l in ls: by[arch_of[l]].append(l)
    whole = sum(tree[a] for a in by); pre = 0
    for a, lg in by.items():
        order = arch_logs[a]; fr = [nfr[x] for x in order]; last = max(order.index(x) for x in lg)
        pre += tree[a] * sum(fr[:last + 1]) / sum(fr)
    return {"archives": len(by), "whole_bytes": whole, "prefix_bytes_E": round(pre)}
def dbs(ls):
    need = [dbr[l] for l in ls if not dbr[l]["local"]]
    return {"n_to_fetch": len(need), "n_local": len(ls) - len(need), "compressed_bytes": sum(r["csz"] for r in need), "uncompressed_bytes": sum(r["usz"] for r in need),
            "zips": sorted({r["zip"] for r in need})}
def frames_floor(ls, anchors_key):
    n = sum(pk[l][anchors_key] if anchors_key != "frames" else pk[l]["n_frames"] for l in ls)
    return {"unique_current_frames": n, "bytes_4cams_E": n * mean4}
out = {}
for name, ls in subsets.items():
    out[name] = {"logs": len(ls), "sessions": len({l.rsplit("_", 2)[0] for l in ls}), "frames_2Hz": sum(pk[l]["n_frames"] for l in ls),
                 "route_anchors": sum(pk[l]["anchors_route"] for l in ls), "known_cmd_anchors": sum(pk[l]["anchors_known_cmd"] for l in ls),
                 "mirror_4cam_whole_bytes": mirror(ls), "mirror_F0_only_bytes": sum(mir[l]["files"][CAMF[0]]["size"] for l in ls),
                 "openscene": openscene(ls), "db": dbs(ls),
                 "floor_route_anchor_frames": frames_floor(ls, "anchors_route"), "floor_known_cmd_anchor_frames": frames_floor(ls, "anchors_known_cmd"), "floor_all_2Hz_frames": frames_floor(ls, "frames")}
out["_mean_4cam_bytes_per_frame"] = mean4
out["_K20_logs"] = K20; out["_M13_logs"] = M13
json.dump(out, open(os.path.join(RAW, "cost_matrix.json"), "w"), indent=1)
for k in subsets:
    o = out[k]; print(k, o["logs"], "logs", o["sessions"], "sessions", o["frames_2Hz"], "frames", o["route_anchors"], "route anchors", o["known_cmd_anchors"], "known-cmd anchors")
    print("   mirror 4cam %.3f GB | OpenScene %d archives whole %.2f GB prefix(E) %.2f GB | DB fetch %d (+%d local) %.3f GB (unc %.3f) | floor(anchors) %.2f GB, floor(known) %.2f GB" % (
        o["mirror_4cam_whole_bytes"] / 1e9, o["openscene"]["archives"], o["openscene"]["whole_bytes"] / 1e9, o["openscene"]["prefix_bytes_E"] / 1e9, o["db"]["n_to_fetch"], o["db"]["n_local"], o["db"]["compressed_bytes"] / 1e9, o["db"]["uncompressed_bytes"] / 1e9,
        o["floor_route_anchor_frames"]["bytes_4cams_E"] / 1e9, o["floor_known_cmd_anchor_frames"]["bytes_4cams_E"] / 1e9))
