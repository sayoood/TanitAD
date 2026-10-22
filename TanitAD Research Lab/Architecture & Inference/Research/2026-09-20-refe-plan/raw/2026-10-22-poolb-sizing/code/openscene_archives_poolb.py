#!/usr/bin/env python3
"""Which OpenScene trainval camera archives hold pool B, what they cost whole, and what a PREFIX scan (stop after the last needed
log's CAM_B0) costs. Inputs are LOCAL files only (TanitAstra source_contracts map, camera_header_inventory, compact_metadata,
pool_lists) plus the HF tree listing fetched by hf_tree_pinned.py. Writes raw/openscene_archives_poolb.json.
Model: archive members are written log by log, cameras in the order F0,L0,L1,L2,R0,R1,R2,B0 (MEASURED on a local test archive,
raw/members_test_camera_30.json), one jpg per 2 Hz frame per camera. Bytes of a log inside its archive ~ n_frames x 8 x (archive_bytes / sum n_frames)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "..", "raw")
TA = "D:/TanitAstraResearch/data/e024_corpus_20261010/"
mp = json.load(open(TA + "source_contracts/openscene_sensor_trainval_0-199.json", encoding="utf-8"))
hdr = {int(r["path"].rsplit("_", 1)[1].split(".")[0]): r for r in json.load(open(TA + "camera_header_inventory.json"))["rows"]}
tree = {int(i["path"].rsplit("_", 1)[1].split(".")[0]): i for i in json.load(open(os.path.join(RAW, "hf_tree_openscene_trainval_camera_pinned.json")))}
B = json.load(open(os.path.join(RAW, "poolB_logs.json")))
nfr = {}
CM = TA + "compact_metadata/"
for fn in os.listdir(CM):
    if fn.endswith(".json"):
        nfr[fn[:-5]] = len(json.load(open(CM + fn))["frames"])
for r in json.load(open("C:/Users/Admin/par18/unseenpool/raw/pool_lists.json"))["trainval_not_in_navtrain"]:
    nfr.setdefault(r["log"], r["n_frames_2Hz"])
# consistency: map order vs the archive's first member (TanitAstra header probe) -- is "position in the map list" = tar order?
order_ok = {int(k.rsplit("_", 1)[1]): (hdr[int(k.rsplit("_", 1)[1])]["first_segment"] == v[0]) for k, v in mp.items()}
arch = {int(k.rsplit("_", 1)[1]): v for k, v in mp.items()}
need = {}
for r in B:
    a = [k for k, v in arch.items() if r["log"] in v]
    assert len(a) == 1, (r["log"], a)
    need.setdefault(a[0], []).append(r)
rows = []
for a, rs in sorted(need.items()):
    logs = arch[a]; fr = [nfr[l] for l in logs]
    size = tree[a]["size"]; per_frame = size / sum(fr)           # bytes per (frame, all 8 cams) inside this archive
    last = max(logs.index(r["log"]) for r in rs)
    prefix_frames = sum(fr[: last + 1])
    rows.append({"archive": a, "bytes_whole": size, "lfs_sha256": tree[a]["lfs"]["oid"], "n_logs": len(logs), "n_frames_total": sum(fr),
                 "pool_b_logs": [r["log"] for r in rs], "positions": [logs.index(r["log"]) for r in rs],
                 "routed": [r["scorable_has_route"] for r in rs],
                 "prefix_bytes_est": round(size * prefix_frames / sum(fr)), "first_member_matches_map_order": order_ok[a]})
def tot(rows, routed_only=False):
    sel = [r for r in rows if (not routed_only) or any(r["routed"])]
    return {"archives": len(sel), "bytes_whole": sum(r["bytes_whole"] for r in sel), "prefix_bytes_est": sum(r["prefix_bytes_est"] for r in sel)}
out = {"order_check_all_200": {"first_member_equals_map_first": sum(order_ok.values()), "of": len(order_ok)},
       "pool_b_60": tot(rows), "pool_b_routed_56": tot(rows, True), "rows": rows}
json.dump(out, open(os.path.join(RAW, "openscene_archives_poolb.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
