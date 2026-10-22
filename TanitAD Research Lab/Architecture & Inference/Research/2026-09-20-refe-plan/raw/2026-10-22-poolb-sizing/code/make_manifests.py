#!/usr/bin/env python3
"""Machine-readable download manifests (one row per file / archive / DB member), generated from the listings in raw/. No network.
Tier K20 = P6-readable logs; tier R36 = the other 36 routed logs (UNKNOWN command; P2 goal-mode only); tier N4 = the 4 route-less logs (not requested)."""
import os, json, csv
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "..", "raw")
J = lambda n: json.load(open(os.path.join(RAW, n)))
cm = J("cost_matrix.json"); K = set(cm["_K20_logs"]); pk = J("poolB_pkl_summary.json")["per_log"]
mir = J("mirror_poolB_sizes.json"); dbr = {r["log"]: r for r in J("db_poolB_rows.json")}
REV = "484d9d14e18fdf712dbc70962997ba2d53617bce"; OSREV = "a76f840b65e972bc45e56c2adced897498e9a026"
tier = lambda l: "K20" if l in K else ("R36" if pk[l]["anchors_route"] > 0 else "N4")
with open(os.path.join(RAW, "MANIFEST_mirror_poolB.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["tier", "log", "split", "camera", "bytes", "lfs_sha256", "xet_hash", "url"])
    for l in sorted(mir, key=lambda x: (tier(x), x)):
        for c in ("f0", "l0", "r0", "b0"):
            fn = "camera.pcam_%s.arrow" % c; v = mir[l]["files"][fn]
            w.writerow([tier(l), l, mir[l]["split"], "CAM_" + c.upper(), v["size"], v["lfs_oid"], v["xet"],
                        "https://huggingface.co/datasets/kesai-labs/nuplan/resolve/%s/logs/nuplan_%s/%s/%s" % (REV, mir[l]["split"], l, fn)])
tree = {int(i["path"].rsplit("_", 1)[1].split(".")[0]): i for i in J("hf_tree_openscene_trainval_camera_pinned.json")}
TA = "D:/TanitAstraResearch/data/e024_corpus_20261010/"
mp = json.load(open(TA + "source_contracts/openscene_sensor_trainval_0-199.json", encoding="utf-8"))
arch_of = {l: int(k.rsplit("_", 1)[1]) for k, v in mp.items() for l in v}
with open(os.path.join(RAW, "MANIFEST_openscene_archives_poolB.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["archive", "bytes_whole", "lfs_sha256", "pool_b_logs_in_archive(tier:pos)", "url"])
    by = {}
    for l in mir: by.setdefault(arch_of[l], []).append(l)
    for a in sorted(by):
        order = mp["openscene_sensor_trainval_camera_%d" % a] if ("openscene_sensor_trainval_camera_%d" % a) in mp else [x for k, v in mp.items() if int(k.rsplit("_", 1)[1]) == a for x in v]
        w.writerow([a, tree[a]["size"], tree[a]["lfs"]["oid"], ";".join("%s:%s:%d" % (tier(l), l, order.index(l)) for l in by[a]),
                    "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/%s/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_%d.tgz" % (OSREV, a)])
BASE = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/"
with open(os.path.join(RAW, "MANIFEST_db_range_poolB.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["tier", "log", "already_local", "zip_url", "zip_member", "local_header_offset", "method", "compressed_bytes", "uncompressed_bytes", "crc32"])
    d = J("db_sizes_poolB.json")["logs"]
    for l in sorted(mir, key=lambda x: (tier(x), x)):
        e = d[l][0]
        w.writerow([tier(l), l, "yes" if dbr[l]["local"] else "no", BASE + e["zip"], e["name"], e["lho"], e["method"], e["csz"], e["usz"], e["crc32"]])
print("manifests written")
