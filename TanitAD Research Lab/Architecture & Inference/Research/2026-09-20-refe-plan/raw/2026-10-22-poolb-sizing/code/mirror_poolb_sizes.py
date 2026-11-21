#!/usr/bin/env python3
"""TREE LISTINGS ONLY (no file body) of the 60 pool-B per-log directories in the third-party HF mirror kesai-labs/nuplan,
pinned to revision 484d9d14... Writes raw/mirror_poolB_sizes.json {log: {split, files:{name:{size, lfs_oid, xet}}}}."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netlog
REV = "484d9d14e18fdf712dbc70962997ba2d53617bce"
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
split = json.load(open(os.path.join(here, "poolB_mirror_split.json")))
out = {}
for log, sp in split.items():
    url = "https://huggingface.co/api/datasets/kesai-labs/nuplan/tree/%s/logs/nuplan_%s/%s" % (REV, sp, log)
    body, _ = netlog.get_listing(url, "tree-listing")
    items = json.loads(body)
    out[log] = {"split": sp, "files": {i["path"].rsplit("/", 1)[-1]: {"size": i["size"], "lfs_oid": (i.get("lfs") or {}).get("oid"), "xet": i.get("xetHash")}
                                        for i in items if i["type"] == "file"}}
json.dump(out, open(os.path.join(here, "mirror_poolB_sizes.json"), "w"), indent=0)
print(len(out), "logs listed")
