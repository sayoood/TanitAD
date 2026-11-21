#!/usr/bin/env python3
"""HF dataset SEARCH (metadata JSON only) for other public sources of nuPlan / OpenScene / NAVSIM camera frames. Writes raw/hf_search_routes.json"""
import sys, json, os, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netlog
Q = ["nuplan", "openscene", "navsim", "nuplan camera", "nuplan sensor", "navtrain", "navtest", "nuplan trainval", "OpenScene sensor", "nuplan jpg", "nuplan 123D",
     "nuplan front camera", "navsim sensor blobs", "navsim trainval", "nuplan images", "nuplan lidar camera"]
out = {}
for q in Q:
    url = "https://huggingface.co/api/datasets?search=%s&limit=60&sort=downloads&direction=-1" % urllib.parse.quote(q)
    body, _ = netlog.get_listing(url, "search")
    out[q] = [[d["id"], d.get("downloads"), (d.get("lastModified") or d.get("createdAt") or "")[:10], d.get("gated")] for d in json.loads(body)]
allids = {}
for q, rows in out.items():
    for r in rows:
        allids.setdefault(r[0], r)
json.dump({"queries": out, "unique": sorted(allids.values(), key=lambda r: -(r[1] or 0))}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "hf_search_routes.json"), "w"), indent=0)
print(len(allids), "unique datasets")
