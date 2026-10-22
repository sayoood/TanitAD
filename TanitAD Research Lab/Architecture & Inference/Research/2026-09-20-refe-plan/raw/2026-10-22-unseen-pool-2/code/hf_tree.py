#!/usr/bin/env python3
"""HF TREE LISTING (metadata only; no file body) of a dataset repo path, logged to raw/requests_log.jsonl. Usage: hf_tree.py <repo> <rev> <path> <out.json> [recursive]"""
import sys, json, time, urllib.request, os
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "requests_log.jsonl")
repo, rev, path, out = sys.argv[1:5]; rec = len(sys.argv) > 5
url = "https://huggingface.co/api/datasets/%s/tree/%s%s%s" % (repo, rev, ("/" + path) if path else "", "?recursive=true&limit=1000" if rec else "?limit=1000")
items = []
while url:
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "tanitad-unseenpool-listing"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read(); link = r.headers.get("Link"); status = r.status
    with open(LOG, "a") as f:
        f.write(json.dumps({"url": url, "method": "GET(tree-listing)", "status": status, "bytes": len(body), "secs": round(time.time() - t0, 2), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    items += json.loads(body)
    url = None
    if link and 'rel="next"' in link:
        url = link.split("<")[1].split(">")[0]
json.dump(items, open(out, "w"), indent=1)
print(len(items), "entries ->", out)
