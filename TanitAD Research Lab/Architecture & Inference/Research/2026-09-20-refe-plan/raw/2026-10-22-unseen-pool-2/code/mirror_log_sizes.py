#!/usr/bin/env python3
"""TREE LISTINGS ONLY (no file body) of per-log directories in the third-party HF mirror kesai-labs/nuplan, for a list of logs.
Writes raw/mirror_log_sizes.json: {log: {split, files:{name:size}}}. Every request is logged by hf_tree.py's logger (same file)."""
import sys, json, time, urllib.request, os
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "requests_log.jsonl")
splits = {}
for sp in ("train", "val", "test"):
    for i in json.load(open("raw/hf_other/kesai_logs_nuplan_%s.json" % sp)):
        splits[i["path"].rsplit("/", 1)[-1]] = sp
logs = json.load(open(sys.argv[1]))["logs"]; out = {}
for l in logs:
    sp = splits[l]
    url = "https://huggingface.co/api/datasets/kesai-labs/nuplan/tree/main/logs/nuplan_%s/%s" % (sp, l)
    req = urllib.request.Request(url, headers={"User-Agent": "tanitad-unseenpool-listing"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read(); st = r.status
    open(LOG, "a").write(json.dumps({"url": url, "method": "GET(tree-listing)", "status": st, "bytes": len(body), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    out[l] = {"split": sp, "files": {i["path"].rsplit("/", 1)[-1]: i["size"] for i in json.loads(body) if i["type"] == "file"}}
json.dump(out, open(sys.argv[2], "w"), indent=0)
print(len(out), "logs listed")
