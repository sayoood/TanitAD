#!/usr/bin/env python3
"""MM read-only check of the private HF dataset after DATASCALE-10K: every shard's cache and members tar on the hub
carries an LFS sha256 equal to banked10k.jsonl. Token read in place from /root/.hf_token, never printed. No address in this file."""
import json
import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

REPO = "Sayood/tanitad-datascale-10k"
api = HfApi(token=open("/root/.hf_token", encoding="utf-8").read().strip())
info = api.dataset_info(REPO, files_metadata=True)
hub = {s.rfilename: (s.lfs.sha256 if s.lfs else None, s.size) for s in info.siblings}
banked = [json.loads(l) for l in open("/root/datascale/banked10k.jsonl", encoding="utf-8") if l.strip()]
rep = {"repo_private": bool(info.private), "hub_files": len(hub), "shards": len(banked), "mismatch": [], "missing": []}
for b in banked:
    for kind, key in (("cache", "cache_tar_sha256"), ("members", "members_tar_sha256")):
        f = f"{kind}/{b['shard']}.{kind}.tar"
        if f not in hub:
            rep["missing"].append(f)
        elif hub[f][0] != b[key]:
            rep["mismatch"].append([f, hub[f][0], b[key]])
rep["manifest_files"] = sorted(f for f in hub if f.startswith("manifest/"))
rep["cache_bytes_hub"] = sum(v[1] or 0 for f, v in hub.items() if f.startswith("cache/"))
json.dump(rep, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("HFVERIFY", "OK" if not rep["mismatch"] and not rep["missing"] else "BAD", json.dumps({k: rep[k] for k in ("repo_private", "hub_files", "shards")}), len(rep["mismatch"]), len(rep["missing"]))
