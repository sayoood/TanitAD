"""Pull the aug120 label trees + reconstruction inputs (labels only, no video).

Payload: 49 batch label JSONs (~5 MB) + fused_w120val/_summary.json +
bridged clips/failures.json + w120val_600/clips.json + records.parquet (26 MB)
+ metadata listing of the w120 corpus (for the todo reconstruction).
"""
import json
import os
import re
import shutil

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"
TOKEN = re.search(r"hf_[A-Za-z0-9]+",
                  open(KEYS, encoding="utf-8", errors="replace").read()).group(0)

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi(token=TOKEN)
DS = "Sayood/tanitad-ph0-aug120"
ROOT = os.path.dirname(os.path.abspath(__file__))


def pull(repo, rf, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = hf_hub_download(repo, rf, repo_type="dataset", token=TOKEN)
    shutil.copyfile(p, dst)
    return os.path.getsize(dst)


total = 0
info = api.dataset_info(DS)
batch_files = [f.rfilename for f in info.siblings
               if f.rfilename.startswith("batch_")]
for rf in sorted(batch_files):
    total += pull(DS, rf, os.path.join(ROOT, "labels", rf))
print(f"batch label files: {len(batch_files)}  bytes={total}")

for rf in ("fused_w120val/_summary.json",
           "bridged_w120train_2400/clips.json",
           "bridged_w120train_2400/failures.json",
           "w120val_600/clips.json"):
    try:
        sz = pull(DS, rf, os.path.join(ROOT, "aux", rf.replace("/", "__")))
        print(f"aux {rf}: {sz} B")
    except Exception as e:  # noqa: BLE001
        print(f"aux MISS {rf}: {type(e).__name__}: {e}")

sz = pull("Sayood/tanitad-alpamayo2-augmentation", "records.parquet",
          os.path.join(ROOT, "aux", "records.parquet"))
print(f"records.parquet: {sz} B")

# w120 corpus shard listing (metadata only) -> loc for the todo reconstruction
i = api.dataset_info("Sayood/tanitad-physicalai-w120-256x640cyl")
loc = {}
for f in i.siblings:
    if f.rfilename.endswith(".v2ep.pt"):
        cid = f.rfilename.split("/")[-1][: -len(".v2ep.pt")]
        loc[cid] = f.rfilename
json.dump(loc, open(os.path.join(ROOT, "aux", "w120_loc.json"), "w"))
segs = {}
for v in loc.values():
    segs[v.split("/")[0]] = segs.get(v.split("/")[0], 0) + 1
print(f"w120 corpus shards: {len(loc)} by segment {segs}")
print("PULL_DONE")
