"""S2 strategic-gap review — pull the pushed VLM/SAM3/Ego pipeline artifacts.

Pulls (labels only, no video/cache):
  1. Sayood/tanitad-ph0-aug120 far-side listing (file inventory, sizes)
  2. fused_aug120/  — ALL records (record count = fact, file count is not)
  3. fused_w120val/ — meta + v2/sam3 inputs to identify the 4 silent-empty
     records, then those 4 + a 40-record sample
  4. Sayood/tanitad-alpamayo2-augmentation records.parquet (row count + samples)
"""
import json
import os
import random
import re
import shutil
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"
TOKEN = re.search(r"hf_[A-Za-z0-9]+",
                  open(KEYS, encoding="utf-8", errors="replace").read()).group(0)

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

api = HfApi(token=TOKEN)
DS = "Sayood/tanitad-ph0-aug120"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_pull")
os.makedirs(ROOT, exist_ok=True)


def pull(repo, rf, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = hf_hub_download(repo, rf, repo_type="dataset", token=TOKEN)
    shutil.copyfile(p, dst)
    return os.path.getsize(dst)


# ---- 1. far-side inventory ------------------------------------------------ #
info = api.dataset_info(DS, files_metadata=True)
inv = [{"path": f.rfilename, "size": f.size} for f in info.siblings]
json.dump(inv, open(os.path.join(ROOT, "farside_inventory.json"), "w"),
          indent=1)
by_dir = {}
for f in inv:
    d = f["path"].split("/")[0] if "/" in f["path"] else "(root)"
    by_dir[d] = by_dir.get(d, 0) + 1
print("FARSIDE_DIRS " + json.dumps(by_dir), flush=True)

# ---- 2. fused_aug120: ALL records ---------------------------------------- #
aug = sorted(f["path"] for f in inv if f["path"].startswith("fused_aug120/"))
n = 0
for rf in aug:
    pull(DS, rf, os.path.join(ROOT, rf))
    n += 1
    if n % 50 == 0:
        print(f"aug120 {n}/{len(aug)}", flush=True)
print(f"AUG120_FILES {len(aug)}", flush=True)

# ---- 3. fused_w120val: meta + inputs, then targeted records --------------- #
pull(DS, "fused_w120val/_summary.json",
     os.path.join(ROOT, "fused_w120val/_summary.json"))
# v2 + sam3 production inputs to compute the 4 missing-perception clips
v2p = s3p = None
for f in inv:
    p = f["path"]
    if "w120val" in p and p.endswith(".json") and "/v2/" in p:
        v2p = p
    if "w120val" in p and p.endswith("sam3.json"):
        s3p = p
print(f"V2_INPUT {v2p}  SAM3_INPUT {s3p}", flush=True)
gap_ids = []
if v2p and s3p:
    pull(DS, v2p, os.path.join(ROOT, "w120val_inputs/v2.json"))
    pull(DS, s3p, os.path.join(ROOT, "w120val_inputs/sam3.json"))
    v2d = json.load(open(os.path.join(ROOT, "w120val_inputs/v2.json")))
    v2rows = v2d if isinstance(v2d, list) else v2d.get("clips", [])
    s3d = json.load(open(os.path.join(ROOT, "w120val_inputs/sam3.json")))
    s3rows = s3d.get("clips") if isinstance(s3d, dict) else s3d
    v2ids = {r["clip_id"] for r in v2rows if isinstance(r, dict)}
    s3ids = {r["clip_id"] for r in s3rows if isinstance(r, dict)}
    gap_ids = sorted(v2ids - s3ids)
    print(f"W120VAL v2={len(v2ids)} sam3={len(s3ids)} GAP={gap_ids}",
          flush=True)

val = sorted(f["path"] for f in inv
             if f["path"].startswith("fused_w120val/")
             and not f["path"].endswith("_summary.json"))
random.seed(42)
sample = sorted(set(random.sample(val, min(40, len(val)))
                    | {f"fused_w120val/{c}.json" for c in gap_ids
                       if f"fused_w120val/{c}.json" in set(val)}))
for rf in sample:
    pull(DS, rf, os.path.join(ROOT, rf))
print(f"W120VAL_SAMPLE {len(sample)} of {len(val)} record files", flush=True)

# ---- 4. Alpamayo-2 parquet ------------------------------------------------ #
sz = pull("Sayood/tanitad-alpamayo2-augmentation", "records.parquet",
          os.path.join(ROOT, "records.parquet"))
print(f"RECORDS_PARQUET {sz} B", flush=True)
print("PULL_DONE", flush=True)
