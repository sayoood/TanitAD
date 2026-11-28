"""S2 pull, part 2: all fused_w120val records + the A2 parquet. Resume-safe."""
import json
import os
import re
import shutil

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"
TOKEN = re.search(r"hf_[A-Za-z0-9]+",
                  open(KEYS, encoding="utf-8", errors="replace").read()).group(0)

from huggingface_hub import hf_hub_download  # noqa: E402

DS = "Sayood/tanitad-ph0-aug120"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_pull")


def pull(repo, rf, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = hf_hub_download(repo, rf, repo_type="dataset", token=TOKEN)
    shutil.copyfile(p, dst)
    return os.path.getsize(dst)


inv = json.load(open(os.path.join(ROOT, "farside_inventory.json")))
val = sorted(f["path"] for f in inv if f["path"].startswith("fused_w120val/"))
n = 0
for rf in val:
    pull(DS, rf, os.path.join(ROOT, rf))
    n += 1
    if n % 100 == 0:
        print(f"w120val {n}/{len(val)}", flush=True)
print(f"W120VAL_FILES {len(val)}", flush=True)

sz = pull("Sayood/tanitad-alpamayo2-augmentation", "records.parquet",
          os.path.join(ROOT, "records.parquet"))
print(f"RECORDS_PARQUET {sz} B", flush=True)
print("PULL2_DONE", flush=True)
