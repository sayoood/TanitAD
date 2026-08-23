"""S2 pull, part 3: the aug120 batch_* v2+sam3 label files (raw VLM records
with _calls -> ENGINE_A prompt values)."""
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

inv = json.load(open(os.path.join(ROOT, "farside_inventory.json")))
bt = sorted(f["path"] for f in inv if f["path"].startswith("batch_"))
for rf in bt:
    dst = os.path.join(ROOT, rf)
    if os.path.exists(dst):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = hf_hub_download(DS, rf, repo_type="dataset", token=TOKEN)
    shutil.copyfile(p, dst)
print(f"BATCH_FILES {len(bt)}")
for rf in bt[:6]:
    print(rf)
print("PULL3_DONE")
