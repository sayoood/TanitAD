"""S2 v1 label build — pull the ego npz for every labeled clip (CPU-only leg).

Pulls from Sayood/tanitad-ph0-aug120:
  bridged_w120train_2400/ego/<clip>.npz  for the 201 fused aug120 clips
  w120val_600/ego/<clip>.npz             for the 600 fused w120val clips
  bridged_w120train_2400/clips.json + w120val_600/clips.json (join manifests)

Membership is checked against the far-side inventory BEFORE pulling; a clip
whose npz does not exist on the far side is a loud failure, never a silent
skip (absence at one location is not absence — but presence in the inventory
is the owning tool's answer).
"""
import concurrent.futures as cf
import json
import os
import re
import shutil
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"
TOKEN = re.search(r"hf_[A-Za-z0-9]+",
                  open(KEYS, encoding="utf-8", errors="replace").read()).group(0)

from huggingface_hub import hf_hub_download  # noqa: E402

DS = "Sayood/tanitad-ph0-aug120"
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
PULL = os.path.join(SP, "s2_pull")
EGO = os.path.join(SP, "s2_ego")
os.makedirs(EGO, exist_ok=True)


def clip_ids(dirname: str) -> list[str]:
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(os.path.join(PULL, dirname))
                  if f.endswith(".json") and not f.startswith("_"))


aug_ids = clip_ids("fused_aug120")
val_ids = clip_ids("fused_w120val")
print(f"aug120 clips {len(aug_ids)}  val clips {len(val_ids)}", flush=True)

inv = {f["path"] for f in json.load(
    open(os.path.join(PULL, "farside_inventory.json"), encoding="utf-8"))}

jobs: list[tuple[str, str]] = []          # (repo_file, dst)
missing: list[str] = []
for cid in aug_ids:
    rf = f"bridged_w120train_2400/ego/{cid}.npz"
    (jobs.append((rf, os.path.join(EGO, "aug120", f"{cid}.npz")))
     if rf in inv else missing.append(rf))
for cid in val_ids:
    rf = f"w120val_600/ego/{cid}.npz"
    (jobs.append((rf, os.path.join(EGO, "w120val", f"{cid}.npz")))
     if rf in inv else missing.append(rf))
for rf in ("bridged_w120train_2400/clips.json", "w120val_600/clips.json"):
    if rf in inv:
        jobs.append((rf, os.path.join(EGO, rf.replace("/", "__"))))
    else:
        missing.append(rf)
if missing:
    print(f"MISSING_ON_FARSIDE {len(missing)}: {missing[:10]}", flush=True)
    sys.exit(2)

os.makedirs(os.path.join(EGO, "aug120"), exist_ok=True)
os.makedirs(os.path.join(EGO, "w120val"), exist_ok=True)


def pull(job):
    rf, dst = job
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "cached"
    p = hf_hub_download(DS, rf, repo_type="dataset", token=TOKEN)
    shutil.copyfile(p, dst)
    return "pulled"


done = 0
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for res in ex.map(pull, jobs):
        done += 1
        if done % 100 == 0:
            print(f"{done}/{len(jobs)}", flush=True)
print(f"EGO_PULL_DONE {done}/{len(jobs)} "
      f"(aug {len(aug_ids)}, val {len(val_ids)})", flush=True)
