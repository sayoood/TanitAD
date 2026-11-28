"""Is the far-side `bridged_w120train_2400/videos/<cid>.mp4` the SAME video the
backfill pipeline feeds SAM3?

The pipeline calls `bridge_batch` -> `v2_to_pilot` on the w120 shard and writes
its own mp4 into a scratch dir it then deletes. The overlay renderer and the
eq3 experiment both pulled the PRE-BRIDGED mp4 from HF instead. If the two
differ, then (a) eq3's "cross-VM" difference was really a frame-source
difference, and (b) an overlay drawn on the far-side mp4 shows boxes on frames
SAM3 never saw.

CPU only. Bridges one clip locally and compares decoded frames pixel-wise.
"""
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np
import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\mp4chk")
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
sys.path.insert(0, os.path.join(REPO, "stack"))
TOK = re.search(r"hf_[A-Za-z0-9]+",
                open(os.path.join(REPO, "Keys.txt"), encoding="utf-8",
                     errors="replace").read()).group(0)
DS = "Sayood/tanitad-ph0-aug120"
DSV = "Sayood/tanitad-physicalai-w120-256x640cyl"
DSA = "Sayood/tanitad-alpamayo2-augmentation"
CID = json.load(open(os.path.join(REPO, "colab", "fixtures",
                                  "sam3_backfill_expected.json"),
                     encoding="utf-8"))["clips"][0]
os.makedirs(SP, exist_ok=True)
from huggingface_hub import HfApi, hf_hub_download                # noqa: E402
api = HfApi(token=TOK)


def dl(repo, rf, dst=None):
    p = hf_hub_download(repo, rf, repo_type="dataset", token=TOK)
    if dst:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(p, dst)
        return dst
    return p


far_mp4 = dl(DS, f"bridged_w120train_2400/videos/{CID}.mp4",
             os.path.join(SP, "far.mp4"))
loc = {f.rfilename.split("/")[-1][:-len(".v2ep.pt")]: f.rfilename
       for f in api.dataset_info(DSV).siblings
       if f.rfilename.endswith(".v2ep.pt")}
rf = loc[CID]
seg = rf.split("/")[0]
cdir = os.path.join(SP, "corpus")
os.makedirs(cdir, exist_ok=True)
for side in ("_geometry.json", "_v2manifest.pt"):
    try:
        dl(DSV, f"{seg}/{side}", os.path.join(cdir, side))
    except Exception as e:
        print("[side] MISS", side, type(e).__name__)
dl(DSV, rf, os.path.join(cdir, f"{CID}.v2ep.pt"))
recs = dl(DSA, "records.parquet")

import v2_to_pilot                                                # noqa: E402
out = os.path.join(SP, "bridge")
rc = v2_to_pilot.main(["--corpus", cdir, "--records", recs,
                       "--out", out, "--n", "1"])
print("[bridge] rc", rc)
own_mp4 = os.path.join(out, "videos", f"{CID}.mp4")

for tag, p in (("far-side", far_mp4), ("pipeline", own_mp4)):
    b = open(p, "rb").read()
    print(f"[mp4] {tag:9s} {len(b):>9d} B  md5={hashlib.md5(b).hexdigest()}")

import ph0_pilot                                                  # noqa: E402
fa = ph0_pilot.sample_clip_frames(far_mp4, t0_s=8.0)[0]
fb = ph0_pilot.sample_clip_frames(own_mp4, t0_s=8.0)[0]
print(f"[frames] far {len(fa)} {fa[0].shape} | pipeline {len(fb)} "
      f"{fb[0].shape}")
n = min(len(fa), len(fb))
diffs = []
for i in range(n):
    a, b = fa[i].astype(np.int16), fb[i].astype(np.int16)
    if a.shape != b.shape:
        diffs.append((i, "SHAPE", a.shape, b.shape))
        continue
    d = np.abs(a - b)
    diffs.append((i, float(d.mean()), int(d.max()),
                  float((d > 2).mean() * 100)))
print("[cmp] per-frame |diff| mean / max / %px>2 :")
for r in diffs[:8]:
    print("   frame", r[0], r[1:])
means = [r[1] for r in diffs if isinstance(r[1], float)]
print(f"[cmp] over {len(means)} frames: mean |diff| "
      f"{np.mean(means):.3f}, worst-frame mean {max(means):.3f}")
print("[cmp] IDENTICAL:", all(m == 0.0 for m in means))
json.dump({"clip_id": CID,
           "far_md5": hashlib.md5(open(far_mp4, "rb").read()).hexdigest(),
           "pipeline_md5": hashlib.md5(open(own_mp4, "rb").read()).hexdigest(),
           "far_bytes": os.path.getsize(far_mp4),
           "pipeline_bytes": os.path.getsize(own_mp4),
           "n_frames": n,
           "per_frame_mean_abs_diff": [r[1] for r in diffs],
           "per_frame_max_abs_diff": [r[2] for r in diffs],
           "pct_px_gt2": [r[3] for r in diffs],
           "identical": all(m == 0.0 for m in means),
           "evidence_class": "MEASURED"},
          open(os.path.join(SP, "mp4_source_check.json"), "w",
               encoding="utf-8"), indent=1)
print("MP4CHK_DONE")
