#!/usr/bin/env python3
# Batch keyframe download + montage for scenario classification.
# Usage: kf_batch.py <start> <step> <count>  (selects pool_2604.txt[start::step][:count])
import os, sys, subprocess, math, json
import concurrent.futures as cf
from pathlib import Path
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HOME"] = "/workspace/.hf"
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
pool = [l.strip() for l in open("/workspace/pool_2604.txt") if l.strip()]
start, step, count = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
clips = pool[start::step][:count]
KFDIR = sys.argv[4] if len(sys.argv) > 4 else "/workspace/kf_batch"
MDIR = sys.argv[5] if len(sys.argv) > 5 else "/workspace/montages"
os.makedirs(KFDIR, exist_ok=True); os.makedirs(MDIR, exist_ok=True)

def fetch(c):
    try:
        p = hf_hub_download(REPO, f"sample_set/26.04_release/{c}/camera_front_wide_120fov.mp4",
                            repo_type="dataset", revision="26.04", local_dir="/workspace/kf_dl")
        out = f"{KFDIR}/{c[:8]}.jpg"
        if not os.path.exists(out):
            subprocess.run(["ffmpeg", "-y", "-ss", "2", "-i", p, "-frames:v", "1",
                            "-vf", "scale=480:-1", out], capture_output=True)
        return c[:8] if os.path.exists(out) else None
    except Exception:
        return None

with cf.ThreadPoolExecutor(max_workers=6) as ex:
    got = [r for r in ex.map(fetch, clips) if r]
print(f"got {len(got)}/{len(clips)} keyframes")

imgs = sorted(Path(KFDIR).glob("*.jpg"))
COLS, ROWS = 5, 3; PER = COLS * ROWS; TW, TH, LB = 384, 230, 18
manifest = {}
for gi in range(math.ceil(len(imgs) / PER)):
    batch = imgs[gi * PER:(gi + 1) * PER]
    grid = Image.new("RGB", (COLS * TW, ROWS * (TH + LB)), (15, 15, 15))
    d = ImageDraw.Draw(grid); ids = []
    for j, ip in enumerate(batch):
        try:
            im = Image.open(ip).convert("RGB").resize((TW, TH))
        except Exception:
            continue
        r, c = divmod(j, COLS); x, y = c * TW, r * (TH + LB)
        grid.paste(im, (x, y + LB)); d.text((x + 3, y + 3), ip.stem, fill=(0, 255, 0)); ids.append(ip.stem)
    gp = f"{MDIR}/grid_{gi:02d}.jpg"; grid.save(gp, quality=82); manifest[f"grid_{gi:02d}"] = ids
json.dump(manifest, open(f"{MDIR}/manifest.json", "w"), indent=1)
print(f"montages: {len(manifest)} grids, {len(imgs)} keyframes total")
