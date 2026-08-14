"""Bridge pai_epcache (ep_*.pt) -> the ph0 pilot layout (mp4 + ego npz + clips.json).

WHY: the Alpamayo-2 augmentation ran on the PhysicalAI OFFICIAL VAL split
("TanitAD OOD-val", a2_batch.sh), cached here as 290 ep_*.pt with real frames.
Our PH0 VLM/SAM3 pipeline has never seen it -- it processed the 600-clip
w120 val cache, of which only 56 are in the Alpamayo record set.

v2_to_pilot.py cannot read this: it takes a v2 --corpus (provider format), and
the epcache is per-episode tensors. This is the missing adapter.

geometry: frames_u8 is (T, 9, 256, 256) -- NINE channels, 256x256 PINHOLE, not
the 256x640 cylindrical the w120 caches use. Channels 0:3 are the front camera.
Stated, not silently reshaped: the VLM sees a narrower field than it did on the
w120 clips, so sign/agent recall is NOT comparable across the two batches.
"""
import glob, json, os, sys
import numpy as np
import torch

SRC = sys.argv[1]
OUT = sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FPS = 10

os.makedirs(f"{OUT}/videos", exist_ok=True)
os.makedirs(f"{OUT}/ego", exist_ok=True)
import imageio.v2 as iio

eps = sorted(glob.glob(os.path.join(SRC, "ep_*.pt")))
if LIMIT:
    eps = eps[:LIMIT]
ids, n_ok, n_fail = [], 0, 0
for p in eps:
    cid = os.path.splitext(os.path.basename(p))[0]          # ep_00000
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        fr = d["frames_u8"]                                  # (T, 9, H, W)
        rgb = fr[:, :3].permute(0, 2, 3, 1).contiguous().numpy()
        iio.mimwrite(f"{OUT}/videos/{cid}.mp4", rgb, fps=FPS,
                     codec="libx264", quality=7)
        np.savez(f"{OUT}/ego/{cid}.npz",
                 poses=d["poses"].numpy().astype("float32"),
                 actions=d["actions"].numpy().astype("float32"))
        ids.append(cid)
        n_ok += 1
    except Exception as e:                                   # noqa: BLE001
        n_fail += 1
        print(f"FAIL {cid} {type(e).__name__}: {e}", flush=True)
    if n_ok % 25 == 0 and n_ok:
        print(f"BRIDGED {n_ok}/{len(eps)}", flush=True)
json.dump(ids, open(f"{OUT}/clips.json", "w"), indent=1)
print(f"EPC_BRIDGE_DONE ok={n_ok} fail={n_fail} clips={len(ids)}")
