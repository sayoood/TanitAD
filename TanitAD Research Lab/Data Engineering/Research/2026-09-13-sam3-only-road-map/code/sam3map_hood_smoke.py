"""Smoke: can a SAM3 prompt find the ego vehicle's own body (bonnet + mirrors) in the 7 virtual views?

v1 of the SAM3-only map labelled the bonnet as road and accepted the anonymisation patch on it as a lane marking.
Prints, per view and prompt, the instances as (score, pixels, touches the bottom border) and saves a 4x2 overlay sheet.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
from PIL import Image
import sam3_smoke as S
from sam3map_extract import instances, VIEWS, V2

PROMPTS = {"car hood": 0.30, "hood of the car": 0.30, "car body": 0.30}
COL = {"car hood": (255, 0, 0), "hood of the car": (0, 255, 0), "car body": (0, 90, 255)}

c8 = sys.argv[1]; js = [int(x) for x in sys.argv[2].split(",")]
sd = V2 / f"seq_{c8}"; toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
proc, _ = S.build(conf=0.25)
out = Path("/home/nvidia/sam3map/hood_smoke"); out.mkdir(exist_ok=True)
for j in js:
    fd = sd / toks[j]
    tiles = []
    for cam in VIEWS:
        img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
        state = proc.set_image(img)
        a = np.asarray(img).astype(np.float32)
        rec = {}
        for p, t in PROMPTS.items():
            ins = instances(proc, state, p, t)
            rec[p] = [(round(s, 2), int(m.sum()), bool(m[-4:].any())) for s, m in ins]
            for s, m in ins:
                a[m] = 0.55 * a[m] + 0.45 * np.array(COL[p], np.float32)
        print(j, cam, json.dumps(rec), flush=True)
        tiles.append(Image.fromarray(a.clip(0, 255).astype(np.uint8)).resize((640, 360)))
    sheet = Image.new("RGB", (640 * 4, 360 * 2))
    for q, tl in enumerate(tiles):
        sheet.paste(tl, ((q % 4) * 640, (q // 4) * 360))
    sheet.save(out / f"hood_{c8[:2]}_{j:03d}.jpg", quality=85)
print("ZZHOOD-SMOKE-DONEZZ")
