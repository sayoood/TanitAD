"""Contact sheet of one v6 extraction frame: the SAM3 classes over every native camera (before refinement)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/nvidia/sam3map/eval")
import sam3map_render_v4 as R  # noqa: E402

c8, j, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
d = np.load(f"/home/nvidia/sam3map/{c8}_v6raw/{j:03d}.npz", allow_pickle=True)
fd = Path(f"/home/nvidia/sam3map/native7/seq_{c8}") / str(d["tok"])
cams = ["CAM_FW", "CAM_CL", "CAM_CR", "CAM_FT", "CAM_RL", "CAM_RR", "CAM_RT"]
W, H = 640, 360
sheet = Image.new("RGB", (4 * W, 2 * H + 40), (14, 17, 16)); dr = ImageDraw.Draw(sheet)
st = json.loads(str(d["stats"]))
for q, cam in enumerate(cams):
    img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
    cls = np.repeat(np.repeat(d[f"cls_{cam}"], 2, axis=0), 2, axis=1)[: img.shape[0], : img.shape[1]]
    pan = R.overlay(img, cls).resize((W, H))
    x, y = (q % 4) * W, (q // 4) * H
    sheet.paste(pan, (x, y))
    frac = {R.NAME[k].split(" ")[0]: round(100 * float((d[f"cls_{cam}"] == k).mean()), 1) for k in range(1, 8)}
    dr.rectangle([x, y, x + W, y + 22], fill=(0, 0, 0))
    dr.text((x + 6, y + 3), f"{cam}  {frac}", fill=(255, 255, 255), font=R.font(13, True))
for q, k in enumerate((1, 2, 3, 6, 4, 5, 7)):
    xx = 3 * W + 10 + (q % 2) * 300; yy = H + 20 + (q // 2) * 30
    dr.rectangle([xx, yy, xx + 20, yy + 20], fill=R.COL[k]); dr.text((xx + 28, yy), R.NAME[k], fill=(230, 232, 231), font=R.font(16))
dr.text((10, 2 * H + 10), f"v6 raw extraction, token {j} ({d['tok']}) -- classes before refinement (ego body not yet removed)", fill=(200, 205, 202), font=R.font(16))
sheet.save(out)
print("ZZCONTACT-OKZZ", out)
