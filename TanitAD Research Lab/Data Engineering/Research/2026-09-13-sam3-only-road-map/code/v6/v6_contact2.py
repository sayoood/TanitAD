"""Before / after contact sheet of one frame: SAM3 classes over every camera, top row = input npz, bottom row = output npz."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/nvidia/sam3map/eval")
import sam3map_render_v4 as R  # noqa: E402

before, after, seq, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
a = np.load(before, allow_pickle=True); b = np.load(after, allow_pickle=True)
fd = seq / str(a["tok"])
cams = [c for c in ("CAM_FW", "CAM_CL", "CAM_CR", "CAM_FT", "CAM_RL", "CAM_RR", "CAM_RT", "CAM_F0") if f"cls_{c}" in a.files]
W, H = 480, 270
sheet = Image.new("RGB", (len(cams) * W, 2 * H + 30), (14, 17, 16)); dr = ImageDraw.Draw(sheet)
for q, cam in enumerate(cams):
    img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
    for row, z in enumerate((a, b)):
        cls = np.repeat(np.repeat(z[f"cls_{cam}"], 2, axis=0), 2, axis=1)[: img.shape[0], : img.shape[1]]
        sheet.paste(R.overlay(img, cls).resize((W, H)), (q * W, row * H))
    ch = int((a[f"cls_{cam}"] != b[f"cls_{cam}"]).sum())
    dr.rectangle([q * W, 0, q * W + W, 20], fill=(0, 0, 0)); dr.text((q * W + 4, 3), f"{cam}  changed px {ch}", fill=(255, 255, 255), font=R.font(13, True))
dr.text((6, 2 * H + 6), f"top: {before.parent.name}   bottom: {after.parent.name}   token {a['tok']}", fill=(200, 205, 202), font=R.font(15))
sheet.save(out)
print("ZZCONTACT2-OKZZ")
