"""Visual check of the agent occluder masks (tracked boxes projected into every camera) on one token."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/nvidia/sam3map/eval")
import sam3map_render_v5 as R5  # noqa: E402
import camera_model as CM  # noqa: E402

c8, tok, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
fd = Path("/home/nvidia/sam3map/native7") / f"seq_{c8}" / tok
c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
boxes = np.load(Path("/home/nvidia/qwendrive/v2") / f"seq_{c8}" / tok / "gt.npz")["boxes"]
cams = ["CAM_FW", "CAM_CL", "CAM_CR", "CAM_FT", "CAM_RL", "CAM_RR", "CAM_RT"]
W, H = 480, 270
sheet = Image.new("RGB", (4 * W, 2 * H), (0, 0, 0)); d = ImageDraw.Draw(sheet)
for q, cam in enumerate(cams):
    C = CM.Camera.from_calib(c, fr["cam_order"].index(cam))
    img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")).astype(np.float32)
    m = R5.agent_occluders(C, boxes)
    mf = np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)
    img[mf] = 0.5 * img[mf] + 0.5 * np.array([255, 0, 0])
    sheet.paste(Image.fromarray(img.astype(np.uint8)).resize((W, H)), ((q % 4) * W, (q // 4) * H))
    d.text(((q % 4) * W + 4, (q // 4) * H + 4), f"{cam} occluded {100 * m.mean():.1f} %", fill=(255, 255, 0))
sheet.save(out)
print("ZZOCCDEBUG-OKZZ", len(boxes), "boxes")
