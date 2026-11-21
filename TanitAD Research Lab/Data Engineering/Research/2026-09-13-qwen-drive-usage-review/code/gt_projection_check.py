"""Geometry check BEFORE any GPU: project OUR obstacle.offline GT boxes into the 8 virtual views.

Independent of the model and of the image synthesis code path: boxes come from the label
parquet, pixels come from the video, and the only thing joining them is calib.npz written
by the builder, consumed through Qwen-Drive's OWN `build_lidar2img` / `box_corners` /
`project_to_image`. If the green boxes sit on the vehicles, intrinsics, extrinsics, the
virtual rotation, the ego z-shift and the time alignment are all right at once.
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r"<scratchpad>\qd\src")
from qwen_drive_perception import geometry  # noqa: E402

fd = Path(sys.argv[1]); out = Path(sys.argv[2])
fr = json.loads((fd / "frame.json").read_text()); c = np.load(fd / "calib.npz"); g = np.load(fd / "gt.npz")
l2e = c["lidar2ego"]
boxes_ego = g["boxes"]
boxes_lidar = boxes_ego.copy(); boxes_lidar[:, 2] -= l2e[2, 3]          # ego -> lidar (pure z shift here)
corners = geometry.box_corners(boxes_lidar)
tiles = []
for i, cam in enumerate(fr["cam_order"]):
    img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
    w, h = img.size
    l2i = geometry.build_lidar2img(c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i])
    uv, ok = geometry.project_to_image(corners, l2i, w, h)
    d = ImageDraw.Draw(img)
    for k in np.where(ok)[0]:
        p = uv[k]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            d.line([tuple(p[a]), tuple(p[b])], fill=(60, 230, 90), width=3)
    d.text((12, 10), f"{cam}  GT boxes in view: {int(ok.sum())}", fill=(255, 255, 0))
    tiles.append(img.resize((640, 360)))
sheet = Image.new("RGB", (640 * 4, 360 * 2))
for j, t in enumerate(tiles):
    sheet.paste(t, ((j % 4) * 640, (j // 4) * 360))
sheet.save(out)
print("wrote", out, "boxes", len(boxes_ego))
