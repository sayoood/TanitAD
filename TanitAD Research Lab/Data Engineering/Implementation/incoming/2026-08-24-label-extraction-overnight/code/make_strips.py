"""Write one montage PNG per sampled clip so the frames can actually be LOOKED at.

Each montage is a 6-frame strip at -4/-2/0/+2/+4/+6 s around the s2 anchor, with
the offset burned into each tile and the clip id in the corner. The label is
NOT drawn on the image on purpose: the whole point is to read the scene first
and compare with the label afterwards, and a caption on the frame would anchor
the judgement it is supposed to test.
"""
from __future__ import annotations

import json
import os
import sys

import av
from PIL import Image, ImageDraw

CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov"
OUT = "C:/Users/Admin/tanitad-wt/_s2build/report/strips"
T0 = 8.0
OFFSETS = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
W = 400

os.makedirs(OUT, exist_ok=True)


def strip(clip_id: str) -> str | None:
    p = f"{CAM}/{clip_id}.mp4"
    if not os.path.exists(p):
        return None
    tiles: dict[float, Image.Image] = {}
    with av.open(p) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate or 30.0)
        want = {int(round((T0 + o) * fps)): o for o in OFFSETS}
        last = max(want)
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                im = fr.to_image()
                im.thumbnail((W, W))
                tiles[want[i]] = im
            if i > last:
                break
    if not tiles:
        return None
    keys = sorted(tiles)
    w, h = tiles[keys[0]].size
    sheet = Image.new("RGB", (w * len(keys), h + 22), "black")
    d = ImageDraw.Draw(sheet)
    for j, o in enumerate(keys):
        sheet.paste(tiles[o], (j * w, 22))
        lab = f"{o:+.0f}s" + ("  <-- KEY (anchor 8.0s)" if o == 0 else "")
        d.text((j * w + 5, 5), lab, fill="#0f0" if o == 0 else "#ccc")
    d.text((sheet.width - 150, 5), clip_id[:8], fill="#ff0")
    out = f"{OUT}/{clip_id[:8]}.png"
    sheet.save(out, quality=88)
    return out


if __name__ == "__main__":
    ids = json.load(open(sys.argv[1] if len(sys.argv) > 1
                         else "C:/Users/Admin/tanitad-wt/_s2build/report/sample.json"))
    for cid in ids:
        try:
            r = strip(cid)
            print(f"  {cid[:8]}: {'OK ' + r if r else 'NO FRAMES'}", flush=True)
        except Exception as e:                            # noqa: BLE001
            print(f"  {cid[:8]}: FAILED {type(e).__name__}: {e}", flush=True)
