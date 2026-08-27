"""Crop each clip's grounding box out of the key frame, at FULL resolution.

My montage tiles are 400 px wide; a real pedestrian box can be 21 px wide in
the 1920x1080 source. "I looked and saw no pedestrian" on a thumbnail is
therefore NOT evidence that the box is wrong — the check has to happen at the
resolution the box was drawn at.

Writes one panel per clip: the key frame with the box drawn, plus a 4x zoom of
the box contents.
"""
from __future__ import annotations

import json
import os
import sys

import av
from PIL import Image, ImageDraw

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.data import alpamayo_records as AR      # noqa: E402

CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov"
OUT = "C:/Users/Admin/tanitad-wt/_s2build/report/boxes2"
#: The box is drawn on ALPAMAYO's anchor frame (5.1 s), not our 8.0 s one.
T_BOX = 5.1
os.makedirs(OUT, exist_ok=True)


def key_frame(clip_id: str, t: float):
    p = f"{CAM}/{clip_id}.mp4"
    if not os.path.exists(p):
        return None
    with av.open(p) as c:
        fps = float(c.streams.video[0].average_rate or 30.0)
        want = int(round(t * fps))
        for i, fr in enumerate(c.decode(video=0)):
            if i >= want:
                return fr.to_image()
    return None


def check(clip_id: str) -> str | None:
    rec = AR.get(clip_id)
    if not rec or not rec.boxes:
        print(f"  {clip_id[:8]}: no boxes")
        return None
    im = key_frame(clip_id, T_BOX)
    if im is None:
        print(f"  {clip_id[:8]}: no frame")
        return None
    W, H = im.size
    full = im.copy()
    d = ImageDraw.Draw(full)
    crops = []
    for q, lab, b in rec.boxes:
        x0, y0, x1, y1 = b
        # Alpamayo emits boxes in a 0-1000 normalised space on some rows and
        # raw pixels on others; decide by whether the box fits the frame.
        # ⚠️ DECIDE THE COORDINATE SPACE BY EVIDENCE, NOT BY "does it fit".
        # A 1920x1080 frame swallows almost any 0-1000 box without complaint,
        # so "it fits" is not a test — it is the reason the first render put a
        # pedestrian box on a blank wall. Both readings are drawn and labelled.
        sx, sy = W / 1000.0, H / 1000.0
        nx0, ny0, nx1, ny1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
        d.rectangle([nx0, ny0, nx1, ny1], outline="#f0f", width=4)
        d.text((nx0, max(0, ny0 - 34)), "norm/1000", fill="#f0f")
        d.rectangle([x0, y0, x1, y1], outline="#0f0", width=4)
        d.text((x0, max(0, y0 - 16)), f"{lab}", fill="#0f0")
        pad = 12
        # crop BOTH readings so the eye can settle it
        c = im.crop((max(0, int(nx0) - pad), max(0, int(ny0) - pad),
                     min(W, int(nx1) + pad), min(H, int(ny1) + pad)))
        if c.width > 2 and c.height > 2:
            k = min(6.0, 260.0 / max(c.width, c.height))
            crops.append((lab, c.resize((int(c.width * k), int(c.height * k)),
                                        Image.LANCZOS)))
    full.thumbnail((900, 520))
    ch = max([c.height for _, c in crops] + [0])
    sheet = Image.new("RGB", (full.width + sum(c.width + 8 for _, c in crops) + 8,
                              max(full.height, ch) + 26), "black")
    sheet.paste(full, (0, 26))
    x = full.width + 8
    dd = ImageDraw.Draw(sheet)
    dd.text((4, 6), f"{clip_id[:8]}  box frame t={T_BOX}s  "
                    f"Q: {rec.boxes[0][0][:70]}", fill="#ff0")
    for lab, c in crops:
        sheet.paste(c, (x, 26))
        dd.text((x + 3, 8), lab, fill="#0f0")
        x += c.width + 8
    out = f"{OUT}/{clip_id[:8]}.png"
    sheet.save(out)
    print(f"  {clip_id[:8]}: {len(crops)} box(es) -> {out}")
    return out


if __name__ == "__main__":
    ids = json.load(open("C:/Users/Admin/tanitad-wt/_s2build/report/sample.json"))
    for cid in ids:
        try:
            check(cid)
        except Exception as e:                            # noqa: BLE001
            print(f"  {cid[:8]}: FAILED {type(e).__name__}: {e}")
