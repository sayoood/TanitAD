"""Correct the camera caption of the delivered step-35,000 map video, frame by frame (CPU only).

The delivered frames say: "bottom 43 rows are ZEROED inside the trunk (--equalize-bottom-rows 43)".
That is FALSE for the built model. The trainer's pin sets `trunk_equalize_bottom_rows` as an
UNDECLARED attribute (refc_v3_train.py:381), and the --image-hw rebuild `dataclasses.replace` at
:459 drops it: the trunk zeroes nothing, while the BEV lift (built from the argv, :6922-6925) still
treats the bottom rows as unobserved (D-REFCV6-EQUALIZE-DROPPED). The step-38,000 render MEASURES
this on the built modules (`render_record.json` -> `equalize_bottom_rows`).

The fix touches ONLY the caption box: a dark label is drawn over the old text and the corrected text
is written on it. Every other pixel is unchanged, which is asserted per frame outside that box. The
video is then re-encoded from the corrected frames with the renderer's own ffmpeg arguments, and
decoded back by verify_mp4.py.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FR = Path("C:/Users/Admin/qland/work/mapvid/frames_refcv6_map_step35000")
SEQ = Path("C:/Users/Admin/qland/work/mapvid/seq_refcv6_map_step35000_fixed")
OUT = Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-map-video/raw")
MP4 = OUT / "refcv6_map_step35000.mp4"
PAD, Y_CAM, CAM_W, CAM_H, EQ = 16, 84, 1024, 416, 43
OLD = "\u2193 bottom 43 rows are ZEROED inside the trunk (--equalize-bottom-rows 43)"
NEW = ("\u2193 bottom 43 rows: NOT zeroed by the trunk (--equalize-bottom-rows 43 dropped by the config "
       "rebuild, refc_v3_train.py:381/:459); the BEV lift treats them as unobserved")
font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 11)
d0 = ImageDraw.Draw(Image.new("RGB", (10, 10)))
ye = Y_CAM + CAM_H - EQ                          # the dashed line row in canvas coordinates
w_old = d0.textlength(OLD, font=font)
w_new = d0.textlength(NEW, font=font)
x_right = PAD + CAM_W - 8
box = (int(x_right - max(w_old, w_new)) - 4, ye + 2, PAD + CAM_W - 2, ye + 18)
frames = sorted(FR.glob("c*_w*.png"))
titles = sorted(FR.glob("c*_title.png"))
if len(frames) != 513 or len(titles) != 3:
    raise SystemExit(f"expected 513 window frames + 3 title cards, found {len(frames)} + {len(titles)}")
n_changed_outside = 0
for f in frames:
    im = Image.open(f).convert("RGB")
    before = np.asarray(im).copy()
    d = ImageDraw.Draw(im)
    d.rectangle(box, fill=(12, 14, 18))
    d.text((x_right - w_new, ye + 3), NEW, fill=(245, 180, 90), font=font)
    after = np.asarray(im)
    mask = np.ones(after.shape[:2], bool)
    mask[box[1]:box[3] + 1, box[0]:box[2] + 1] = False
    n_out = int((before[mask] != after[mask]).any(axis=-1).sum())
    if n_out:                                    # checked BEFORE the save: a failure leaves no half-patched frame
        raise SystemExit(f"{f.name}: {n_out} pixels changed OUTSIDE the caption box -- nothing saved")
    n_changed_outside += n_out
    im.save(f, compress_level=1)
if SEQ.exists():
    shutil.rmtree(SEQ)
SEQ.mkdir(parents=True)
order = []
for ci in range(3):
    order += [FR / f"c{ci}_title.png"] * 25 + sorted(FR.glob(f"c{ci}_w*.png"))
for n, src in enumerate(order):
    os.link(src, SEQ / f"f{n:06d}.png")
ff = shutil.which("ffmpeg")
tmp = MP4.with_name("refcv6_map_step35000.fixed.mp4")
cmd = [ff, "-y", "-r", "10", "-i", str(SEQ / "f%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-crf", "18", "-preset", "slow", "-movflags", "+faststart", str(tmp)]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode or not tmp.exists():
    raise SystemExit(r.stderr[-800:])
v = subprocess.run([sys.executable, "C:/Users/Admin/ev6_82c2331/taniteval/tools/verify_mp4.py", str(tmp)],
                   capture_output=True, text=True)
print(v.stdout[-1500:])
if v.returncode:
    raise SystemExit("verify_mp4 failed on the corrected video")
os.replace(tmp, MP4)
shutil.rmtree(SEQ)
rec = {"what": "caption correction of the step-35,000 map video (D-REFCV6-EQUALIZE-DROPPED)",
       "frames_patched": len(frames), "caption_box_xyxy": list(box),
       "pixels_changed_outside_box": n_changed_outside, "old": OLD, "new": NEW,
       "mp4": str(MP4), "bytes": MP4.stat().st_size}
(OUT / "caption_fix_record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(json.dumps(rec, indent=1))
