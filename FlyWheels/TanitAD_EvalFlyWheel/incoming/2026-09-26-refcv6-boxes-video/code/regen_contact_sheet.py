"""Re-draw the step-38,000 boxes contact sheet from the SAVED frames with the corrected header.

The render wrote the sheet with a header hard-coded to "step 35,000" (fixed in the renderer:
make_contact_sheet(..., step=)). The 12 keyframes and their captions are unchanged; this script
asserts that NO pixel below the 56-px header band changes, then replaces the sheet and writes a record.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PKG = Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-boxes-video")
RAW = PKG / "raw"
FR = Path("C:/Users/Admin/qland/work/mapvid/frames_refcv6_boxes_step38000")


def by_path(name, p):
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = json.loads((RAW / "render_record.json").read_text(encoding="utf-8"))
rmv = by_path("rmv_regen", "D:/Projects/TanitAD/taniteval/tools/render_refcv6_map_video.py")
rcv3 = by_path("rcv3_regen", rr["reused"]["render_refcv3_video"])
F = {"hud": rcv3.font(17), "tiny": rcv3.font(13)}
pc = json.loads((RAW / "per_clip.json").read_text(encoding="utf-8"))
clips_out = [{"clip_sha12": c["clip_sha12"], "nav": c["nav"], "n_windows_rendered": c["n_windows_rendered"]}
             for c in pc["clips"]]
# captions from the render's OWN keyframe record (per_frame.jsonl rounds t_label_s to 4 dp)
ci_of = {c["clip_sha12"]: i for i, c in enumerate(clips_out)}
iou_by = {(ci_of[k["clip_sha12"]], int(k["window"]) - 1): (k["iou"], k["t_label_s"])
          for k in rr["contact_sheet"]["keyframes"]}
old = RAW / "contact_sheet.png"
new = RAW / "contact_sheet.regen.png"
step = int(pc["step"])
assert step == 38000, step
rec = rmv.make_contact_sheet(new, FR, clips_out, F, "render", iou_by, step=step)
a = np.asarray(Image.open(old).convert("RGB"))
b = np.asarray(Image.open(new).convert("RGB"))
assert a.shape == b.shape, (a.shape, b.shape)
diff = np.any(a != b, axis=-1)
rows = np.nonzero(diff.any(axis=1))[0]
below = int(diff[56:].sum())
kf_same = rec["keyframes"] == rr["contact_sheet"]["keyframes"]
out = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "why": "header hard-coded 'step 35,000' on the step-38,000 sheet",
       "step_in_header": step, "n_pixels_changed": int(diff.sum()),
       "changed_rows": [int(rows.min()), int(rows.max())] if rows.size else None,
       "n_pixels_changed_below_header_band_56px": below, "must_read": 0,
       "keyframes_identical_to_render_record": bool(kf_same)}
print(json.dumps(out))
if below != 0 or not kf_same:
    new.unlink()
    raise SystemExit("REFUSED: the regenerated sheet differs outside the header band")
new.replace(old)
(RAW / "logs" / "contact_sheet_regen_record.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("replaced", old)
