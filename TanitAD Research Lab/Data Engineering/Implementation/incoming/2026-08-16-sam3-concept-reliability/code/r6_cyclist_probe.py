"""STEP 6 — settle the NAMED single-frame observation, by looking at it.

`SAM3_DTYPE_FIX.md` §5 records, explicitly as *"VISUAL / SINGLE FRAME — not a
measurement"*: on clip `8f5df500` frame 12 a cyclist is prominent mid-road and
the boxes over that region read `car 0.72 / 0.83`. That is a HYPOTHESIS about
`car`↔`cyclist` confusion and it is the right way to have recorded it.

⛔ The shared-box test in `r5_precision.py` (max IoU between a `cyclist` box and
any `car` box on the same frame = **0.0**, n=10) does NOT settle it, and saying
it did would be the error. That test asks *"does SAM3 give ONE object TWO
labels?"*. The hypothesis is the opposite shape: *"does the `cyclist` prompt stay
silent while the `car` prompt fires on the same rider?"* — which produces exactly
zero overlap, because only one box is ever emitted.

⇒ The only thing that answers it is looking at the frame the observation names,
with every detection drawn and labelled. This renders that frame (and its
neighbours) from the pipeline's own bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
KEYS = os.path.join(REPO, "Keys.txt")
DS = "Sayood/tanitad-ph0-aug120"
PRE = "bridged_w120train_2400/videos/"
CACHE = (r"C:\Users\Admin\AppData\Local\Temp\claude"
         r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
         r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\sam3rel\records")
LOCAL_BRIDGE = (r"C:\Users\Admin\AppData\Local\Temp\claude"
                r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
                r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad"
                r"\sam3fix_assets\videos")
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
COL = {"car": (255, 90, 60), "cyclist": (60, 230, 120),
       "pedestrian": (250, 220, 60), "truck": (160, 120, 255),
       "bus": (60, 200, 255), "traffic sign": (255, 150, 220),
       "traffic light": (200, 200, 200)}


def token() -> str:
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("r6_cyclist_probe")
    ap.add_argument("--clip", default="8f5df500")
    ap.add_argument("--frames", default="8,12,16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--scale", type=int, default=3)
    a = ap.parse_args(argv)

    from PIL import Image, ImageDraw
    import ph0_pilot
    from huggingface_hub import hf_hub_download

    cid = next(fn[:-5] for fn in os.listdir(CACHE)
               if fn.startswith(a.clip) and fn.endswith(".json"))
    rec = json.load(open(os.path.join(CACHE, f"{cid}.json"), encoding="utf-8"))
    # ⭐ PREFER the pipeline's OWN bridge if it is on this box (C79): for this
    # one named frame the exact bytes exist locally, so there is no reason to
    # accept even the 1.4/255 photometric delta of the pre-bridged copy.
    loc = os.path.join(LOCAL_BRIDGE, f"{cid}.mp4")
    src = loc if os.path.exists(loc) else hf_hub_download(
        DS, f"{PRE}{cid}.mp4", repo_type="dataset", token=token())
    print(f"[probe] {cid[:8]} frames from "
          f"{'PIPELINE BRIDGE (exact bytes)' if src == loc else 'pre-bridged'}")
    fr, _t, _n = ph0_pilot.sample_clip_frames(src, t0_s=8.0, px=640)
    fr448, _, _ = ph0_pilot.sample_clip_frames(src, t0_s=8.0)
    sc = fr[0].shape[1] / float(fr448[0].shape[1])

    want = [int(x) for x in a.frames.split(",")]
    banked = {}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    panels = []
    for fi in want:
        ds = [d for d in (rec.get("frames") or {}).get(str(fi), {}).get(
            "det", []) if "score" in d and d.get("box_xyxy")]
        banked[str(fi)] = [{"concept": d["concept"], "score": d["score"],
                            "box_xyxy": d["box_xyxy"]} for d in ds]
        img = Image.fromarray(fr[fi]).resize(
            (fr[fi].shape[1] * a.scale, fr[fi].shape[0] * a.scale),
            Image.BICUBIC)
        d = ImageDraw.Draw(img)
        for det in ds:
            x0, y0, x1, y1 = [v * sc * a.scale for v in det["box_xyxy"]]
            col = COL.get(det["concept"], (255, 255, 255))
            d.rectangle([x0, y0, x1, y1], outline=col, width=2)
            d.text((x0 + 2, max(0, y0 - 11)),
                   f"{det['concept']} {det['score']:.2f}", fill=col)
        d.text((6, 4), f"{cid[:8]} frame {fi} — {len(ds)} detections, "
                       f"ALL concepts drawn", fill=(255, 255, 255))
        panels.append(img)
        print(f"[probe] frame {fi}: " + ", ".join(
            f"{x['concept']}:{x['score']}" for x in banked[str(fi)]))
    W = max(p.width for p in panels)
    H = sum(p.height + 4 for p in panels)
    sheet = Image.new("RGB", (W, H), (8, 8, 8))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + 4
    sheet.save(a.out)
    json.dump({"clip_id": cid, "frames": want,
               "frame_source": ("pipeline bridge (exact bytes SAM3 scored)"
                                if src == loc else "pre-bridged HF copy"),
               "sam3_px": list(fr448[0].shape[:2][::-1]),
               "render_px": list(fr[0].shape[:2][::-1]),
               "detections": banked},
              open(a.out_json, "w", encoding="utf-8"), indent=1)
    print(f"[probe] wrote {a.out}")
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
