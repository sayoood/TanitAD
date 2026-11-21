"""STEP 4 — draw the sample and render BLIND contact sheets for adjudication.

⭐ THIS IS THE ONLY STEP THAT CAN MEASURE PRECISION. Everything upstream is
description: a score histogram cannot say whether a box contains a sign, and
asking the detector's own score whether the detector was right is circular —
precisely the failure this study exists to catch.

⛔ **BLIND BY CONSTRUCTION.** The rendered cell carries **only an index**. The
score, the clip, the frame and the box geometry are banked in
`adjudication_sample.json` and joined to the verdicts *afterwards*. A sheet
labelled `0.94` would buy the adjudicator's agreement, and the threshold
recommendation in §4 is derived FROM the score↔correctness relation, so letting
the score reach the eye would make that recommendation self-fulfilling.
One concept per sheet, because the claim under test *is* the concept.

**Sampling.** Stratified by concept, uniform at random WITHIN concept, seed
fixed. Rare classes (`cyclist` n=10, `bus` n=26) are taken as a **CENSUS** — for
them "precision on a sample" and "precision" are the same number, and a class
whose whole population is 10 must not be described by a sample of 6.

⚠️ Detections are CLUSTERED IN CLIPS (a clip with one false sign tends to carry
several: same pole, several frames). The sample is drawn over detections, so the
point estimate is unbiased for the corpus of detections — but its interval must
be an **episode-cluster bootstrap over clips**, never a binomial. That is step 5.

**Frames.** Obtained by calling `ph0_pilot.sample_clip_frames` itself, at the
pipeline's own `t0_s=8.0` — not by a reimplementation — so the moment is the
pipeline's by construction. Boxes are in the 448-px frame SAM3 scored; the CROP
is taken from the same frame at native 640 px (scale 640/448) purely so a human
can see a 70-px² object at all. See `r3_encode_equivalence.py` for why the
pre-bridged encode is admissible HERE (photometric Δ 1.36/255, frame-aligned)
while remaining inadmissible for re-detection counts (C79).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import numpy as np
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
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))

# n to adjudicate per concept. `None` = CENSUS (take the whole population).
TARGETS = {"traffic sign": 64, "car": 48, "traffic light": 32,
           "pedestrian": 32, "truck": 32, "bus": None, "cyclist": None}
SAM3_PX = 448.0            # the frame SAM3 scored (ph0_pilot.VIDEO_PX)
COLS, ROWS = 4, 4          # 16 cells per sheet
CELL = 320                 # px per cell
PAD = 26                   # label strip height


def token() -> str:
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def load_complete(cache: str) -> dict:
    recs = {}
    for fn in sorted(os.listdir(cache)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(cache, fn), encoding="utf-8"))
        if r.get("liveness") is not None:
            recs[r["clip_id"]] = r
    return recs


def population(recs: dict) -> list[dict]:
    pop = []
    for cid, r in sorted(recs.items()):
        for fk in sorted((r.get("frames") or {}), key=lambda k: int(k)):
            for j, d in enumerate(r["frames"][fk].get("det", [])):
                if "score" not in d or not d.get("box_xyxy"):
                    continue
                pop.append({"clip_id": cid, "frame_idx": int(fk), "det_i": j,
                            "concept": d["concept"], "score": float(d["score"]),
                            "box_xyxy": d["box_xyxy"],
                            "mask_area_px": d.get("mask_area_px")})
    return pop


def crop_cell(frame_native, box448, scale, cell=CELL, ctx=6.0, minpx=96):
    """Context window around the banked box, upscaled to a square cell.

    ⚠️ The upscale is BICUBIC and that is a real choice: a 70-px² object is not
    adjudicable at native size, and NEAREST would make every small object a
    mosaic that no reviewer could call either way. Smoothing invents no content
    — it interpolates the pixels that are there — but it does make an ambiguous
    blob look tidier, so anything I cannot resolve is scored `unclear`, never
    `correct`, and the `unclear` rate is reported per concept."""
    from PIL import Image
    H, W = frame_native.shape[:2]
    x0, y0, x1, y1 = [v * scale for v in box448]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(minpx / 2.0, ctx * max(x1 - x0, y1 - y0) / 2.0)
    a0, b0 = int(round(cx - half)), int(round(cy - half))
    a1, b1 = int(round(cx + half)), int(round(cy + half))
    a0, b0 = max(0, a0), max(0, b0)
    a1, b1 = min(W, max(a1, a0 + 2)), min(H, max(b1, b0 + 2))
    sub = Image.fromarray(frame_native[b0:b1, a0:a1])
    sc = cell / max(sub.width, sub.height)
    out = Image.new("RGB", (cell, cell), (16, 16, 16))
    rs = sub.resize((max(1, int(round(sub.width * sc))),
                     max(1, int(round(sub.height * sc)))), Image.BICUBIC)
    ox, oy = (cell - rs.width) // 2, (cell - rs.height) // 2
    out.paste(rs, (ox, oy))
    bx = [ox + (x0 - a0) * sc, oy + (y0 - b0) * sc,
          ox + (x1 - a0) * sc, oy + (y1 - b0) * sc]
    return out, bx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("r4_sample_and_render")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--seed", type=int, default=20260816)
    a = ap.parse_args(argv)

    from PIL import ImageDraw
    from PIL import Image
    import ph0_pilot
    from huggingface_hub import hf_hub_download
    tok = token()

    recs = load_complete(a.cache)
    pop = population(recs)
    by_c: dict[str, list] = {}
    for p in pop:
        by_c.setdefault(p["concept"], []).append(p)
    print(f"[samp] {len(recs)} complete clips · {len(pop)} detections with a box")

    rng = random.Random(a.seed)
    picked = []
    for c, want in TARGETS.items():
        xs = by_c.get(c, [])
        if want is None or want >= len(xs):
            sel, mode = list(xs), "CENSUS"
        else:
            sel, mode = rng.sample(xs, want), "SAMPLE"
        for s in sel:
            s["_mode"] = mode
        picked.extend(sel)
        print(f"[samp] {c:<14} population {len(xs):>5} -> {len(sel):>3} "
              f"({mode})")
    # stable, concept-major order; the INDEX is what the sheet shows
    picked.sort(key=lambda p: (p["concept"], p["clip_id"], p["frame_idx"],
                               p["det_i"]))
    for i, p in enumerate(picked):
        p["idx"] = i

    need = sorted({p["clip_id"] for p in picked})
    print(f"[samp] {len(need)} distinct clips to pull", flush=True)
    frames: dict[str, list] = {}
    for k, cid in enumerate(need):
        v = hf_hub_download(DS, f"{PRE}{cid}.mp4", repo_type="dataset",
                            token=tok)
        fr, _t, _n = ph0_pilot.sample_clip_frames(v, t0_s=8.0, px=640)
        frames[cid] = fr
        if (k + 1) % 10 == 0:
            print(f"[samp] decoded {k+1}/{len(need)}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    sheets = []
    for c in TARGETS:
        xs = [p for p in picked if p["concept"] == c]
        for s0 in range(0, len(xs), COLS * ROWS):
            chunk = xs[s0:s0 + COLS * ROWS]
            name = (f"{c.replace(' ', '_')}_{s0 // (COLS * ROWS) + 1:02d}.png")
            W = COLS * CELL
            H = ROWS * (CELL + PAD) + PAD
            sheet = Image.new("RGB", (W, H), (10, 10, 10))
            d = ImageDraw.Draw(sheet)
            d.text((8, 6), f"CONCEPT UNDER TEST: '{c}'   "
                           f"(sheet {s0 // (COLS * ROWS) + 1}) "
                           f"— scores hidden on purpose", fill=(235, 235, 235))
            for k, p in enumerate(chunk):
                r, cc = divmod(k, COLS)
                fr = frames[p["clip_id"]]
                fi = p["frame_idx"]
                if fi >= len(fr):
                    continue
                nat = fr[fi]
                scale = nat.shape[1] / SAM3_PX
                cellimg, bx = crop_cell(nat, p["box_xyxy"], scale)
                x, y = cc * CELL, PAD + r * (CELL + PAD)
                sheet.paste(cellimg, (x, y))
                dd = ImageDraw.Draw(sheet)
                dd.rectangle([x + bx[0], y + bx[1], x + bx[2], y + bx[3]],
                             outline=(255, 215, 0), width=2)
                dd.text((x + 5, y + CELL + 5), f"#{p['idx']}",
                        fill=(255, 215, 0))
                dd.rectangle([x, y, x + CELL - 1, y + CELL - 1],
                             outline=(60, 60, 60), width=1)
            sheet.save(os.path.join(a.out_dir, name))
            sheets.append({"file": name, "concept": c,
                           "idx": [p["idx"] for p in chunk]})
            print(f"[sheet] {name} {len(chunk)} cells", flush=True)

    out = {"seed": a.seed, "sam3_px": SAM3_PX,
           "n_clips_complete": len(recs), "n_population": len(pop),
           "targets": {k: (v if v is not None else "CENSUS")
                       for k, v in TARGETS.items()},
           "population_per_concept": {c: len(v) for c, v in
                                      sorted(by_c.items())},
           "n_picked": len(picked), "n_clips_touched": len(need),
           "sheets": sheets,
           "detections": picked}
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
    print(f"[samp] {len(picked)} detections · {len(sheets)} sheets")
    print("SAMPLE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
