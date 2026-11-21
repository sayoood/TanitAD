"""STEP 3z — ESCALATION RENDER for cells a 320-px contact-sheet cell cannot
resolve.

⚠️ **This is not a third protocol.** It re-renders the SAME detections at a
larger cell in the SAME two renderings the reliability study already uses and
already sanctions viewing together: its own 6×-box context window with the gold
box outlined, and G1's tight 4× LANCZOS crop. `raw/g1_reconcile_verdicts.json`
states the practice verbatim — *"BOTH renderings of each detection were viewed
before the verdict was fixed. That is deliberately the best available evidence
rather than a single protocol."*

⛔ **STILL BLIND.** It takes indices only. Score, clip, frame and box remain in
the sample JSON and are joined afterwards, so escalating a hard cell cannot leak
the withheld metadata into its verdict.

⚠️ **It can only move a verdict toward `correct` or leave it `unclear`** — more
pixels never manufacture a false positive. Reporting the escalation is what makes
the `unclear` rate honest rather than a measure of how hard I looked.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
sys.path.insert(0, os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                                "Implementation", "incoming",
                                "2026-08-16-sam3-concept-reliability", "code"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w3z_zoom")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--idx", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="zoomA")
    ap.add_argument("--cell", type=int, default=600)
    ap.add_argument("--cols", type=int, default=2)
    a = ap.parse_args(argv)

    import w3_sample_and_render as w3
    from PIL import Image, ImageDraw
    from r4_sample_and_render import crop_cell

    s = json.load(open(a.sample, encoding="utf-8"))
    by = {int(d["idx"]): d for d in s["detections"]}
    want = [int(x) for x in a.idx.split(",")]
    sel = [by[i] for i in want]
    frames = w3.frames_for(sorted({p["clip_id"] for p in sel}), w3.token())

    C, PAD, COLS = a.cell, 26, a.cols
    os.makedirs(a.out_dir, exist_ok=True)
    for style in ("context", "g1tight"):
        rows = (len(sel) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * C, rows * (C + PAD) + PAD),
                          (10, 10, 10))
        ImageDraw.Draw(sheet).text(
            (8, 6), f"ESCALATION — same detections, {C}px cell "
                    f"[{'6x context, gold box' if style == 'context' else 'G1 tight 4x LANCZOS'}]"
                    "  — scores still hidden", fill=(235, 235, 235))
        for k, p in enumerate(sel):
            fr = frames.get(p["clip_id"])
            if fr is None:
                continue
            f640, f448 = fr
            fi = p["frame_idx"]
            if fi >= len(f640):
                continue
            if style == "context":
                cellimg, bx = crop_cell(f640[fi], p["box_xyxy"],
                                        f640[fi].shape[1] / 448.0, cell=C)
            else:
                x0, y0, x1, y1 = [int(round(v)) for v in p["box_xyxy"]]
                src = Image.fromarray(f448[fi]).crop(
                    (max(0, x0), max(0, y0), max(x0 + 1, x1), max(y0 + 1, y1)))
                src = src.resize((src.width * 4, src.height * 4), Image.LANCZOS)
                cellimg = Image.new("RGB", (C, C), (16, 16, 16))
                sc = min(4.0, C / max(src.width, src.height))
                src = src.resize((max(1, int(src.width * sc)),
                                  max(1, int(src.height * sc))), Image.LANCZOS)
                cellimg.paste(src, ((C - src.width) // 2,
                                    (C - src.height) // 2))
                bx = None
            r_, c_ = divmod(k, COLS)
            x, y = c_ * C, PAD + r_ * (C + PAD)
            sheet.paste(cellimg, (x, y))
            dd = ImageDraw.Draw(sheet)
            if bx:
                dd.rectangle([x + bx[0], y + bx[1], x + bx[2], y + bx[3]],
                             outline=(255, 215, 0), width=3)
            dd.text((x + 6, y + C + 6), f"#{p['idx']}", fill=(255, 215, 0))
            dd.rectangle([x, y, x + C - 1, y + C - 1], outline=(60, 60, 60))
        name = f"{a.tag}_{style}.png"
        sheet.save(os.path.join(a.out_dir, name))
        print(f"[zoom] {name} {len(sel)} cells", flush=True)
    print("ZOOM_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
