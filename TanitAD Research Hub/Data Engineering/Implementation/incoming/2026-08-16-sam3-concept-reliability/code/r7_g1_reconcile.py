"""STEP 7 — reconcile with G1, because two MEASURED numbers disagree by a lot.

`Project Steering/G1_RESULT.md` (2026-08-14) reports that **~two thirds of the
crops contained no sign at all** — the claim `NEXT_4472_BUILD_INPUTS.md` §3 cites
as the reason this study must precede the 4,472 build. My uniform sample of the
same class reads **0.88 precision on resolvable crops** (n=64). Both are MEASURED.
⛔ A disagreement that large is not settled by preferring the newer number; it is
settled by finding what differs.

**Two candidate differences, both in the PROTOCOL, both testable here.**

  **A. THE SELECTION.** G1 cropped *"the largest-area sign/light detection"* and
     the second-largest, **one pair per clip**. That is a MAX-AREA pick, not a
     uniform draw — and it is drawn from a class whose box-area distribution is
     extremely skewed (median 71 px², p90 0.004 of the frame). If SAM3's sign
     false positives are preferentially LARGE (a blank wall, a patch of sky, an
     advertising hoarding all present a big uniform region), then a max-area pick
     concentrates them and a uniform pick dilutes them. **The two numbers would
     then both be right and be about different populations.**

  **B. THE RENDERING.** G1 used a TIGHT crop of the box, 4× LANCZOS, taken from
     the 448-wide bridge. This study uses a 6×-box CONTEXT window from the same
     frames at native 640. A tight 8×8-px crop shows the sign and nothing else —
     no pole, no roadside, no scale — so a genuine small sign can read as
     "uniform brown" while a context crop shows a signpost.

This renders the SAME max-area detections BOTH ways, on one sheet each, so the
selection effect and the rendering effect are separated instead of confounded.
⚠️ This is the C79 lesson applied in advance: **an arm that differs in more ways
than the one under test settles nothing.**
"""
from __future__ import annotations

import argparse
import json
import os
import random
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
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
COLS, ROWS, CELL, PAD = 4, 4, 320, 26


def token() -> str:
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("r7_g1_reconcile")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260816)
    a = ap.parse_args(argv)

    from PIL import Image, ImageDraw
    import ph0_pilot
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from r4_sample_and_render import crop_cell
    tok = token()

    # G1's rule, applied to THIS corpus: per clip, the LARGEST-AREA `traffic
    # sign` detection. (G1 said "sign/light"; restricting to `traffic sign`
    # makes the comparison to my sign sample exact rather than approximate.)
    picks = []
    for fn in sorted(os.listdir(CACHE)):
        if not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(CACHE, fn), encoding="utf-8"))
        if r.get("liveness") is None:
            continue
        best = None
        for fk, f in (r.get("frames") or {}).items():
            for j, d in enumerate(f.get("det", [])):
                if d.get("concept") != "traffic sign" or not d.get("box_xyxy"):
                    continue
                b = d["box_xyxy"]
                ar = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                if best is None or ar > best["area_px"]:
                    best = {"clip_id": r["clip_id"], "frame_idx": int(fk),
                            "det_i": j, "concept": d["concept"],
                            "score": float(d["score"]), "box_xyxy": b,
                            "area_px": round(ar, 1)}
        if best:
            picks.append(best)
    picks.sort(key=lambda p: p["clip_id"])
    print(f"[g1] {len(picks)} clips carry a `traffic sign`; max-area pick each")
    rng = random.Random(a.seed)
    sel = sorted(rng.sample(picks, min(a.n, len(picks))),
                 key=lambda p: (p["clip_id"], p["frame_idx"]))
    for i, p in enumerate(sel):
        p["idx"] = 1000 + i          # index space disjoint from the main sample

    frames = {}
    for cid in sorted({p["clip_id"] for p in sel}):
        v = hf_hub_download(DS, f"{PRE}{cid}.mp4", repo_type="dataset",
                            token=tok)
        frames[cid] = (ph0_pilot.sample_clip_frames(v, t0_s=8.0, px=640)[0],
                       ph0_pilot.sample_clip_frames(v, t0_s=8.0)[0])

    os.makedirs(a.out_dir, exist_ok=True)
    sheets = []
    for style in ("g1tight", "context"):
        for s0 in range(0, len(sel), COLS * ROWS):
            chunk = sel[s0:s0 + COLS * ROWS]
            sheet = Image.new("RGB", (COLS * CELL,
                                      ROWS * (CELL + PAD) + PAD), (10, 10, 10))
            d0 = ImageDraw.Draw(sheet)
            d0.text((8, 6),
                    ("G1 PROTOCOL: tight box crop, 4x LANCZOS, from the 448-px "
                     "bridge" if style == "g1tight" else
                     "THIS STUDY: 6x-box context window from native 640") +
                    "   — concept 'traffic sign', MAX-AREA per clip, scores hidden",
                    fill=(235, 235, 235))
            for k, p in enumerate(chunk):
                r_, c_ = divmod(k, COLS)
                nat, f448 = frames[p["clip_id"]]
                fi = p["frame_idx"]
                if fi >= len(nat):
                    continue
                if style == "context":
                    cellimg, bx = crop_cell(nat[fi], p["box_xyxy"],
                                            nat[fi].shape[1] / 448.0)
                else:
                    x0, y0, x1, y1 = [int(round(v)) for v in p["box_xyxy"]]
                    src = Image.fromarray(f448[fi]).crop(
                        (max(0, x0), max(0, y0),
                         max(x0 + 1, x1), max(y0 + 1, y1)))
                    src = src.resize((src.width * 4, src.height * 4),
                                     Image.LANCZOS)
                    cellimg = Image.new("RGB", (CELL, CELL), (16, 16, 16))
                    sc = min(1.0, CELL / max(src.width, src.height))
                    src = src.resize((max(1, int(src.width * sc)),
                                      max(1, int(src.height * sc))),
                                     Image.LANCZOS)
                    cellimg.paste(src, ((CELL - src.width) // 2,
                                        (CELL - src.height) // 2))
                    bx = None
                x, y = c_ * CELL, PAD + r_ * (CELL + PAD)
                sheet.paste(cellimg, (x, y))
                dd = ImageDraw.Draw(sheet)
                if bx:
                    dd.rectangle([x + bx[0], y + bx[1], x + bx[2], y + bx[3]],
                                 outline=(255, 215, 0), width=2)
                dd.text((x + 5, y + CELL + 5), f"#{p['idx']}",
                        fill=(255, 215, 0))
                dd.rectangle([x, y, x + CELL - 1, y + CELL - 1],
                             outline=(60, 60, 60), width=1)
            name = f"g1recon_{style}_{s0 // (COLS * ROWS) + 1:02d}.png"
            sheet.save(os.path.join(a.out_dir, name))
            sheets.append(name)
            print(f"[sheet] {name} {len(chunk)} cells")

    areas = sorted(p["area_px"] for p in picks)
    json.dump({"selection_rule": "G1's: the LARGEST-AREA `traffic sign` "
                                 "detection per clip",
               "n_clips_with_a_sign": len(picks),
               "n_rendered": len(sel), "seed": a.seed,
               "maxarea_pick_area_px": {
                   "min": areas[0], "median": areas[len(areas) // 2],
                   "max": areas[-1]},
               "sheets": sheets, "detections": sel},
              open(a.out_json, "w", encoding="utf-8"), indent=1)
    print("G1RECON_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
