"""STEP 4 — ARM C: re-read G1's OWN banked crops, blind.

⭐ **THIS IS THE SHARPEST INSTRUMENT AVAILABLE AND IT COSTS ~0.2 MB.** G1's
evidence was never thrown away: `Sayood/tanitad-ph0-aug120` →
`g1_evidence/crops/row01.jpg … row54.jpg` plus the assembled `g1sheet.jpg`.
Re-reading those exact JPEGs holds the corpus, the selection rule, the rendering
protocol AND the pixels fixed, so the ONLY remaining variable is **the
adjudicator**.

⛔ **BLINDING.** The tiles are named `rowNN`, and `G1_SIGN_OCR_GRADING_SHEET.md`
maps row → clip → claimed OCR text. Reading them in filename order would carry
G1's own ordering (and, via the sheet, its claimed sign text) straight into the
eye. So the tiles are **SHUFFLED under a fixed seed and re-indexed 3000+k**; the
row→index map is written to the sample JSON and joined only AFTER the verdicts
are fixed.

⚠️ **What this arm can and cannot say.** It measures whether a second reader
agrees with G1's *"no sign visible in the crop at all"* call on G1's own
evidence. It is NOT a fresh sample of the corpus (arm A is), and it inherits
whatever selection G1 made. Its n is 54 tiles over ~30 clips.
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
PRE = "g1_evidence/crops/"
COLS, ROWS, CELL, PAD = 4, 4, 320, 26


def token() -> str:
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w4_g1_tiles")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260817)
    a = ap.parse_args(argv)

    from PIL import Image, ImageDraw
    from huggingface_hub import HfApi, hf_hub_download
    tok = token()
    api = HfApi(token=tok)
    rfs = sorted(f.rfilename for f in api.dataset_info(DS).siblings
                 if f.rfilename.startswith(PRE))
    print(f"[g1tiles] {len(rfs)} banked G1 crops on the far side")
    assert rfs, "g1_evidence/crops is EMPTY — refusing to report an arm C"

    tiles = []
    for rf in rfs:
        p = hf_hub_download(DS, rf, repo_type="dataset", token=tok)
        tiles.append({"row_file": os.path.basename(rf), "local": p})

    rng = random.Random(a.seed)
    rng.shuffle(tiles)                       # ⛔ break G1's own ordering
    for i, t in enumerate(tiles):
        t["idx"] = 3000 + i

    os.makedirs(a.out_dir, exist_ok=True)
    sheets = []
    for s0 in range(0, len(tiles), COLS * ROWS):
        chunk = tiles[s0:s0 + COLS * ROWS]
        sheet = Image.new("RGB", (COLS * CELL, ROWS * (CELL + PAD) + PAD),
                          (10, 10, 10))
        ImageDraw.Draw(sheet).text(
            (8, 6), "ARM C — G1's OWN banked crops, re-read blind. Question: "
                    "does the tile contain a TRAFFIC SIGN at all?",
            fill=(235, 235, 235))
        for k, t in enumerate(chunk):
            r_, c_ = divmod(k, COLS)
            im = Image.open(t["local"]).convert("RGB")
            t["src_wh"] = [im.width, im.height]
            sc = min(1.0 * CELL / max(im.width, im.height), 8.0)
            im2 = im.resize((max(1, int(im.width * sc)),
                             max(1, int(im.height * sc))), Image.LANCZOS)
            cellimg = Image.new("RGB", (CELL, CELL), (16, 16, 16))
            cellimg.paste(im2, ((CELL - im2.width) // 2,
                                (CELL - im2.height) // 2))
            x, y = c_ * CELL, PAD + r_ * (CELL + PAD)
            sheet.paste(cellimg, (x, y))
            dd = ImageDraw.Draw(sheet)
            dd.text((x + 5, y + CELL + 5), f"#{t['idx']}", fill=(255, 215, 0))
            dd.rectangle([x, y, x + CELL - 1, y + CELL - 1],
                         outline=(60, 60, 60), width=1)
        name = f"w120sign_C_g1tiles_{s0 // (COLS * ROWS) + 1:02d}.png"
        sheet.save(os.path.join(a.out_dir, name))
        sheets.append(name)
        print(f"[sheet] {name} {len(chunk)} cells", flush=True)

    out = {"arm": "C", "source": f"{DS} :: {PRE}",
           "what_is_held_fixed": "corpus, selection rule, rendering protocol "
                                 "AND the exact JPEG bytes G1 read. The only "
                                 "variable is the ADJUDICATOR.",
           "blinding": f"tiles shuffled with seed {a.seed} and re-indexed "
                       "3000+k; the row->index map below is joined only AFTER "
                       "the verdicts are fixed",
           "n_tiles": len(tiles), "sheets": sheets,
           "tiles": [{k: v for k, v in t.items() if k != "local"}
                     for t in tiles]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
    print(f"G1TILES_DONE n={len(tiles)} sheets={len(sheets)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
