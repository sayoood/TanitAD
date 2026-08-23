"""STEP 8 — bank the rendered evidence into the repo.

⚠️ **THE SHEETS ARE THE ONLY COPY OF THE EVIDENCE THAT SURVIVES THE SESSION.**
The scratchpad PNGs are session-local; `crops/` is the durable record, exactly as
`…/2026-08-16-sam3-concept-reliability/crops/` is for that package. Written as
**JPEG q=90 with 4:4:4 chroma (no subsampling)** — the same setting the study
used — so the repo copy is faithful-but-not-bit-identical to the PNGs the
verdicts were fixed on. ⚠️ That is stated rather than hidden: every sheet
re-renders deterministically from the banked sample JSON if a bit-exact re-read
is ever needed.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w8_bank_crops")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--quality", type=int, default=90)
    a = ap.parse_args(argv)
    from PIL import Image
    os.makedirs(a.dst, exist_ok=True)
    tot_in = tot_out = 0
    for fn in sorted(os.listdir(a.src)):
        if not fn.endswith(".png"):
            continue
        src = os.path.join(a.src, fn)
        dst = os.path.join(a.dst, fn[:-4] + ".jpg")
        Image.open(src).convert("RGB").save(dst, "JPEG", quality=a.quality,
                                            subsampling=0, optimize=True)
        tot_in += os.path.getsize(src)
        tot_out += os.path.getsize(dst)
        print(f"  {fn} -> {os.path.basename(dst)} "
              f"{os.path.getsize(dst)/1e6:.2f} MB")
    print(f"[bank] {tot_in/1e6:.1f} MB PNG -> {tot_out/1e6:.1f} MB JPEG "
          f"(q={a.quality}, 4:4:4)")
    print("BANK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
