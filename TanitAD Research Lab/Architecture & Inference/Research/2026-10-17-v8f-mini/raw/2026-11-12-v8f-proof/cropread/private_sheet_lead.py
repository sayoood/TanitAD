#!/usr/bin/env python3
"""private_sheet_lead.py -- lead's PRIVATE sheet for the PI: fixed windows of card ADD99 (W32 and W16 of arm D) on frames with a lead in the label, each with the label gap, the gap of the
fixed-window head (D-H5, fit seed 0) and the gap of the present reader written under it. Tier T0 (a probe on logged frames). Camera pixels are private: the sheet is shown to the PI as a local
file only; it is never landed, uploaded or published. Reads the windows kept by an earlier smoke run (no new pixel file except the sheet itself).
Usage: private_sheet_lead.py --win-dir <smokeN>/win --dump-head <D_H5/reader_s0_score.npz> --dump-present <both/reader_s0_score.npz> --out <png> [--per-clip 6] [--tile 256]
Prints the number of tiles and the clip ids (8 characters); prints no gap value."""
import argparse
import os
import sys

import numpy as np


def rows_of(z, clip):
    clips = [str(c) for c in z["clips"]]
    if clip not in clips:
        return None
    ci = clips.index(clip)
    m = np.asarray(z["cid"]) == ci
    return {k: np.asarray(z[k])[m] for k in ("frame", "gap", "lab_gap", "lab_present")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--win-dir", required=True)
    ap.add_argument("--dump-head", required=True)
    ap.add_argument("--dump-present", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-clip", type=int, default=6)
    ap.add_argument("--tile", type=int, default=256)
    a = ap.parse_args(argv)
    from PIL import Image, ImageDraw
    if os.path.exists(a.out):
        sys.exit("REFUSED: %s exists" % a.out)
    zh, zp = np.load(a.dump_head, allow_pickle=False), np.load(a.dump_present, allow_pickle=False)
    tiles, used = [], []
    for f in sorted(os.listdir(a.win_dir)):
        if not f.endswith(".npy"):
            continue
        clip = f[:-4]
        rh, rp = rows_of(zh, clip), rows_of(zp, clip)
        if rh is None or rp is None:
            continue
        win = np.load(os.path.join(a.win_dir, f), mmap_mode="r")          # [n_frames, 4, 3, 256, 256] uint8: D W32, D W16, B W32, B W16
        pres_gap = {int(fr): float(g) for fr, g in zip(rp["frame"], rp["gap"])}
        ok = (rh["lab_present"].astype(bool)) & np.isfinite(rh["gap"]) & np.isfinite(rh["lab_gap"]) & (rh["frame"] < win.shape[0])
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        pick = idx[np.unique(np.linspace(0, len(idx) - 1, min(a.per_clip, len(idx))).round().astype(int))]
        used.append(clip[:8])
        for i in pick:
            fr = int(rh["frame"][i])
            pair = []
            for w in (0, 1):
                im = Image.fromarray(np.transpose(np.asarray(win[fr, w]), (1, 2, 0)))
                if a.tile != im.size[0]:
                    im = im.resize((a.tile, a.tile), Image.BILINEAR)
                pair.append(im)
            cap = "clip %s  frame %d\nlabel %.1f m   crop head %.1f m   present reader %s" % (
                clip[:8], fr, float(rh["lab_gap"][i]), float(rh["gap"][i]), ("%.1f m" % pres_gap[fr]) if fr in pres_gap and np.isfinite(pres_gap[fr]) else "none")
            tiles.append((pair, cap))
    if not tiles:
        sys.exit("STOP: no clip of the window folder is in the dumps with a label lead")
    T, CAP, GAP = a.tile, 40, 10
    cols = 2                                                                # two tiles (each a W32 | W16 pair) per row
    nrow = (len(tiles) + cols - 1) // cols
    tw = 2 * T + 4
    W, H = cols * tw + (cols + 1) * GAP, 54 + nrow * (T + CAP + GAP) + GAP
    sheet = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(sheet)
    d.text((GAP, 8), "PRIVATE - not to be landed or published. Fixed windows of card ADD99 (left 32 degrees, right 16 degrees) on logged frames; tier T0.", fill=(11, 11, 11))
    d.text((GAP, 26), "Under each pair: label gap, gap of the fixed-window head (D-H5, fit seed 0), gap of the present reader (seed 0).", fill=(82, 81, 78))
    for n, (pair, cap) in enumerate(tiles):
        r, c = divmod(n, cols)
        x0, y0 = GAP + c * (tw + GAP), 54 + r * (T + CAP + GAP)
        sheet.paste(pair[0], (x0, y0))
        sheet.paste(pair[1], (x0 + T + 4, y0))
        d.multiline_text((x0, y0 + T + 4), cap, fill=(11, 11, 11), spacing=3)
    sheet.save(a.out)
    print("ZZ PRIVATE-SHEET tiles %d clips %s size %dx%d" % (len(tiles), ",".join(used), W, H))


if __name__ == "__main__":
    main()
