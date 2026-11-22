#!/usr/bin/env python3
"""private_sheet_lookat.py -- lead's PRIVATE sheet for the PI, run on the pod (CPU): where the windows look when the lead is off the camera axis.
Row = one logged frame of an evaluation clip with a lead in the label more than 8 degrees off the axis. Tiles: the fixed wide window (32 degrees, arm D), the fixed narrow window (16 degrees,
arm D), the narrow window at the direction of the head of card ADD107 (fit seed 0; the direction files of card ADD108, stage A, evaluation clips), the narrow window at the label's box centre
(the bound, not admissible at inference). Tier T0 (recorded frames, no rollout); not driving performance. Camera pixels are private: the sheet is shown to the PI as a local file only; it is
never landed, uploaded or published. The windows are made by the pilot's own function (window_encode.make_windows, imported unchanged); no encoder runs.
Selection, fixed before any picture was seen (no choice by result): in the order of the evaluation list, the first N clips with at least MIN frames (frame number 4 or later) that have a lead
in the label with |azimuth| > 8 degrees; of each clip the frames at the 1/6, 3/6 and 5/6 positions of its ordered list of such frames.
Usage: private_sheet_lookat.py --out <png> [--clips 4] [--min-frames 20]
Prints the number of rows and the clip ids (8 characters) and, per row, whether the lead's box centre is inside the head's window."""
import argparse
import os
import sys

import numpy as np

KIT = "/workspace/v8proof/cropread/kit/enc"
STAGE = "/workspace/v8proof/cropread/stage"
BOX = "/workspace/v8proof/cropread/enc_out/box"
DIRS = "/workspace/v8proof/lookgap/dirs"
EVAL = "/workspace/v8proof/cropread/run1/lists/eval141.txt"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--min-frames", type=int, default=20)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        sys.exit("REFUSED: %s exists" % a.out)
    sys.path.insert(0, KIT)
    import window_encode as WE
    from enc_common import Layout
    from PIL import Image, ImageDraw
    L = Layout(STAGE)
    ev = [l.split()[0] for l in open(EVAL) if l.strip()]
    rows, used = [], []
    for clip in ev:
        if len(used) >= a.clips:
            break
        zb = np.load(os.path.join(BOX, clip + ".npz"))
        has = zb["has_lead"].astype(bool) & np.isfinite(zb["ctr_az"]) & np.isfinite(zb["ctr_el"])
        off = [int(j) for j in np.nonzero(has & (np.abs(np.where(has, zb["ctr_az"], 0.0)) > 8.0))[0] if j >= 4]
        if len(off) < a.min_frames or not os.path.isfile(L.mp4(clip)):
            continue
        zd = np.load(os.path.join(DIRS, clip + ".npz"))
        pick = [off[(len(off) * k) // 6] for k in (1, 3, 5)]
        w_lab, _h, _p, _s = WE.make_windows(clip, L, BOX)
        w_dir, _h2, _p2, _s2 = WE.make_windows(clip, L, DIRS)
        used.append(clip)
        for j in pick:
            laz, lel = float(zb["ctr_az"][j]), float(zb["ctr_el"][j])
            haz, hel = float(zd["ctr_az"][j]), float(zd["ctr_el"][j])
            inside = abs(haz - laz) <= 8.0 and abs(hel - lel) <= 8.0
            tiles = [w_lab[j, 0], w_lab[j, 1], w_dir[j, 3], w_lab[j, 3]]
            cap = "clip %s  frame %d   lead in the label at azimuth %+.1f, elevation %+.1f degrees   head's direction %+.1f, %+.1f   lead inside the head's window: %s" % (
                clip[:8], j, laz, lel, haz, hel, "yes" if inside else "no")
            rows.append((tiles, cap, inside))
        del w_lab, w_dir
    if not rows:
        sys.exit("STOP: no evaluation clip has enough frames with the lead off the axis")
    T, CAP, GAP, TOP = 256, 22, 8, 78
    heads = ["fixed wide window (32 degrees)", "fixed narrow window (16 degrees)", "narrow window at the head's direction", "narrow window at the label (bound)"]
    W = 4 * T + 5 * GAP
    H = TOP + len(rows) * (T + CAP + GAP) + GAP
    sheet = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(sheet)
    d.text((GAP, 8), "PRIVATE - not to be landed or published. Where the windows look when the lead is more than 8 degrees off the camera axis. Tier T0 (recorded frames), not driving performance.", fill=(11, 11, 11))
    d.text((GAP, 26), "Frames chosen by a fixed rule from the label only (first clips of the evaluation list with enough such frames; three frames per clip), not by how well the head does.", fill=(82, 81, 78))
    for i, h in enumerate(heads):
        d.text((GAP + i * (T + GAP), 54), h, fill=(11, 11, 11))
    for r, (tiles, cap, _inside) in enumerate(rows):
        y0 = TOP + r * (T + CAP + GAP)
        for i, t in enumerate(tiles):
            sheet.paste(Image.fromarray(np.transpose(t, (1, 2, 0))), (GAP + i * (T + GAP), y0))
        d.text((GAP, y0 + T + 5), cap, fill=(11, 11, 11))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sheet.save(a.out)
    print("ZZ PRIVATE-SHEET rows %d clips %s inside %s size %dx%d" % (len(rows), ",".join(c[:8] for c in used), "".join("y" if x[2] else "n" for x in rows), W, H))


if __name__ == "__main__":
    main()
