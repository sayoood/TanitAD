"""lv2_frames.py: level L0 of the sheets. Decodes ONLY frame t and frame t + 2.0 s of the windows of WINDOWS_LV2.json, with the earlier prep code (rule R4 of latvis/LATVIS_DEFS.md and
latvis.py:cmd_prep, l.235): `fullres_encode.load_ep` / `decode` (full/fullres_encode.py:65, :73) of raw frame 2 g of the 139-episode eval corpus, area-downsampled by
F.interpolate(mode="area") to 64 x 160, rounded to uint8. Frame t of window w = grid step g0 of the pack (frame 4 of the window), t + 2.0 s = grid step g0 + 10 (frame 14).

  python lv2_frames.py --windows WINDOWS_LV2.json --out-dir private/frames [--src E:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl]

PRIVATE: camera content. The output folder holds a PRIVATE_DO_NOT_LAND.txt and never goes into any other folder. If the clips are not reachable on the machine, the level stays empty (the sheet
draws an empty tile with the reason) and this tool says so and exits 3."""
import argparse
import json
import os
import sys

import numpy as np

SRC_DEFAULT = "E:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl"
FULL_DIR = "F:/TanitAD-scratch/v8proof/full"
H, WD = 64, 160
S0, T2 = 4, 14


def raw_index(step):
    """grid step -> raw frame of the episode file (the prep decodes raw frames 0, 2, 4, ...: raw = 2 g)."""
    return 2 * int(step)


def steps_of(win):
    return int(win["g0"]), int(win["g0"]) + (T2 - S0)


def downsample(x):
    import torch
    import torch.nn.functional as F
    y = F.interpolate(x.float(), size=(H, WD), mode="area")
    return y.round().clamp(0, 255).to(torch.uint8).numpy()


def decode_window(FE, src, win):
    path = f"{src}/{win['eid']}.v2ep.pt"
    o, offs = FE.load_ep(path)
    n_raw = len(o["jpeg_len"])
    st = steps_of(win)
    raw = [raw_index(s) for s in st]
    if max(raw) >= n_raw:
        raise SystemExit(f"REFUSED: window {win['window']} needs raw frames {raw}, the clip has {n_raw}")
    x = FE.decode(o, offs, raw)
    return downsample(x).transpose(0, 2, 3, 1).copy()           # [2, 64, 160, 3] uint8


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    if not os.path.isdir(a.src):
        print(f"lv2_frames: the clips are not reachable on this machine ({a.src}); level L0 stays empty")
        return 3
    if FULL_DIR not in sys.path:
        sys.path.append(FULL_DIR)
    import fullres_encode as FE
    W = json.load(open(a.windows, encoding="utf-8"))["windows"]
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        W = [w for w in W if int(w["window"]) in keep]
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "PRIVATE_DO_NOT_LAND.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("Decoded camera frames. Private: dev box only, shown to the PI only, never landed or banked.\n")
    done = 0
    for w in W:
        fr = decode_window(FE, a.src, w)
        np.savez(os.path.join(a.out_dir, f"frames_{int(w['window'])}.npz"), frames=fr, eid=str(w["eid"]), steps=np.array(steps_of(w)))
        done += 1
    print(f"lv2_frames: {done} windows -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
