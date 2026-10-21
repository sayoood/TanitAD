"""build_gt6.py -- the S6 GT (1,2,3,4,5,6 s = model slots 1,3,4,5,6,7) for a captured dump, FROM THE DUMP ITSELF (CPU, no model).

`l7_targets_pai.py --s6 <dir>` wants `<dir>/ep*.npz` with `clip_index`, `ws`, `g [n,6,2]`. The battery builds that with `r7_panel.build_s6_dump`,
which needs banked baseline dumps that a TRAIN139 capture does not have. The GT, however, is a pure function of what the roll already wrote in
`decisions/ep*.npz`: `waypoint_targets(pose_last, gt_future_ext, horizons)` (refb_labels.py:210, UNCHANGED) on the windows whose 60-step future is
valid -- the exact rule `build_s6_dump` applies (`fv[:, 59]`). Windows without a valid 6 s future are omitted and the target builder falls back to
the 2 s grid for them (grid = 4), as it does for eval139.

    python build_gt6.py --dump <dump_dir> --out <gt6_dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("L7_STACK_SCRIPTS", "C:/lgs/Lfb47a58376/stack/scripts"))
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)       # the model's slot ticks (config.horizons)
S6_SLOTS = (1, 3, 4, 5, 6, 7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import refb_labels
    man = json.load(open(os.path.join(a.dump, "manifest.json"), encoding="utf-8"))
    hz = tuple(int(h) for h in man["model"]["horizons"])
    assert hz == HORIZONS, f"model horizons {hz} != {HORIZONS}"
    os.makedirs(a.out, exist_ok=True)
    n_tot = n_keep = 0
    for f in sorted(glob.glob(os.path.join(a.dump, "ep*.npz"))):
        with np.load(f, allow_pickle=True) as z:
            ws, ci = np.asarray(z["ws"]), np.asarray(z["clip_index"])
        with np.load(os.path.join(a.dump, "decisions", os.path.basename(f)), allow_pickle=True) as dz:
            fv = np.asarray(dz["gt_future_valid_ext"]) > 0.5
            keep = fv[:, max(hz) - 1]
            n_tot += len(keep)
            if not keep.any():
                continue
            pl = torch.as_tensor(np.asarray(dz["pose_last"])[keep]).float()
            fut = torch.as_tensor(np.asarray(dz["gt_future_ext"])[keep]).float()
        g = refb_labels.waypoint_targets(pl, fut, hz)[:, list(S6_SLOTS)].numpy().astype(np.float32)
        n_keep += int(keep.sum())
        np.savez(os.path.join(a.out, os.path.basename(f)), g=g, ws=ws[keep], clip_index=ci)
    print("[gt6] windows %d, with a valid 6 s future %d -> %s ZZGT6OKZZ" % (n_tot, n_keep, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
