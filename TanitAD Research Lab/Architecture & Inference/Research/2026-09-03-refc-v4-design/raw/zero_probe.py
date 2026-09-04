# -*- coding: utf-8 -*-
"""Second, independently-sampled probe of the X15 zero-collision base rate.

The 4.4531 % figure the X15 argument rests on is MEASURED by a sibling package
(`2026-09-03-ego-zero-collision/`) over 781,635 TRAIN windows. For this package
it is INHERITED, and it is load-bearing (it is the whole reason
`ego_valid_channel` is a precondition rather than an option). This re-measures
the exact-zero rate of every ego channel on a DIFFERENT sample -- the 40 val
episodes -- so the claim rests on two probes with different path binding.

Frames are read and discarded one episode at a time; only poses/actions are kept.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = sys.argv[1]
OUT = sys.argv[2]
WHEELBASE = 2.9

files = sorted(glob.glob(os.path.join(ROOT, "ep_*.pt")))
V, A, K = [], [], []
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    V.append(d["poses"][:, 3].numpy().astype(np.float64))
    A.append(d["actions"][:, 1].numpy().astype(np.float64))
    K.append(np.tan(d["actions"][:, 0].numpy().astype(np.float64)) / WHEELBASE)
    del d
v = np.concatenate(V); a = np.concatenate(A); k = np.concatenate(K)
n = int(len(v))
rep = {
    "_provenance": {"root": ROOT, "n_episodes": len(files), "n_frames": n,
                    "evidence_class": "MEASURED",
                    "_purpose": ("second probe, different sample, of the X15 "
                                 "exact-zero base rates the sibling package "
                                 "measured on 781,635 TRAIN windows")},
    "exact_zero_frac": {
        "v0": float((v == 0.0).mean()),
        "a_long": float((a == 0.0).mean()),
        "curvature": float((k == 0.0).mean()),
        "yaw_rate": float(((v * k) == 0.0).mean()),
    },
    "near_zero_frac": {
        "v0_lt_0p5": float((v < 0.5).mean()),
        "a_long_abs_lt_0p05": float((np.abs(a) < 0.05).mean()),
    },
    "atom_check_v0": {
        "_reads": ("the [0, 0.1) bin against the next bin -- a hard atom at "
                   "exactly 0 is what makes a withheld zero indistinguishable "
                   "from a genuine standstill"),
        "bin_0_0p1": float(((v >= 0.0) & (v < 0.1)).mean()),
        "bin_0p1_0p2": float(((v >= 0.1) & (v < 0.2)).mean()),
    },
}
b0, b1 = rep["atom_check_v0"]["bin_0_0p1"], rep["atom_check_v0"]["bin_0p1_0p2"]
rep["atom_check_v0"]["ratio"] = float(b0 / b1) if b1 > 0 else None
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(rep, fh, indent=1)
print(json.dumps(rep, indent=1))
