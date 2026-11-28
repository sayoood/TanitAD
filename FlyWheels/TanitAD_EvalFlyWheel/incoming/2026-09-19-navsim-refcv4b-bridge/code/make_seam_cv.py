#!/usr/bin/env python3
"""K4 seam-transparency control: a seam whose EVERY row is the devkit CV agent's own
poses (exported per token by export_agent_inputs.py, `cv_poses`). Scored through the
seam agent, it must reproduce the official `constant_velocity_agent` run EXACTLY —
which proves token keying, the float32 round trip and the TrajectorySampling are
transparent. Any venv (numpy only).  ->  raw/seam_K4_seam_cv.npz
"""
import json
import os
import sys

import numpy as np

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")


def main() -> int:
    doc = json.load(open(os.path.join(RAW, "navsim_agent_inputs.json"), encoding="utf-8"))
    toks = sorted(doc["tokens"])
    if any("cv_poses" not in doc["tokens"][t] for t in toks):
        sys.exit("⛔ export has no cv_poses — re-run export_agent_inputs.py")
    poses = np.asarray([doc["tokens"][t]["cv_poses"] for t in toks], dtype=np.float32)
    np.savez(os.path.join(RAW, "seam_K4_seam_cv.npz"), token=np.asarray(toks),
             fingerprint=np.asarray([doc["tokens"][t]["fingerprint"] for t in toks]),
             source=np.asarray(["precomputed"] * len(toks)), poses=poses,
             knots=np.full((len(toks), 8, 2), np.nan, np.float32),
             sampling=np.asarray([8, 0.5]), arm=np.asarray("K4_seam_cv"))
    print(f"seam_K4_seam_cv: {len(toks)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
