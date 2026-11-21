#!/usr/bin/env python3
"""STOP control — POST-HOC DIAGNOSTIC (not a bar). Any venv (numpy only).

WHY: the vision-pure arm A2 collapses to a near-standstill plan (202/204 stage-2 scenes
move < 1 m in 4 s) and out-scores every moving arm on S2-EPDMS-u. The clean question is
therefore "what does DOING NOTHING score on this protocol?". The STOP plan is the
all-zero trajectory (x = y = heading = 0 at every NavSim time) for all 220 tokens; it
reads no input at all, so it also runs stage 1 and its OFFICIAL two-stage EPDMS exists.
-> raw/seam_STOP_zero.npz
"""
import json
import os
import sys

import numpy as np

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")


def main() -> int:
    doc = json.load(open(os.path.join(RAW, "navsim_agent_inputs.json"), encoding="utf-8"))
    toks = sorted(doc["tokens"])
    np.savez(os.path.join(RAW, "seam_STOP_zero.npz"), token=np.asarray(toks),
             fingerprint=np.asarray([doc["tokens"][t]["fingerprint"] for t in toks]),
             source=np.asarray(["precomputed"] * len(toks)),
             poses=np.zeros((len(toks), 8, 3), np.float32),
             knots=np.full((len(toks), 8, 2), np.nan, np.float32),
             sampling=np.asarray([8, 0.5]), arm=np.asarray("STOP_zero"))
    print(f"seam_STOP_zero: {len(toks)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
