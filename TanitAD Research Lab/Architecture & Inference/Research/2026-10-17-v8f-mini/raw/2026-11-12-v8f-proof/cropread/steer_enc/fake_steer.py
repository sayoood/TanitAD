#!/usr/bin/env python3
"""Smoke helper only: writes made-up steering files <s-root>/steer/<clip>.npy from the box files of card ADD99 (box direction where the frame
has a lead, steer_encode.NO_LEAD_FILL_DEG elsewhere). The real steering files come from the localiser, not from this tool."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import steer_encode as SE   # noqa: E402
from enc_common import read_ids   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--box-dir", required=True)
    ap.add_argument("--s-root", required=True)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.s_root, "steer"), exist_ok=True)
    for c in read_ids(a.list):
        st, has = SE.box_steering(a.box_dir, c)
        p = SE.steer_path(a.s_root, c)
        if os.path.exists(p):
            print(f"exists, kept: {c[:8]}")
            continue
        np.save(p, st)
        print(f"wrote {c[:8]} n={len(st)} lead={int(has.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
