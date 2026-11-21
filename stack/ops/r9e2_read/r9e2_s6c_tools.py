#!/usr/bin/env python3
"""r9e2_s6c_tools.py -- the one tiny helper r9e2_roll.sh needs around the MAPSEL rolls.  NEW file (E2-S6C-WIRE, 2026-11-08).  No torch, no model: numpy + json only.

    python r9e2_s6c_tools.py final <trace.npz> <DEV_HOLD_SPLIT.json>
        rc 0 iff the trace record is FINAL and holds exactly the DEV split's own S2 window count (``DEV.n_windows_s2``) with NO window of a HOLD clip's count to spare;
        rc 1 otherwise (partial / wrong count / unreadable).  Prints ``n_windows`` on rc 0, ``BAD <why>`` on rc 1.  Used (a) to skip a roll that is already complete and
        (b) to refuse a roll whose trace could carry anything but the DEV windows.  The roll itself already restricts to DEV (mapsel_roll_r9_e2.dev_windows_arg); this is the
        independent count check on its output.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def check_final(trace, split) -> tuple:
    try:
        z = np.load(trace, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        n = int(len(z["ws"]))
        want = int(json.loads(open(split, encoding="utf-8").read())["DEV"]["n_windows_s2"])
    except Exception as e:                                                  # noqa: BLE001 -- any unreadable input is a BAD, never a pass
        return False, f"unreadable ({type(e).__name__}: {e})"
    if meta.get("final") is not True:
        return False, "not final"
    if n != want or int(meta.get("n_windows", -1)) != want:
        return False, f"{n} windows (meta {meta.get('n_windows')}) != the DEV split's {want}"
    return True, str(n)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 3 or a[0] != "final":
        print("usage: r9e2_s6c_tools.py final <trace.npz> <DEV_HOLD_SPLIT.json>", file=sys.stderr)
        return 2
    ok, msg = check_final(a[1], a[2])
    print(msg if ok else "BAD " + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
