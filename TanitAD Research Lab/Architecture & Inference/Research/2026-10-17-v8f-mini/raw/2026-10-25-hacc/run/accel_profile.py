#!/usr/bin/env python3
"""accel_profile.py (HACC copy) -- the per-step acceleration instrument, extended to HACC. The original is `hspec/accel_profile.py` (untouched).

DESCRIPTIVE (no bar, no arm). Per 0.2 s accel cell k = 1..29: mean |a_plan - a_gt| over windows, for any set of named plans (HACC, H, P10, B).

Definition (the family's, as written in test_v8f_oacc.py's oracle): speed_k = |p_k - p_(k-1)| / 0.2 with p_0 = the origin (the ego at NOW),
accel_k = (speed_(k+1) - speed_k) / 0.2.  The family computes in FLOAT32 (original script's amendment: float64 read 5.7e-5 .. 9.3e-5 off the banked values;
float32 reproduces them to 9.5e-7 (H) / 4.8e-7 (P10)), so this does too, and the control bar is 2e-6.

PREREG_V8F_HACC control 7: the instrument reproduces the banked `fw_lon_accel_mae` in float32 to 2e-6 (H and P10 against the banked per-window arrays).

Importable with numpy alone (the tests do); `main()` is a stand-alone re-profile from the per-window plans HACC banked (needs the eval world, so it lazily imports it).
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402

DT = 0.2
CONTROL_BAR = 2e-6
# the bands the PREREG's "Why" table names (cell k is at t = 0.2 k s): inclusive cell ranges
BANDS = {"0.8-1.6 s (cells 4-8)": (4, 8), "2.2-2.6 s (cells 11-13)": (11, 13), "2.8-5.0 s (cells 14-25)": (14, 25), "5.2-5.8 s (cells 26-29)": (26, 29)}


def accel(p) -> np.ndarray:
    """[N, 30, 2] plan -> [N, 29] acceleration cells, float32 (the family's arithmetic)."""
    p = np.asarray(p).astype(np.float32)
    q = np.concatenate([np.zeros_like(p[:, :1]), p], 1)
    v = np.linalg.norm(np.diff(q, axis=1), axis=-1) / np.float32(DT)
    return np.diff(v, axis=1) / np.float32(DT)


def err_cells(plan, gt) -> np.ndarray:
    """[N, 29] |a_plan - a_gt| in float64 (computed in float32, widened)."""
    return np.abs(accel(plan) - accel(gt)).astype(np.float64)


def window_mae(plan, gt) -> np.ndarray:
    """[N] per-window mean over the 29 cells -- what must equal the banked per-window `fw_lon_accel_mae`."""
    return np.abs(accel(plan) - accel(gt)).mean(1).astype(np.float64)


def control(plan, gt, banked_per_window) -> float:
    """max |instrument - banked| over the windows where the banked value is finite."""
    ref = np.asarray(banked_per_window, np.float64)
    ok = np.isfinite(ref)
    return float(np.abs(window_mae(plan, gt)[ok] - ref[ok]).max())


def profile(plans: dict, gt, ref_name: str = "P10") -> dict:
    """plans: {name: [N, 30, 2]} -> {'cells': {name: [29]}, 'minus_ref': {name: [29]}, 'bands': {band: {name: mean over the band's cells, name_minus_ref}}}."""
    cells = {n: err_cells(p, gt).mean(0) for n, p in plans.items()}
    ref = cells[ref_name]
    out = {"t_s": [round(0.2 * (k + 1), 1) for k in range(29)], "cells": {n: c.tolist() for n, c in cells.items()},
           "minus_" + ref_name: {n: (c - ref).tolist() for n, c in cells.items() if n != ref_name}, "bands": {}}
    for bn, (k0, k1) in BANDS.items():
        sl = slice(k0 - 1, k1)
        out["bands"][bn] = {n: float(c[sl].mean()) for n, c in cells.items()}
        out["bands"][bn].update({f"{n}_minus_{ref_name}": float((c - ref)[sl].mean()) for n, c in cells.items() if n != ref_name})
    out["all_cells_mean"] = {n: float(c.mean()) for n, c in cells.items()}
    return out


def main() -> int:
    """Stand-alone: re-profile the banked per-window plans of one HACC out dir (default: this dir) and print the band table."""
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = sys.argv[1] if len(sys.argv) > 1 else here
    sys.path.insert(0, here)
    import v8f_hacc as A  # noqa: E402
    res = {}
    for seed in A.SEEDS:
        z = np.load(f"{out_dir}/score_{seed}_perwindow.npz")
        W = A.S.load_world(seed)
        plans = {"HACC": z["HACC"].astype(np.float64), "H": W["H"], "P10": W["P_file"], "B": W["tau"]}
        res[seed] = profile(plans, W["gt"])
    A.H.jdump(res, f"{out_dir}/accel_profile_restandalone.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
