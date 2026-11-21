"""v3_lib.py -- lean library of the v8F-mini evidence video v3 (HACC re-render).  CPU only, read-only on every input except its own OUT.

v2_lib loaded the model code to run the nav counterfactual; v3 re-uses v2's banked counterfactual arrays (data/cf_windows.npz, unchanged copy) and draws the HACC
plans from the banked per-window arrays, so no model is imported here.  Only the frame accessor needs torch (decode of the cached PNG frames).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
OUT = f"{SC}/viz3"
V1 = f"{SC}/viz"
V2 = f"{SC}/viz2"
HACC = f"{SC}/hacc"
HBLEND = f"{SC}/hblend"
HSPEC = f"{SC}/hspec"
REBUILD = "C:/Users/Admin/tanitad-data/refav1-eval141-256rb"

SEL = json.load(open(f"{V1}/out/selection.json"))
KEYS = ["JUNCTION", "CURVE", "NO-TURN", "FAILURE(worst)"]
WINDOWS = {k: SEL[k] for k in KEYS}
SEEDS = ("M2b_s0", "M2b_s1")
LAT5 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHG_L", "LANE_CHG_R")
LON7 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECEL", "ACCEL", "KEEP")
CMDS = ("FOLLOW", "TURN_L", "TURN_R")
CONFIGS = {"a_all": ("S", "T", "O"), "b_O": ("O",), "c_T": ("T",), "d_S": ("S",)}

# ------------------------------------------------------------------------------------------------ frames (v1 / v2 access, unchanged)
_FR: dict = {}


def v2ep_frame(clip: str, j: int) -> np.ndarray:
    """cache step j = native frame 2j of the v2ep episode (v1 `render_video.load_clip`); [256, 640, 3] uint8."""
    import torch
    import torchvision
    if clip not in _FR:
        torch.set_num_threads(1)
        o = torch.load(f"{REBUILD}/{clip}.v2ep.pt", map_location="cpu", weights_only=False)
        buf, lens = o["jpeg_buf"], o["jpeg_len"]
        offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
        _FR[clip] = (buf, offs)
    buf, offs = _FR[clip]
    return torchvision.io.decode_png(buf[offs[2 * j]:offs[2 * j + 1]].clone()).permute(1, 2, 0).numpy()


# ------------------------------------------------------------------------------------------------ source registry (binding number checks)
SOURCES = {
    "hacc_read": f"{HACC}/READ.json",
    "hacc_score_s0": f"{HACC}/score_M2b_s0.json",
    "hacc_score_s1": f"{HACC}/score_M2b_s1.json",
    "hacc_accel": f"{HACC}/accel_profile.json",
    "hacc_dk": f"{HACC}/dk_read.json",
    "hacc_w_s0": f"{HACC}/weights_HACC_M2b_s0.json",
    "hacc_w_s1": f"{HACC}/weights_HACC_M2b_s1.json",
    "hblend_w_s0": f"{HBLEND}/weights_M2b_s0.json",
    "hblend_w_s1": f"{HBLEND}/weights_M2b_s1.json",
    "hspec_s0": f"{HSPEC}/score_hs2_M2b_s0.json",
    "hspec_s1": f"{HSPEC}/score_hs2_M2b_s1.json",
    "sfresh": f"{SC}/sfresh/sfresh_numbers.json",
    "sfresh_extra": f"{SC}/sfresh/sfresh_extra.json",
    "tables_v2": f"{OUT}/data/tables.json",
    "cf_evalwide": f"{OUT}/data/cf_evalwide.json",
    "aux3": f"{OUT}/data/aux3.json",
}
_JC: dict = {}


def J(key: str):
    if key not in _JC:
        _JC[key] = json.load(open(SOURCES[key]))
    return _JC[key]


def getp(obj, ptr: str):
    """pointer 'a/b/0/c' into nested dict / list."""
    for k in ptr.split("/"):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return obj


REG: list = []      # every registered on-screen number of the scene being drawn: {src, ptr, spec, shown}
CUR = {"scene": ""}


def N(src: str, ptr: str, spec: str = "{:.4f}", scale: float = 1.0) -> str:
    """Return the formatted value of source `src` at `ptr` AND register it for the binding control (the control re-reads the file in another process,
    re-formats with `spec` and requires equality with `shown`, and that `shown` occurs in the scene's drawn text)."""
    v = getp(J(src), ptr) * scale
    s = spec.format(v)
    REG.append({"scene": CUR["scene"], "src": src, "ptr": ptr, "spec": spec, "scale": scale, "shown": s})
    return s


def NV(src: str, ptr: str) -> float:
    return float(getp(J(src), ptr))


def ND(src: str, ptr_a: str, ptr_b: str, spec: str = "{:+.3f}") -> str:
    """formatted (value at ptr_a) - (value at ptr_b) of one source, registered for the control (kind 'diff')."""
    v = getp(J(src), ptr_a) - getp(J(src), ptr_b)
    s = spec.format(v)
    REG.append({"scene": CUR["scene"], "src": src, "ptr": ptr_a, "ptr2": ptr_b, "spec": spec, "scale": 1.0, "shown": s})
    return s
