#!/usr/bin/env python3
"""Shared loaders for the aux-drivable navtrain -> navtest shift location (2026-10-16). READ-ONLY on every shared artifact.

Run with:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I <script>.py
(-I ignores PYTHONPATH, so the repo package is put on sys.path here, explicitly and read-only.)

Everything here is a thin reuse of the programme's OWN code (eval/aux_s1_readout.py, refe/aux_drivable_head.py): the point is to re-derive the banked P1 numbers
with the programme's own functions first (REPRO), and only then decompose them.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# <PK>/raw/2026-10-16-aux-shift-locate -> <PK>
PK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PK, "eval"))
sys.path.insert(0, os.path.join(PK, "refe"))
import torch  # noqa: E402

torch.set_num_threads(2)
try:                                              # low CPU priority: the dev box is scoring a P2 table on 6 workers
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL_PRIORITY_CLASS
except Exception:
    pass
import aux_drivable_head as AUX  # noqa: E402
import aux_s1_readout as RO  # noqa: E402

ART = "D:/Projects/TanitAD-artifacts"
S1CACHE = f"{ART}/s1_read_cache"
AUXDIR = f"{ART}/aux_drivable"
CTX_DIR = f"{ART}/aux_navtest_forward/c3_full_ctx"
C3_TOKENS = os.path.join(PK, "raw", "2026-10-11-c3-selection-headroom", "tokens.json")
HELDOUT_LOGS = f"{ART}/_tlane_work/ho/refe_heldout/heldout_logs.txt"
SCRATCH = os.environ.get("SHF_SCRATCH", "C:/Users/Admin/AppData/Local/Temp/shf_scratch")
os.makedirs(SCRATCH, exist_ok=True)
ARMS = ["H1-s0", "H1-s1", "H1-s2", "H2-s0", "H2-s1", "H2-s2"]
NCX, NCY = RO.NCX, RO.NCY


def jdump(obj, path):
    def _d(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_d)


def open_train_labels():
    return RO.open_labels("train")


def open_navtest_labels():
    return RO.open_labels("navtest")


def heldout_logs():
    return [l.strip() for l in open(HELDOUT_LOGS, encoding="utf-8") if l.strip()]


def c3():
    return json.load(open(C3_TOKENS, encoding="utf-8"))


def load_heldout_pred(arm: str, which: str = "update400") -> dict:
    return RO.load_head_pred(f"{S1CACHE}/arms/{arm}/dump/head_pred_{which}.npz")


def navtest_pred_path(arm: str) -> str:
    return f"{SCRATCH}/navtest_head_pred_{arm}.npz"


def regenerate_navtest_pred(arm: str, limit: int | None = None, log=print) -> str:
    """Re-run the arm's head (aux_drv.pt, 400 updates) on the banked fp32 planner ctx of the 1,058 C3 tokens (CPU). This is exactly what
    eval/aux_navtest_forward.py's cache pass does (apply_head on cache_load), minus the trunk (the ctx is read from disk)."""
    import time
    sys.path.insert(0, os.path.join(PK, "eval"))
    import aux_navtest_forward as FW
    out = navtest_pred_path(arm)
    if os.path.exists(out) and limit is None:
        return out
    head, info = FW.load_aux_head(f"{S1CACHE}/arms/{arm}/aux_drv.pt")
    tj = c3()
    toks = tj["tokens"] if limit is None else tj["tokens"][:limit]
    logit = np.zeros((len(toks), NCX, NCY), np.float32)
    sdf = np.zeros_like(logit)
    t0 = time.time()
    for i, t in enumerate(toks):
        ctx = torch.from_numpy(np.load(FW.cache_path(CTX_DIR, t)))[None]
        lg, sd = FW.apply_head(head, ctx, autocast=False)
        logit[i], sdf[i] = lg[0], sd[0]
        if (i + 1) % 100 == 0:
            log(f"  {arm}: {i + 1}/{len(toks)}  {time.time() - t0:.0f}s")
    if limit is None:
        FW.write_head_pred(out, [tj["token_log"][t] for t in toks], toks, logit, sdf, {"arm": arm, "head": info["sha256"], "ctx": "fp32 planner cache"})
    return out


def load_navtest_pred(arm: str) -> dict:
    return RO.load_head_pred(navtest_pred_path(arm))
