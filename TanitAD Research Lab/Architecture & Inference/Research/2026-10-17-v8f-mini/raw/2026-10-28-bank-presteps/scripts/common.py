"""Shared loaders (numpy only)."""
import json
from pathlib import Path
import numpy as np
PKG = Path("E:/Projects/TanitAD-artifacts/dagland/bank_presteps")
RAW = PKG / "raw"
FROZEN_SHA = open(PKG / "FROZEN_CARDS.sha256").read().split()[0]
AMEND_SHA = open(PKG / "FROZEN_AMENDMENT_1.sha256").read().split()[0]
OK, OFF_LAT, OFF_HEAD, END_EDGE, NONFINITE = range(5)
REASONS = ("ok", "off_lat", "off_head", "end_edge", "nonfinite")
DAGGER_N = {"s0": 14688, "s1": 14839}      # v8f-dagoacc/M2b_s{0,1}_dagoacc/summary.json dagger_n (MEASURED)
N_WINDOWS = 306566

def load_table(seed):
    z = np.load(RAW / f"bank_table_{seed}.npz")
    return {k: z[k] for k in z.files}

def load_logged():
    z = np.load(RAW / "logged_cache.npz")
    return {k: z[k] for k in z.files}

def attach_logged(tab, lg):
    """Add per-record: cidx (index into the logged cache), T of the clip, v_gt_r (logged speed at the MATCHED row r'), and helpers."""
    names = lg["names"].astype(str)
    tn = tab["clip_names"].astype(str)
    cmap = np.searchsorted(names, tn)
    assert (names[cmap] == tn).all()
    cidx = cmap[tab["clip_g"]]
    tab["cidx"] = cidx
    tab["T"] = lg["Ts"][cidx]
    r = tab["r"].astype(np.int64)
    okr = (r >= 0) & (r < tab["T"])
    vg = np.full(len(r), np.nan)
    vg[okr] = lg["V"][lg["offs"][cidx[okr]] + r[okr]]
    tab["v_gt_r"] = vg
    return tab

def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)
