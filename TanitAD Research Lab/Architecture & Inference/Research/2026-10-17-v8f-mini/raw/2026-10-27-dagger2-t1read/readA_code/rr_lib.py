"""rr_lib.py -- shared loaders for the PREREG_V8F_DAG_REREAD reads (analysis-only, CPU, zero training).

REUSE, NEVER RE-IMPLEMENT (the same AST-extraction t1klc_read.py / hmc_lib.py use):
  comps / paired / ROWS   <- raw/2026-10-26-t1kin-hacc/t1k_read.py
  lat_rows                <- raw/2026-10-26-t1-latcond/t1klc_read.py
  pair_err                <- raw/2026-10-26-hmc-build/ref/lcdiag/common.py   (M1's pair speeds)
All three sources are the files committed at bdf8d561 (exported by `git archive`, CR-preserved), their sha256 recorded in raw/INPUT_SHA256.txt.
The instruments (taniteval.four_families / ci) come from the tool tree E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a, whose key modules were
verified CR-normalised identical to bdf8d561 (raw/TREE_PARITY.txt).
"""
import ast
import glob
import hashlib
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

PKG = "E:/Projects/TanitAD-artifacts/dagreread"
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
RD = os.environ.get("RR_READERS", "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/rd")
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", os.path.join(TREE, "stack").replace("/", os.sep))
sys.path.insert(0, os.path.join(TREE, "taniteval"))
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

DT, VMIN, F, TWO_F = 0.2, 5.2, 0.1207, 0.2415
SEEDS = (0, 1)
_ns = {"np": np, "torch": torch, "FF": FF, "PB": PB, "math": math, "DT": DT, "VMIN": VMIN}


def _exec_from(src, funcs=(), assigns=()):
    tree = ast.parse(open(src, encoding="utf-8").read())
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef) and node.name in funcs) or \
           (isinstance(node, ast.Assign) and any(getattr(t, "id", "") in assigns for t in node.targets)):
            exec(compile(ast.Module([node], []), src, "exec"), _ns)


_exec_from(f"{RD}/t1k_read.py", funcs=("comps", "paired"), assigns=("ROWS",))
_exec_from(f"{RD}/t1klc_read.py", funcs=("lat_rows",))
_exec_from(f"{RD}/lcdiag_common.py", funcs=("pair_err",))
comps, paired, ROWS, lat_rows, pair_err = _ns["comps"], _ns["paired"], _ns["ROWS"], _ns["lat_rows"], _ns["pair_err"]
LROWS = ("L1_lat_accel_mae_mps2", "L2_curvature_mae_fast_1pm", "L3_yaw_rate_mae_fast_degps")
KEYS = ("g", "cl", "ol", "fl", "ha0", "v0", "eid", "clip_index", "ws")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_dir(d):
    """The per-episode ep*.npz of one T1 run, concatenated in sorted order (as t1k_read.load does)."""
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == 141, (d, len(fs))
    zs = [np.load(f) for f in fs]
    out = {}
    for k in KEYS:
        if k in zs[0].files:
            out[k] = np.concatenate([z[k] for z in zs])
    return out


def load_npz(p):
    z = np.load(p)
    return {k: z[k] for k in z.files}


def pack(src_dir, dst):
    d = load_dir(src_dir)
    np.savez_compressed(dst, **d)
    return sha256_file(dst)


def rows_of(X, G):
    """Per-window [n] rows of one executed track: the nine canonical comps rows + L1, L2, L3."""
    C = comps(X, G)
    C.update({k: v for k, v in lat_rows(X, G).items() if not k.startswith("_")})
    return C


def m1_window(X, G):
    """M1 as a per-window share (PREREG_V8F_DAG_REREAD section 2: 'a paired per-window share'): per window, the GT-fast pairs
    (hmc_controls.m_rows: both pair-valid, GT pair speed >= 5.2) whose OWN pair speed is < 5.2, divided by the window's GT-fast pairs.
    NaN where the window has no GT-fast pair.  Also returns the pooled M1 (hmc_controls.m_rows definition) for the reproduction control."""
    pa = pair_err(X, G)
    slow = pa["fast"] & (pa["vX"] < VMIN)
    c = pa["fast"].sum(1)
    share = np.where(c > 0, slow.sum(1) / np.maximum(c, 1), np.nan)
    return share, float(slow.sum() / pa["fast"].sum()), int(slow.sum()), int(pa["fast"].sum())


def sepworse(r, higher_is_worse=True):
    return bool(r["separated"] and ((r["delta"] > 0) if higher_is_worse else (r["delta"] < 0)))
