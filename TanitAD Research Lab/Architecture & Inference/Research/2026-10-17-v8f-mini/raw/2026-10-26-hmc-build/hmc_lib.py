"""hmc_lib.py -- shared loaders for the PREREG_V8F_HACC_MC build (bake, reader, smoke).  CPU only, read-only on every tree it names.

REUSE, NEVER RE-IMPLEMENT: ``comps`` / ``paired`` / ``ROWS`` are AST-extracted from ``ref/t1k_read_repo.py`` and ``lat_rows`` from ``ref/t1klc_read.py``
(both are byte-snapshots of the repo files at 8d4b936d, CR-stripped sha256 identical: ``ref_SHA256.txt``), exactly the way ``t1klc_read.py`` /
``lcdiag/common.py`` exec them.  ``FF`` (four_families) and ``PB`` (the episode-cluster bootstrap) come from READ_ROOT's ``taniteval``.
"""
import ast
import hashlib
import json
import math
import os
import sys

import numpy as np

HMC = "C:/Users/Admin/hmc"
REF = f"{HMC}/ref"
READ_ROOT = os.environ.get("HMC_READ_ROOT", "C:/Users/Admin/t1kc")           # read-only tip export (taniteval at the tip)
ART = "E:/Projects/TanitAD-artifacts/v8f-mini-hacc-mc"                       # large files (bakes) live here
HACC_ART = "E:/Projects/TanitAD-artifacts/v8f-mini-hacc"                     # the shipped HACC checkpoints
BANKED_CAUSAL = f"{HMC}/inputs/banked_causal"                                # git 8d4b936d: raw/2026-10-26-t1kin-causalbucket/run/dumps (float32 consolidated)
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
DT, VMIN = 0.2, 5.2
F = 0.1207
F_HALF = F / 2.0                                                            # 0.06035; the PREREG prints 0.0604 for the TRAIN bar (bar_sensitivity is reported)
F_HALF_PRINTED = 0.0604
SEEDS = (0, 1)

os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", os.path.join(READ_ROOT, "stack").replace("/", os.sep))     # taniteval's stack guard: the stack IS the READ_ROOT's
sys.path.insert(0, os.path.join(READ_ROOT, "taniteval"))
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402

_ns = {"np": np, "torch": torch, "FF": FF, "PB": PB, "math": math, "DT": DT, "VMIN": VMIN}


def _exec_from(src, funcs=(), assigns=()):
    tree = ast.parse(open(src, encoding="utf-8").read())
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef) and node.name in funcs) or \
           (isinstance(node, ast.Assign) and any(getattr(t, "id", "") in assigns for t in node.targets)):
            exec(compile(ast.Module([node], []), src, "exec"), _ns)


_exec_from(f"{REF}/t1k_read_repo.py", funcs=("comps", "paired"), assigns=("ROWS",))
_exec_from(f"{REF}/t1klc_read.py", funcs=("lat_rows",))
comps, paired, ROWS, lat_rows = _ns["comps"], _ns["paired"], _ns["ROWS"], _ns["lat_rows"]
LROWS = ("L1_lat_accel_mae_mps2", "L2_curvature_mae_fast_1pm", "L3_yaw_rate_mae_fast_degps")
DOM_ROWS = ("ade2_m", "speed_mae_mps", "along_mae_m", "accel_mae_mps2", "heading_mae_deg", "cross_mae_m") + LROWS


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(w):
    """sha256 of the float64 C-order bytes of a table (hlat's ``canon``: the pinned identity of a [30, 2] table)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(w, np.float64)).tobytes()).hexdigest()


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def load_run_dumps(run_dir, tag):
    """The per-episode ``ep*.npz`` of one T1 run, concatenated (float64 originals), as ``t1k_read.load`` does."""
    import glob
    fs = sorted(glob.glob(os.path.join(run_dir, f"dump_{tag}", "ep*.npz")))
    assert fs, f"no dumps for {tag} under {run_dir}"
    zs = [np.load(f) for f in fs]
    d = {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "eid", "clip_index", "ws", "v0")}
    d["n_files"] = len(fs)
    return d


def rows_of(X, G):
    """Every row the PREREG reads for one executed track: the nine canonical ``comps`` rows plus L1, L2, L3 (per window [n])."""
    C = comps(X, G)
    C.update({k: v for k, v in lat_rows(X, G).items() if not k.startswith("_")})
    return C


def banked_hacc(s):
    return load_npz(f"{BANKED_CAUSAL}/dumps_HACC_s{s}_m10_causal.npz")


def c6_reader_control():
    """C-6: the reader reproduces HACC's banked L2 (HACC - P10, causal bucket, paired episode-cluster bootstrap B 2000) from the banked consolidated_c dumps.
    PASS bar: +0.0210 / +0.0082, +-1e-4 per seed."""
    want = {0: 0.0210, 1: 0.0082}
    out = {}
    for s in SEEDS:
        d = banked_hacc(s)
        a, b = lat_rows(d["cl"], d["g"])["L2_curvature_mae_fast_1pm"], lat_rows(d["fl"], d["g"])["L2_curvature_mae_fast_1pm"]
        r = paired(a, b, d["eid"])
        out[f"s{s}"] = {**r, "banked": want[s], "abs_diff": abs(r["delta"] - want[s]), "PASS": abs(r["delta"] - want[s]) <= 1e-4}
    out["PASS"] = all(out[f"s{s}"]["PASS"] for s in SEEDS)
    return out


if __name__ == "__main__":
    r = c6_reader_control()
    print(json.dumps(r, indent=1))
