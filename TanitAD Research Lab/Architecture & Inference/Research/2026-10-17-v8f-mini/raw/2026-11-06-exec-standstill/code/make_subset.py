#!/usr/bin/env python3
"""make_subset.py -- the gate subset: a grid json (same format as eval_grid_windows.json) holding a class-enriched subset of the 2,399 eval-141 windows.

Classes are gapdecomp.classify's (L launch, S stay, O other-stopped, M moving) computed on the banked C0_s0 array (v0, g). Selection is deterministic:
all L with v0 < 0.5, N_LR evenly spaced L with v0 >= 0.5, the first N_S of S, the first N_O of O (by window index), and N_M moving windows evenly spaced over the M windows. Writes
  <out>/grid_subset.json     the grid (order + windows per clip), in the full grid's order
  <out>/subset_idx.json      the window indices into the full 2,399-window list, the class of each, and the sha256 of the full grid json
usage: python -I make_subset.py --out C:/Users/Admin/exec_standstill_scratch/subset
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
GRID = f"{DATA}/eval_grid_windows.json"
JOIN = f"{DATA}/join_eval141.npz"
GD_PATH = f"{ART}/dagland/gapdecomp/gapdecomp.py"
BANKED_C0_S0 = f"{ART}/dagland/recover_next/read/raw/arrays/C0_s0.npz"


def load_gd():
    spec = importlib.util.spec_from_file_location("gapdecomp", GD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-s", type=int, default=10)
    ap.add_argument("--n-o", type=int, default=10)
    ap.add_argument("--n-m", type=int, default=28)
    ap.add_argument("--n-lr", type=int, default=16)
    a = ap.parse_args(argv)
    gd = load_gd()
    z = np.load(BANKED_C0_S0, allow_pickle=True)
    cls, _ = gd.classify(z["v0"], z["g"])
    names = gd.CLASSES
    idx = {c: np.where(cls == i)[0] for i, c in enumerate(names)}
    v0 = np.asarray(z["v0"], np.float64)
    L = idx["L"]
    L_stop = L[v0[L] < 0.5]                       # launches from (near) standstill: the windows the executor artifact lives on
    L_run = L[v0[L] >= 0.5]                       # rolling launches: sampled
    pick = list(L_stop) + list(L_run[np.linspace(0, len(L_run) - 1, a.n_lr).round().astype(int)]) + list(idx["S"][:a.n_s]) + list(idx["O"][:a.n_o])
    m = idx["M"]
    pick += list(m[np.linspace(0, len(m) - 1, a.n_m).round().astype(int)])
    pick = sorted(set(int(i) for i in pick))
    J = np.load(JOIN, allow_pickle=True)
    clips = [str(c) for c in J["clip"]]
    ci, ws = z["clip_index"], z["ws"]
    g = json.load(open(GRID, encoding="utf-8"))
    flat = [(c, int(t)) for c in g["order"] for t in g["windows"][c]]
    assert len(flat) == len(ci) == 2399
    for i in range(len(flat)):          # the banked array and the grid walk the SAME window order
        assert clips[int(ci[i])] == flat[i][0] and int(ws[i]) == flat[i][1], i
    sub = {"order": [], "windows": {}}
    for i in pick:
        c, t = flat[i]
        if c not in sub["windows"]:
            sub["order"].append(c)
            sub["windows"][c] = []
        sub["windows"][c].append(t)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/grid_subset.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sub, f)
    info = {"n_windows": len(pick), "idx": pick, "class": [names[int(cls[i])] for i in pick], "class_counts_full": {c: int(len(idx[c])) for c in names},
            "class_counts_subset": {c: int(sum(1 for i in pick if names[int(cls[i])] == c)) for c in names},
            "full_grid_sha256": hashlib.sha256(open(GRID, "rb").read()).hexdigest(), "banked_ref": BANKED_C0_S0}
    with open(f"{a.out}/subset_idx.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "idx"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
