"""Shared helpers of the LEADCOND checks: tree on the path, a tiny synthetic world and batch, model builders. Usage: `import lc_common as C; C.use_tree("patch")` BEFORE importing tanitad.
Every check script of this folder runs under the CPU env (CUDA_VISIBLE_DEVICES=-1, one python job at a time)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LC = "F:/TanitAD-scratch/v8proof/leadcond"
TMP = f"{LC}/work/tmp"
BASE_CKPT = f"{TMP}/base_ls/ckpt_last.pt"          # tiny LS-shape checkpoint written by tree_base (30 steps of the synthetic world; stand-in of LS_s0's ckpt_last.pt)


def use_tree(which: str):
    root = f"{LC}/work/tree_{which}"
    for p in (f"{root}/stack", f"{root}/taniteval", f"{root}/stack/scripts"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tanitad
    assert os.path.abspath(tanitad.__file__).lower().replace("\\", "/").startswith(root.lower()), (tanitad.__file__, root)
    return root


def world_batch(tag: str, hmax: int = 16, n: int = 8):
    """The tiny synthetic world and a lead-sup batch (the same recipe as the test fixture); deterministic. Returns (batch, J)."""
    import numpy as np
    from tanitad.data import v8f_mini_data as D
    root = Path(f"{TMP}/world_{tag}")
    root.mkdir(parents=True, exist_ok=True)
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    D.synthetic_lead_labels(J, 0)
    ds = D.V8FMiniWindows(store, J, wins, hmax=hmax, lead_sup=True)
    idx = np.random.default_rng(0).choice(len(wins), n, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    return batch, J


def build_from_args(extra, J, d: int = 32, seed: int = 0, hmax: int = 10):
    """A tiny model built the way the trainer builds it (argv -> make_config), global seed set before the build."""
    import torch
    import v8f_mini_train as TR
    from tanitad.models import v8f_mini as M
    a = TR.build_parser().parse_args(["--out", "x", "--d", str(d), "--n-q", "2", "--arm", "M2b", "--wm-state", "grid", "--wm-cell-dim", "16", *extra])
    c0 = TR.make_config(a)
    cfg = M.V8FMiniConfig(**{**c0.__dict__, "d": d, "ff": 2 * d, "heads": 4, "depth_s": 1, "depth_t": 1, "depth_o": 1})
    mu, sd = M.cons_stats(J)
    torch.manual_seed(seed)
    return a, cfg, M.V8FMini(cfg, M.P10Prior.synthetic(0), mu, sd)


def dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
