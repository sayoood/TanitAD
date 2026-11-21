"""Shared helpers of the ACTCONS checks: tree on the path, a tiny synthetic world and batch. Every check script runs under the CPU env (CUDA_VISIBLE_DEVICES=-1, one python job at a time)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

AC_ROOT = "F:/TanitAD-scratch/v8proof/actcons"
TMP = f"{AC_ROOT}/work/tmp"
#: the tiny LS-shape checkpoint written by the leadcond checks (30 steps of the synthetic world; stand-in of the LS_s0 checkpoint); read only
BASE_CKPT = "F:/TanitAD-scratch/v8proof/leadcond/work/tmp/base_ls/ckpt_last.pt"


def use_tree(which: str):
    root = f"{AC_ROOT}/work/tree_{which}"
    for p in (f"{root}/stack/tests", f"{root}/stack", f"{root}/taniteval", f"{root}/stack/scripts"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import tanitad
    assert os.path.abspath(tanitad.__file__).lower().replace("\\", "/").startswith(root.lower()), (tanitad.__file__, root)
    return root


def world(tag: str):
    """The tiny synthetic world the unit tests use (hmax 16, lead_sup labels); deterministic. Returns {'batch', 'J'}."""
    import numpy as np
    from tanitad.data import v8f_mini_data as D
    root = Path(f"{TMP}/world_{tag}")
    root.mkdir(parents=True, exist_ok=True)
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    J = D.load_join(Path(root) / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, w["names"]["train"])
    D.synthetic_lead_labels(J, 0)
    ds = D.V8FMiniWindows(store, J, wins, hmax=16, lead_sup=True)
    idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
    batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
    return {"batch": batch, "J": J}


def dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
