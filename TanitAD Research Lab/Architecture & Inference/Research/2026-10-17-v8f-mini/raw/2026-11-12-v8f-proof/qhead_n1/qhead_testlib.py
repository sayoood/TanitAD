"""Test helpers for card ADD42 (QHEAD): where the sources are (QHEAD_SRC lets the mutation runner point at a mutated copy) and the tiny synthetic fixtures."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("QHEAD_SRC", HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
DISTRULE = os.environ.get("DISTRULE_DIR", os.path.normpath(os.path.join(HERE, "..", "distrule")))


def planted(n_clips=40, T=60, D=16, seed=0, amp=5.0):
    """States with a planted gap and noise that grows with the gap.
    Row layout: every clip has frames 0..T-1; dim 0 = gap/50 * amp (the 'read' of the gap), dim 1 = u (a per-clip noise-scale code NOT in the read), the rest small noise.
    Label gap = gap + 0.1 * gap * u * eta (eta standard normal): the 20 % quantile of the label given the state is gap + 0.1 gap u (-0.8416).
    Returns dict(states, pack, gap_true, u, read)."""
    rng = np.random.RandomState(seed)
    names = np.array(["clip%03d" % i for i in range(n_clips)])
    ci = np.repeat(np.arange(n_clips), T)
    t = np.tile(np.arange(T), n_clips)
    g_c = rng.uniform(20, 60, n_clips)
    u_c = np.where(rng.rand(n_clips) < 0.5, 0.3, 1.0)
    gap_true = g_c[ci] + 0.05 * (t - T / 2.0)
    u = u_c[ci]
    Z = rng.randn(len(t), D).astype(np.float32) * 0.05
    Z[:, 0] = gap_true / 50.0 * amp
    Z[:, 1] = u
    label = gap_true + 0.1 * gap_true * u * rng.randn(len(t))
    has = rng.rand(len(t)) < 0.8
    gap = np.where(has, label, np.nan)
    pres = np.where(has, 1.0, 0.0)
    pres[(~has) & (rng.rand(len(t)) < 0.3)] = 1.0         # presence 1 with a missing gap: not a fit row
    return {"states": {"Z": Z.astype(np.float16), "clip_idx": ci, "t": t, "clips": names}, "pack": {"clips": names, "clip_idx": ci.copy(), "t": t.copy(), "gap": gap, "pres": pres,
            "v": np.zeros(len(t)), "meta": np.array("synthetic")}, "gap_true": gap_true, "u": u, "read": gap_true / 50.0 * amp}


def write_files(d, P, split="train", arm="X_s0"):
    os.makedirs(d, exist_ok=True)
    sp = os.path.join(d, arm + "_train.npz")
    np.savez(sp, **P["states"])
    json.dump({"ckpt_sha256": "ab" * 32, "split": split, "lo": 600, "hi": 1800, "step": 19164}, open(sp + ".json", "w"))
    pp = os.path.join(d, "train_pack.npz")
    np.savez(pp, **P["pack"])
    return sp, pp
