"""Test helpers for the dose step (note PREREG_V8F_PROOF_ADD42_N2 section 3): where the sources are (DOSE_SRC lets the mutation runner point at a mutated copy) and tiny synthetic fixtures."""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("DOSE_SRC", HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)
_GITBASH = "C:/Program Files/Git/usr/bin/bash.exe"
BASH = _GITBASH if os.path.isfile(_GITBASH) else shutil.which("bash")


def sh(p):
    return str(p).replace("\\", "/")


def make_layout(n_clips=8, seed=0, frames=None):
    """clips c000.., frames per clip 6..9, kin [N, 3] float32 with distinct values."""
    rng = np.random.RandomState(seed)
    fr = np.array(frames if frames is not None else rng.randint(6, 10, n_clips), np.int32)
    ci = np.repeat(np.arange(len(fr)), fr).astype(np.int32)
    t = np.concatenate([np.arange(f) for f in fr]).astype(np.int16)
    kin = rng.randn(len(t), 3).astype(np.float32)
    return {"clips": np.array(["c%03d" % i for i in range(len(fr))]), "clip_idx": ci, "t": t, "kin": kin, "frames": fr}


def make_join(layout, seed=1, shuffle=True, extra_clips=2):
    """A label join holding the layout's clips in another order plus `extra_clips` more; per row lead_valid in {0, 1} and lead_gap incl. NaN, negative and above 60 m."""
    rng = np.random.RandomState(seed)
    names = list(layout["clips"]) + ["x%03d" % i for i in range(extra_clips)]
    fr = {c: int(f) for c, f in zip(layout["clips"], layout["frames"])}
    for c in names:
        fr.setdefault(c, int(rng.randint(5, 9)))
    order = list(names)
    if shuffle:
        rng.shuffle(order)
    n = np.array([fr[c] for c in order], np.int64)
    off = np.r_[0, np.cumsum(n)[:-1]].astype(np.int64)
    tot = int(n.sum())
    lv = (rng.rand(tot) < 0.6).astype(np.int8)
    g = rng.uniform(-5, 70, tot).astype(np.float32)
    g[rng.rand(tot) < 0.1] = np.nan
    return {"clip": np.array(order), "off": off, "n": n, "lead_valid": lv, "lead_gap": g}


def expected_pack(layout, join):
    """Row by row in plain python: the rule of leadhead_b (valid, finite, 0 <= gap <= 60)."""
    pos = {c: i for i, c in enumerate(join["clip"])}
    gaps, press = [], []
    for c, t in zip(layout["clip_idx"], layout["t"]):
        j = pos[str(layout["clips"][c])]
        k = int(join["off"][j]) + int(t)
        v = int(join["lead_valid"][k])
        g = float(join["lead_gap"][k])
        ok = v == 1 and np.isfinite(g) and 0.0 <= g <= 60.0
        gaps.append(g if ok else np.nan)
        press.append(1.0 if v == 1 else 0.0)
    return np.array(gaps), np.array(press)


def write_npz(path, d):
    np.savez(path, **d)


def make_states(layout, D=12, seed=2, scale=1.0):
    rng = np.random.RandomState(seed)
    return (rng.randn(len(layout["t"]), D) * scale).astype(np.float16)


def write_states(path, layout, Z, ckpt="a" * 64, lo=0, hi=None, split="train", record=True):
    np.savez(path, Z=Z, clip_idx=layout["clip_idx"], t=layout["t"], clips=layout["clips"])
    if record:
        rec = {"ckpt_sha256": ckpt, "split": split, "lo": lo, "hi": hi if hi is not None else len(layout["clips"]), "n_clips": len(layout["clips"]), "D": int(Z.shape[1]), "n_frames": int(len(Z))}
        with open(str(path) + ".json", "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)


def banked_from_new(layout, Z, lo, hi):
    """The banked-file arrays of clips lo..hi-1 of a new file: rows, clip_idx - lo, t, clips."""
    rows = np.flatnonzero((layout["clip_idx"] >= lo) & (layout["clip_idx"] < hi))
    sub = {"clips": layout["clips"][lo:hi], "clip_idx": (layout["clip_idx"][rows] - lo).astype(np.int32), "t": layout["t"][rows], "kin": layout["kin"][rows],
           "frames": layout["frames"][lo:hi]}
    return rows, sub
