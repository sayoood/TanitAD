#!/usr/bin/env python3
"""Build ``native_tracks.pt``: the raw 10 Hz pose track of every train and eval clip, keyed by clip id.

WHY (PI ruling D-PI-YAWRATE-1, 2026-10-17: *"the model should get the measured yaw rate"*): the S4 pooled-feature cache holds the
0.2 s grid only (``pose_grid`` = poses[2j]), and the measured yaw rate / ``p10_pose``'s kinematics are ONE-FRAME (0.1 s) backward
differences of the heading -- they need the odd frames. MEASURED (``p10_yaw_dt_ablation_540.json``): a P10 fitted on the 0.2 s
grid-derived yaw is 0.032 m worse at 6 s on eval-141 than one on the native track, more than the P10 training-seed floor (0.0213 m).
So ``v8f_mini_data.ClipStore`` / ``ra1_pooled.load_clip`` (``kin_mode="yaw"``) read the native track from this file and REFUSE one that
does not reproduce the cache's own ``pose_grid`` on its even frames.

Sources (both read-only): the train v2 MANIFEST (``train4572_v2manifest.pt``; row ``i`` is raw frame ``i + 2``, so two NaN rows are put
in front -- ``v8f_mini_data.full_from_manifest``) and the eval-141 kinematics file (``eval141_kin.npz``: raw frame-indexed poses).
Output: ``{"tracks": {clip_id: float32 [n_raw, 4]}, "meta": {...}}`` (``torch.load(weights_only=True)``-safe) + ``<out>.json``.
Only POSES are stored: no action channel, so the file cannot reintroduce the steer.

    python build_native_tracks.py --out native_tracks.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tanitad.data import v8f_mini_data as D  # noqa: E402

PKG = os.path.join(REPO, "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini", "raw")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build(train_manifest, eval_kin) -> tuple:
    m = torch.load(train_manifest, weights_only=False)
    tracks = {}
    for c, p in zip(m["clip_id"], m["poses"]):
        P, _ = D.full_from_manifest(p.numpy(), np.zeros((len(p), 2)))        # actions are never read: a zero stand-in
        tracks[str(c)] = torch.as_tensor(P, dtype=torch.float32)
    n_train = len(tracks)
    z = np.load(eval_kin)
    off = np.concatenate([[0], np.cumsum(z["n"])])
    ev = {}
    for i, c in enumerate(z["clip"].tolist()):
        ev[str(c)] = torch.as_tensor(z["poses"][off[i]:off[i + 1]].astype(np.float32))
    both = set(tracks) & set(ev)
    if both:
        raise AssertionError(f"train/eval clip ids overlap: {sorted(both)[:3]} ({len(both)})")
    tracks.update(ev)
    return tracks, n_train, len(ev)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-manifest", default="C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt")
    ap.add_argument("--eval-kin", default=os.path.join(PKG, "eval141_kin.npz"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tracks, n_train, n_eval = build(a.train_manifest, a.eval_kin)
    meta = {"schema": "native-tracks-v1", "n_train": n_train, "n_eval": n_eval, "columns": "x,y,yaw,v",
            "train_rows": "raw frame i+2 of the v2 manifest; rows 0-1 are NaN", "eval_rows": "raw frame-indexed"}
    torch.save({"tracks": tracks, "meta": meta}, a.out)
    rec = dict(meta, out=os.path.abspath(a.out), sha256=file_sha256(a.out), bytes=os.path.getsize(a.out),
               train_manifest=a.train_manifest, train_manifest_sha256=file_sha256(a.train_manifest),
               eval_kin=a.eval_kin, eval_kin_sha256=file_sha256(a.eval_kin))
    json.dump(rec, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
