#!/usr/bin/env python3
"""D-VT1 step 6 — mint the leak-guarded target-speed label for the PARITY TRAIN
corpus, on Thor. Runs THERE; ships back a small npz.

⛔ PARITY IS UNTOUCHED. Read-only over
`/home/nvidia/epcache/epcache-256px-phase0/physicalai-train-e438721ae894`. No
episode is selected, deselected, reordered or rewritten; the file list is hashed
so the exact set this ran over is quotable. Only `poses` is read
(`weights_only=True`, `mmap=True`) — `frames_u8` is never touched.

⭐ WHY THE OUTPUT IS A FULL PER-POSE TRACK, NOT A WINDOW TABLE. A window grid is a
property of the trainer (`stride`, `WINDOW`, `K_MAX`), not of the label. Emitting
`vt[t]` for every pose index lets any grid — stride 5, stride 8, a future one —
index straight in, and removes the class of bug where a label table was built on
one grid and silently read on another.

⚠️ The mint is imported from `stack/tanitad/lake/vtarget.py`, shipped verbatim
beside this script, so train and val labels come from the SAME code. A re-
implementation here would be exactly the drift this programme keeps retracting.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vtarget import (VT_GUARD_STEPS, VT_MIN_LOOKAHEAD, vtarget_guarded,  # noqa: E402
                     vtarget_v2)

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/home/nvidia/epcache/epcache-256px-phase0/physicalai-train-e438721ae894")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/train_vtarget_guarded.npz")

eps = sorted(SRC.glob("ep_*.pt"))
listing = hashlib.sha256("\n".join(p.name for p in eps).encode()).hexdigest()
print(f"{len(eps)} episodes under {SRC}\nfile-list sha256 {listing}", flush=True)

blob, meta, t0 = {}, [], time.time()
n_valid = n_valid_oracle = n_pose = 0
for i, p in enumerate(eps):
    d = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    poses = d["poses"].to(torch.float64).numpy()
    v = poses[:, 3]
    t_len = v.shape[0]
    allt = np.arange(t_len)
    vt_g, ok_g, look_g, _ = vtarget_guarded(v, allt, guard_steps=VT_GUARD_STEPS,
                                            min_lookahead=VT_MIN_LOOKAHEAD)
    vt_o, ok_o, _look, _ = vtarget_v2(v, allt, min_lookahead=VT_MIN_LOOKAHEAD)
    k = p.stem
    blob[f"{k}__vt_guarded"] = vt_g.astype(np.float32)
    blob[f"{k}__valid_guarded"] = ok_g
    blob[f"{k}__lookahead_guarded"] = look_g.astype(np.int16)
    blob[f"{k}__vt_oracle"] = vt_o.astype(np.float32)
    blob[f"{k}__valid_oracle"] = ok_o
    blob[f"{k}__v"] = v.astype(np.float32)
    n_valid += int(ok_g.sum())
    n_valid_oracle += int(ok_o.sum())
    n_pose += t_len
    meta.append({"file": p.name, "episode_id": int(d["episode_id"]),
                 "T": int(t_len),
                 "poses_sha256": hashlib.sha256(
                     d["poses"].to(torch.float32).numpy().tobytes()).hexdigest(),
                 "n_valid_guarded": int(ok_g.sum()),
                 "n_valid_oracle": int(ok_o.sum())})
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{len(eps)} ({time.time() - t0:.0f}s)", flush=True)

blob["_meta_json"] = np.frombuffer(json.dumps({
    "_what": "leak-guarded target-speed labels, PARITY TRAIN corpus, FULL per-pose track",
    "_src": str(SRC), "_file_list_sha256": listing,
    "_guard_steps": VT_GUARD_STEPS, "_min_lookahead": VT_MIN_LOOKAHEAD,
    "_mint": "tanitad.lake.vtarget.vtarget_guarded (shipped verbatim)",
    "_parity": "READ-ONLY; no episode selected, deselected or reordered",
    "n_episodes": len(meta), "n_pose_indices": n_pose,
    "n_valid_guarded": n_valid, "n_valid_oracle": n_valid_oracle,
    "episodes": meta,
}).encode(), dtype=np.uint8)
np.savez_compressed(OUT, **blob)
print(f"wrote {OUT} ({OUT.stat().st_size} B) in {time.time() - t0:.0f}s")
print(f"guarded valid {n_valid}/{n_pose} = {n_valid / n_pose:.4f}; "
      f"oracle valid {n_valid_oracle}/{n_pose} = {n_valid_oracle / n_pose:.4f}")
