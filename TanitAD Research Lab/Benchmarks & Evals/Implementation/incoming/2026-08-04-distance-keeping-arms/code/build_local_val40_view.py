#!/usr/bin/env python3
"""Rebuild the poses-only val40 view on the dev box from Thor's relay, and PROVE it is the
canonical cache by hashing the poses against the committed manifest.

Writes 40 `ep_*.pt` stubs that `tanitad.data.mixing.load_episode` accepts. `frames_u8` is a
[T,9,1,1] zero placeholder: the lead scorer reads only `poses` and `episode_id`, and the window
grid is a function of T alone. Nothing here writes into any real cache, so parity
(`physicalai-val-0c5f7dac3b11`) is untouched by construction.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
MANIFEST = REPO / ("TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
                   "2026-07-26-s3-decision-grade/artifacts/manifest_EVALPOD_val40.json")
NPZ = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

z = np.load(NPZ, allow_pickle=False)
meta = json.loads(bytes(z["_meta_json"]).decode())
man = {e["file"]: e for e in json.load(open(MANIFEST))["episodes"]}

rows, n_ok, n_bad = [], 0, 0
for m in meta:
    stem = Path(m["file"]).stem
    poses = torch.from_numpy(z[f"{stem}__poses"]).clone()
    acts = torch.from_numpy(z[f"{stem}__actions"]).clone()
    T = int(poses.shape[0])
    d = {"frames_u8": torch.zeros((T, 9, 1, 1), dtype=torch.uint8),
         "actions": acts, "poses": poses, "episode_id": int(m["episode_id"])}
    k = f"{stem}__maneuvers"
    if k in z.files:
        d["maneuvers"] = torch.from_numpy(z[k]).clone()
    torch.save(d, OUT / m["file"])

    local_sha = hashlib.sha256(poses.numpy().tobytes()).hexdigest()
    exp = man.get(m["file"])
    ok = exp is not None and local_sha == exp["poses_sha256"] and int(exp["episode_id"]) == int(m["episode_id"]) and int(exp["T"]) == T
    n_ok += bool(ok)
    n_bad += (not ok)
    rows.append({"file": m["file"], "episode_id": m["episode_id"], "T": T,
                 "poses_sha256": local_sha,
                 "manifest_poses_sha256": (exp or {}).get("poses_sha256"),
                 "sha_ok": bool(exp is not None and local_sha == exp["poses_sha256"]),
                 "eid_ok": bool(exp is not None and int(exp["episode_id"]) == int(m["episode_id"])),
                 "T_ok": bool(exp is not None and int(exp["T"]) == T),
                 "thor_frames_u8_shape": m["frames_u8_shape"]})

summary = {"_what": "poses-only val40 view rebuilt on the dev box from Thor's relay",
           "_source": "tanitad-thor:/home/nvidia/valdata/physicalai-val-0c5f7dac3b11 (40/40 sha256 OK, 2026-08-03)",
           "_manifest": str(MANIFEST.relative_to(REPO)),
           "n_episodes": len(rows), "n_match_manifest": n_ok, "n_mismatch": n_bad,
           "total_windows_from_T": int(sum(max(t - 28, 0 + 0) // 8 + (1 if max(t - 28, 0) > 0 else 0) for t in [])),
           "rows": rows}
# window count, computed the way rollout.collect does
starts = lambda t: np.arange(0, max(int(t) - 8 - 20, 0), 8)
summary["total_windows_from_T"] = int(sum(starts(r["T"]).size for r in rows))
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
(OUT.parent / "val40_view_verify.json").write_text(json.dumps(summary, indent=1))
print("wrote", OUT.parent / "val40_view_verify.json")
