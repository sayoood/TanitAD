"""The 3-D join's LAST HOP, measured end to end and banked as JSON.

``24065b6`` landed the join with one hop it could not check: its author's
worktree predated ``box3d_head.py``, so ``3-D join -> zh_targets -> n_z > 0``
shipped UNVERIFIED. This script closes it, and the two tests appended to
``tests/test_refcv6_perception_realdata.py`` keep it closed.

⛔ Clip ids never enter the output: every identity is a sha12.

Run:
    TANITAD_AGENT_JOIN3D=<...>.jsonl.xz python lasthop_zh_evidence.py <out.json>
"""
from __future__ import annotations

import json
import lzma
import os
import sys

import numpy as np
import torch

from tanitad.data.agent_cuboid_gt import (
    base_from_centre, open_join3d, zh_for_frame,
)
from tanitad.data.semantic_map_gt import sha12
from tanitad.models.agent_slots import match_slots, targets_from_join
from tanitad.models.box3d_head import (
    Box3DSlotDecoder, box3d_set_loss, zh_targets,
)


def main(out_path: str) -> dict:
    path = os.environ["TANITAD_AGENT_JOIN3D"]
    store = open_join3d(path)
    if store is None:
        raise SystemExit("no join at TANITAD_AGENT_JOIN3D")

    with lzma.open(path, "rt", encoding="utf-8") as fh:
        rec = json.loads(fh.readline())
    agents = rec["agents"]
    tracks = [a["track_id"] for a in agents]
    fields = sorted(agents[0].keys())

    arr = np.asarray(
        [[a["cx"], a["cy"], a["yaw"], a["l"], a["w"], a["occ"]] for a in agents],
        dtype=np.float32)
    tgt = targets_from_join(arr, classes=[a["cls"] for a in agents])
    cz, h, mask = zh_for_frame(rec["clip_id"], rec["frame"], tracks,
                               join3d=store)
    t3 = zh_targets(tgt, cz, h, mask=mask)
    n_z = int(t3["zh_mask"].sum())

    zz = t3["cz"][t3["zh_mask"]].numpy()
    hh = t3["h"][t3["zh_mask"]].numpy()
    base = base_from_centre(zz, hh)

    torch.manual_seed(1)
    dec = Box3DSlotDecoder(d_memory=16, n_memory=8,
                           n_queries=max(32, len(tracks)), d_model=32,
                           depth=1, n_heads=4, enforce_band=False)
    pred = dec(torch.randn(1, 8, 16))
    m = match_slots(pred, t3)
    with torch.no_grad():
        withl = box3d_set_loss(pred, t3, match=m)
        without = box3d_set_loss(pred, zh_targets(tgt), match=m)

    seen = missed = lines = empty = 0
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ids = [a["track_id"] for a in r["agents"]]
            if not ids:
                empty += 1
                continue
            lines += 1
            _, _, mm = zh_for_frame(r["clip_id"], r["frame"], ids, join3d=store)
            seen += int(mm.sum())
            missed += int((~mm).sum())

    out = {
        "artifact": os.path.basename(path),
        "evidence_class": "MEASURED",
        "closes": "24065b6 -- the last hop the join agent could not verify",
        "join": {
            "clips": store.n_clips, "index_entries": len(store),
            "agents_indexed": store.n_agents, "agent_fields": fields,
            "lines_with_agents": lines, "lines_without_agents": empty,
        },
        "probe_frame": {
            "clip_sha12": sha12(rec["clip_id"]), "n_tracks": len(tracks),
            "n_z": n_z,
            "cz_m": [float(zz.min()), float(zz.max())],
            "h_m": [float(hh.min()), float(hh.max())],
            "base_median_m": float(np.median(base)),
        },
        "loss": {
            "with_labels_n": {k: int(v) for k, v in withl["n"].items()},
            "without_labels_n": {k: int(v) for k, v in without["n"].items()},
            "loss_z": float(withl["loss_z"]), "loss_h": float(withl["loss_h"]),
            "total_with": float(withl["total"]),
            "total_without": float(without["total"]),
            "delta": float(withl["total"]) - float(without["total"]),
        },
        "coverage": {
            "agent_frames_with_zh": seen, "agent_frames_without": missed,
            "total": seen + missed,
            "pct": round(100.0 * seen / max(1, seen + missed), 4),
        },
        "verdict": ("VERIFIED: the 3-D join reaches box3d_head's height targets "
                    "and the z/h terms are consumed by the loss"),
    }
    assert n_z > 0 and withl["n"]["z"] > 0
    assert without["n"]["z"] == 0 and float(without["loss_z"]) == 0.0
    assert abs(out["loss"]["delta"]) > 1e-6
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(json.dumps(out, indent=1, sort_keys=True))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lasthop_zh.json")
