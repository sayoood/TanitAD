#!/usr/bin/env python3
"""Obstacle join for the OLD-format epcache — unblocks the REF-C v2.1 RL pilot.

WHY THIS EXISTS (PI directive 2026-08-29)
-----------------------------------------
*"Take the existing refc model … and try to apply the RL training until refcv3
is finished."* The only local corpus in the format the v2.1 model consumes
(``frames_u8 [T, 9, 256, 256]``) is the dev-box epcache — which carries NO
``clip_id`` and NO agents join. Without a join, ``collision``/``headway`` can
never fire and the reward degenerates toward the hackable progress arm, which
the A0 gate (LAUNCH_PLAN §0) rightly refuses to launch on.

HOW IDENTITY IS RECOVERED
-------------------------
``physicalai.build_episode`` derives ``episode_id = int.from_bytes(
clip_id.encode()[:4].ljust(4, b"\\0"), "big")`` (physicalai.py:740) — the first
FOUR CHARACTERS of the clip uuid as a big-endian int. That is invertible against
a clip universe: here, the union of member names of the locally cached
``obstacle.offline.chunk_*.zip`` files (each member is
``<clip_uuid>.obstacle.offline.parquet``). An episode whose 4-char prefix
matches ZERO local clips is reported unjoined (its chunk is not cached); one
matching MORE THAN ONE clip is DROPPED LOUDLY (ambiguous identity is not
identity).

⚠️ DECLARED TIME-BASE HYPOTHESIS (``--t0-offset``, default 0.2 s)
-----------------------------------------------------------------
Stacked frame ``i`` is built from pre-stack frames ``i, i+1, i+2`` and carries
``poses[i+2]`` (build_episode: ``poses[k:n]``, k = n_stack-1 = 2), so its clip
time is ``t_i = (i + 2) * 0.1 s`` under the 10 Hz decode from clip start.
Obstacle rows are matched at ``|t_row - t_i| <= tol`` (0.06 s, the join family's
default). A wrong offset does not reduce coverage (rows are ~10 Hz, tol 60 ms)
— it displaces agents by ego-motion x error, so the offset is a FLAG, stated in
the pre-registration, not silently baked in.

OUTPUT
------
Same jsonl schema as ``val130_agents.jsonl`` (what ``rl_a0_coverage.py`` and the
reward context builder consume): one line per (episode, frame) carrying
``agents: [{cx, cy, yaw, l, w, occ, track_id, cls}]`` in the ego frame of that
frame, +x forward. ``clip_id`` is set to the EPISODE FILE STEM (``ep_00042``) so
downstream tools key by file, with the recovered uuid kept as ``real_clip_id``.

Evidence class of the output: MEASURED (ours; NVIDIA autolabels v2, rig frame).
⚠️ NON-PARITY corpus (key 14231cd29c74) — pilot-only, stated everywhere.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import zipfile

import numpy as np
import torch


def clip_prefix_int(clip_id: str) -> int:
    return int.from_bytes(clip_id.encode()[:4].ljust(4, b"\0"), "big")


def yaw_from_quat(oz: float, ow: float) -> float:
    """Yaw about +z from a (mostly-planar) quaternion."""
    return 2.0 * math.atan2(oz, ow)


def local_clip_universe(chunks_dir: str) -> dict[str, tuple[str, str]]:
    """clip_uuid -> (zip_path, member_name) over every locally cached chunk."""
    out: dict[str, tuple[str, str]] = {}
    for z in sorted(glob.glob(os.path.join(chunks_dir, "*.zip"))):
        with zipfile.ZipFile(z) as f:
            for m in f.namelist():
                if m.endswith(".obstacle.offline.parquet"):
                    out[m.split(".")[0]] = (z, m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epdir", required=True, help="old-format epcache dir (ep_*.pt)")
    ap.add_argument("--chunks-dir", required=True,
                    help="dir of obstacle.offline.chunk_*.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-s", type=float, default=0.06)
    ap.add_argument("--t0-offset", type=float, default=0.2,
                    help="clip time of stacked frame 0 (the n_stack hypothesis)")
    ap.add_argument("--hz", type=float, default=10.0)
    a = ap.parse_args()

    import pandas as pd

    universe = local_clip_universe(a.chunks_dir)
    by_prefix: dict[int, list[str]] = {}
    for cid in universe:
        by_prefix.setdefault(clip_prefix_int(cid), []).append(cid)
    eps = sorted(glob.glob(os.path.join(a.epdir, "ep_*.pt")))
    print(f"[join] {len(eps)} episodes · {len(universe)} clips in "
          f"{len(set(z for z, _ in universe.values()))} local chunks", flush=True)

    n_match = n_ambig = n_unjoined = 0
    lines: list[str] = []
    stats = {"frames": 0, "frames_with_agents": 0, "agents_total": 0,
             "vehicles_ahead_inlane": 0}

    for path in eps:
        stem = os.path.basename(path).rsplit(".", 1)[0]
        d = torch.load(path, map_location="cpu", weights_only=False)
        ep_id = int(d["episode_id"])
        T = int(d["poses"].shape[0])
        cands = by_prefix.get(ep_id, [])
        if not cands:
            n_unjoined += 1
            continue
        if len(cands) > 1:
            print(f"[join] ⛔ AMBIGUOUS {stem}: episode_id {ep_id} matches "
                  f"{len(cands)} clips — dropped", flush=True)
            n_ambig += 1
            continue
        cid = cands[0]
        z, m = universe[cid]
        with zipfile.ZipFile(z) as f:
            df = pd.read_parquet(io.BytesIO(f.read(m)))
        t_row = df["timestamp_us"].to_numpy(dtype=np.float64) / 1e6
        cx = df["center_x"].to_numpy(dtype=np.float64)
        cy = df["center_y"].to_numpy(dtype=np.float64)
        sx = df["size_x"].to_numpy(dtype=np.float64)
        sy = df["size_y"].to_numpy(dtype=np.float64)
        oz = df["orientation_z"].to_numpy(dtype=np.float64)
        ow = df["orientation_w"].to_numpy(dtype=np.float64)
        tid = df["track_id"].astype(str).to_numpy()
        cls = df["label_class"].astype(str).to_numpy()

        n_match += 1
        for i in range(T):
            t_i = a.t0_offset + i / a.hz
            sel = np.abs(t_row - t_i) <= a.tol_s
            agents = []
            if sel.any():
                # nearest row per track inside the tolerance window
                idxs = np.nonzero(sel)[0]
                best: dict[str, int] = {}
                for j in idxs:
                    k = tid[j]
                    if k not in best or abs(t_row[j] - t_i) < abs(t_row[best[k]] - t_i):
                        best[k] = j
                for j in best.values():
                    agents.append({"cx": round(float(cx[j]), 4),
                                   "cy": round(float(cy[j]), 4),
                                   "yaw": round(yaw_from_quat(oz[j], ow[j]), 5),
                                   "l": round(float(sx[j]), 3),
                                   "w": round(float(sy[j]), 3),
                                   "occ": 0, "track_id": str(tid[j]),
                                   "cls": str(cls[j])})
                    if 0.0 < cx[j] < 80.0 and abs(cy[j]) <= 2.0:
                        stats["vehicles_ahead_inlane"] += 1
            stats["frames"] += 1
            stats["frames_with_agents"] += int(bool(agents))
            stats["agents_total"] += len(agents)
            lines.append(json.dumps({"clip_id": stem, "real_clip_id": cid,
                                     "frame_idx": i,
                                     "t_s": round(t_i, 4), "agents": agents}))

    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    meta = {
        "_what": "obstacle.offline join for the OLD-format dev-box epcache (RL pilot)",
        "_corpus": os.path.basename(a.epdir.rstrip("/\\")),
        "_parity": "⚠️ NON-PARITY (pilot only)",
        "_identity": "episode_id = first-4-chars-of-clip-uuid int (physicalai.py:740), "
                     "inverted against local chunk members; ambiguous prefixes dropped",
        "_time_base": {"t0_offset_s": a.t0_offset, "hz": a.hz, "tol_s": a.tol_s,
                       "note": "HYPOTHESIS, declared — see module docstring"},
        "episodes_total": len(eps), "episodes_joined": n_match,
        "episodes_ambiguous": n_ambig, "episodes_unjoined_no_local_chunk": n_unjoined,
        **stats,
        "_evidence_class": "MEASURED (ours; NVIDIA autolabels v2, rig frame)",
    }
    with open(a.out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    fwa = stats["frames_with_agents"] / max(stats["frames"], 1)
    print(f"[join] joined {n_match}/{len(eps)} episodes "
          f"(ambiguous {n_ambig}, no-local-chunk {n_unjoined})", flush=True)
    print(f"[join] frames with agents {fwa:.1%} · mean agents/frame "
          f"{stats['agents_total'] / max(stats['frames'], 1):.2f} · "
          f"in-lane-ahead vehicle-frames {stats['vehicles_ahead_inlane']:,}", flush=True)
    if n_match == 0:
        print("[join] ⛔ ZERO episodes joined — the pilot has no reward context; "
              "do not launch.")
        return 2
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
