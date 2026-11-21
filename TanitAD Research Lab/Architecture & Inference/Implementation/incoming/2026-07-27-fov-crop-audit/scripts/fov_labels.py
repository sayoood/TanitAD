"""FOV crop audit — PART 2 step 1: situation labels for the LOCALLY DECODABLE clips.

⚠️ WHY THIS FILE EXISTS AND WHAT IT IS NOT. The dev box does **not** hold the parity episode cache
(`physicalai-train-e438721ae894`); its local cache is keyed `14231cd29c74`. The FOV sweep therefore
cannot reuse the situation classifier's cached episodes and instead rebuilds everything from the raw
mp4s. This file reproduces `physicalai.build_episode`'s **exact** query grid —

    t_query   = linspace(t_frames[0], t_frames[-1], n_target),  n_target = int(span_s * 10)
    poses     = signals_at(ego, t_query)[1]        then dropped by n_stack-1 = 2

— so the label index is the frame-stack index, index for index, with no alignment step (the
structural property the situation-classifier stream exploited).  The detectors are imported from
`sc_situations.py` **unmodified**; nothing is re-implemented and no threshold is swept.

⚠️ The intersection label here is the **TURN half only** — the cross-traffic half needs
`obstacle.offline` + calibration for the clip, which is available for only a small minority of the
locally decodable set. `2026-07-26-situation-classifier` §4 V4 is what licenses the turn half
standing alone (perpendicular cross traffic is 2.415x [1.057, 7.931] more common on a tight turn
than on a matched-heading large-radius curve, separated from 1.0). Declared, not hidden.

🔒 No clip UUID leaves this script: the output is keyed by an integer index and the uuid map is
written to a `_LOCAL_ONLY_*` file that never enters the repo.

usage:  python fov_labels.py <out_dir> [limit]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get(
    "TANITAD_STACK", r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack"))
sys.path.insert(0, os.environ.get(
    "SC_SCRIPTS",
    r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
    r"\Architecture & Inference\Implementation\incoming\2026-07-26-situation-classifier\scripts"))

from sc_situations import (anticipation_target, detect_intersection,  # noqa: E402
                           detect_lane_change, detect_roundabout, kinematics)
from tanitad.data.physicalai import (discover_r0_clips, load_egomotion,  # noqa: E402
                                     signals_at)

ROOT = os.environ.get("TANITAD_PAI_ROOT", r"C:\Users\Admin\tanitad-data\physicalai")
TARGET_HZ = 10.0
N_STACK = 3
VAL_FRAC_CHUNKS = 0.65        # chunk-grouped split; HELD-OUT is the larger side (as in sitclf)


def episode_grid(ts_path):
    ts = pd.read_parquet(ts_path)
    tcol = next(c for c in ts.columns if "time" in c.lower())
    t_frames = ts[tcol].to_numpy(np.float64)
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * TARGET_HZ), N_STACK + 1)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    frame_idx = np.searchsorted(t_frames, t_query).clip(0, len(t_frames) - 1)
    return t_query, frame_idx, len(t_frames)


def main():
    out = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    os.makedirs(out, exist_ok=True)
    clips = discover_r0_clips(ROOT)
    sel = pd.read_parquet(os.path.join(ROOT, "r0", "r0_selection.parquet"))
    chunk_of = dict(zip(sel["clip_id"].astype(str), sel["chunk"].astype(int)))
    if limit:
        clips = clips[:limit]
    packs, meta, k2clip = {}, [], {}
    for i, c in enumerate(clips):
        try:
            t_query, frame_idx, n_frames = episode_grid(c["timestamps"])
            ego = load_egomotion(c["ego_zip"], c["clip_id"])
            _actions, poses = signals_at(ego, t_query)
        except Exception as exc:                                     # noqa: BLE001
            print(f"  [{i}] label FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        n = min(len(frame_idx), poses.shape[0])
        P = poses[N_STACK - 1:n]
        T = len(P)
        if T < 60:
            continue
        K = kinematics(P)
        ev = {"lane_change": detect_lane_change(K),
              "roundabout": detect_roundabout(K, bracket=True),
              "intersection": detect_intersection(K, cross=None)[0]}
        for s, e in ev.items():
            y, val = anticipation_target(T, e)
            ong = np.zeros(T, bool)
            for a, b in e:
                ong[a:b + 1] = True
            packs[f"c{i}_y_{s}"] = y
            packs[f"c{i}_valid_{s}"] = val
            packs[f"c{i}_ongoing_{s}"] = ong
        packs[f"c{i}_poses"] = P.astype(np.float32)
        packs[f"c{i}_frame_idx"] = frame_idx[:n].astype(np.int32)
        ch = int(chunk_of[c["clip_id"]])
        meta.append({"k": i, "T": int(T), "chunk": f"{ch:04d}", "n_frames_native": int(n_frames),
                     "n_events": {s: len(e) for s, e in ev.items()}})
        k2clip[str(i)] = c["clip_id"]
        if (i + 1) % 100 == 0:
            print(f"[labels] {i+1}/{len(clips)}", flush=True)

    # ---- chunk-grouped split: a chunk is a geographic/temporal cluster; never split within one ----
    chunks = sorted({m["chunk"] for m in meta})
    h = {c: int(hashlib.sha1(c.encode()).hexdigest(), 16) % 1000 for c in chunks}
    held = {c for c in chunks if h[c] < int(VAL_FRAC_CHUNKS * 1000)}
    for m in meta:
        m["side"] = "HELDOUT" if m["chunk"] in held else "TRAIN"

    np.savez_compressed(os.path.join(out, "fov_labels.npz"), **packs)
    json.dump(meta, open(os.path.join(out, "fov_meta.json"), "w"))
    json.dump(k2clip, open(os.path.join(out, "_LOCAL_ONLY_k2clip.json"), "w"))
    summ = {"n_clips": len(meta), "n_chunks": len(chunks),
            "split": {"TRAIN": sum(m["side"] == "TRAIN" for m in meta),
                      "HELDOUT": sum(m["side"] == "HELDOUT" for m in meta)},
            "events": {s: int(sum(m["n_events"][s] for m in meta))
                       for s in ("lane_change", "roundabout", "intersection")},
            "clips_with_event": {s: int(sum(m["n_events"][s] > 0 for m in meta))
                                 for s in ("lane_change", "roundabout", "intersection")},
            "heldout_clips_with_event": {
                s: int(sum(m["n_events"][s] > 0 and m["side"] == "HELDOUT" for m in meta))
                for s in ("lane_change", "roundabout", "intersection")},
            "intersection_label": "TURN HALF ONLY (no cross-traffic half) — declared",
            "grid": "build_episode's exact t_query/frame_idx; n_stack=3 drop of 2"}
    json.dump(summ, open(os.path.join(out, "labels_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
