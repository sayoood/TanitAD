"""INDEPENDENT cross-check of the B1 TRAIN join against `train2400_agents.jsonl`.

WHY THIS IS THE STRONGEST CONTROL AVAILABLE ON B1 TRAIN. The frame gate proves
the join's time base sits on the right frame; it does not prove the AGENT
GEOMETRY is right. `train2400_agents.jsonl` was built by a DIFFERENT builder
(`build_obstacle_join.py`), from a DIFFERENT pose source (the parity corpus's
v2ep episode cache, not a reconstruction), for a DIFFERENT corpus -- and 182 of
its clips are also in B1 TRAIN. On those clips the two joins must agree box for
box, and a one-frame mis-join must break that agreement.

Both joins key on `frame_idx` in the SAME post-n_stack-trim space (n_stack 3;
train2400's per-clip `n_frames` 199 = 201 - 2 confirms it), so the comparison is
direct and needs no index translation.

Every print is ASCII (cp1252 box).
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import sys

import numpy as np


def load(path, clips, frame_key="frame_idx"):
    """{(clip, frame_idx): {track_id: (cx, cy, yaw)}} for `clips` only."""
    op = lzma.open if str(path).endswith(".xz") else open
    out = {}
    n_lines = n_kept = 0
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            rec = json.loads(line)
            cid = rec["clip_id"]
            if cid not in clips:
                continue
            f = rec.get(frame_key)
            if f is None:
                continue
            out[(cid, int(f))] = {
                str(a["track_id"]): (float(a["cx"]), float(a["cy"]),
                                     float(a["yaw"]))
                for a in rec["agents"] if a.get("track_id") is not None}
            n_kept += 1
    return out, n_lines, n_kept


def compare(mine, ref, shift=0):
    """|dcx|, |dcy|, |dyaw| over every track shared at (clip, f) vs ref (clip, f+shift)."""
    dcx, dcy, dyaw = [], [], []
    n_keys = n_tracks = 0
    for (cid, f), am in mine.items():
        ar = ref.get((cid, f + shift))
        if ar is None:
            continue
        n_keys += 1
        for tid, (cx, cy, yaw) in am.items():
            r = ar.get(tid)
            if r is None:
                continue
            n_tracks += 1
            dcx.append(abs(cx - r[0]))
            dcy.append(abs(cy - r[1]))
            d = yaw - r[2]
            d -= 2.0 * math.pi * math.floor((d + math.pi) / (2.0 * math.pi))
            dyaw.append(abs(d))
    if not n_tracks:
        return {"n_keys": n_keys, "n_tracks": 0}
    f = lambda v: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),  # noqa: E731
                   "max": float(np.max(v))}
    return {"n_keys": n_keys, "n_tracks": n_tracks,
            "dcx_m": f(dcx), "dcy_m": f(dcy),
            "dyaw_deg": f(np.degrees(dyaw))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    clips = set(json.load(open(a.clips, encoding="utf-8")))
    print("[xcheck] shared clips: %d" % len(clips), flush=True)
    import os
    import pickle
    cache = "_ref_cache.pkl"
    if os.path.exists(cache):
        ref = pickle.load(open(cache, "rb"))
        nl_r, nk_r = -1, len(ref)
        print("[xcheck] ref from cache: %d keys" % nk_r, flush=True)
    else:
        ref, nl_r, nk_r = load(a.ref, clips)
        print("[xcheck] ref  %s: %d lines scanned, %d kept"
              % (a.ref.split("/")[-1], nl_r, nk_r), flush=True)
    mine, nl_m, nk_m = load(a.mine, clips)
    print("[xcheck] mine %s: %d lines scanned, %d kept" % (a.mine.split("/")[-1],
                                                           nl_m, nk_m), flush=True)

    true = compare(mine, ref, 0)
    mis = compare(mine, ref, 1)
    rep = {"n_shared_clips": len(clips),
           "ref": a.ref, "mine": a.mine,
           "TRUE_same_frame_idx": true,
           "CONTROL_misjoin_plus1": mis,
           "note": "TRUE compares (clip, frame_idx) against the SAME "
                   "frame_idx in a join built by a different builder from a "
                   "different pose source; CONTROL shifts by one frame and "
                   "must be far worse."}
    if true.get("n_tracks") and mis.get("n_tracks"):
        rep["separation_dcx_median_x"] = round(
            mis["dcx_m"]["median"] / max(true["dcx_m"]["median"], 1e-9), 1)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
