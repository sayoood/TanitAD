#!/usr/bin/env python3
"""The drivable-corridor CHANNEL -- emitter + the consumer side for taniteval.

WHAT THIS ADDS THAT ``taniteval/corridor.py`` DOES NOT HAVE
-----------------------------------------------------------
``taniteval.corridor`` scores ``corridor_departure_rate`` against a **scalar**
half-width, ``CORRIDOR_HALFWIDTH_M = 1.75``, and its own docstring is explicit
that this is **PROPOSED** -- *"about half a lane", NOT measured on this corpus* --
and that *"the 'corridor' is not a lane"*: it is a symmetric band about the
REFERENCE PATH, because no lane geometry existed.

This module supplies the missing thing: a **per-timestep, ASYMMETRIC, map-derived
corridor** -- ``(d_left, d_right)`` from the ego to its lane's actual left and
right edge, in metres, from ``trajdata.VectorMap``.

Why asymmetry is the point, not a detail: a symmetric band cannot represent an ego
that is legitimately close to one edge (a lane change, a wide turn, an offset
approach) but has 3 m of room on the other. Scored symmetrically, the first metre
of a normal lane change reads exactly like the first metre of a departure.

MEASURED, this corpus (51 AlpaSim scenes, ``corridor_verdict.json``):
  * lane width          3.359 m  [3.243, 3.488]   episode(scene)-cluster bootstrap
  * measured half-width 1.802 m  [1.686, 1.939]   -- and 1.75 IS inside that CI
  * ego containment     0.9837   [0.9644, 0.9985]

⭐ So the standing 1.75 m constant is **empirically vindicated as a central value**
-- it is not off. What it cannot represent is the **spread**: per-scene half-width
runs **1.40 m to 2.50 m** (p10-p90). A single constant is a good average and a poor
description of any particular scene.

SCOPE LIMIT -- STATED, NOT BURIED
---------------------------------
This channel is derived from **AlpaSim** maps. **PhysicalAI-AV has no map** (settled
at five independent probes), so the channel CANNOT be emitted for PhysicalAI
windows. Two admissible uses:
  1. score corridor departure on **AlpaSim** closed-loop rollouts (map available);
  2. use the MEASURED half-width distribution to replace the *guessed* constant
     used on PhysicalAI -- a transfer, and it must be labelled one.
Emitting a per-timestep PhysicalAI corridor from this is **not** licensed.

LICENCE: reads map geometry only. The NuRec/gsplat renderer
(NGC-DL-CONTAINER-LICENSE, no derivatives) is never imported or modified.
"""
from __future__ import annotations

import numpy as np

BLOCK = "vectormap.corridor_channel"
VERSION = "1.0.0"

# MEASURED on 51 AlpaSim scenes -- corridor_verdict.json, episode(scene)-cluster
# bootstrap, B=2000, unit = scene. Supersedes nothing automatically; see
# INTEGRATION_NOTE below.
MEASURED_HALFWIDTH_M = 1.802          # HALF THE LANE WIDTH (a centred ego)
MEASURED_HALFWIDTH_CI = (1.686, 1.939)
MEASURED_LANE_WIDTH_M = 3.359
# ⭐ The matched threshold: room from the EGO'S OWN position to the NEARER edge.
MEASURED_EFFECTIVE_HALFWIDTH_M = 1.391
MEASURED_EFFECTIVE_HALFWIDTH_CI = (1.289, 1.500)
MEASURED_EFFECTIVE_P10_P90 = (1.065, 1.722)
PROPOSED_HALFWIDTH_M = 1.75           # taniteval.corridor.CORRIDOR_HALFWIDTH_M

INTEGRATION_NOTE = (
    "taniteval.corridor.CORRIDOR_HALFWIDTH_M = 1.75 is PROPOSED. Two different "
    "measured quantities bear on it and they disagree, so state which one is "
    "meant:\n"
    "  (a) HALF THE LANE WIDTH = 1.802 [1.686, 1.939]. 1.75 sits INSIDE this CI, "
    "so as a description of a lane the constant is vindicated.\n"
    "  (b) EFFECTIVE half-width = min(room to left edge, room to right edge) from "
    "the ego's ACTUAL position = 1.391 [1.289, 1.500]. 85.7 % of ego steps have "
    "LESS room than 1.75 m, and 46/51 scenes are tighter than it.\n"
    "(b) is the matched threshold for corridor_departure_rate, because taniteval "
    "measures cross-track error FROM THE REFERENCE (ego) PATH -- the same origin. "
    "The ego does not drive down the lane centreline, so (a) systematically "
    "overstates the room a prediction may consume: 1.75 is ~26 % TOO PERMISSIVE.\n"
    "⚠️ Do NOT silently retune the constant. Every published corridor_departure_rate, "
    "including E1a's headline 0.5877 / 0.8414, is scored at 1.75; moving it "
    "reprices all of them at once. This is a PI-level decision, and it is also an "
    "AlpaSim -> PhysicalAI TRANSFER (PhysicalAI has no map), which must be labelled "
    "as such wherever it is used.")


# ========================================================================== #
# consumer side -- what taniteval calls                                        #
# ========================================================================== #
def departure_from_bounds(lat_signed, d_left, d_right):
    """Per-window fraction of steps OUTSIDE the mapped lane. ``[N, K] -> [N]``.

    The asymmetric analogue of ``taniteval.corridor.corridor_departure``.

    ⚠️ **ORIGIN CONTRACT -- the one way to misuse this.** ``lat_signed`` must be the
    signed cross-track error of the PREDICTION **relative to the reference (ego)
    path** -- exactly what ``taniteval.corridor.cross_track_from_paths`` returns,
    signed (**+ = LEFT**, the ``driving.frenet`` convention). ``d_left`` /
    ``d_right`` are measured **from the same origin**: the ego pose at that step.

    Do **not** pass an offset measured from the lane CENTRELINE. It has a different
    origin and the comparison is then meaningless -- that error read a scene with
    100 % containment as 0 % contained during this instrument's own development.

    A step is a departure iff it crosses either edge -- so a wide lane forgives a
    large offset and a narrow one does not, which a scalar threshold cannot do.
    """
    lat = np.asarray(lat_signed, dtype=np.float64)
    dl = np.asarray(d_left, dtype=np.float64)
    dr = np.asarray(d_right, dtype=np.float64)
    if not (lat.shape == dl.shape == dr.shape):
        raise ValueError(f"shape mismatch: {lat.shape} {dl.shape} {dr.shape}")
    out = (lat > dl) | (-lat > dr)
    return out.mean(-1)


def margin_to_edge(lat_signed, d_left, d_right):
    """Signed metres of remaining room. ``>0`` inside, ``<0`` outside.

    ``min(d_left - lat, d_right + lat)``. This is the quantity D-A intervention #3
    would condition on: it is 0 exactly at the lane edge and degrades smoothly,
    unlike a binary departure flag."""
    lat = np.asarray(lat_signed, dtype=np.float64)
    return np.minimum(np.asarray(d_left, dtype=np.float64) - lat,
                      np.asarray(d_right, dtype=np.float64) + lat)


def effective_halfwidth(d_left, d_right):
    """Symmetric half-width that would score the same as these bounds.

    ``min(d_left, d_right)`` -- the conservative reduction, for comparing a mapped
    corridor against the scalar-threshold instrument on equal terms."""
    return np.minimum(np.asarray(d_left, dtype=np.float64),
                      np.asarray(d_right, dtype=np.float64))


def load_channel(npz_path):
    """Load an emitted channel. Returns ``{scene: {d_left_m, d_right_m, ...}}``."""
    z = np.load(npz_path, allow_pickle=False)
    scenes = sorted({k.rsplit("__", 1)[0] for k in z.files if "__" in k})
    return {s: {k.split("__")[1]: z[k] for k in z.files
                if k.startswith(s + "__")} for s in scenes}


# ========================================================================== #
# emitter -- runs on the pod, where the maps are                               #
# ========================================================================== #
def main():
    import argparse
    import glob
    import json
    import os
    import sys
    import time
    sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vectormap_corridor import probe_scene, SSROOT

    ap = argparse.ArgumentParser("corridor_channel")
    ap.add_argument("--out", default="/workspace/corridor_channel.npz")
    ap.add_argument("--meta", default="/workspace/corridor_channel_meta.json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from alpasim_runtime.scene_loader import ArtifactSceneProvider
    scenes = {}
    for sd in sorted(d for d in glob.glob(SSROOT + "/*") if os.path.isdir(d)):
        try:
            prov = ArtifactSceneProvider.from_path(sd, smooth_trajectories=True)
        except Exception:
            continue
        for sid in sorted(prov.scene_ids):
            scenes.setdefault(sid, prov)
    sids = sorted(scenes)
    if a.limit:
        sids = sids[:a.limit]

    arrays, meta = {}, {}
    t0 = time.time()
    for i, sid in enumerate(sids, 1):
        short = sid[7:15] if sid.startswith("clipgt-") else sid[:8]
        try:
            ds = scenes[sid].get_data_source(sid)
            rec, chan = probe_scene(ds)
            if chan is None:
                continue
            for key in ("d_left_m", "d_right_m", "width_m", "lat_m"):
                arrays[f"{short}__{key}"] = chan[key].astype(np.float32)
            arrays[f"{short}__inside"] = chan["inside"].astype(np.uint8)
            meta[short] = {"n_steps": int(len(chan["lat_m"])),
                           "dt_s": float(chan["dt_s"]),
                           "containment": rec["ego_containment_rate"],
                           "halfwidth_m": rec["corridor"]["halfwidth_m_median"]}
            print("[%2d/%2d] %s n=%d contain=%.4f  %.0fs"
                  % (i, len(sids), short, meta[short]["n_steps"],
                     meta[short]["containment"], time.time() - t0), flush=True)
            try:
                ds.clear_cache()
            except Exception:
                pass
        except Exception as e:
            print("FAIL", short, repr(e)[:120], flush=True)

    np.savez_compressed(a.out, **arrays)
    json.dump({"block": BLOCK, "version": VERSION,
               "measured_halfwidth_m": MEASURED_HALFWIDTH_M,
               "measured_halfwidth_ci": list(MEASURED_HALFWIDTH_CI),
               "proposed_halfwidth_m": PROPOSED_HALFWIDTH_M,
               "integration_note": INTEGRATION_NOTE,
               "scope_limit": ("AlpaSim only. PhysicalAI-AV has no map, so this "
                               "channel cannot be emitted for PhysicalAI windows."),
               "sign_convention": "+lat = LEFT of lane centreline (driving.frenet)",
               "scenes": meta}, open(a.meta, "w"), indent=1)
    print("WROTE", a.out, "and", a.meta, "-- %d scenes" % len(meta))


if __name__ == "__main__":
    main()
