#!/usr/bin/env python3
"""(NAVSIM VENV; run ONCE with PYTHONPATH=<v1.1 tree> on the v1 cache and ONCE on the v2 cache)
Dump, for one token's metric cache, which tracked objects the SCORER's observation holds at each 0.1 s
step t = 0..4.0 s (the NC loop's domain), with their types and footprints — so the two protocols'
object sets can be compared WITHOUT re-running either scorer.

    <py> code/diag_observation_objects.py --pkl <metric_cache.pkl> --out <json> --steps 41

W8, 2026-09-26 — diagnosis of the pre-registered C-NC FAILURE on token 937ca624cc2658a6 (CV: v1 NC 0.5,
v2 NC 1.0). For v2 the objects are what the non-reactive policy hands the scorer
(``observation.detections_tracks[:41]`` minus agents intersecting the ego at t0,
log_replay_traffic_agents.py:35-58); for v1 the cached PDMObservation's occupancy maps are read as the
v1 scorer reads them (``observation[time_idx]``, sample_res 1).
"""
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=41)
    a = ap.parse_args(argv)
    mc = pickle.loads(lzma.decompress(a.pkl.read_bytes()))
    obs = mc.observation
    import navsim
    rec = {"pkl": str(a.pkl), "navsim_file": navsim.__file__, "steps": {}, "types": {}, "footprints": {}}
    ego = mc.ego_state.car_footprint.oriented_box.geometry
    rec["ego_t0_bounds"] = list(ego.bounds)
    uo = obs.unique_objects
    for tok, o in uo.items():
        rec["types"][tok] = str(o.tracked_object_type)
    dts = getattr(obs, "_detections_tracks", None)
    if dts is not None:                          # v2: what the log-replay policy hands the scorer
        rec["source"] = "v2 observation.detections_tracks[:41] (the log-replay policy's input)"
        rec["n_detection_frames"] = len(dts)
        colliding = {o.metadata.track_token for o in dts[0].tracked_objects.tracked_objects
                     if o.box.geometry.intersects(ego)}
        rec["t0_colliding_removed_by_policy"] = sorted(colliding)
        for t in range(min(a.steps, len(dts))):
            objs = [o for o in dts[t].tracked_objects.tracked_objects if o.metadata.track_token not in colliding]
            rec["steps"][t] = sorted(o.metadata.track_token for o in objs)
            for o in objs:
                rec["footprints"].setdefault(o.metadata.track_token, {})[t] = [round(x, 3) for x in o.box.geometry.bounds]
    else:                                        # v1: the cached occupancy maps the scorer reads
        rec["source"] = "v1 PDMObservation occupancy maps (observation[time_idx])"
        for t in range(a.steps):
            om = obs[t]
            toks = list(om.tokens)
            rec["steps"][t] = sorted(toks)
            for tok in toks:
                g = om[tok]
                rec["footprints"].setdefault(tok, {})[t] = [round(x, 3) for x in g.bounds]
    a.out.write_text(json.dumps(rec), encoding="utf-8")
    print(json.dumps({"n_unique": len(uo), "n_steps": len(rec["steps"]), "source": rec["source"],
                      "navsim": rec["navsim_file"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
