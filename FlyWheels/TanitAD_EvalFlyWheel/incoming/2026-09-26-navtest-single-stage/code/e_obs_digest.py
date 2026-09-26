#!/usr/bin/env python3
"""(NAVSIM VENV) Per-token digest of the OBJECTS the scorer's collision checks see at t = 0..4.0 s.

    # v1 side (W3's v1.1 cache) — PYTHONPATH=<v1.1 tree>
    <py> code/e_obs_digest.py --cache D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest \
        --side v1 --tokens-file <json> --out raw/e_obs_v1_<name>.json
    # v2 side (our cache) — PYTHONPATH=<suite devkit_side>
    <py> code/e_obs_digest.py --cache <v2 cache> --side v2 --tokens-file <json> --out raw/e_obs_v2_<name>.json

W8, 2026-09-26 — PREREG AMENDMENT A2. The pre-registered C-NC FAILED on the smoke (token 937ca624…, CV:
v1 NC 0.5, v2 NC 1.0). MEASURED mechanism (both caches + the log): the metric cache's observation is
interpolated over **5.0 s in v1.1** (v1 ``metric_cache_processor.py:101``) and **4.0 s in v2**
(v2 ``:109``), and an object observed EXACTLY ONCE inside the window is placed at EVERY step
(``StateInterpolator`` start == end -> ``initial_detection_track`` appended; v1 ``:165-166``, v2
``:177-178``). Five tracks of that token are observed only at +5.0 s: v1 holds them as GHOSTS at every
t in [0, 4] s, v2 does not hold them at all, and CV's straight path hits one (a GENERIC_OBJECT -> NC 0.5).

This script computes, for each token, the sorted (track_token, footprint bounds rounded to 1 um) per step
t = 0..40 from the CACHE CONTENT the scorer consumes — v1: the cached PDMObservation occupancy maps
(``observation[t]``); v2: ``observation.detections_tracks[:41]`` minus agents intersecting the ego at t0
(what ``log_replay_traffic_agents.py:35-58`` hands the scorer) — and a sha1 per step. E-OBS = the tokens
whose per-step digests differ between the two sides. Nothing here reads a scorer output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import pickle
import sys
import time
from pathlib import Path


def _digest(items) -> str:
    return hashlib.sha1(json.dumps(sorted(items)).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--side", required=True, choices=("v1", "v2"))
    ap.add_argument("--tokens-file", default=None, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=41)
    a = ap.parse_args(argv)
    t0 = time.time()
    want = None
    if a.tokens_file:
        d = json.loads(a.tokens_file.read_text(encoding="utf-8"))
        want = set(d["tokens"] if isinstance(d, dict) else d)
    paths = [p for p in sorted(a.cache.glob("*/*/*/metric_cache.pkl")) if want is None or p.parts[-2] in want]
    out, fails = {}, []
    for p in paths:
        tok = p.parts[-2]
        try:
            mc = pickle.loads(lzma.decompress(p.read_bytes()))
            obs = mc.observation
            steps = []
            if a.side == "v2":
                ego = mc.ego_state.car_footprint.oriented_box.geometry
                dts = obs.detections_tracks[:a.steps]
                col = {o.metadata.track_token for o in dts[0].tracked_objects.tracked_objects if o.box.geometry.intersects(ego)}
                for t in range(a.steps):
                    items = [(o.metadata.track_token, [round(x, 6) for x in o.box.geometry.bounds])
                             for o in dts[t].tracked_objects.tracked_objects if o.metadata.track_token not in col]
                    steps.append(_digest(items))
            else:
                for t in range(a.steps):
                    om = obs[t]
                    items = [(tk, [round(x, 6) for x in om[tk].bounds]) for tk in om.tokens
                             if "red_light" not in tk]
                    steps.append(_digest(items))
            out[tok] = steps
        except Exception as e:                                           # noqa: BLE001
            fails.append({"token": tok, "exception": f"{type(e).__name__}: {e}"[:200]})
    rec = {"schema": "w8-e-obs-digest/1", "side": a.side, "cache": str(a.cache).replace("\\", "/"),
           "n_requested": (len(want) if want is not None else None), "n_digested": len(out), "n_failures": len(fails),
           "failures": fails[:20], "steps": a.steps, "digests": out, "wall_s": round(time.time() - t0, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("digests", "failures")}))
    return 0 if (out and not fails and (want is None or len(out) == len(want))) else 1


if __name__ == "__main__":
    sys.exit(main())
