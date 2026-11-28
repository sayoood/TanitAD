#!/usr/bin/env python3
"""(NAVSIM VENV, v2 tree) PREREG AMENDMENT A2 — classify each token by HOW the two caches' observations differ.

    <py> code/e_obs_classify.py --v1-cache D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest \
        --v2-cache <v2 cache> [--tokens-file <json>] --out raw/e_obs_class_<name>.json

For every token and every step t = 0..40 (the NC loop's domain) it compares the objects the scorer's collision
checks see — v1.1: the cached PDMObservation occupancy maps (``observation[t]``); v2: the log-replay detections
(``observation.detections_tracks[:41]`` minus agents intersecting the ego at t0, log_replay_traffic_agents.py:35-58)
— as (track token, footprint bounds rounded to 1 um), and classifies the token:

  IDENTICAL    every step's object set and footprints are equal            -> NC must be EXACTLY equal
  V1_SUPERSET  at every step v2's objects are a subset of v1's (common      -> NC(v2) >= NC(v1) and
               ones with identical footprints): v1 only holds EXTRA objects     TTC(v2) >= TTC(v1)
               (the +4.5/+5.0 s single-observation GHOSTS of the 5.0 s window)
  V2_EXTRA     at some step v2 holds an object or footprint v1 does not      -> no bar (two-sided)

The v1 pickle is read in the v2 tree (same module paths; a dataclass unpickles by __dict__). ⛔ That is a
second READ PATH for v1, so ``--control-v1-digest`` (e_obs_digest.py's v1-tree output) is compared per step:
any token whose cross-tree digest differs from the v1-tree digest is reported, and a nonzero count voids
the classification (the control must read EXACTLY 0).
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


def v1_items(mc, steps):
    obs = mc.observation
    out = []
    for t in range(steps):
        om = obs[t]
        out.append({tk: tuple(round(x, 6) for x in om[tk].bounds) for tk in om.tokens if "red_light" not in tk})
    return out


def v2_items(mc, steps):
    ego = mc.ego_state.car_footprint.oriented_box.geometry
    dts = mc.observation.detections_tracks[:steps]
    col = {o.metadata.track_token for o in dts[0].tracked_objects.tracked_objects if o.box.geometry.intersects(ego)}
    return [{o.metadata.track_token: tuple(round(x, 6) for x in o.box.geometry.bounds)
             for o in dts[t].tracked_objects.tracked_objects if o.metadata.track_token not in col}
            for t in range(steps)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-cache", required=True, type=Path)
    ap.add_argument("--v2-cache", required=True, type=Path)
    ap.add_argument("--tokens-file", default=None, type=Path)
    ap.add_argument("--control-v1-digest", default=None, type=Path, help="e_obs_digest.py --side v1 output")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=41)
    a = ap.parse_args(argv)
    t0 = time.time()
    v2p = {p.parts[-2]: p for p in a.v2_cache.glob("*/*/*/metric_cache.pkl")}
    if a.tokens_file:
        d = json.loads(a.tokens_file.read_text(encoding="utf-8"))
        want = set(d["tokens"] if isinstance(d, dict) else d)
        v2p = {t: p for t, p in v2p.items() if t in want}
    ctrl = json.loads(a.control_v1_digest.read_text(encoding="utf-8"))["digests"] if a.control_v1_digest else None
    cls, detail, fails, ctrl_bad, n_ctrl = {}, {}, [], [], 0
    for tok, p2 in sorted(v2p.items()):
        p1 = a.v1_cache / p2.parts[-4] / p2.parts[-3] / tok / "metric_cache.pkl"
        try:
            m1 = pickle.loads(lzma.decompress(p1.read_bytes()))
            m2 = pickle.loads(lzma.decompress(p2.read_bytes()))
            i1, i2 = v1_items(m1, a.steps), v2_items(m2, a.steps)
        except Exception as e:                                           # noqa: BLE001
            fails.append({"token": tok, "exception": f"{type(e).__name__}: {e}"[:200]})
            continue
        if ctrl is not None and tok in ctrl:
            n_ctrl += 1
            mine = [_digest([(k, list(v)) for k, v in s.items()]) for s in i1]
            if mine != ctrl[tok]:
                ctrl_bad.append(tok)
        extra1 = extra2 = 0
        for s1, s2 in zip(i1, i2):
            for k, v in s2.items():
                if s1.get(k) != v:
                    extra2 += 1
            for k, v in s1.items():
                if s2.get(k) != v:
                    extra1 += 1
        c = "IDENTICAL" if (extra1 == 0 and extra2 == 0) else ("V1_SUPERSET" if extra2 == 0 else "V2_EXTRA")
        cls[tok] = c
        detail[tok] = {"v1_only_object_steps": extra1, "v2_only_object_steps": extra2}
    counts = {k: sum(1 for v in cls.values() if v == k) for k in ("IDENTICAL", "V1_SUPERSET", "V2_EXTRA")}
    rec = {"schema": "w8-e-obs-class/1", "v1_cache": str(a.v1_cache).replace("\\", "/"),
           "v2_cache": str(a.v2_cache).replace("\\", "/"), "n_tokens": len(cls), "counts": counts,
           "n_failures": len(fails), "failures": fails[:20],
           "control_v1_cross_tree_digest": {"n_compared": n_ctrl, "n_differ": len(ctrl_bad), "first": ctrl_bad[:10],
                                            "must_read": 0, "pass": (ctrl is None) or (n_ctrl > 0 and not ctrl_bad)},
           "class": cls, "detail": detail, "wall_s": round(time.time() - t0, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec), encoding="utf-8")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("class", "detail", "failures")}))
    return 0 if (cls and not fails and rec["control_v1_cross_tree_digest"]["pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
