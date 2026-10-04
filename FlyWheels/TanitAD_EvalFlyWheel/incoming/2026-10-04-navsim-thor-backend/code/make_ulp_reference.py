#!/usr/bin/env python3
"""Dump the AgentInput the devkit's runner builds for given tokens, as exact float64 hex (+ fingerprint).

Run on the DEV BOX it produces the ULP reference for the Thor seam-agent variants; run on Thor it
produces the Thor side of the same table (the two are diffed by ``compare_ulp_reference.py``-style
code in RESULT.md). It calls exactly what ``run_pdm_score`` calls: ``SceneLoader(...)`` built from
the split's own scene-filter YAML (only ``log_names`` / ``tokens`` / ``synthetic_scene_tokens``
narrowed) and ``scene_loader.get_agent_input_from_token(token)``. Works with navsim v1.1 (navtest)
and v2 (navhard two-stage); the devkit on PYTHONPATH decides which.

    python make_ulp_reference.py --version v1 --filter-yaml <…/scene_filter/navtest.yaml> \
        --logs <navsim_logs/test> --tokens tokens.json --out ref.json
    python make_ulp_reference.py --version v2 --filter-yaml <…/scene_filter/navhard_two_stage.yaml> \
        --logs <navsim_logs/test> --synthetic <synthetic_scene_pickles> --tokens tokens.json --out ref.json

``tokens.json``: ``{"tokens": [...], "token_log": {token: log_name}}`` (W3 format); synthetic
(17-char) tokens may omit ``token_log`` -- their scenes are found by scanning the pickles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

if os.name != "nt":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import thor_compat
    thor_compat.install()

from navsim.common.dataclasses import SceneFilter, SensorConfig            # noqa: E402
from navsim.common.dataloader import SceneLoader                           # noqa: E402

FIELDS = ("agent_ego_pose", "agent_ego_velocity", "agent_ego_acceleration", "agent_driving_command")


def hx(a) -> list:
    return [float(x).hex() for x in np.asarray(a, dtype=np.float64).ravel()]


def fp(ess) -> str:
    h = hashlib.sha1()
    for es in ess:
        for arr in (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command):
            h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=("v1", "v2"), required=True)
    ap.add_argument("--filter-yaml", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--synthetic", default="")
    ap.add_argument("--synthetic-map", default="",
                    help="LITE mode for synthetic tokens: JSON {token: pickle file name}. Each scene is "
                         "loaded on its own -- the SAME call the v2 loader makes for a synthetic token, "
                         "Scene.load_from_disk(file, sensor_path, no-sensors).get_agent_input() -- so a "
                         "RAM-tight box never builds the full SceneLoader. Correctness is not argued: the "
                         "ULP agents refuse any reference that does not re-hash to the seam fingerprint.")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-log", action="store_true", help="v1 only: one log per SceneLoader (RAM-tight box)")
    a = ap.parse_args()
    doc = json.load(open(a.tokens, encoding="utf-8"))
    toks = list(doc["tokens"])
    token_log = doc.get("token_log", {})
    if a.synthetic_map:
        from navsim.common.dataclasses import Scene
        smap = json.load(open(a.synthetic_map, encoding="utf-8"))
        if any(t in token_log for t in toks):
            raise SystemExit("--synthetic-map handles synthetic tokens only")
        out = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
               "numpy": np.__version__, "version": a.version, "mode": "lite-synthetic",
               "n_requested": len(toks), "missing": [t for t in toks if t not in smap], "tokens": {}}
        t0 = time.time()
        for t in toks:
            if t not in smap:
                continue
            sc = Scene.load_from_disk(Path(a.synthetic) / smap[t], None, SensorConfig.build_no_sensors())
            assert sc.scene_metadata.initial_token == t, (t, smap[t])
            ai = sc.get_agent_input()
            rec = {k: [] for k in FIELDS}
            for es in ai.ego_statuses:
                for k, arr in zip(FIELDS, (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command)):
                    rec[k].append(hx(arr))
            rec["fingerprint"] = fp(ai.ego_statuses)
            out["tokens"][t] = rec
        out["wall_s"] = round(time.time() - t0, 1)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print(json.dumps({"n_tokens": len(out["tokens"]), "missing": len(out["missing"]), "wall_s": out["wall_s"]}))
        return 0 if not out["missing"] else 1
    cfg = yaml.safe_load(open(a.filter_yaml, encoding="utf-8"))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    orig = [t for t in toks if t in token_log]
    synth = [t for t in toks if t not in token_log]
    cfg["log_names"] = sorted({token_log[t] for t in orig}) or cfg.get("log_names")
    cfg["tokens"] = orig
    t0 = time.time()
    if a.version == "v1" and a.per_log:
        # RAM-tight mode (dev box): ONE log per SceneLoader, so peak RSS is one log's frames, not all
        # 136 (the all-logs navtest loader is ~2.1 GB MEASURED by W3). Same loader, same call per token.
        out = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
               "numpy": np.__version__, "version": a.version, "mode": "v1-per-log",
               "filter_yaml": a.filter_yaml, "n_requested": len(toks), "tokens": {}}
        by_log: dict = {}
        for t in orig:
            by_log.setdefault(token_log[t], []).append(t)
        for ln, lt in sorted(by_log.items()):
            c = dict(cfg, log_names=[ln], tokens=lt)
            sl = SceneLoader(Path(a.logs), None, SceneFilter(**c), SensorConfig.build_no_sensors())
            for t in lt:
                if t not in sl.tokens:
                    continue
                ai = sl.get_agent_input_from_token(t)
                rec = {k: [] for k in FIELDS}
                for es in ai.ego_statuses:
                    for k, arr in zip(FIELDS, (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command)):
                        rec[k].append(hx(arr))
                rec["fingerprint"] = fp(ai.ego_statuses)
                out["tokens"][t] = rec
            del sl
        out["missing"] = sorted(set(toks) - set(out["tokens"]))
        out["wall_s"] = round(time.time() - t0, 1)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print(json.dumps({"n_tokens": len(out["tokens"]), "missing": len(out["missing"]), "wall_s": out["wall_s"]}))
        return 0 if not out["missing"] else 1
    if a.version == "v1":
        sl = SceneLoader(Path(a.logs), None, SceneFilter(**cfg), SensorConfig.build_no_sensors())
    else:
        if synth:
            cfg["synthetic_scene_tokens"] = synth
            cfg["log_names"] = None                      # synthetic scenes are found by token, any log
            cfg["tokens"] = orig or ["__none__"]
        sl = SceneLoader(data_path=Path(a.logs), original_sensor_path=None, scene_filter=SceneFilter(**cfg),
                         synthetic_sensor_path=None,
                         synthetic_scenes_path=(Path(a.synthetic) if a.synthetic else None),
                         sensor_config=SensorConfig.build_no_sensors())
    have = set(sl.tokens)
    out = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
           "numpy": np.__version__, "version": a.version, "filter_yaml": a.filter_yaml,
           "n_requested": len(toks), "missing": sorted(set(toks) - have), "tokens": {}}
    for t in toks:
        if t not in have:
            continue
        ai = sl.get_agent_input_from_token(t)
        rec = {k: [] for k in FIELDS}
        for es in ai.ego_statuses:
            for k, arr in zip(FIELDS, (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command)):
                rec[k].append(hx(arr))
        rec["fingerprint"] = fp(ai.ego_statuses)
        out["tokens"][t] = rec
    out["wall_s"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"n_tokens": len(out["tokens"]), "missing": len(out["missing"]), "wall_s": out["wall_s"]}))
    return 0 if not out["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
