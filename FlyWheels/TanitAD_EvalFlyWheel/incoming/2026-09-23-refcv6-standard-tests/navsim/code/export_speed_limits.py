#!/usr/bin/env python3
"""refcv6's MAX-SPEED input from the nuPlan MAP (RUN IN THE NAVSIM VENV, Python 3.9).

⭐ WHAT IS LOOKED UP, AND WHY IT IS ADMISSIBLE AT INFERENCE. Per scorer token: the ego's GLOBAL
pose at t0 (the last history frame — ``Scene.frames[t0].ego_status.ego_pose``, the pose the devkit
itself converts every AgentInput from), and the POSTED LIMIT (``speed_limit_mps``) of the map lane
the ego is ON at that instant. That is localisation + a map read — what a production car with a
navigation map has. It reads NO future frame, NO route (``roadblock_ids`` are never touched), and
NO human trajectory. The brief records the PI ruling of 2026-09-19 as making nuPlan map speed
limits admissible for this input. ⚠️ The ruling's recorded text (memory
``max-speed-labels-non-parity-ruling``) speaks of LABELS for the max-speed head; its use as an
INFERENCE INPUT is the brief's (Master Mind's) reading, surfaced in RESULT.md, not assumed.

⭐ THE LOOKUP RULE (fixed before any refcv6 number existed; SPEC §3):
  1. candidates = LANE + LANE_CONNECTOR objects within ``R_QUERY_M`` of the ego point;
  2. prefer lanes whose polygon CONTAINS the ego point and whose baseline heading at the nearest
     point is within ``HEADING_TOL_DEG`` of the ego heading — the smallest heading difference wins;
  3. else the nearest lane within ``R_NEAR_M`` with the heading condition;
  4. else ``none`` (no lane) — and a chosen lane whose ``speed_limit_mps`` is null is ``no_limit``.
The field is nuPlan's own ``NuPlanLane.speed_limit_mps`` / ``NuPlanLaneConnector.speed_limit_mps``
(``lanes_polygons`` / ``lane_connectors`` column ``speed_limit_mps``) — ⛔ NEVER
``lanes_polygons.max_speed``, which is a constant-25 placeholder on every lane of every city
(DataEng FS19-2, MEASURED).

Measured coverage by city is written to the output (FS19-2 measured it by LANE LENGTH: LV 100 %,
PIT 100 %, BOS 7.1 %, SG 0 %; this is the per-SCENE figure on the split actually scored).

    python code/export_speed_limits.py --inputs <navsim_agent_inputs.json> --split warmup_two_stage \
        --out raw/speed_limits_warmup_two_stage.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "C:/Users/Admin/navsim-crun/data/maps")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "C:/Users/Admin/navsim-crun/data/openscene")
os.environ.setdefault("NAVSIM_EXP_ROOT", "C:/Users/Admin/navsim/exp")
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", "C:/Users/Admin/navsim-crun/devkit")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from nuplan.common.actor_state.state_representation import Point2D  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # noqa: E402

from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

R_QUERY_M = 5.0
R_NEAR_M = 2.0
HEADING_TOL_DEG = 60.0
DEVKIT = pathlib.Path(os.environ["NAVSIM_DEVKIT_ROOT"])
CFG = DEVKIT / "navsim/planning/script/config/common/train_test_split"
DATA = pathlib.Path(os.environ["OPENSCENE_DATA_ROOT"])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def lookup(map_api, x: float, y: float, heading: float) -> dict:
    """The rule in the module docstring, returning the chosen lane and why."""
    pt = Point2D(float(x), float(y))
    objs = map_api.get_proximal_map_objects(
        pt, R_QUERY_M, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    cands = []
    for layer, lst in objs.items():
        for lane in lst:
            try:
                near = lane.baseline_path.get_nearest_pose_from_position(pt)
                dh = abs(math.degrees(_wrap(float(near.heading) - float(heading))))
                poly = lane.polygon
                from shapely.geometry import Point as _SP
                sp = _SP(float(x), float(y))
                contains = bool(poly.contains(sp))
                dist = float(poly.distance(sp))
            except Exception as e:                                   # noqa: BLE001
                cands.append({"id": str(lane.id), "error": repr(e)[:120]})
                continue
            cands.append({"id": str(lane.id), "layer": layer.name, "contains": contains,
                          "dist_m": dist, "dheading_deg": dh,
                          "speed_limit_mps": lane.speed_limit_mps})
    ok = [c for c in cands if "error" not in c and c["dheading_deg"] <= HEADING_TOL_DEG]
    inside = sorted((c for c in ok if c["contains"]), key=lambda c: c["dheading_deg"])
    method, pick = "none", None
    if inside:
        method, pick = "contains", inside[0]
    else:
        near = sorted((c for c in ok if c["dist_m"] <= R_NEAR_M), key=lambda c: c["dist_m"])
        if near:
            method, pick = "nearest", near[0]
    if pick is None:
        status, lim = "none", None
    elif pick["speed_limit_mps"] is None:
        status, lim = "no_limit", None
    else:
        status, lim = "limit", float(pick["speed_limit_mps"])
    return {"status": status, "speed_limit_mps": lim, "method": method,
            "lane_id": pick["id"] if pick else None,
            "lane_layer": pick.get("layer") if pick else None,
            "dheading_deg": pick.get("dheading_deg") if pick else None,
            "dist_m": pick.get("dist_m") if pick else None,
            "n_candidates": len(cands)}


def build_loader(split: str, syn_sensors: pathlib.Path):
    sp = yaml.safe_load(open(CFG / f"{split}.yaml", encoding="utf-8"))
    sf = yaml.safe_load(open(CFG / "scene_filter" / f"{split}.yaml", encoding="utf-8"))
    kw = {k: v for k, v in sf.items() if not k.startswith("_")}
    filt = SceneFilter(**kw)
    loader = SceneLoader(data_path=DATA / "navsim_logs" / "test",
                         original_sensor_path=DATA / "sensor_blobs" / "test",
                         scene_filter=filt, synthetic_sensor_path=syn_sensors,
                         synthetic_scenes_path=DATA / split / "synthetic_scene_pickles",
                         sensor_config=SensorConfig.build_no_sensors())
    return loader, filt


def pose_from_log_frame(f: dict) -> tuple:
    """GLOBAL (x, y, heading) of a raw log frame — the devkit's own construction
    (``navsim.common.dataclasses``: ``ego2global_translation[:2]`` and the yaw of the
    ``ego2global_rotation`` quaternion, (w, x, y, z) order)."""
    from pyquaternion import Quaternion           # the devkit's own call (dataclasses.py:459-461)
    t = [float(v) for v in f["ego2global_translation"]]
    yaw = float(Quaternion(*f["ego2global_rotation"]).yaw_pitch_roll[0])
    return t[0], t[1], yaw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="the E2-format export JSON (token set)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--syn-sensors", default=None)
    ap.add_argument("--from-logs", default="", help="single-stage (navtest): read each token's t0 "
                    "pose straight from its LOG pickle in this dir (no SceneLoader)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if a.inputs.endswith(".gz"):
        import gzip
        doc = json.load(gzip.open(a.inputs, "rt", encoding="utf-8"))
    else:
        doc = json.load(open(a.inputs, encoding="utf-8"))
    toks = doc["tokens"]
    maps: dict = {}
    rows: dict = {}
    t0 = time.time()
    if a.from_logs:
        import pickle

        class _U(pickle.Unpickler):
            def find_class(self, m, n):
                if m == "pathlib" and n == "PosixPath":
                    return pathlib.PurePosixPath
                return super().find_class(m, n)
        by_log: dict = {}
        for tok, r in toks.items():
            by_log.setdefault(r["log_name"], []).append(tok)
        for i, (ln, tl) in enumerate(sorted(by_log.items())):
            fr = _U(open(os.path.join(a.from_logs, f"{ln}.pkl"), "rb")).load()
            idx = {f["token"]: f for f in fr}
            for tok in tl:
                f = idx[tok]
                mname = f["map_location"]
                if mname != toks[tok]["map_name"]:
                    raise SystemExit(f"{tok}: map {mname} != export {toks[tok]['map_name']}")
                x, y, h = pose_from_log_frame(f)
                if mname not in maps:
                    maps[mname] = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"],
                                               os.environ["NUPLAN_MAP_VERSION"], mname)
                rec = lookup(maps[mname], x, y, h)
                rec.update({"stage": int(toks[tok].get("stage", 1)), "map_name": mname,
                            "log_name": ln, "ego_global_xyh": [x, y, h],
                            "pose_source": "log pickle ego2global"})
                rows[tok] = rec
            print(f"[speed] log {i+1}/{len(by_log)} {time.time()-t0:.0f}s", flush=True)
    syn = pathlib.Path(a.syn_sensors or (DATA / a.split / "sensor_blobs"))
    loader = filt = None
    if not a.from_logs:
        loader, filt = build_loader(a.split, syn)
    nh = filt.num_history_frames if filt is not None else None
    for i, (tok, r) in enumerate(sorted(toks.items()) if not a.from_logs else []):
        if r["stage"] == 2:
            path, _ln = loader.synthetic_scenes[tok]
            sc = Scene.load_from_disk(path, syn, SensorConfig.build_no_sensors())
            fr = sc.frames[sc.scene_metadata.num_history_frames - 1]
            mname = sc.scene_metadata.map_name
        else:
            frames = loader.scene_frames_dicts[tok]
            sc = Scene.from_scene_dict_list(frames, DATA / "sensor_blobs" / "test",
                                            num_history_frames=nh,
                                            num_future_frames=filt.num_future_frames,
                                            sensor_config=SensorConfig.build_no_sensors())
            fr = sc.frames[nh - 1]
            mname = sc.scene_metadata.map_name
        if mname != r["map_name"]:
            raise SystemExit(f"{tok}: map {mname} != export {r['map_name']}")
        x, y, h = (float(v) for v in fr.ego_status.ego_pose)
        if mname not in maps:
            maps[mname] = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"],
                                       os.environ["NUPLAN_MAP_VERSION"], mname)
        rec = lookup(maps[mname], x, y, h)
        rec.update({"stage": int(r["stage"]), "map_name": mname, "log_name": r["log_name"],
                    "ego_global_xyh": [x, y, h]})
        rows[tok] = rec
        if (i + 1) % 500 == 0:
            print(f"[speed] {i+1}/{len(toks)} {time.time()-t0:.0f}s", flush=True)
    by_city: dict = {}
    for rec in rows.values():
        c = by_city.setdefault(rec["map_name"], {"n": 0, "limit": 0, "no_limit": 0, "none": 0,
                                                 "values_mps": {}})
        c["n"] += 1
        c[rec["status"]] += 1
        if rec["speed_limit_mps"] is not None:
            k = f"{rec['speed_limit_mps']:.4f}"
            c["values_mps"][k] = c["values_mps"].get(k, 0) + 1
    for c in by_city.values():
        c["coverage"] = c["limit"] / c["n"] if c["n"] else None
    n = len(rows)
    out = {"_what": "per-token posted limit of the lane the ego occupies at t0 (nuPlan map)",
           "rule": {"R_QUERY_M": R_QUERY_M, "R_NEAR_M": R_NEAR_M,
                    "HEADING_TOL_DEG": HEADING_TOL_DEG,
                    "field": "NuPlanLane/NuPlanLaneConnector.speed_limit_mps (never max_speed)"},
           "split": a.split, "inputs": os.path.abspath(a.inputs),
           "maps_root": os.environ["NUPLAN_MAPS_ROOT"],
           "map_version": os.environ["NUPLAN_MAP_VERSION"],
           "n_tokens": n, "n_limit": sum(1 for r in rows.values() if r["status"] == "limit"),
           "coverage_by_city": by_city, "seconds": round(time.time() - t0, 1), "tokens": rows}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "tokens"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
