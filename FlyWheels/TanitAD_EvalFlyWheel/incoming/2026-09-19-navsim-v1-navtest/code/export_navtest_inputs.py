#!/usr/bin/env python3
"""Export, per navtest SCORER TOKEN, the EXACT v1.1 ``AgentInput`` the official scorer hands an
agent (NAVSIM VENV, PYTHONPATH = the v1.1 tree).

The v1.1 port of E2's ``export_agent_inputs.py`` stage-1 branch (same record schema, so E2's
bridge functions consume it unchanged): the ego statuses come from the devkit's own
``SceneLoader.get_agent_input_from_token`` (the call ``run_pdm_score.py:79`` makes), the human
future from the devkit's own ``Scene.get_future_trajectory`` (exactly what ``HumanAgent``
returns), the CV plan from the devkit's own ``ConstantVelocityAgent``. Re-deriving these from the
raw pickles would be a second implementation of ``get_agent_input`` — and two implementations of
one quantity is how two "independent" checks agree on a wrong answer (E2's rule).

One log at a time (a SceneLoader over all 136 logs would hold every log pickle in RAM).
Output: a gzipped JSON on D: (``--out``) + a small manifest (``--manifest``) with its sha256.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
V11_TREE = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
for k, v in {"NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
             "NUPLAN_MAPS_ROOT": "D:/Archive/devbox-C/navsim/data/maps",
             "OPENSCENE_DATA_ROOT": "D:/Archive/devbox-C/navsim/data/openscene",
             "NAVSIM_EXP_ROOT": "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1",
             "NAVSIM_DEVKIT_ROOT": V11_TREE}.items():
    os.environ.setdefault(k, v)

import navsim  # noqa: E402
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent  # noqa: E402
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402

import run_v1  # noqa: E402  (navtest_tokens: the yaml parse pinned against yaml.safe_load)
from w3_agents_v1 import fingerprint  # noqa: E402  (E2's, verbatim; pinned by a test)

DATA = pathlib.Path(os.environ["OPENSCENE_DATA_ROOT"])
LOGS = DATA / "navsim_logs" / "test"
SENSORS = pathlib.Path("D:/Archive/devbox-C/navsim/data/openscene-v1.1/sensor_blobs/test")
CAMS = ("cam_l0", "cam_f0", "cam_r0")


def es_dict(es) -> dict:
    """VERBATIM copy of E2's ``export_agent_inputs.es_dict``."""
    return {"ego_pose": np.asarray(es.ego_pose, dtype=np.float64).tolist(),
            "ego_velocity": np.asarray(es.ego_velocity, dtype=np.float64).tolist(),
            "ego_acceleration": np.asarray(es.ego_acceleration, dtype=np.float64).tolist(),
            "driving_command": np.asarray(es.driving_command, dtype=np.float64).tolist(),
            "_dtypes": [str(np.asarray(es.ego_pose).dtype), str(np.asarray(es.ego_velocity).dtype),
                        str(np.asarray(es.ego_acceleration).dtype),
                        str(np.asarray(es.driving_command).dtype)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="gzipped JSON (D:)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--logs", default="", help="comma list; default = all 136 navtest logs")
    a = ap.parse_args(argv)
    if not os.path.normcase(navsim.__file__).replace("\\", "/").startswith(
            os.path.normcase(V11_TREE).replace("\\", "/")):
        sys.exit(f"⛔ navsim imported from {navsim.__file__}, not the v1.1 tree")
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:                                                  # noqa: BLE001
        pass
    all_logs, all_toks = run_v1.navtest_tokens()
    logs = [x for x in a.logs.split(",") if x] or all_logs
    tokset = set(all_toks)
    t0 = time.time()
    rec, fps, per_log = {}, {}, {}
    cv = ConstantVelocityAgent()
    for li, ln in enumerate(logs):
        sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                         has_route=True, max_scenes=None, log_names=[ln], tokens=list(all_toks))
        loader = SceneLoader(data_path=LOGS, sensor_blobs_path=SENSORS, scene_filter=sf,
                             sensor_config=SensorConfig.build_no_sensors())
        toks = [t for t in loader.tokens if t in tokset]
        per_log[ln] = len(toks)
        for tok in toks:
            ai = loader.get_agent_input_from_token(tok)
            frames = loader.scene_frames_dicts[tok]
            nh = sf.num_history_frames
            sc = Scene.from_scene_dict_list(frames, SENSORS, num_history_frames=nh,
                                            num_future_frames=sf.num_future_frames,
                                            sensor_config=SensorConfig.build_no_sensors())
            human = sc.get_future_trajectory(8)
            fp = fingerprint(ai.ego_statuses)
            rec[tok] = {"stage": 1, "frame_type": "ORIGINAL",
                        "log_name": frames[nh - 1]["log_name"],
                        "map_name": frames[nh - 1]["map_location"],
                        "scene_token": frames[nh - 1]["scene_token"],
                        "frame_tokens": [frames[i]["token"] for i in range(nh)],
                        "timestamps_us": [int(frames[i]["timestamp"]) for i in range(nh)],
                        "ego_statuses": [es_dict(e) for e in ai.ego_statuses],
                        "cams": {c: [str(frames[i]["cams"][c.upper()]["data_path"]) for i in range(nh)]
                                 for c in CAMS},
                        "human_future_poses": np.asarray(human.poses, dtype=np.float64).tolist(),
                        "human_future_sampling": [human.trajectory_sampling.num_poses,
                                                  human.trajectory_sampling.interval_length],
                        "cv_poses": np.asarray(cv.compute_trajectory(ai).poses,
                                               dtype=np.float64).tolist(),
                        "fingerprint": fp}
            fps.setdefault(fp, []).append(tok)
        del loader
        if (li + 1) % 10 == 0 or li + 1 == len(logs):
            print(f"[export] {li + 1}/{len(logs)} logs, {len(rec)} tokens, "
                  f"{time.time() - t0:.0f} s", flush=True)
    want = {t for t in all_toks} if not a.logs else None
    coll = {fp: ts for fp, ts in fps.items() if len(ts) > 1}
    doc = {"split": "navtest", "devkit": "navsim v1.1 @ 3e8291b", "navsim_file": navsim.__file__,
           "n_tokens": len(rec), "n_logs": len(logs), "tokens_per_log": per_log,
           "fingerprint_unique": not coll, "fingerprint_collisions": coll,
           "fingerprint_note": ("a CONSISTENCY check, never the lookup key (E2 D3): the seam agent "
                                "keys by scene_metadata.initial_token"),
           "tokens": rec}
    if want is not None:
        doc["n_missing_vs_yaml"] = len(want - set(rec))
        doc["n_unexpected_vs_yaml"] = len(set(rec) - want)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    raw = json.dumps(doc).encode("utf-8")
    with gzip.open(a.out, "wb", compresslevel=6) as fh:
        fh.write(raw)
    man = {"out": a.out, "bytes": os.path.getsize(a.out),
           "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
           "n_tokens": len(rec), "n_logs": len(logs),
           "fingerprint_unique": not coll, "n_fingerprint_collision_groups": len(coll),
           "n_missing_vs_yaml": doc.get("n_missing_vs_yaml"),
           "n_unexpected_vs_yaml": doc.get("n_unexpected_vs_yaml"),
           "wall_s": round(time.time() - t0, 1), "navsim_file": navsim.__file__}
    json.dump(man, open(a.manifest, "w", encoding="utf-8"), indent=1)
    print(json.dumps(man, indent=1))
    bad = (not rec) or (want is not None and (doc["n_missing_vs_yaml"] or doc["n_unexpected_vs_yaml"]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
