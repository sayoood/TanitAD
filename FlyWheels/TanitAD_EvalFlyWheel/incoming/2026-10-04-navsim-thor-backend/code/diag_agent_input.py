#!/usr/bin/env python3
"""Diagnose a cross-platform AgentInput fingerprint mismatch, bit by bit (NAVSIM v2 venv, CPU only).

For each synthetic token: load its scene pickle exactly as the devkit does
(``Scene.load_from_disk``), then dump -- as ``float.hex`` -- every input and intermediate of
``Scene.get_agent_input`` -> ``get_history_trajectory`` -> ``convert_absolute_to_relative_se2_array``:
the global ego poses, the origin, ``cos(theta)`` / ``sin(theta)``, the subtraction, the 2x2 matmul
(numpy ``@`` = BLAS) next to a plain scalar re-evaluation of the same products, the angle
normalisation, and the fingerprint. Run it on BOTH machines and diff the JSON: the first differing
field names the operation.

    python diag_agent_input.py --scenes <synthetic_scene_pickles> --tokens t1 t2 ... --out diag.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

if os.name != "nt":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import thor_compat
    thor_compat.install()

from navsim.common.dataclasses import Scene, SensorConfig                               # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (   # noqa: E402
    normalize_angle)
from nuplan.common.actor_state.state_representation import StateSE2                    # noqa: E402


def hx(a) -> list:
    return [float(x).hex() for x in np.asarray(a, dtype=np.float64).ravel()]


def fingerprint(ego_statuses) -> str:
    h = hashlib.sha1()
    for es in ego_statuses:
        for arr in (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command):
            h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--tokens", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
           "numpy": np.__version__, "tokens": {}}
    try:
        out["numpy_blas"] = {k: str(v) for k, v in np.__config__.CONFIG.items()} if hasattr(np.__config__, "CONFIG") else None
    except Exception:                                                     # noqa: BLE001
        out["numpy_blas"] = None
    for spec in a.tokens:                     # "token=file.pkl" (synthetic pickles are not named by token)
        tok, fname = spec.split("=", 1) if "=" in spec else (spec, f"{spec}.pkl")
        sc = Scene.load_from_disk(Path(a.scenes) / fname, None, SensorConfig.build_no_sensors())
        assert sc.scene_metadata.initial_token == tok, (tok, fname)
        n = sc.scene_metadata.num_history_frames
        glob = np.array([sc.frames[i].ego_status.ego_pose for i in range(n)], dtype=np.float64)
        o = sc.frames[n - 1].ego_status.ego_pose
        origin = StateSE2(*o)
        theta = -origin.heading
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rel = glob - np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)
        mm = rel[..., :2] @ R.T                                           # what the devkit does (BLAS)
        sc_x = rel[:, 0] * R.T[0, 0] + rel[:, 1] * R.T[1, 0]              # same products, no BLAS
        sc_y = rel[:, 0] * R.T[0, 1] + rel[:, 1] * R.T[1, 1]
        head = normalize_angle(rel[:, 2])
        ai = sc.get_agent_input()
        out["tokens"][tok] = {
            "global_poses": hx(glob), "origin": hx(o), "theta": hx([theta]), "cos": hx([c]), "sin": hx([s]),
            "rel_sub": hx(rel), "matmul_xy": hx(mm), "scalar_xy": hx(np.stack([sc_x, sc_y], -1)),
            "matmul_equals_scalar": bool(np.array_equal(mm, np.stack([sc_x, sc_y], -1))),
            "heading_norm": hx(head),
            "agent_ego_pose": [hx(e.ego_pose) for e in ai.ego_statuses],
            "agent_ego_velocity": [hx(e.ego_velocity) for e in ai.ego_statuses],
            "agent_ego_acceleration": [hx(e.ego_acceleration) for e in ai.ego_statuses],
            "agent_driving_command": [hx(e.driving_command) for e in ai.ego_statuses],
            "fingerprint": fingerprint(ai.ego_statuses),
        }
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({t: v["fingerprint"] for t, v in out["tokens"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
