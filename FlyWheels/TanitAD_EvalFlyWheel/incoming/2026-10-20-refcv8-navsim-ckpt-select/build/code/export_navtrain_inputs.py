#!/usr/bin/env python3
"""export_navtrain_inputs.py -- per-token NavSim ``AgentInput`` export for the navtrain sample (NAVSIM VENV; PYTHONPATH = the v1.1 devkit tree).

PORT of W3's ``export_navtest_inputs.py`` (``2026-09-19-navsim-v1-navtest/code``), record schema UNCHANGED so the refcv8 bridge consumes it
exactly as it consumes ``navtest_inputs.json.gz`` (``run_bridge8.py --inputs``).  Same principle as W3/E2: the ego statuses come from the
devkit's own ``SceneLoader.get_agent_input_from_token``, the human future from the devkit's own ``Scene.get_future_trajectory``, the CV plan
from the devkit's own ``ConstantVelocityAgent`` -- nothing is re-derived (two implementations of one quantity is how two "independent"
checks agree on a wrong answer).

What differs from the navtest original, and nothing else:

* the universe is the **registered navtrain sample** (``NAVTRAIN_SAMPLE_TOKENS.json``: status FIXED, sha256 re-derived), not the navtest yaml;
  the SceneFilter of each log is restricted to that log's sample tokens;
* the logs are the OpenScene **trainval** pickles (``--logs-root navsim_logs/trainval``); the record's ``stage`` is 1 and ``split`` ``navtrain``;
* the scene-filter geometry (4 history / 10 future / has_route) comes from the navtrain yaml header (``--filter-yaml``), the same numbers the
  sample fixer used, so the two cannot disagree about which tokens exist;
* ``--no-maps`` (default ON for a navtrain run): ``Scene.from_scene_dict_list`` builds a nuPlan map API per scene (``Scene._build_map_api``)
  although NOTHING exported reads it (``get_future_trajectory`` reads ego poses only).  Thor holds the Las Vegas map only and navtrain spans
  four cities, so the map constructor is replaced by a no-op.  MEASURED control (``tests/test_export_navtrain_inputs.py``): the stubbed export
  of 334 real navtest tokens (4 logs) equals, record for record, W3's banked export of the same tokens, which was built WITH the real map API.

Output: gzipped JSON ``--out`` + a small manifest ``--manifest`` with its sha256.  ``--split navtest`` exists ONLY for that control.

    python export_navtrain_inputs.py --sample NAVTRAIN_SAMPLE_TOKENS.json --logs-root <navsim_logs/trainval> --navsim-tree <v1.1 tree> \
        --w3-code <.../2026-09-19-navsim-v1-navtest/code> --filter-yaml navtrain.yaml --out inputs/navtrain_sample_inputs.json.gz --manifest m.json
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

MUTATION = os.environ.get("R8EXP_MUTATION", "")
MUTATIONS = ("stage_2", "frame_tokens_shifted", "cams_not_history", "human_horizon_6", "es_dtype_float32", "sample_unchecked",
             "velocity_acceleration_swapped", "refused_export_written", "prefix_tree_check")
CAMS = ("cam_l0", "cam_f0", "cam_r0")
HUMAN_HORIZON_POSES = 6 if MUTATION == "human_horizon_6" else 8        # W3: ``sc.get_future_trajectory(8)`` = 4 s at 0.5 s (the bridge's scoring horizon)
HERE = os.path.dirname(os.path.abspath(__file__))


def es_dict(es) -> dict:
    """VERBATIM copy of W3's / E2's ``export_agent_inputs.es_dict`` (float64 lists + the source dtypes)."""
    dt = np.float32 if MUTATION == "es_dtype_float32" else np.float64
    v, acc = (es.ego_acceleration, es.ego_velocity) if MUTATION == "velocity_acceleration_swapped" else (es.ego_velocity, es.ego_acceleration)
    return {"ego_pose": np.asarray(es.ego_pose, dtype=dt).tolist(),
            "ego_velocity": np.asarray(v, dtype=dt).tolist(),
            "ego_acceleration": np.asarray(acc, dtype=dt).tolist(),
            "driving_command": np.asarray(es.driving_command, dtype=dt).tolist(),
            "_dtypes": [str(np.asarray(es.ego_pose).dtype), str(np.asarray(es.ego_velocity).dtype),
                        str(np.asarray(es.ego_acceleration).dtype), str(np.asarray(es.driving_command).dtype)]}


def build_record(tok: str, frames: list, nh: int, ego_statuses, human_poses, human_sampling, cv_poses, fp: str, split_stage: int = 1) -> dict:
    """One token's record, W3's schema.  ``frames`` = the scene's frame dicts (``frames[nh - 1]`` is the token's frame)."""
    f0 = frames[nh - 1]
    lo = 1 if MUTATION == "frame_tokens_shifted" else 0
    return {"stage": 2 if MUTATION == "stage_2" else split_stage, "frame_type": "ORIGINAL",
            "log_name": f0["log_name"], "map_name": f0["map_location"], "scene_token": f0["scene_token"],
            "frame_tokens": [frames[i]["token"] for i in range(lo, nh + lo)],
            "timestamps_us": [int(frames[i]["timestamp"]) for i in range(nh)],
            "ego_statuses": [es_dict(e) for e in ego_statuses],
            "cams": {c: [str(frames[i]["cams"][c.upper()]["data_path"]) for i in range(nh if MUTATION != "cams_not_history" else nh + 1)]
                     for c in CAMS},
            "human_future_poses": np.asarray(human_poses, dtype=np.float64).tolist(),
            "human_future_sampling": [human_sampling[0], human_sampling[1]],
            "cv_poses": np.asarray(cv_poses, dtype=np.float64).tolist(), "fingerprint": fp}


def write_outputs(doc: dict, out: str, manifest: str, bad: bool) -> dict:
    """Write the gzipped export ONLY when the export is complete.  A refused export (a sample token the devkit does not produce, or one it
    produces that the sample does not name) leaves NO file at ``out``: a partial export in the canonical path is read downstream as complete."""
    man = {"out": out, "n_tokens": doc["n_tokens"], "fingerprint_unique": doc["fingerprint_unique"],
           "n_fingerprint_collision_groups": len(doc["fingerprint_collisions"]), "n_missing_vs_sample": doc.get("n_missing_vs_sample"),
           "n_unexpected_vs_sample": doc.get("n_unexpected_vs_sample"), "refused": bool(bad), "bytes": None, "sha256": None}
    if bad and MUTATION != "refused_export_written":
        return man
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with gzip.open(out, "wb", compresslevel=6) as fh:
        fh.write(json.dumps(doc).encode("utf-8"))
    man["bytes"] = os.path.getsize(out)
    man["sha256"] = hashlib.sha256(open(out, "rb").read()).hexdigest()
    return man


def navsim_is_from(tree: str, navsim_file: str) -> bool:
    """True iff the imported ``navsim`` IS ``<tree>/navsim/__init__.py``.  A PREFIX test is wrong: the venv also carries the v1.0 devkit at
    ``C:/Users/Admin/navsim-crun/devkit`` and a tree argument of ``C:/Users/Admin`` would accept it (MEASURED while writing this port --
    the v1.0 ``SceneLoader`` then died on ``sensor_blobs_path``, after the check had passed)."""
    got = os.path.normcase(os.path.realpath(navsim_file))
    if MUTATION == "prefix_tree_check":
        return got.startswith(os.path.normcase(os.path.realpath(tree)))
    return got == os.path.normcase(os.path.realpath(os.path.join(tree, "navsim", "__init__.py")))


def check_fresh(out: str) -> None:
    """An export that already exists is never overwritten (a rerun after DONE is a mistake; after REFUSED there is no file)."""
    if os.path.exists(out):
        sys.exit(f"[export] {out} already exists -- delete it deliberately if you mean to rebuild")


def load_sample(path: str, split: str) -> dict:
    s = json.load(open(path, encoding="utf-8"))
    toks = list(s["tokens"])
    if split == "navtrain" and MUTATION != "sample_unchecked":
        if s.get("status") != "FIXED":
            sys.exit(f"[export] sample status {s.get('status')!r} is not FIXED -- availability is fixed BEFORE any export")
        if hashlib.sha256("\n".join(sorted(toks)).encode()).hexdigest() != s.get("sha256_sorted_tokens") or len(toks) != s.get("count"):
            sys.exit("[export] the sample's sha256 / count do not re-derive")
    if not s.get("token_log") or any(t not in s["token_log"] for t in toks):
        sys.exit("[export] the sample carries no complete token_log (token -> log_name)")
    return s


def main(argv=None) -> int:
    if MUTATION:
        sys.exit("[export] R8EXP_MUTATION is set -- a mutation-test environment never writes an export")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--logs-root", required=True)
    ap.add_argument("--navsim-tree", required=True, help="the v1.1 devkit tree (3e8291b) that must be the imported navsim")
    ap.add_argument("--w3-code", required=True, help="W3's code dir (w3_agents_v1.fingerprint is imported from it)")
    ap.add_argument("--filter-yaml", default="")
    ap.add_argument("--split", choices=("navtrain", "navtest"), default="navtrain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--logs", default="", help="comma list of logs (default: every log of the sample)")
    ap.add_argument("--maps-root", default="")
    ap.add_argument("--no-maps", dest="no_maps", action="store_true", default=None)
    ap.add_argument("--with-maps", dest="no_maps", action="store_false")
    a = ap.parse_args(argv)
    check_fresh(a.out)
    no_maps = (a.split == "navtrain") if a.no_maps is None else a.no_maps
    sample = load_sample(a.sample, a.split)
    nh, nf, has_route = 4, 10, True
    if a.filter_yaml:
        sys.path.insert(0, HERE)
        import navtrain_common as N
        h = N.parse_scene_filter(a.filter_yaml)["header"]
        nh, nf, has_route = h["num_history_frames"], h["num_future_frames"], h["has_route"]
    tree = a.navsim_tree.replace("\\", "/").rstrip("/")
    for k, v in {"NUPLAN_MAP_VERSION": "nuplan-maps-v1.0", "NUPLAN_MAPS_ROOT": a.maps_root or ("/nonexistent_maps" if no_maps else ""),
                 "OPENSCENE_DATA_ROOT": os.path.dirname(os.path.dirname(a.logs_root.replace("\\", "/").rstrip("/"))),
                 "NAVSIM_EXP_ROOT": os.path.dirname(os.path.abspath(a.out)), "NAVSIM_DEVKIT_ROOT": tree}.items():
        if v:
            os.environ[k] = v
    if "NUPLAN_MAPS_ROOT" not in os.environ:
        sys.exit("[export] --with-maps needs NUPLAN_MAPS_ROOT (or --maps-root)")
    sys.path.insert(0, tree)
    sys.path.insert(0, a.w3_code)
    import navsim  # noqa: E402
    from navsim.agents.constant_velocity_agent import ConstantVelocityAgent  # noqa: E402
    from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig  # noqa: E402
    from navsim.common.dataloader import SceneLoader  # noqa: E402
    from w3_agents_v1 import fingerprint  # noqa: E402  (E2's, verbatim; pinned by a test)
    if not navsim_is_from(tree, navsim.__file__):
        sys.exit(f"[export] navsim imported from {navsim.__file__}, not the v1.1 tree {tree}")
    if no_maps:
        Scene._build_map_api = classmethod(lambda cls, map_name: None)
    logs_dir = pathlib.Path(a.logs_root)
    sensors = pathlib.Path(a.logs_root).parent.parent / "sensor_blobs" / ("trainval" if a.split == "navtrain" else "test")   # never read (no sensors)
    by_log: dict = {}
    for t in sample["tokens"]:
        by_log.setdefault(sample["token_log"][t], []).append(t)
    logs = [x for x in a.logs.split(",") if x] or sorted(by_log)
    t0 = time.time()
    rec, fps, per_log = {}, {}, {}
    cv = ConstantVelocityAgent()
    for li, ln in enumerate(logs):
        want = by_log.get(ln, [])
        sf = SceneFilter(num_history_frames=nh, num_future_frames=nf, frame_interval=1, has_route=has_route, max_scenes=None,
                         log_names=[ln], tokens=list(want))
        loader = SceneLoader(data_path=logs_dir, sensor_blobs_path=sensors, scene_filter=sf, sensor_config=SensorConfig.build_no_sensors())
        toks = [t for t in loader.tokens if t in set(want)]
        per_log[ln] = len(toks)
        for tok in toks:
            ai = loader.get_agent_input_from_token(tok)
            frames = loader.scene_frames_dicts[tok]
            sc = Scene.from_scene_dict_list(frames, sensors, num_history_frames=nh, num_future_frames=nf,
                                            sensor_config=SensorConfig.build_no_sensors())
            human = sc.get_future_trajectory(HUMAN_HORIZON_POSES)
            fp = fingerprint(ai.ego_statuses)
            rec[tok] = build_record(tok, frames, nh, ai.ego_statuses, human.poses,
                                    (human.trajectory_sampling.num_poses, human.trajectory_sampling.interval_length),
                                    cv.compute_trajectory(ai).poses, fp)
            fps.setdefault(fp, []).append(tok)
        del loader
        if (li + 1) % 10 == 0 or li + 1 == len(logs):
            print(f"[export] {li + 1}/{len(logs)} logs, {len(rec)} tokens, {time.time() - t0:.0f} s", flush=True)
    want_all = set(sample["tokens"]) if not a.logs else None
    coll = {fp: ts for fp, ts in fps.items() if len(ts) > 1}
    doc = {"split": a.split, "devkit": "navsim v1.1 @ 3e8291b", "navsim_file": navsim.__file__, "n_tokens": len(rec), "n_logs": len(logs),
           "tokens_per_log": per_log, "fingerprint_unique": not coll, "fingerprint_collisions": coll, "maps_stubbed": bool(no_maps),
           "scene_filter": {"num_history_frames": nh, "num_future_frames": nf, "has_route": has_route},
           "sample_sha256_sorted_tokens": sample.get("sha256_sorted_tokens"),
           "fingerprint_note": "a CONSISTENCY check, never the lookup key (E2 D3): the seam agent keys by scene_metadata.initial_token",
           "tokens": rec}
    if want_all is not None:
        doc["n_missing_vs_sample"] = len(want_all - set(rec))
        doc["n_unexpected_vs_sample"] = len(set(rec) - want_all)
    bad = (not rec) or (want_all is not None and bool(doc["n_missing_vs_sample"] or doc["n_unexpected_vs_sample"]))
    man = write_outputs(doc, a.out, a.manifest, bad)
    man.update({"n_logs": len(logs), "maps_stubbed": bool(no_maps), "wall_s": round(time.time() - t0, 1), "navsim_file": navsim.__file__,
                "split": a.split})
    json.dump(man, open(a.manifest, "w", encoding="utf-8"), indent=1)
    print(f"ZZEXPORT_{'REFUSED' if bad else 'DONE'} tokens={len(rec)} missing={doc.get('n_missing_vs_sample')} sha256={man['sha256']}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
