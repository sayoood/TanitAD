#!/usr/bin/env python3
"""NavSim side of the seam (RUN IN THE NAVSIM VENV, Python 3.9).

Exports, per SCORER TOKEN of `warmup_two_stage`, the EXACT `AgentInput` ego
statuses the official scorer hands an agent -- built through the devkit's own
`SceneLoader.get_agent_input_from_token` (the call `run_pdm_score.py:90/144`
makes) -- plus the provenance the TanitAD side needs to join frames, and the
logged HUMAN future for the stage-1 scenes (the devkit's own
`Scene.get_future_trajectory`, i.e. exactly what `HumanAgent` returns).

WHY A SEPARATE EXPORT (and not a re-parse of the pickles on the TanitAD side):
the declared-input manifest is only evidence if the numbers it declares are the
numbers the devkit would hand ANY agent. Re-deriving them from the raw pickles
in another venv would be a second implementation of `get_agent_input` -- and two
implementations of one quantity is how two "independent" checks agree on a
wrong answer.

It also writes a per-token FINGERPRINT of the AgentInput ego statuses, which the
NavSim-side seam agent re-computes on the object the scorer actually hands it and
REQUIRES to match (so the declared values in the manifest are provably the values
the devkit delivered). ⚠️ It is NOT unique across tokens (measured: identical ego
histories on distinct synthetic renders), so it is a consistency check, never the
lookup key.

Also exports every log window of the 7 warmup logs that has 3 history + 8
future frames (for the GT round-trip / lateral-sign / command-order controls).

Outputs (under --out):
  navsim_agent_inputs.json        220 tokens (16 stage-1 + 204 stage-2)
  log_windows_human_future.npz    all admissible log windows (human future etc.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "C:/Users/Admin/navsim/data/maps")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "C:/Users/Admin/navsim/data/openscene")
os.environ.setdefault("NAVSIM_EXP_ROOT", "C:/Users/Admin/navsim/exp")
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", "C:/Users/Admin/navsim/devkit")

import yaml  # noqa: E402
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent  # noqa: E402

DEVKIT = pathlib.Path(os.environ["NAVSIM_DEVKIT_ROOT"])
CFG = DEVKIT / "navsim/planning/script/config/common/train_test_split"
DATA = pathlib.Path(os.environ["OPENSCENE_DATA_ROOT"])
LOGS = DATA / "navsim_logs" / "test"          # default_dataset_paths.yaml: navsim_logs/${data_split}
ORIG_SENSORS = pathlib.Path(os.environ.get("E2_ORIG_SENSORS", DATA / "sensor_blobs" / "test"))  # default_dataset_paths.yaml (ABSENT on this box)
SPLIT = os.environ.get("E2_SPLIT", "warmup_two_stage")   # navhard_two_stage: the next run
WARMUP = DATA / SPLIT
SYN_SENSORS = pathlib.Path(os.environ.get("E2_SYN_SENSORS", WARMUP / "sensor_blobs"))  # existence checks only (no pixel is read here)
SYN_SCENES = WARMUP / "synthetic_scene_pickles"
CAMS = ("cam_l0", "cam_f0", "cam_r0")


def fingerprint(ego_statuses) -> str:
    """sha1 over the float64 bytes of every EgoStatus field, frame by frame.

    Shared VERBATIM with `tanitad_seam_agent.py` (asserted there by a
    self-test against this module's output, not by trust)."""
    h = hashlib.sha1()
    for es in ego_statuses:
        for arr in (es.ego_pose, es.ego_velocity, es.ego_acceleration, es.driving_command):
            h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()


class _PosixSafe(pickle.Unpickler):
    """The devkit's own Windows shim (dataclasses.py load_from_disk)."""

    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        return super().find_class(module, name)


def build_scene_filter() -> SceneFilter:
    split = yaml.safe_load(open(CFG / f"{SPLIT}.yaml", encoding="utf-8"))
    sf = yaml.safe_load(open(CFG / "scene_filter" / f"{SPLIT}.yaml", encoding="utf-8"))
    kw = {k: v for k, v in sf.items() if not k.startswith("_")}
    return SceneFilter(**kw), split


def es_dict(es) -> dict:
    return {"ego_pose": np.asarray(es.ego_pose, dtype=np.float64).tolist(),
            "ego_velocity": np.asarray(es.ego_velocity, dtype=np.float64).tolist(),
            "ego_acceleration": np.asarray(es.ego_acceleration, dtype=np.float64).tolist(),
            "driving_command": np.asarray(es.driving_command, dtype=np.float64).tolist(),
            "_dtypes": [str(np.asarray(es.ego_pose).dtype), str(np.asarray(es.ego_velocity).dtype),
                        str(np.asarray(es.ego_acceleration).dtype),
                        str(np.asarray(es.driving_command).dtype)]}


# --- W1 ADDITION [W8 2026-09-26: SINGLE-STAGE splits (navtest); the two-stage path is untouched] ---
def _stage1_record(loader, scene_filter, tok, cv_agent) -> dict:
    """The stage-1 record, field for field the two-stage loop's (same devkit calls)."""
    ai = loader.get_agent_input_from_token(tok)
    frames = loader.scene_frames_dicts[tok]
    nh = scene_filter.num_history_frames
    sc = Scene.from_scene_dict_list(frames, ORIG_SENSORS, num_history_frames=nh,
                                    num_future_frames=scene_filter.num_future_frames,
                                    sensor_config=SensorConfig.build_no_sensors())
    human = sc.get_future_trajectory(8)
    cams = {c: [str(frames[i]["cams"][c.upper()]["data_path"]) for i in range(nh)] for c in CAMS}
    exists = {c: [bool((ORIG_SENSORS / p).exists()) for p in v] for c, v in cams.items()}
    return {"stage": 1, "frame_type": "ORIGINAL",
            "log_name": frames[nh - 1]["log_name"],
            "map_name": frames[nh - 1]["map_location"],
            "scene_token": frames[nh - 1]["scene_token"],
            "frame_tokens": [frames[i]["token"] for i in range(nh)],
            "timestamps_us": [int(frames[i]["timestamp"]) for i in range(nh)],
            "ego_statuses": [es_dict(e) for e in ai.ego_statuses],
            "cams": cams, "cam_files_exist": exists,
            "human_future_poses": np.asarray(human.poses, dtype=np.float64).tolist(),
            "human_future_sampling": [human.trajectory_sampling.num_poses,
                                      human.trajectory_sampling.interval_length],
            "cv_poses": np.asarray(cv_agent.compute_trajectory(ai).poses, dtype=np.float64).tolist(),
            "fingerprint": fingerprint(ai.ego_statuses)}


def export_single_stage(a, out: pathlib.Path, t0: float) -> int:
    """Every token of a SINGLE-STAGE split (original log frames only; no synthetic scenes, no mapping),
    through the same ``SceneLoader.get_agent_input_from_token`` the one-stage runner calls
    (run_pdm_score_one_stage.py:87). Logs are read from ``--logs`` (the runner's ``navsim_log_path``);
    ``--tokens-file`` ({"tokens": [...], "log_names": [...]}) restricts to a subset."""
    sf = yaml.safe_load(open(CFG / "scene_filter" / f"{SPLIT}.yaml", encoding="utf-8"))
    kw = {k: v for k, v in sf.items() if not k.startswith("_")}
    if kw.get("include_synthetic_scenes") or kw.get("reactive_synthetic_initial_tokens"):
        print(f"⛔ {SPLIT} carries two-stage content — not a single-stage split", flush=True)
        return 2
    subset = None
    if a.tokens_file:
        subset = json.load(open(a.tokens_file, encoding="utf-8"))
        kw["tokens"] = list(subset["tokens"])
        kw["log_names"] = list(subset["log_names"])
    scene_filter = SceneFilter(**kw)
    logs = pathlib.Path(a.logs)
    loader = SceneLoader(data_path=logs, original_sensor_path=ORIG_SENSORS, scene_filter=scene_filter,
                         sensor_config=SensorConfig.build_no_sensors())
    s1 = sorted(loader.tokens)
    want = set(kw["tokens"])
    print(f"[export] single-stage {SPLIT}: tokens {len(s1)} (requested {len(want)}) from {logs}", flush=True)
    if not s1 or set(s1) != want:
        print(f"⛔ the loader returned {len(s1)} tokens for {len(want)} requested "
              f"(missing {len(want - set(s1))}) — refusing a partial export", flush=True)
        return 2
    rec, fps = {}, {}
    cv_agent = ConstantVelocityAgent()
    for i, tok in enumerate(s1):
        rec[tok] = _stage1_record(loader, scene_filter, tok, cv_agent)
        fps.setdefault(rec[tok]["fingerprint"], []).append(tok)
        if i % 2000 == 0:
            print(f"[export] {i + 1}/{len(s1)} ({time.time() - t0:.0f} s)", flush=True)
    coll = {k: v for k, v in fps.items() if len(v) > 1}
    doc = {"_what": ("EXACT AgentInput ego statuses per scorer token, via the devkit's "
                     "SceneLoader.get_agent_input_from_token (run_pdm_score_one_stage.py:87) — SINGLE-STAGE."),
           "devkit": str(DEVKIT), "split": SPLIT, "single_stage": True, "logs_dir": str(logs).replace("\\", "/"),
           "subset": bool(subset), "n_stage1": len(s1), "n_stage2": 0,
           "fingerprint_unique": not coll, "fingerprint_collisions": coll,
           "reactive_all_mapping": [],
           "stage1_cam_files_present": int(sum(sum(v) for r in rec.values() for v in r["cam_files_exist"].values())),
           "stage1_cam_files_expected": int(len(s1) * 4 * len(CAMS)),
           "stage2_cam_files_present": 0, "stage2_cam_files_expected": 0,
           "tokens": rec,
           "fingerprint_note": ("NOT guaranteed unique across tokens — a consistency check per token, never the "
                                "lookup key (the seam is keyed on the scorer token)")}
    with open(out / "navsim_agent_inputs.json", "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"[export] wrote {len(rec)} single-stage tokens; fingerprint unique={not coll}; "
          f"stage-1 cam files present {doc['stage1_cam_files_present']}/{doc['stage1_cam_files_expected']} "
          f"({time.time() - t0:.1f} s)", flush=True)
    return 0


# --- end W1 ADDITION ---
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-log-windows", action="store_true")
    # --- W1 ADDITION [W8 2026-09-26: single-stage flags] ---
    ap.add_argument("--single-stage", action="store_true", help="W8: a single-stage split (navtest)")
    ap.add_argument("--logs", default=None, help="W8: the navsim_log_path of a single-stage run")
    ap.add_argument("--tokens-file", default=None, help="W8: {tokens, log_names} subset of a single-stage split")
    # --- end W1 ADDITION ---
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # --- W1 ADDITION [W8 2026-09-26: single-stage dispatch] ---
    if a.single_stage:
        if not a.logs:
            print("⛔ --single-stage needs --logs (the runner's navsim_log_path)", flush=True)
            return 2
        return export_single_stage(a, out, t0)
    # --- end W1 ADDITION ---

    scene_filter, split = build_scene_filter()
    loader = SceneLoader(data_path=LOGS, original_sensor_path=ORIG_SENSORS,
                         scene_filter=scene_filter, synthetic_sensor_path=SYN_SENSORS,
                         synthetic_scenes_path=SYN_SCENES,
                         sensor_config=SensorConfig.build_no_sensors())
    s1 = sorted(loader.tokens_stage_one)
    s2 = sorted(loader.reactive_tokens_stage_two)
    print(f"[export] stage-1 tokens {len(s1)} | stage-2 reactive tokens {len(s2)} "
          f"| loader.tokens {len(loader.tokens)}", flush=True)
    if len(s1) == 0 or len(s2) == 0:
        print("⛔ EMPTY SET — refusing to export nothing", flush=True)
        return 2

    rec: dict = {}
    fps: dict = {}
    CV = ConstantVelocityAgent()   # the devkit's own agent: stand-in + seam-transparency control
    for tok in s1:
        ai = loader.get_agent_input_from_token(tok)
        frames = loader.scene_frames_dicts[tok]
        nh = scene_filter.num_history_frames
        sc = Scene.from_scene_dict_list(frames, ORIG_SENSORS, num_history_frames=nh,
                                        num_future_frames=scene_filter.num_future_frames,
                                        sensor_config=SensorConfig.build_no_sensors())
        human = sc.get_future_trajectory(8)
        cams = {c: [str(frames[i]["cams"][c.upper()]["data_path"]) for i in range(nh)] for c in CAMS}
        exists = {c: [bool((ORIG_SENSORS / p).exists() or (SYN_SENSORS / p).exists()) for p in v]
                  for c, v in cams.items()}
        fp = fingerprint(ai.ego_statuses)
        rec[tok] = {"stage": 1, "frame_type": "ORIGINAL",
                    "log_name": frames[nh - 1]["log_name"],
                    "map_name": frames[nh - 1]["map_location"],
                    "scene_token": frames[nh - 1]["scene_token"],
                    "frame_tokens": [frames[i]["token"] for i in range(nh)],
                    "timestamps_us": [int(frames[i]["timestamp"]) for i in range(nh)],
                    "ego_statuses": [es_dict(e) for e in ai.ego_statuses],
                    "cams": cams, "cam_files_exist": exists,
                    "human_future_poses": np.asarray(human.poses, dtype=np.float64).tolist(),
                    "human_future_sampling": [human.trajectory_sampling.num_poses,
                                              human.trajectory_sampling.interval_length],
                    "cv_poses": np.asarray(CV.compute_trajectory(ai).poses, dtype=np.float64).tolist(),
                    "fingerprint": fp}
        fps.setdefault(fp, []).append(tok)

    for tok in s2:
        path, log_name = loader.synthetic_scenes[tok]
        sc = Scene.load_from_disk(path, SYN_SENSORS, SensorConfig.build_no_sensors())
        ai = sc.get_agent_input()
        raw = _PosixSafe(open(path, "rb")).load()
        nh = sc.scene_metadata.num_history_frames
        cams = {c: [str(raw["frames"][i]["camera_dict"][c]["data_path"]) for i in range(nh)]
                for c in CAMS}
        exists = {c: [bool((SYN_SENSORS / p).exists()) for p in v] for c, v in cams.items()}
        fp = fingerprint(ai.ego_statuses)
        m = sc.scene_metadata
        rec[tok] = {"stage": 2, "frame_type": "SYNTHETIC",
                    "log_name": m.log_name, "map_name": m.map_name,
                    "scene_token": m.scene_token,            # == pickle stem == frame-bank name
                    "pickle": path.name,
                    "initial_token": m.initial_token,
                    "corresponding_original_scene": m.corresponding_original_scene,
                    "corresponding_original_initial_token": m.corresponding_original_initial_token,
                    "frame_tokens": [f.token for f in sc.frames[:nh]],
                    "timestamps_us": [int(f.timestamp) for f in sc.frames[:nh]],
                    "num_future_frames": int(m.num_future_frames),
                    "ego_statuses": [es_dict(e) for e in ai.ego_statuses],
                    "cams": cams, "cam_files_exist": exists,
                    "cv_poses": np.asarray(CV.compute_trajectory(ai).poses, dtype=np.float64).tolist(),
                    "fingerprint": fp}
        fps.setdefault(fp, []).append(tok)

    coll = {k: v for k, v in fps.items() if len(v) > 1}
    mapping = [[m[0], m[1], [list(p) for p in m[2]]] for m in split["reactive_all_mapping"]]
    doc = {"_what": ("EXACT AgentInput ego statuses per scorer token, via the devkit's "
                     "SceneLoader.get_agent_input_from_token (run_pdm_score.py:90,144)."),
           "devkit": str(DEVKIT), "split": SPLIT,
           "n_stage1": len(s1), "n_stage2": len(s2),
           "fingerprint_unique": not coll, "fingerprint_collisions": coll,
           "reactive_all_mapping": mapping,
           "stage1_cam_files_present": int(sum(sum(v) for r in rec.values() if r["stage"] == 1
                                               for v in r["cam_files_exist"].values())),
           "stage1_cam_files_expected": int(len(s1) * 4 * len(CAMS)),
           "stage2_cam_files_present": int(sum(sum(v) for r in rec.values() if r["stage"] == 2
                                               for v in r["cam_files_exist"].values())),
           "stage2_cam_files_expected": int(len(s2) * 4 * len(CAMS)),
           "tokens": rec}
    # ⚠️ MEASURED 2026-09-19 (first run of this script, rc=3): 8 groups / 19
    # tokens of DISTINCT synthetic renders share a byte-identical 4-frame ego
    # history, so an AgentInput-only key is AMBIGUOUS. The seam is therefore
    # keyed on the SCORER TOKEN (scene.scene_metadata.initial_token, the
    # HumanAgent pattern) and the fingerprint is kept only as a CONSISTENCY
    # check (the AgentInput the scorer hands the agent must equal this export).
    doc["fingerprint_note"] = ("NOT unique across tokens (identical ego histories on "
                               "distinct synthetic renders) — used as a consistency "
                               "check per token, never as the lookup key")
    with open(out / "navsim_agent_inputs.json", "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"[export] wrote {len(rec)} tokens; fingerprint unique={not coll} "
          f"({sum(len(v) for v in coll.values())} tokens in {len(coll)} shared-history "
          f"groups); stage-1 cam files present {doc['stage1_cam_files_present']}/"
          f"{doc['stage1_cam_files_expected']}; stage-2 {doc['stage2_cam_files_present']}/"
          f"{doc['stage2_cam_files_expected']} ({time.time()-t0:.1f} s)", flush=True)

    if a.skip_log_windows:
        return 0
    # ---- every admissible log window of the 7 warmup logs (controls only) ------
    rows = {k: [] for k in ("log", "idx", "token", "human", "heading_hist", "vel0",
                            "acc0", "cmd0", "has_route")}
    for ln in scene_filter.log_names:
        frames = pickle.load(open(LOGS / f"{ln}.pkl", "rb"))
        for i in range(3, len(frames) - 8):
            win = frames[i - 3:i + 9]
            sc = Scene.from_scene_dict_list(win, ORIG_SENSORS, num_history_frames=4,
                                            num_future_frames=8,
                                            sensor_config=SensorConfig.build_no_sensors())
            hum = sc.get_future_trajectory(8).poses
            hist = sc.get_history_trajectory(4).poses
            ds = frames[i]["ego_dynamic_state"]
            rows["log"].append(ln)
            rows["idx"].append(i)
            rows["token"].append(frames[i]["token"])
            rows["human"].append(np.asarray(hum, dtype=np.float64))
            rows["heading_hist"].append(np.asarray(hist, dtype=np.float64))
            rows["vel0"].append(np.asarray(ds[:2], dtype=np.float64))
            rows["acc0"].append(np.asarray(ds[2:], dtype=np.float64))
            rows["cmd0"].append(np.asarray(frames[i]["driving_command"], dtype=np.float64))
            rows["has_route"].append(len(frames[i]["roadblock_ids"]) > 0)
        print(f"[export] log {ln}: cumulative windows {len(rows['idx'])}", flush=True)
    np.savez_compressed(out / "log_windows_human_future.npz",
                        log=np.asarray(rows["log"]), idx=np.asarray(rows["idx"]),
                        token=np.asarray(rows["token"]),
                        human=np.stack(rows["human"]), history=np.stack(rows["heading_hist"]),
                        vel0=np.stack(rows["vel0"]), acc0=np.stack(rows["acc0"]),
                        cmd0=np.stack(rows["cmd0"]), has_route=np.asarray(rows["has_route"]))
    print(f"[export] log windows: {len(rows['idx'])} ({time.time()-t0:.1f} s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
