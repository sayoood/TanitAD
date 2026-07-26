"""nuScenes adapter tests — on a SCHEMA-FAITHFUL synthetic fixture (2026-07-26).

Why a fixture and not real bytes: nuScenes cannot be fetched by an agent (free
account + human acceptance of the Terms of Use), so the corpus is not on disk.
The fixture therefore reproduces the REAL table schema — the same 13 JSON tables,
the same token graph, the same ``[w,x,y,z]`` quaternion + ``translation``
conventions, the same ego(x-fwd, y-left, z-up) / camera(x-right, y-down, z-fwd)
frames, the same 6-camera rig geometry and 1600x900 intrinsics — so every
transform, projection and split rule is exercised for real. The only thing not
proven here is that the released JSON matches its own documented schema.

The headline test is ``test_agent_behind_ego_is_off_front_only``: it proves the
cross-camera visibility label is a PROJECTION, not an approximation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tanitad.data import nuscenes as ns
from tanitad.data.calib import F_REF


# --------------------------------------------------------------------------- #
# Fixture builder — real nuScenes schema, synthetic content                    #
# --------------------------------------------------------------------------- #
def _rotmat_to_quat(r: np.ndarray) -> list[float]:
    """3x3 rotation -> [w, x, y, z] (nuScenes order)."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# camera optical axes expressed in the ego frame (ego: x fwd, y left, z up):
#   cam +x (right) = ego -y ; cam +y (down) = ego -z ; cam +z (fwd) = ego +x
_CAM_PERM = np.array([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])

# realistic nuScenes-like mounting yaws (rad), +ve = toward ego-left
_CAM_YAW = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_LEFT": math.radians(55),
    "CAM_FRONT_RIGHT": math.radians(-55),
    "CAM_BACK_LEFT": math.radians(110),
    "CAM_BACK_RIGHT": math.radians(-110),
    "CAM_BACK": math.radians(180),
}

# nuScenes CAM_FRONT-like intrinsics on 1600x900 (the height-bound camera)
_K = [[1266.417203046554, 0.0, 816.2670197447984],
      [0.0, 1266.417203046554, 491.50706579294757],
      [0.0, 0.0, 1.0]]
# CAM_BACK is the wide one in the real rig (fx~809)
_K_BACK = [[809.2209905677063, 0.0, 829.2196003259838],
           [0.0, 809.2209905677063, 481.77842384512485],
           [0.0, 0.0, 1.0]]


def build_fixture(root: Path, n_scenes: int = 3, n_keyframes: int = 8,
                  turn_deg_per_scene=(0.0, 90.0, 260.0)) -> Path:
    """Write a schema-faithful ``v1.0-trainval`` metadata tree. Returns ``root``.

    Scene 0 drives straight, scene 1 turns 90 deg (a left turn), scene 2 turns
    260 deg (a roundabout-like traversal). Scenes 0 and 1 share a log so the
    drive-disjoint split rule has something to bite on.
    """
    base = root / "v1.0-trainval"
    base.mkdir(parents=True, exist_ok=True)

    sensors, calibs = [], []
    for ch in ns.CAMERAS:
        st = f"sensor-{ch}"
        sensors.append({"token": st, "channel": ch, "modality": "camera"})
        r = _rz(_CAM_YAW[ch]) @ _CAM_PERM
        calibs.append({
            "token": f"cs-{ch}",
            "sensor_token": st,
            "translation": [1.5, 0.0, 1.5] if "FRONT" in ch else [0.0, 0.0, 1.5],
            "rotation": _rotmat_to_quat(r),
            "camera_intrinsic": _K_BACK if ch == "CAM_BACK" else _K,
        })

    visibility = [
        {"token": "1", "level": "v0-40", "description": "0-40% visible"},
        {"token": "2", "level": "v40-60", "description": "40-60% visible"},
        {"token": "3", "level": "v60-80", "description": "60-80% visible"},
        {"token": "4", "level": "v80-100", "description": "80-100% visible"},
    ]
    categories = [
        {"token": "cat-car", "name": "vehicle.car", "description": "car"},
        {"token": "cat-ped", "name": "human.pedestrian.adult", "description": "ped"},
    ]

    logs, scenes, samples, sample_data, ego_poses = [], [], [], [], []
    annotations, instances, maps = [], [], []

    for si in range(n_scenes):
        log_tok = f"log-{si // 2}"                      # scenes 0,1 share a log
        if not any(l["token"] == log_tok for l in logs):
            logs.append({
                "token": log_tok, "logfile": f"n008-{log_tok}",
                "vehicle": "n008", "date_captured": "2018-08-01",
                "location": "boston-seaport" if si < 2 else "singapore-onenorth",
            })
        sc_tok = f"scene-{si}"
        turn = math.radians(turn_deg_per_scene[si % len(turn_deg_per_scene)])
        desc = ["Parked truck, wait for pedestrian",
                "Turn left at intersection, traffic light",
                "Roundabout, yield to traffic"][si % 3]
        scenes.append({
            "token": sc_tok, "log_token": log_tok, "name": f"scene-{si:04d}",
            "description": desc, "nbr_samples": n_keyframes,
            "first_sample_token": f"sample-{si}-0",
            "last_sample_token": f"sample-{si}-{n_keyframes - 1}",
        })
        maps.append({"token": f"map-{si}", "log_tokens": [log_tok],
                     "category": "semantic_prior", "filename": f"maps/{si}.png"})

        # ego drives a constant-speed arc, total heading change = `turn`
        speed, dt = 8.0, 0.5
        for k in range(n_keyframes):
            frac = k / max(1, n_keyframes - 1)
            yaw = turn * frac
            arc = speed * dt * k
            if abs(turn) > 1e-6:
                radius = speed * dt * (n_keyframes - 1) / turn
                x = radius * math.sin(yaw)
                y = radius * (1 - math.cos(yaw))
            else:
                x, y = arc, 0.0
            ts = 1_500_000_000_000_000 + int((si * 100 + k) * dt * 1e6)
            ep_tok = f"ego-{si}-{k}"
            ego_poses.append({"token": ep_tok, "timestamp": ts,
                              "translation": [x, y, 0.0],
                              "rotation": _rotmat_to_quat(_rz(yaw))})
            s_tok = f"sample-{si}-{k}"
            samples.append({
                "token": s_tok, "timestamp": ts, "scene_token": sc_tok,
                "next": f"sample-{si}-{k + 1}" if k < n_keyframes - 1 else "",
                "prev": f"sample-{si}-{k - 1}" if k else "",
            })
            for ch in ns.CAMERAS:
                sample_data.append({
                    "token": f"sd-{si}-{k}-{ch}", "sample_token": s_tok,
                    "ego_pose_token": ep_tok, "calibrated_sensor_token": f"cs-{ch}",
                    "filename": f"samples/{ch}/{si}_{k}.jpg", "fileformat": "jpg",
                    "width": 1600, "height": 900, "timestamp": ts,
                    "is_key_frame": True, "next": "", "prev": "",
                })

            # --- agents placed in the EGO frame, then mapped to global --------
            # a: 20 m straight ahead  -> CAM_FRONT must see it
            # b: 20 m straight behind -> ONLY the rear cameras
            # c: 15 m to the left     -> the left cameras, not CAM_FRONT
            r_e = _rz(yaw)
            for name, ego_xyz, cat, vis in (
                    ("a", [20.0, 0.0, 0.0], "cat-car", "4"),
                    ("b", [-20.0, 0.0, 0.0], "cat-car", "4"),
                    ("c", [0.0, 15.0, 0.0], "cat-ped", "3")):
                g = r_e @ np.asarray(ego_xyz) + np.array([x, y, 0.0])
                inst_tok = f"inst-{si}-{name}"
                annotations.append({
                    "token": f"ann-{si}-{k}-{name}", "sample_token": s_tok,
                    "instance_token": inst_tok, "visibility_token": vis,
                    "translation": [float(v) for v in g],
                    "size": [1.9, 4.6, 1.7], "rotation": _rotmat_to_quat(r_e),
                    "num_lidar_pts": 42, "num_radar_pts": 3,
                    "next": "", "prev": "", "attribute_tokens": [],
                })
                if not any(i["token"] == inst_tok for i in instances):
                    instances.append({
                        "token": inst_tok, "category_token": cat,
                        "nbr_annotations": n_keyframes,
                        "first_annotation_token": f"ann-{si}-0-{name}",
                        "last_annotation_token": f"ann-{si}-{n_keyframes-1}-{name}",
                    })

    payload = {
        "sensor": sensors, "calibrated_sensor": calibs, "visibility": visibility,
        "category": categories, "log": logs, "scene": scenes, "sample": samples,
        "sample_data": sample_data, "ego_pose": ego_poses,
        "sample_annotation": annotations, "instance": instances, "map": maps,
        "attribute": [],
    }
    for t, rows in payload.items():
        (base / f"{t}.json").write_text(json.dumps(rows), encoding="utf-8")
    return root


@pytest.fixture
def idx(tmp_path):
    build_fixture(tmp_path)
    return ns.NuScenesIndex(ns.load_tables(tmp_path))


# --------------------------------------------------------------------------- #
# Acquisition guard                                                            #
# --------------------------------------------------------------------------- #
def test_missing_metadata_raises_actionable_terms_error(tmp_path):
    """No silent fallback, no mirror: an explicit human-action error."""
    with pytest.raises(ns.NuScenesTermsError) as e:
        ns.load_tables(tmp_path / "nope")
    msg = str(e.value)
    assert "terms-of-use" in msg and "HUMAN" in msg
    assert "unofficial mirror" in msg
    assert "reachability is NOT permission" in msg      # the AWS-bucket nuance
    assert "CC BY-NC-SA 4.0" in msg
    # the cheapest-first acquisition order must stay in the message
    assert "0.46 GB" in msg and "16 MiB" in msg and "44.9 GB" in msg


# --------------------------------------------------------------------------- #
# Index + chains                                                               #
# --------------------------------------------------------------------------- #
def test_index_builds_scene_chains(idx):
    assert len(idx.scenes()) == 3
    for sc in idx.scenes():
        samples = idx.samples_of_scene(sc["token"])
        assert len(samples) == 8, "the `next` chain must yield every keyframe"
        # chain order, not timestamp sort
        assert [s["token"] for s in samples] == [
            f"sample-{sc['token'].split('-')[1]}-{k}" for k in range(8)]


def test_six_cameras_resolve_per_sample(idx):
    s = idx.samples_of_scene("scene-0")[0]
    for ch in ns.CAMERAS:
        sd = idx.sample_data(s["token"], ch)
        assert sd is not None, ch
        intr = idx.camera_intrinsics_of(sd)
        assert intr.width == 1600 and intr.height == 900
        assert intr.dist == (0.0,) * 5, "nuScenes ships no distortion coeffs"


def test_intrinsics_are_per_sample_not_the_nominal_constant(idx):
    """CAM_BACK really is a different lens — proving we read per-sample calib."""
    s = idx.samples_of_scene("scene-0")[0]
    f_front = idx.camera_intrinsics_of(idx.sample_data(s["token"], "CAM_FRONT")).fx
    f_back = idx.camera_intrinsics_of(idx.sample_data(s["token"], "CAM_BACK")).fx
    assert abs(f_front - 1266.417) < 1e-2
    assert abs(f_back - 809.221) < 1e-2
    assert f_front != f_back


# --------------------------------------------------------------------------- #
# Ego track + actions                                                          #
# --------------------------------------------------------------------------- #
def test_ego_track_shape_and_speed(idx):
    poses, t = ns.ego_track(idx, "scene-0")
    assert poses.shape == (8, 4) and t.shape == (8,)
    # fixture drives 8 m/s
    assert np.allclose(poses[:, 3], 8.0, atol=0.5), poses[:, 3]
    assert np.isfinite(poses).all()


def test_ego_track_recovers_the_turn(idx):
    """scene-1 turns 90 deg, scene-2 turns 260 deg — the yaw must show it."""
    for scene, expect in (("scene-1", 90.0), ("scene-2", 260.0)):
        poses, _ = ns.ego_track(idx, scene)
        yaw = np.unwrap(poses[:, 2].astype(np.float64))
        assert abs(math.degrees(yaw[-1] - yaw[0]) - expect) < 5.0


def test_actions_finite_and_shaped(idx):
    poses, t = ns.ego_track(idx, "scene-2")
    a = ns.actions_from_track(poses, t)
    assert a.shape == (8, 2)
    assert np.isfinite(a).all(), "no NaN/inf from the standstill guard"
    assert (np.abs(a[:, 0]) > 1e-3).any(), "a 260 deg turn must produce steer"


def test_actions_no_blowup_at_standstill(idx):
    poses, t = ns.ego_track(idx, "scene-2")
    poses[:, 3] = 0.0                              # parked
    a = ns.actions_from_track(poses, t)
    assert np.isfinite(a).all()
    assert np.allclose(a[:, 0], 0.0), "steer must be 0, not inf, at v=0"


# --------------------------------------------------------------------------- #
# 3D agent tracks                                                              #
# --------------------------------------------------------------------------- #
def test_agent_tracks_carry_instance_ids_and_visibility(idx):
    tr = ns.agent_tracks(idx, "scene-0")
    assert len(tr) == 8 * 3
    inst = {r["instance_token"] for r in tr}
    assert len(inst) == 3, "an agent is a TRACK across keyframes, not a detection"
    assert {r["category"] for r in tr} == {"vehicle.car", "human.pedestrian.adult"}
    assert {r["visibility_level"] for r in tr} <= {"v0-40", "v40-60", "v60-80",
                                                   "v80-100"}


# --------------------------------------------------------------------------- #
# THE HEADLINE — cross-camera visibility is a PROJECTION                       #
# --------------------------------------------------------------------------- #
def test_agent_ahead_is_seen_by_cam_front(idx):
    s = idx.samples_of_scene("scene-0")[0]
    rows = {r["instance_token"]: r for r in ns.camera_visibility(idx, s["token"])}
    a = rows["inst-0-a"]                       # 20 m straight ahead
    assert a["in_ego_camera"] is True
    assert "CAM_FRONT" in a["cameras"]
    assert a["off_front_only"] is False
    assert 15.0 < a["range_m"] < 25.0


def test_agent_behind_ego_is_off_front_only(idx):
    """THE LABEL: an agent 20 m BEHIND the ego is invisible to CAM_FRONT and
    visible to the rear cameras — computed by projection, not approximated."""
    s = idx.samples_of_scene("scene-0")[0]
    rows = {r["instance_token"]: r for r in ns.camera_visibility(idx, s["token"])}
    b = rows["inst-0-b"]
    assert b["in_ego_camera"] is False, "CAM_FRONT cannot see behind"
    assert b["off_front_only"] is True
    assert b["n_cameras"] >= 1
    assert any(c.startswith("CAM_BACK") for c in b["cameras"]), b["cameras"]


def test_agent_to_the_left_is_off_front_only(idx):
    s = idx.samples_of_scene("scene-0")[0]
    rows = {r["instance_token"]: r for r in ns.camera_visibility(idx, s["token"])}
    c = rows["inst-0-c"]                        # 15 m to ego-LEFT
    assert c["in_ego_camera"] is False
    assert c["off_front_only"] is True
    assert any("LEFT" in ch for ch in c["cameras"]), c["cameras"]
    assert not any("RIGHT" in ch for ch in c["cameras"]), c["cameras"]


def test_visibility_holds_through_a_turn(idx):
    """The projection must track the ego's rotation, not a fixed world axis."""
    for k in range(8):
        s = idx.samples_of_scene("scene-2")[k]
        rows = {r["instance_token"]: r
                for r in ns.camera_visibility(idx, s["token"])}
        assert rows["inst-2-a"]["in_ego_camera"] is True, f"keyframe {k}"
        assert rows["inst-2-b"]["off_front_only"] is True, f"keyframe {k}"


def test_corpus_visibility_report_aggregates(idx):
    rep = ns.corpus_camera_visibility_report(idx)
    assert rep["n_annotations"] == 3 * 8 * 3
    assert rep["n_off_front_only"] > 0
    assert 0.0 < rep["frac_off_front_only"] < 1.0
    assert "CAM_FRONT" in rep["per_camera_hits"]


# --------------------------------------------------------------------------- #
# Scenario statistics (heuristic — clearly labelled as such)                   #
# --------------------------------------------------------------------------- #
def test_scenario_stats_flag_turn_and_roundabout(idx):
    by = {s["name"]: s for s in
          (ns.scene_scenario_stats(idx, sc) for sc in idx.scenes())}
    assert by["scene-0000"]["is_turn_heuristic"] is False
    assert by["scene-0001"]["is_left_turn_heuristic"] is True
    assert by["scene-0002"]["is_roundabout_heuristic"] is True
    # the independent description signal
    assert by["scene-0001"]["description_traffic_light"] is True
    assert by["scene-0002"]["description_roundabout"] is True


def test_corpus_scenario_report_counts(idx):
    rep = ns.corpus_scenario_report(idx)
    assert rep["n_scenes"] == 3 and rep["n_error"] == 0
    assert rep["counts"]["roundabout_heuristic"] == 1
    assert rep["counts"]["left_turn_heuristic"] == 1
    assert set(rep["locations"]) == {"boston-seaport", "singapore-onenorth"}


# --------------------------------------------------------------------------- #
# Geometry — the height-bound wall, and the fix                                #
# --------------------------------------------------------------------------- #
def test_build_episode_lands_canonical_feff(idx, tmp_path):
    """The whole point of the R1 fold-in: f_eff == 266, not 360."""
    def fake_decode(_p):
        return torch.randint(0, 256, (3, 900, 1600), dtype=torch.uint8)

    ep, geo = ns.build_episode(idx, "scene-0", tmp_path, episode_id=0,
                               decode_fn=fake_decode)
    assert geo["f_eff_px"] == F_REF
    assert geo["canon"] == "pinhole_rectify(D-016 R1)"
    assert 0.5 < geo["observed_frac"] < 1.0, "honest masked periphery"
    assert ep.frames.shape == (6, 9, 256, 256)      # 8 keyframes - (n_stack-1)*2
    assert ep.frames.dtype == torch.uint8
    assert ep.actions.shape == (6, 2) and ep.poses.shape == (6, 4)


def test_missing_image_blobs_raise_actionable_error(idx, tmp_path):
    with pytest.raises(ns.NuScenesTermsError, match="keyframe blob missing"):
        ns.build_episode(idx, "scene-0", tmp_path, episode_id=0)


# --------------------------------------------------------------------------- #
# I3 split unit = the LOG, not the scene                                       #
# --------------------------------------------------------------------------- #
def test_split_unit_is_the_log(idx):
    scenes = {s["token"]: s for s in idx.scenes()}
    assert ns.split_unit_of(idx, scenes["scene-0"]) == "log-0"
    assert ns.split_unit_of(idx, scenes["scene-1"]) == "log-0"   # same drive
    assert ns.split_unit_of(idx, scenes["scene-2"]) == "log-1"


# --------------------------------------------------------------------------- #
# END-TO-END through the REAL lake ingest + the REAL guards                    #
# --------------------------------------------------------------------------- #
def _fake_decode(_p):
    return torch.randint(0, 256, (3, 900, 1600), dtype=torch.uint8)


def test_ingest_routes_nuscenes_to_segregated_copyleft_shard(tmp_path):
    """The integration proof: real ingestor -> real ShardWriter -> SA subtree."""
    from tanitad.lake.ingest import NuScenesIngestor, ingest_source
    root = build_fixture(tmp_path / "ns")
    lake = tmp_path / "lake"
    ing = NuScenesIngestor(size=64, decode_fn=_fake_decode, val_frac=0.5)
    summary = ingest_source(ing, root, lake, verbose=False)

    assert summary["license_class"] == "nc-research"
    assert summary["license_name"] == "CC-BY-NC-SA-4.0"
    assert summary["share_alike"] is True
    built = sum(v["built"] for v in summary["per_split"].values())
    assert built == 3, summary
    for shard in summary["shards"]:
        assert shard.startswith("shards/nc-research/sharealike/nuscenes/"), shard
    assert not list((lake / "shards" / "owned-safe").rglob("*.tar")), \
        "nuScenes must never land in an owned-safe shard"


def test_ingested_nuscenes_is_refused_from_commercial_tier_C(tmp_path):
    """End-to-end: ingest for real, then run the ACTUAL tier-C export guard."""
    import pyarrow.dataset as pads
    from tanitad.lake.catalog import resolve_members
    from tanitad.lake.ingest import NuScenesIngestor, ingest_source
    from tanitad.lake.license_guard import LicenseScopeError, verify_license_scope
    from tanitad.lake.view import owned_safe_commercial_view

    root = build_fixture(tmp_path / "ns")
    lake = tmp_path / "lake"
    ingest_source(NuScenesIngestor(size=64, decode_fn=_fake_decode, val_frac=0.5),
                  root, lake, verbose=False)

    rows = resolve_members(lake, pads.field("source") == "nuscenes")
    assert rows, "fixture must have produced catalog rows"
    for r in rows:
        assert r["license_class"] == "nc-research" and r["share_alike"]
        assert not r["commercial_ok"]
    with pytest.raises(LicenseScopeError):
        verify_license_scope(rows, allowed_classes={"owned-safe"},
                             require_commercial_ok=True, forbid_share_alike=True,
                             context="tier-C")
    # and the C VIEW itself resolves to nothing at all in an NC-only lake
    with pytest.raises(ValueError, match="0 episodes"):
        owned_safe_commercial_view(lake).resolve()


def test_ingest_split_is_drive_disjoint(tmp_path):
    """No log may appear in both splits (scenes 0,1 share log-0)."""
    from tanitad.lake.ingest import NuScenesIngestor, ingest_source
    root = build_fixture(tmp_path / "ns")
    ing = NuScenesIngestor(size=64, decode_fn=_fake_decode, val_frac=0.5)
    ingest_source(ing, root, tmp_path / "lake", verbose=False)
    units = ing.discover(root)
    split = ing.split_units(units, seed=0)
    logs_tr = {ns.split_unit_of(ing._idx, u) for u in split["train"]}
    logs_va = {ns.split_unit_of(ing._idx, u) for u in split["val"]}
    assert not (logs_tr & logs_va), f"drive leak: {logs_tr & logs_va}"
