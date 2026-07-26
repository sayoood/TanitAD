"""nuScenes adapter — METADATA-FIRST, image-optional (D-nuScenes, 2026-07-26).

WHY METADATA-FIRST
------------------
The H2 survey's cheap-discriminating step: the ~0.4 GB ``v1.0-trainval`` metadata
archive answers *"is this corpus worth 60 GB of keyframes"* offline, before a
single image byte moves. Everything in this module except :func:`build_episode`
runs on metadata alone — ego tracks, 3D agent tracks, per-camera calibration,
scenario statistics, and the cross-camera visibility projection that is the whole
reason nuScenes ranks first for the camera-attention workstream.

⚠️ LICENSE — READ BEFORE USING THIS MODULE
------------------------------------------
nuScenes is **CC BY-NC-SA 4.0** (``nc-research``, ``share_alike=True`` in
``lake.schema.SOURCE_REGISTRY``, corrected 2026-07-26). Consequences that this
module cannot enforce on its own and that the guard enforces downstream:

* it can **NEVER** enter TanitDataSet-C or any commercial artifact;
* it is **copyleft** — records route to the segregated shard
  ``shards/nc-research/sharealike/nuscenes/…`` and must never co-mingle;
* **derivatives inherit it**: a model or label set built on nuScenes is itself
  ``nc``/SA;
* we ship **pointers + derived features, never source bytes**.

The bytes additionally require a human to register at nuscenes.org and accept the
Terms of Use. **An agent must not do either** (account creation and terms
acceptance are outside an agent's boundary). ``load_tables`` therefore fails with
an explicit, actionable error instead of fetching anything itself.

⚠️ Precision on the channel, because it is easy to get wrong in both directions:
Motional also publishes the corpus through an **AWS Open Data** bucket
(``motional-nuscenes``, listed at registry.opendata.aws) whose object listing is
publicly readable — so the sizes below are MEASURED, and that bucket is an
**official channel, not a third-party mirror**. Technical reachability is
nevertheless **not** licence permission: the Terms of Use remain the operative
instrument, nuscenes.org gates the download behind accepting them, and that page
returned EMPTY on every fetch attempt (4×) while being documented to carry
*modifications* to the CC grant. Conservative branch, which is what we follow: a
human accepts the terms, then downloads. Do not treat bucket readability as
consent, and do not substitute an unofficial third-party mirror either.

MEASURED download sizes (public bucket listing, 2026-07-26):
  ``v1.0-trainval_meta.tgz``            461,678,030 B  (0.46 GB)  <- metadata
  ``nuScenes-map-expansion-v1.2.zip``    17,136,555 B  (16 MiB)   <- LANE GRAPH
  trainval keyframe blobs (10 files)  44,902,690,772 B (44.9 GB)
  ``can_bus.zip``                       780,974,697 B  (745 MiB)  <- routes
So the entire value question is answerable for **0.48 GB**, and the keyframe tier
is **44.9 GB**, not the ~60 GB the H2 survey estimated.

GEOMETRY
--------
nuScenes CAM_FRONT is ``fx≈1266`` on 1600×900. Our canonical square crop is
height-bound at 900 and would land ``f_eff≈360`` against the canonical **266** —
the PandaSet-class wall. This module therefore canonicalizes with
:func:`tanitad.data.calib.pinhole_rectify` (D-016 R1, folded into ``calib.py``
2026-07-26), which lands ``f_eff == 266`` exactly and reports an honest
``observed_frac`` mask instead of a silent 1.35× zoom. Intrinsics are read
**per sample** from ``calibrated_sensor``; the module-level nominal constant is
never asserted on a record (the PhysicalAI two-rig ``cy`` lesson).

NO NEW DEPENDENCY
-----------------
The nuScenes devkit is Apache-2.0 but heavyweight; the released metadata is plain
JSON with a stable, documented schema, so this module reads it directly with the
stdlib + numpy. That keeps the ingest path identical in shape to the comma2k19 /
PandaSet / ZOD loaders.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from tanitad.data._contract import assert_contract, finite_diff_accel
from tanitad.data.calib import F_REF, PinholeIntrinsics, pinhole_rectify
from tanitad.data.toy_driving import ToyEpisode

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
DEFAULT_VERSION = "v1.0-trainval"

#: The 13 JSON tables shipped in the metadata archive.
TABLES = ("attribute", "calibrated_sensor", "category", "ego_pose", "instance",
          "log", "map", "sample", "sample_annotation", "sample_data", "scene",
          "sensor", "visibility")

#: The 6-camera surround rig. CAM_FRONT is our ego view; the other five are the
#: "was this camera necessary" candidates for the camera-attention workstream.
CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")
EGO_CAMERA = "CAM_FRONT"

#: Keyframe (annotated) rate. Non-keyframe `sample_data` runs at ~12 Hz.
KEYFRAME_HZ = 2.0

#: nuScenes stores timestamps in MICROseconds.
US = 1e-6


class NuScenesTermsError(RuntimeError):
    """Raised when the metadata is absent — with the (human) acquisition steps."""


_ACQUIRE_MSG = """\
nuScenes metadata not found at {root!r} (looked for {version}/scene.json).

nuScenes CANNOT be fetched by an agent. It requires, as a HUMAN:
  1. a free account at https://www.nuscenes.org/sign-up
  2. explicit acceptance of https://www.nuscenes.org/terms-of-use
  3. download (MEASURED sizes), cheapest-first:
       v1.0-trainval_meta.tgz          0.46 GB  <- answers the value question
       nuScenes-map-expansion-v1.2.zip 16 MiB   <- the ROUTABLE LANE GRAPH
       can_bus.zip                     745 MiB  <- per-scene route paths
       keyframe blobs                  44.9 GB  <- only if the above justify it
  4. extract so that {root}/{version}/scene.json exists
     (map expansion -> {root}/maps/expansion/, CAN bus -> {root}/can_bus/)

The Terms of Use are the operative instrument and carry documented modifications
on top of the CC BY-NC-SA 4.0 grant which we have not been able to read (the page
returned empty on 4 fetch attempts). Motional's AWS Open Data bucket is an
official channel and its listing is public, but reachability is NOT permission —
a human accepts the terms first. Do not substitute an unofficial mirror.

License once acquired: CC BY-NC-SA 4.0 -> nc-research + share_alike -> NEVER in
TanitDataSet-C, segregated copyleft shard only, derivatives inherit NC+SA.
"""


# --------------------------------------------------------------------------- #
# Quaternion helpers (nuScenes stores rotation as [w, x, y, z])                #
# --------------------------------------------------------------------------- #
def quat_to_rotmat(q: Iterable[float]) -> np.ndarray:
    """[w,x,y,z] unit quaternion -> 3x3 rotation matrix (right-handed)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_to_yaw(q: Iterable[float]) -> float:
    """Heading (rad) about +z of a [w,x,y,z] quaternion, in the global frame."""
    r = quat_to_rotmat(q)
    return float(math.atan2(r[1, 0], r[0, 0]))


def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# Table loading + indexing                                                     #
# --------------------------------------------------------------------------- #
def load_tables(root: str | Path, version: str = DEFAULT_VERSION
                ) -> dict[str, list[dict]]:
    """Load the nuScenes JSON tables. Metadata only — touches no image bytes."""
    root = Path(root)
    base = root / version
    if not (base / "scene.json").exists():
        raise NuScenesTermsError(_ACQUIRE_MSG.format(root=str(root),
                                                     version=version))
    out: dict[str, list[dict]] = {}
    for t in TABLES:
        p = base / f"{t}.json"
        out[t] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    return out


@dataclass
class NuScenesIndex:
    """Token-indexed view over the tables + the scene->sample->sample_data chains.

    Pure metadata. Construct once per corpus; every accessor is O(1).
    """

    tables: dict[str, list[dict]]
    by: dict[str, dict[str, dict]] = field(default_factory=dict)
    _samples_of_scene: dict[str, list[dict]] = field(default_factory=dict)
    _sd_of_sample: dict[str, dict[str, dict]] = field(default_factory=dict)
    _ann_of_sample: dict[str, list[dict]] = field(default_factory=dict)

    def __post_init__(self):
        for t, rows in self.tables.items():
            self.by[t] = {r["token"]: r for r in rows}

        # sensor channel per calibrated_sensor (camera identity)
        self._channel_of_cs = {
            cs["token"]: self.by["sensor"][cs["sensor_token"]]["channel"]
            for cs in self.tables.get("calibrated_sensor", [])
            if cs.get("sensor_token") in self.by.get("sensor", {})}

        # scene -> ordered samples (follow the `next` chain, not timestamp sort:
        # the chain is the authoritative ordering)
        for sc in self.tables.get("scene", []):
            chain, tok = [], sc.get("first_sample_token", "")
            seen = set()
            while tok and tok in self.by["sample"] and tok not in seen:
                seen.add(tok)
                s = self.by["sample"][tok]
                chain.append(s)
                tok = s.get("next", "")
            self._samples_of_scene[sc["token"]] = chain

        # sample -> {channel: sample_data} for KEYFRAMES only
        for sd in self.tables.get("sample_data", []):
            if not sd.get("is_key_frame"):
                continue
            ch = self._channel_of_cs.get(sd.get("calibrated_sensor_token"))
            if ch is None:
                continue
            self._sd_of_sample.setdefault(sd["sample_token"], {})[ch] = sd

        for a in self.tables.get("sample_annotation", []):
            self._ann_of_sample.setdefault(a["sample_token"], []).append(a)

    # -- accessors ---------------------------------------------------------- #
    def scenes(self) -> list[dict]:
        return list(self.tables.get("scene", []))

    def samples_of_scene(self, scene_token: str) -> list[dict]:
        return self._samples_of_scene.get(scene_token, [])

    def sample_data(self, sample_token: str, channel: str = EGO_CAMERA):
        return self._sd_of_sample.get(sample_token, {}).get(channel)

    def annotations_of(self, sample_token: str) -> list[dict]:
        return self._ann_of_sample.get(sample_token, [])

    def log_of_scene(self, scene: dict) -> dict:
        return self.by["log"].get(scene.get("log_token", ""), {})

    def location_of_scene(self, scene: dict) -> str:
        return self.log_of_scene(scene).get("location", "")

    def category_of(self, ann: dict) -> str:
        inst = self.by["instance"].get(ann.get("instance_token", ""), {})
        return self.by["category"].get(inst.get("category_token", ""), {}
                                       ).get("name", "")

    def visibility_of(self, ann: dict) -> dict:
        """nuScenes' OWN visibility attribute: fraction of the box visible across
        ALL SIX cameras, binned (``level`` v0-40 / v40-60 / v60-80 / v80-100)."""
        return self.by["visibility"].get(ann.get("visibility_token", ""), {})

    def ego_pose_of(self, sd: dict) -> dict:
        return self.by["ego_pose"].get(sd.get("ego_pose_token", ""), {})

    def calibrated_sensor_of(self, sd: dict) -> dict:
        return self.by["calibrated_sensor"].get(
            sd.get("calibrated_sensor_token", ""), {})

    def camera_intrinsics_of(self, sd: dict) -> PinholeIntrinsics:
        """REAL per-sample intrinsics. Never the module-level nominal constant.

        nuScenes' ``camera_intrinsic`` is a bare 3x3 with no distortion coeffs, so
        ``dist`` stays zero and :func:`pinhole_rectify` degrades to its honest
        pad-crop half — no invented lens model.
        """
        cs = self.calibrated_sensor_of(sd)
        k = cs.get("camera_intrinsic") or []
        if not k:
            raise ValueError(f"sample_data {sd.get('token')!r} has no "
                             f"camera_intrinsic (is it a camera?)")
        return PinholeIntrinsics(
            fx=float(k[0][0]), fy=float(k[1][1]),
            cx=float(k[0][2]), cy=float(k[1][2]),
            width=int(sd.get("width", 1600)), height=int(sd.get("height", 900)))


# --------------------------------------------------------------------------- #
# Ego track -> the (x, y, yaw, v) pose contract                                #
# --------------------------------------------------------------------------- #
def ego_track(idx: NuScenesIndex, scene_token: str,
              channel: str = EGO_CAMERA) -> tuple[np.ndarray, np.ndarray]:
    """Per-keyframe ego poses ``[T,4] (x, y, yaw, v)`` + timestamps ``[T]`` (s).

    Speed is the central finite difference of global translation w.r.t. the real
    (non-uniform) timestamps — nuScenes keyframes are nominally 2 Hz but not
    exactly uniform, so a constant ``dt`` would bias speed.
    """
    xs, ys, yaws, ts = [], [], [], []
    for s in idx.samples_of_scene(scene_token):
        sd = idx.sample_data(s["token"], channel)
        if sd is None:
            continue
        ep = idx.ego_pose_of(sd)
        tr = ep.get("translation") or [0.0, 0.0, 0.0]
        xs.append(float(tr[0]))
        ys.append(float(tr[1]))
        yaws.append(quat_to_yaw(ep.get("rotation") or [1, 0, 0, 0]))
        ts.append(float(sd.get("timestamp", ep.get("timestamp", 0))) * US)

    n = len(xs)
    if n < 2:
        raise ValueError(f"scene {scene_token!r}: {n} keyframes — too short")
    x = np.asarray(xs, np.float64)
    y = np.asarray(ys, np.float64)
    t = np.asarray(ts, np.float64)
    yaw = np.unwrap(np.asarray(yaws, np.float64))

    dt = np.gradient(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if (dt > 0).any() else 1.0 / KEYFRAME_HZ
    v = np.hypot(np.gradient(x) / dt, np.gradient(y) / dt)

    poses = np.stack([x, y, wrap_pi(yaw), v], axis=1).astype(np.float32)
    return poses, t.astype(np.float32)


def actions_from_track(poses: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``[T,2] (steer rad, accel m/s^2)`` from the ego track.

    ``steer`` is the per-step yaw RATE scaled to a bicycle-model steering angle
    proxy (``atan(L * yawrate / v)``, L = wheelbase); at standstill it is 0 rather
    than a divide-by-zero blow-up. ``accel`` is the contract's forward finite
    difference of speed. Both are POSE-DERIVED — nuScenes ships no CAN, so the
    ingestor sets ``action_source='pose_derived'``/``has_can=False``.
    """
    L = 2.588                                   # Renault Zoe wheelbase [m]
    yaw = np.unwrap(poses[:, 2].astype(np.float64))
    v = poses[:, 3].astype(np.float64)
    dt = np.gradient(t.astype(np.float64))
    dt[dt <= 0] = 1.0 / KEYFRAME_HZ
    yawrate = np.gradient(yaw) / dt
    with np.errstate(divide="ignore", invalid="ignore"):
        steer = np.arctan(np.where(v > 0.5, L * yawrate / np.maximum(v, 1e-6), 0.0))
    steer = np.nan_to_num(steer, nan=0.0, posinf=0.0, neginf=0.0)
    accel = finite_diff_accel(v, float(np.median(dt)))
    return np.stack([steer, accel], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3D agent tracks                                                              #
# --------------------------------------------------------------------------- #
def agent_tracks(idx: NuScenesIndex, scene_token: str) -> list[dict]:
    """Per-keyframe 3D agent boxes with INSTANCE ids (a track, not a detection).

    Returns one dict per (sample, annotation): ``instance_token`` (the track id),
    ``category``, global ``translation``/``size``/``rotation``, the nuScenes
    ``visibility`` bin, and lidar/radar point counts (an occlusion proxy).
    """
    out: list[dict] = []
    for si, s in enumerate(idx.samples_of_scene(scene_token)):
        for a in idx.annotations_of(s["token"]):
            vis = idx.visibility_of(a)
            out.append({
                "sample_index": si,
                "sample_token": s["token"],
                "instance_token": a.get("instance_token", ""),
                "category": idx.category_of(a),
                "translation": [float(v) for v in a.get("translation", [0, 0, 0])],
                "size": [float(v) for v in a.get("size", [0, 0, 0])],
                "rotation": [float(v) for v in a.get("rotation", [1, 0, 0, 0])],
                "visibility_level": vis.get("level", ""),
                "num_lidar_pts": int(a.get("num_lidar_pts", 0)),
                "num_radar_pts": int(a.get("num_radar_pts", 0)),
            })
    return out


# --------------------------------------------------------------------------- #
# THE CROSS-CAMERA VISIBILITY PROJECTION                                       #
# --------------------------------------------------------------------------- #
def global_to_camera(points_global: np.ndarray, ego_pose: dict,
                     calib_sensor: dict) -> np.ndarray:
    """``[N,3]`` global points -> the camera frame (+z forward, +x right, +y down).

    Two rigid transforms, exactly as the devkit composes them:
    global -> ego (inverse of ``ego_pose``) -> camera (inverse of
    ``calibrated_sensor``, which is expressed ego->sensor).
    """
    p = np.asarray(points_global, dtype=np.float64).reshape(-1, 3)
    r_e = quat_to_rotmat(ego_pose.get("rotation") or [1, 0, 0, 0])
    t_e = np.asarray(ego_pose.get("translation") or [0, 0, 0], dtype=np.float64)
    p_ego = (p - t_e) @ r_e                      # r_e.T @ (p - t_e), row-vector form

    r_c = quat_to_rotmat(calib_sensor.get("rotation") or [1, 0, 0, 0])
    t_c = np.asarray(calib_sensor.get("translation") or [0, 0, 0], dtype=np.float64)
    return (p_ego - t_c) @ r_c


def project_to_pixels(points_cam: np.ndarray, intr: PinholeIntrinsics
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points -> ``([N,2]`` pixels, ``[N]`` in-front-of-camera mask)."""
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    z = p[:, 2]
    infront = z > 1e-3
    zz = np.where(infront, z, 1.0)
    u = intr.fx * p[:, 0] / zz + intr.cx
    v = intr.fy * p[:, 1] / zz + intr.cy
    return np.stack([u, v], axis=1), infront


def camera_visibility(idx: NuScenesIndex, sample_token: str,
                      cameras: Iterable[str] = CAMERAS,
                      margin_px: float = 0.0) -> list[dict]:
    """Per-annotation, per-camera FRUSTUM MEMBERSHIP for one keyframe.

    This is the mechanism the camera-attention workstream needs, and it is a
    **projection, not an approximation**: nuScenes publishes ``calibrated_sensor``
    intrinsics+extrinsics for all six cameras and ``ego_pose`` per keyframe, so
    *"agent A is inside CAM_BACK_LEFT's frustum at t and NOT inside CAM_FRONT's"*
    is computed, not estimated.

    Each returned dict carries:
      ``in_frustum``   {camera: bool}   — box CENTER projects inside the image
      ``cameras``      sorted list of cameras that see it
      ``in_ego_camera``bool             — visible from CAM_FRONT specifically
      ``off_front_only`` bool           — **the label**: seen by some camera, but
                                          NOT by CAM_FRONT
      ``visibility_level``              — nuScenes' own across-camera bin
      ``range_m``                       — distance from the ego camera
    """
    anns = idx.annotations_of(sample_token)
    if not anns:
        return []
    centers = np.array([a.get("translation", [0, 0, 0]) for a in anns], np.float64)

    per_cam_hits: dict[str, np.ndarray] = {}
    for ch in cameras:
        sd = idx.sample_data(sample_token, ch)
        if sd is None:
            continue
        try:
            intr = idx.camera_intrinsics_of(sd)
        except ValueError:
            continue
        pc = global_to_camera(centers, idx.ego_pose_of(sd),
                              idx.calibrated_sensor_of(sd))
        px, infront = project_to_pixels(pc, intr)
        inside = (infront
                  & (px[:, 0] >= -margin_px) & (px[:, 0] < intr.width + margin_px)
                  & (px[:, 1] >= -margin_px) & (px[:, 1] < intr.height + margin_px))
        per_cam_hits[ch] = inside

    # range measured from the ego camera
    sd_front = idx.sample_data(sample_token, EGO_CAMERA)
    if sd_front is not None:
        pc_front = global_to_camera(centers, idx.ego_pose_of(sd_front),
                                    idx.calibrated_sensor_of(sd_front))
        rng = np.linalg.norm(pc_front, axis=1)
    else:
        rng = np.full(len(anns), np.nan)

    out = []
    for i, a in enumerate(anns):
        hits = {ch: bool(m[i]) for ch, m in per_cam_hits.items()}
        seen = sorted(ch for ch, ok in hits.items() if ok)
        in_ego = hits.get(EGO_CAMERA, False)
        out.append({
            "instance_token": a.get("instance_token", ""),
            "category": idx.category_of(a),
            "in_frustum": hits,
            "cameras": seen,
            "n_cameras": len(seen),
            "in_ego_camera": in_ego,
            "off_front_only": bool(seen) and not in_ego,
            "visibility_level": idx.visibility_of(a).get("level", ""),
            "range_m": float(rng[i]) if np.isfinite(rng[i]) else None,
        })
    return out


# --------------------------------------------------------------------------- #
# Scenario statistics — the T3 value question, answered on metadata alone      #
# --------------------------------------------------------------------------- #
#: Sustained heading change (deg) that marks a junction manoeuvre.
TURN_DEG = 60.0
#: A roundabout traversal turns much further than a single turn, at low speed.
ROUNDABOUT_DEG = 200.0


def scene_scenario_stats(idx: NuScenesIndex, scene: dict) -> dict:
    """HEURISTIC scenario labels for one scene, from the ego track + description.

    ⚠️ EVIDENCE CLASS: these are **DERIVED HEURISTICS over the ego track**, not
    nuScenes ground truth. nuScenes ships no roundabout/turn label. Two
    independent signals are returned so they can be cross-checked rather than
    trusted singly:
      * ``turn_deg`` / ``max_window_turn_deg`` — integrated heading change
      * ``description_*`` — keyword hits in the human-written ``scene.description``
    A count quoted from either alone is ESTIMATED. Agreement between them is the
    strongest claim available without the map layers.
    """
    poses, t = ego_track(idx, scene["token"])
    yaw = np.unwrap(poses[:, 2].astype(np.float64))
    v = poses[:, 3].astype(np.float64)
    total_turn = float(np.degrees(yaw[-1] - yaw[0]))

    # widest heading swing over any contiguous window (a roundabout's signature)
    span = float(np.degrees(yaw.max() - yaw.min()))

    desc = str(scene.get("description", "")).lower()
    kw = {
        "description_intersection": any(k in desc for k in
                                        ("intersection", "junction", "crossing")),
        "description_turn": "turn" in desc,
        "description_left_turn": "left turn" in desc or "turn left" in desc,
        "description_roundabout": "roundabout" in desc or "rotary" in desc,
        "description_traffic_light": any(k in desc for k in
                                         ("traffic light", "stoplight", "signal")),
        "description_wait": "wait" in desc,
        "description_peds": "ped" in desc,
        "description_rain": "rain" in desc,
        "description_night": "night" in desc,
    }
    # A roundabout traversal is a large sustained turn, so it would otherwise
    # also satisfy the plain turn test. The buckets must be DISJOINT: a
    # roundabout is not an "unprotected left", and double-counting it would
    # inflate exactly the two numbers the PI is deciding on.
    is_roundabout = bool(span >= ROUNDABOUT_DEG)
    is_turn = bool(abs(total_turn) >= TURN_DEG) and not is_roundabout
    return {
        "scene_token": scene["token"],
        "name": scene.get("name", ""),
        "location": idx.location_of_scene(scene),
        "description": scene.get("description", ""),
        "n_keyframes": int(poses.shape[0]),
        "duration_s": float(t[-1] - t[0]),
        "mean_speed_mps": float(v.mean()),
        "min_speed_mps": float(v.min()),
        "turn_deg": total_turn,
        "yaw_span_deg": span,
        "is_turn_heuristic": is_turn,
        "is_left_turn_heuristic": is_turn and total_turn >= TURN_DEG,
        "is_right_turn_heuristic": is_turn and total_turn <= -TURN_DEG,
        "is_roundabout_heuristic": is_roundabout,
        # an unprotected-left PROXY: a left turn that involved slowing to a
        # near-stop (yielding). Still a heuristic — nuScenes has no such label.
        "is_yielding_left_heuristic": (is_turn and total_turn >= TURN_DEG
                                       and bool(v.min() < 2.0)),
        "has_stop_heuristic": bool(v.min() < 0.5),
        **kw,
    }


def corpus_scenario_report(idx: NuScenesIndex) -> dict:
    """Scenario stats over every scene + the aggregate counts the PI asked for."""
    rows = []
    for sc in idx.scenes():
        try:
            rows.append(scene_scenario_stats(idx, sc))
        except Exception as e:                       # F-6: one bad scene never kills
            rows.append({"scene_token": sc.get("token"), "error":
                         f"{type(e).__name__}: {e}"})
    ok = [r for r in rows if "error" not in r]
    def n(k):
        return int(sum(bool(r.get(k)) for r in ok))
    locs: dict[str, int] = {}
    for r in ok:
        locs[r.get("location", "")] = locs.get(r.get("location", ""), 0) + 1
    return {
        "n_scenes": len(rows), "n_ok": len(ok), "n_error": len(rows) - len(ok),
        "locations": locs,
        "counts": {
            "turn_heuristic": n("is_turn_heuristic"),
            "left_turn_heuristic": n("is_left_turn_heuristic"),
            "right_turn_heuristic": n("is_right_turn_heuristic"),
            "roundabout_heuristic": n("is_roundabout_heuristic"),
            "yielding_left_heuristic": n("is_yielding_left_heuristic"),
            "has_stop_heuristic": n("has_stop_heuristic"),
            "description_intersection": n("description_intersection"),
            "description_roundabout": n("description_roundabout"),
            "description_left_turn": n("description_left_turn"),
            "description_traffic_light": n("description_traffic_light"),
            "description_night": n("description_night"),
            "description_rain": n("description_rain"),
        },
        "scenes": rows,
    }


def corpus_camera_visibility_report(idx: NuScenesIndex, max_scenes: int | None = None,
                                    categories: Iterable[str] | None = None) -> dict:
    """Aggregate the off-front-only statistic over the corpus (metadata only).

    ``categories``: restrict to decision-relevant agents (e.g. vehicles + VRUs);
    ``None`` counts every annotated agent.
    """
    cats = set(categories) if categories else None
    n_ann = n_seen = n_front = n_offfront = 0
    per_cam: dict[str, int] = {}
    scenes = idx.scenes()[:max_scenes] if max_scenes else idx.scenes()
    for sc in scenes:
        for s in idx.samples_of_scene(sc["token"]):
            for r in camera_visibility(idx, s["token"]):
                if cats is not None and not any(r["category"].startswith(c)
                                                for c in cats):
                    continue
                n_ann += 1
                if r["n_cameras"]:
                    n_seen += 1
                if r["in_ego_camera"]:
                    n_front += 1
                if r["off_front_only"]:
                    n_offfront += 1
                for ch in r["cameras"]:
                    per_cam[ch] = per_cam.get(ch, 0) + 1
    return {
        "n_scenes": len(scenes), "n_annotations": n_ann,
        "n_visible_any_camera": n_seen, "n_in_ego_camera": n_front,
        "n_off_front_only": n_offfront,
        "frac_off_front_only": (n_offfront / n_seen) if n_seen else None,
        "per_camera_hits": per_cam,
    }


# --------------------------------------------------------------------------- #
# Episode build (needs the IMAGE blobs — the 60 GB keyframe pull)              #
# --------------------------------------------------------------------------- #
def canonicalize_frames(vid: torch.Tensor, intr: PinholeIntrinsics,
                        size: int = 256, f_ref: float = F_REF
                        ) -> tuple[torch.Tensor, dict]:
    """[T,3,H,W] native -> [T,3,size,size] canonical at ``f_eff == f_ref``.

    Uses D-016 R1 :func:`pinhole_rectify`, NOT ``focal_crop_resize``: nuScenes
    CAM_FRONT is height-bound and the crop path would land ``f_eff≈360``. Returns
    the frames plus the geometry provenance for the data card.
    """
    out = pinhole_rectify(vid, intr, size=size, f_ref=f_ref)
    return out, {
        "canon": "pinhole_rectify(D-016 R1)",
        "f_eff_px": float(pinhole_rectify.last_f_eff),
        "observed_frac": float(pinhole_rectify.last_observed_frac),
        "native_fx": intr.fx, "native_fy": intr.fy,
        "native_cx": intr.cx, "native_cy": intr.cy,
        "native_width": intr.width, "native_height": intr.height,
    }


def load_keyframe_images(idx: NuScenesIndex, scene_token: str, root: str | Path,
                         channel: str = EGO_CAMERA,
                         decode_fn=None) -> tuple[torch.Tensor, PinholeIntrinsics]:
    """Decode a scene's keyframe JPEGs for one camera -> ``[T,3,H,W]`` uint8.

    ``decode_fn(path) -> [3,H,W] uint8`` is injectable so CI exercises the whole
    path with no image bytes (the ``comma2k19`` fixture pattern).
    """
    root = Path(root)
    sds, frames = [], []
    for s in idx.samples_of_scene(scene_token):
        sd = idx.sample_data(s["token"], channel)
        if sd is not None:
            sds.append(sd)
    if not sds:
        raise ValueError(f"scene {scene_token!r}: no {channel} keyframes")

    # The blob-existence check belongs to the REAL decoder only: an injected
    # decode_fn (CI fixtures) synthesizes pixels and must not require files.
    require_blobs = decode_fn is None
    if decode_fn is None:
        from PIL import Image

        def decode_fn(p):                                   # noqa: F811
            with Image.open(p) as im:
                a = np.asarray(im.convert("RGB"), dtype=np.uint8)
            return torch.from_numpy(a).permute(2, 0, 1).contiguous()

    for sd in sds:
        p = root / sd["filename"]
        if require_blobs and not p.exists():
            raise NuScenesTermsError(
                f"keyframe blob missing: {p}\nThe METADATA archive alone does not "
                f"contain images. Pull the keyframe blobs (~60 GB) only after the "
                f"value case justifies it — and only via the terms-accepted "
                f"official download.")
        frames.append(decode_fn(p))
    return torch.stack(frames), idx.camera_intrinsics_of(sds[0])


def build_episode(idx: NuScenesIndex, scene_token: str, root: str | Path,
                  episode_id: int, size: int = 256, n_stack: int = 3,
                  channel: str = EGO_CAMERA, decode_fn=None
                  ) -> tuple[ToyEpisode, dict]:
    """One nuScenes scene -> the canonical ``ToyEpisode`` + geometry provenance.

    Frames are D-016-R1-rectified to ``f_eff == 266`` and stacked into the 9-channel
    D-015 layout; poses/actions come from ``ego_pose`` (POSE-DERIVED — no CAN).
    """
    from tanitad.data.comma2k19 import stack_frames

    poses_all, t_all = ego_track(idx, scene_token, channel)
    vid, intr = load_keyframe_images(idx, scene_token, root, channel, decode_fn)
    n = min(vid.shape[0], poses_all.shape[0])
    vid, poses_all, t_all = vid[:n], poses_all[:n], t_all[:n]

    rect, geo = canonicalize_frames(vid, intr, size=size)
    stacked = stack_frames(rect, n_stack)                  # [n-(n_stack-1),9,S,S]
    off = n_stack - 1
    poses = poses_all[off:]
    actions = actions_from_track(poses_all, t_all)[off:]

    # NOTE: built directly, not via `_contract.assemble_episode` — that helper
    # casts frames to float32 and asserts the 1-channel BEV contract, whereas the
    # real corpora emit the 9-channel D-015 uint8 stack (comma2k19 does the same).
    ep = ToyEpisode(
        frames=stacked,
        actions=torch.from_numpy(np.asarray(actions)).float(),
        poses=torch.from_numpy(np.asarray(poses)).float(),
        episode_id=int(episode_id))
    assert_contract(ep, channels=3 * n_stack)
    return ep, geo


# --------------------------------------------------------------------------- #
# Discovery + the I3 split unit                                                #
# --------------------------------------------------------------------------- #
def discover_scenes(idx: NuScenesIndex) -> list[dict]:
    """Ingest units = scenes (one 20 s recording == one episode)."""
    return idx.scenes()


def split_unit_of(idx: NuScenesIndex, scene: dict) -> str:
    """I3 split unit = the LOG, not the scene.

    Scenes from one log are consecutive slices of the same drive on the same
    roads; splitting on ``scene`` would leak road segments across train/val
    exactly as the comma2k19 route-vs-segment lesson predicts. The log is the
    drive-disjoint unit.
    """
    return str(scene.get("log_token", scene.get("token", "")))
