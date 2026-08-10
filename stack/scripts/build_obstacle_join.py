"""I1a — `obstacle.offline` -> episode JOIN for P8/P4 (V58F_INTERP_PLAN.md, 2026-08-10).

Produces the jsonl that ``train_p8_occupancy.py --raster-source join-file`` consumes
(the schema is train_p8_occupancy.py:30-41 / ``JoinFileReader``, :189-286 — matched
EXACTLY and proven compatible by test, ``tests/test_obstacle_join.py`` imports the
reader), with the P4 per-agent visibility flag so the visible/occluded split needs
no re-derivation downstream.

LINE SCHEMA (one line per LABELLED episode frame)::

    {"clip_id": str, "frame_idx": int, "t_s": float,
     "agents": [{"cx": f, "cy": f, "yaw": f, "l": f, "w": f, "occ": 0|1,
                 "track_id": str, "cls": str}]}

  * ``cx/cy/yaw`` are in the EGO frame of that frame (+x fwd, +y LEFT — the
    ``refb_labels.ego_frame`` convention, scripts/refb_labels.py:86-90, applied via
    ``tanitad.data.bev_raster.ego_frame_agents``, bev_raster.py:190-212).
  * ``frame_idx`` is EPISODE index space (the post-n_stack-trim pose index —
    exactly the space ``register_poses_to_time`` fits its time grid over and the
    space ``train_p8_occupancy.window_frame`` looks up).
  * ``l = size_x``, ``w = size_y`` (bev_raster.agents_at_time's mapping, :253).
  * ``occ``: 0 = center inside the 120 deg front-camera frustum ("visible"),
    1 = outside it while the track continues ("occluded / out of view") — P4.
  * ``t_s`` / ``track_id`` / ``cls`` are AUDIT extras; ``JoinFileReader`` /
    ``agents_to_array`` (bev_raster.py:129-155) read only the keys they know, so
    extras are ignored by construction (and the roundtrip test proves it).
  * An ABSENT (clip, frame) line is NO_LABEL (skip+count downstream, never "road
    clear"); an EMPTY ``agents`` list IS a label: labelled clear
    (train_p8_occupancy.py:39-41; 2026-08-03-obstacle-offline-join doc §4).

WHY THIS SCRIPT EXISTS — the corpus facts (each MEASURED/verified from source):
  * The episode build never ingests ``obstacle.offline``: ``physicalai.py`` reads
    the camera mp4 + timestamps parquet (physicalai.py:705-707), egomotion
    (:708, ``load_egomotion`` :479-483) and the calibration features
    camera_intrinsics / sensor_extrinsics / vehicle_dimensions (:233-235); the
    episode contract is frames/actions/poses/episode_id/maneuvers only
    (physicalai.py:742-746; tanitad/data/_contract.py:8-12). ``grep obstacle
    tanitad/data/physicalai.py`` -> zero matches. So the join is a POD-SIDE step.
  * Raw layout: ``<root>/labels/obstacle.offline/obstacle.offline.chunk_{c:04d}.zip``
    holding ONE parquet per clip (scripts/lead_state_gate.py:308-338) with columns
    ``timestamp_us, source, track_id, center_{x,y,z}, size_{x,y,z},
    orientation_{x,y,z,w}, label_class, reference_frame,
    reference_frame_timestamp_us`` (bev_raster.py:21-26; join doc §1, MEASURED
    from bytes).
  * FRAME: ``reference_frame == "rig"`` on every row and
    ``reference_frame_timestamp_us == timestamp_us`` — each cuboid lives in the
    EGO/RIG frame AT ITS OWN TIMESTAMP, axis x-fwd / y-left / z-up (MEASURED by
    the parked-car experiment, 7.4x over the nearest alternative — join doc §2;
    bev_raster.py:27-33). A cuboid is therefore NOT clip-local world: to place it
    at a DIFFERENT time it must be composed through the world frame.
  * CLOCK/RATE: ~10 Hz per track, tracks STAGGERED (1.000-1.005 rows per unique
    timestamp -> per-track temporal lookup, never a frame index; bev_raster.py:34-36),
    labels span ~20 s while egomotion runs 48-140 s (join doc §1) — frames past the
    span are NO_LABEL, a state of their own (join doc §4). The EPISODE grid is an
    affine reparametrisation of the clip clock with spacing ~0.1007 s, NOT 0.1
    (``build_episode``: t_query = linspace(t0, t1, int(span*10)) — physicalai.py:717-718;
    MEASURED 0.10066-0.10084, lead_source.py:141-144; join doc §5). Frame times are
    therefore RECOVERED per episode by content registration
    (``taniteval.lead_source.register_poses_to_time``, lead_source.py:131-229 —
    worst 25.9 ms over 500 clips), never assumed.

THE TWO TRANSFORMS (both cited, neither invented):
  1. rig@sample -> world, using the egomotion pose AT THE CUBOID'S OWN TIMESTAMP
     (the frame it is expressed in — lead_source.py:308-316)::

         L_w = ego_xy(t_samp) + R(yaw_ego(t_samp)) @ [cx, cy]
         yaw_w = yaw_ego(t_samp) + yaw_cuboid

  2. world -> ego frame of the episode pose at the frame (``poses[i]``), via
     ``bev_raster.ego_frame_agents`` — EXACTLY refb_labels.ego_frame
     (scripts/refb_labels.py:86-90), already pinned against the torch original in
     tests/test_p8.py. The episode pose IS egomotion interpolated at the frame
     time (physicalai.signals_at), residual sub-cm (join doc §5), so using it
     keeps the raster in the same frame the model's own poses live in.

  This is the EGO-COMPENSATION bev_raster.py:69-77 prescribes for a join builder
  ("wants exactness -> ego-compensate via egomotion instead of widening the
  tolerance"): the nearest sample within DEFAULT_TOL_S (0.06 s) is mis-registered
  by ~|v_rel|*dt if read raw (~0.5 m at 10 m/s); composed through the world frame
  the EGO part of that error vanishes (the residual is only the AGENT's own motion
  over dt — `obstacle.offline` carries no velocity column to remove it, join doc §1).

VISIBILITY (P4 split, WM_PHYSICS_PROOF.md P4/P8): a cuboid is VISIBLE at a frame
iff its CENTER's azimuth in that frame's ego frame, ``atan2(cy, cx)``, lies within
+-hfov/2 of straight ahead (default hfov 120 deg — the sensor: `camera_front_wide_
120fov`, physicalai.py:232; the v5.8f eval frame is CanonicalFrame.from_hfov(120,
256, 640, "cylindrical"), where azimuth maps LINEARLY to columns, calib.py:88-94,
so the frustum bound IS an azimuth bound). Stated limits: (a) no object-object
occlusion is modelled — "occluded" here means OUT OF VIEW of the front camera
(behind / beyond +-60 deg) while the track continues, which is the P4 selection
("visible at t, not visible at t+k, track continues"); (b) no vertical bound —
the join schema drops z, and the ~45 deg VFOV of the eval frame excludes
ground-level agent centers only at <~2 m range.

EPISODE <-> CLIP IDENTITY (v2 corpora): each ``*.v2ep.pt`` payload stores the FULL
``clip_id`` (v2_dataset._scan_meta:358); the ``_v2manifest.pt`` sidecar (version 3)
carries ``clip_id`` + ``episode_uid`` per clip (v2_dataset.py:413-416). Provider
order == SORTED filename order (v2_dataset._list_clips:366-368, concatenated by
build_v2_providers), and ``train_p8_occupancy.mini_eval`` selects ``e < episodes``
over that order — so "the first N episodes" HERE is the first N sorted files,
matching the eval's own set. The join emits the full clip_id string;
``JoinFileReader`` resolves it from the provider's 63-bit uid
(blake2b(clip_id, digest_size=8) >> 1 — v2_dataset.py:95-96) or the legacy 4-byte
id (physicalai.py:740), refusing ambiguous legacy ids.

Usage (pod4 — CPU only, no GPU; ⛔ never on a training pod)::

  OMP_NUM_THREADS=6 PYTHONPATH=/workspace/TanitAD/stack python3 \
      stack/scripts/build_obstacle_join.py \
      --hf-cache /workspace/hf-cache \
      --corpus /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --episodes 40 \
      --out /workspace/data/p8_join/val40_agents.jsonl.xz

Prints per-episode stats while streaming, self-verifies the written file with the
REAL consumer (``JoinFileReader``), writes a ``<out>.meta.json`` provenance
sidecar, and ends with one ``JOIN_DONE {...}`` line carrying n_episodes /
n_frames / n_agent_boxes / visible_frac (+ md5 for the pod->pod relay check).
Missing obstacle data for a clip is REPORTED AND SKIPPED, never fatal — absence
at one clip is not corpus absence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parents[1]
sys.path.insert(0, str(_SCRIPTS))                    # sibling scripts
if str(_SCRIPTS.parent) not in sys.path:             # stack/ -> tanitad importable
    sys.path.insert(1, str(_SCRIPTS.parent))         # (pods also set PYTHONPATH)

from tanitad.data.bev_raster import (DEFAULT_TOL_S,  # noqa: E402
                                     ego_frame_agents, yaw_from_quaternion)

#: the gated dataset (physicalai.py:231) and the two label features joined here.
HF_DATASET = "nvidia/PhysicalAI-Autonomous-Vehicles"
OBSTACLE = "obstacle.offline"
EGOMOTION = "egomotion"

#: the front camera's field: `camera_front_wide_120fov` (physicalai.py:232); the
#: v5.8f eval frame retains exactly this via from_hfov(120, 256, 640, "cyl").
HFOV_DEG_DEFAULT = 120.0

#: v2 manifest sidecar contract mirrored from v2_dataset.py:60-63 (the module
#: itself imports torchvision at module level and is absent on some hosts —
#: same fallback pattern as train_p8_occupancy.episode_uid_of_clip).
_MANIFEST_NAME = "_v2manifest.pt"
_MANIFEST_VERSION = 3


def _lead_source():
    """``taniteval.lead_source`` (registration by content) — imported, never
    re-implemented; path-appended like lead_state_gate.py:436-438."""
    try:
        from taniteval import lead_source
    except ModuleNotFoundError:
        te = str(_REPO / "taniteval")
        if te not in sys.path:
            sys.path.append(te)
        from taniteval import lead_source
    return lead_source


# ============================================================================
# pure geometry (CPU-tested in tests/test_obstacle_join.py)
# ============================================================================
def rig_to_world(agents_rig, ego_xyyaw) -> np.ndarray:
    """rig-frame agents ``[A, >=3] (cx, cy, yaw, ...)`` -> world frame.

    ``ego_xyyaw``: ``[A, 3]`` per-row ego pose (x, y, yaw) — per-row because each
    cuboid is expressed in the rig frame AT ITS OWN TIMESTAMP (bev_raster.py:27-29),
    so two samples carry two different ego poses — or a single ``(3,)`` pose.
    The composition is lead_source.lead_track_in_window's (lead_source.py:314-316)::

        wx = ex + cx*cos(eyaw) - cy*sin(eyaw)
        wy = ey + cx*sin(eyaw) + cy*cos(eyaw)
        wyaw = eyaw + yaw

    and is the exact INVERSE of ``bev_raster.ego_frame_agents`` (world->ego,
    refb_labels.ego_frame's rotation) — pinned by roundtrip in the tests.
    Extra columns (l, w, occ) pass through untouched.
    """
    ag = np.asarray(agents_rig, dtype=np.float64).copy()
    if ag.size == 0:
        return ag.reshape(0, ag.shape[1] if ag.ndim == 2 else 6)
    pose = np.asarray(ego_xyyaw, dtype=np.float64)
    if pose.ndim == 1:
        pose = np.broadcast_to(pose, (ag.shape[0], 3))
    ex, ey, eyaw = pose[:, 0], pose[:, 1], pose[:, 2]
    c, s = np.cos(eyaw), np.sin(eyaw)
    wx = ex + ag[:, 0] * c - ag[:, 1] * s
    wy = ey + ag[:, 0] * s + ag[:, 1] * c
    ag[:, 0] = wx
    ag[:, 1] = wy
    ag[:, 2] = eyaw + ag[:, 2]           # ego_frame_agents wraps on the way back
    return ag


def visibility_occ(agents_ego, hfov_deg: float = HFOV_DEG_DEFAULT) -> np.ndarray:
    """P4 flag per EGO-FRAME agent row: 0 = center azimuth within +-hfov/2
    (visible to the front camera), 1 = outside (occluded / out of view).

    Azimuth = ``atan2(cy, cx)`` in the +x-fwd/+y-left frame; under the eval's
    cylindrical projection azimuth maps linearly to image columns (calib.py:88-94),
    so this IS the horizontal frustum bound. Closed inequality at the edge.
    """
    ag = np.asarray(agents_ego, dtype=np.float64)
    if ag.size == 0:
        return np.zeros((0,), dtype=np.int64)
    az = np.arctan2(ag[:, 1], ag[:, 0])
    return np.where(np.abs(az) <= math.radians(float(hfov_deg)) / 2.0, 0, 1
                    ).astype(np.int64)


class EgoTrack:
    """One clip's egomotion in its own clock (seconds): sorted ``t``, ``x``,
    ``y`` and UNWRAPPED quaternion yaw (interpolating a wrapped yaw across the
    +-pi seam smears it — same rule as physicalai.signals_at:637-640).
    Duck-typed columns (DataFrame or dict of arrays): ``timestamp`` [us],
    ``x``, ``y``, ``qx/qy/qz/qw`` — the egomotion schema (physicalai.py:9-10)."""

    def __init__(self, ego):
        t = np.asarray(ego["timestamp"], dtype=np.float64) / 1e6
        o = np.argsort(t)
        self.t = t[o]
        self.x = np.asarray(ego["x"], dtype=np.float64)[o]
        self.y = np.asarray(ego["y"], dtype=np.float64)[o]
        self.yaw_u = np.unwrap(yaw_from_quaternion(
            *(np.asarray(ego[c], dtype=np.float64)[o]
              for c in ("qx", "qy", "qz", "qw"))))

    def at(self, ts) -> np.ndarray:
        """Interpolated ``[N, 3] (x, y, yaw_unwrapped)`` at times ``ts`` [s]."""
        ts = np.atleast_1d(np.asarray(ts, dtype=np.float64))
        return np.column_stack([np.interp(ts, self.t, self.x),
                                np.interp(ts, self.t, self.y),
                                np.interp(ts, self.t, self.yaw_u)])


def clip_tracks(obs) -> list[dict]:
    """Raw `obstacle.offline` rows of ONE clip -> per-track sorted arrays.

    Duck-typed columns as bev_raster.agents_at_time (:229-231). Refuses any row
    whose ``reference_frame`` is not ``"rig"`` when that column is present —
    "every row is rig" is a MEASURED invariant (join doc §1), and a silent
    exception to it would mis-place every box of that clip.
    """
    ts = np.asarray(obs["timestamp_us"], dtype=np.float64) / 1e6
    try:
        ref = np.asarray(obs["reference_frame"]).astype(str)
        bad = ref != "rig"
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())}/{ref.size} obstacle rows have "
                f"reference_frame != 'rig' ({sorted(set(ref[bad]))[:3]}) — the "
                f"measured every-row-rig invariant (join doc §1) is violated; "
                f"refusing to join this clip rather than mis-place its boxes")
    except (KeyError, IndexError):
        pass                                   # synthetic inputs may omit it
    tid = np.asarray(obs["track_id"]).astype(str)
    cx = np.asarray(obs["center_x"], dtype=np.float64)
    cy = np.asarray(obs["center_y"], dtype=np.float64)
    sx = np.asarray(obs["size_x"], dtype=np.float64)
    sy = np.asarray(obs["size_y"], dtype=np.float64)
    yaw = yaw_from_quaternion(obs["orientation_x"], obs["orientation_y"],
                              obs["orientation_z"], obs["orientation_w"])
    try:
        cls = np.asarray(obs["label_class"]).astype(str)
    except (KeyError, IndexError):
        cls = np.full(ts.shape, "", dtype=object)
    out = []
    for track in np.unique(tid):
        m = tid == track
        o = np.argsort(ts[m], kind="stable")
        out.append({"tid": str(track), "cls": str(cls[m][o[0]]),
                    "t": ts[m][o], "cx": cx[m][o], "cy": cy[m][o],
                    "yaw": np.asarray(yaw)[m][o],
                    "l": sx[m][o], "w": sy[m][o]})
    return out


def world_agents_at(tracks: list[dict], t_s: float, ego: EgoTrack,
                    tol_s: float = DEFAULT_TOL_S
                    ) -> tuple[np.ndarray, list[str], list[str]]:
    """WORLD-frame agents near time ``t_s``: per track, the single sample
    nearest ``t_s`` within ``tol_s`` (bev_raster.agents_at_time's rule — tracks
    are STAGGERED, a per-track lookup is mandatory), composed rig->world with
    the egomotion pose at the SAMPLE'S OWN timestamp (lead_source.py:308-316).

    Returns ``(agents_world [A, 6] (wx, wy, wyaw, l, w, -1), track_ids, classes)``.
    The raster is a GT LABEL (a scoring input, like GT waypoints — lead_source
    docstring), so NEAREST — not causal-last — is the right sample rule here.
    """
    rows, samp_t, tids, clss = [], [], [], []
    for tr in tracks:
        j = int(np.argmin(np.abs(tr["t"] - float(t_s))))
        if abs(float(tr["t"][j]) - float(t_s)) > float(tol_s):
            continue
        rows.append((float(tr["cx"][j]), float(tr["cy"][j]),
                     float(tr["yaw"][j]), float(tr["l"][j]),
                     float(tr["w"][j]), -1.0))
        samp_t.append(float(tr["t"][j]))
        tids.append(tr["tid"])
        clss.append(tr["cls"])
    if not rows:
        return np.zeros((0, 6), dtype=np.float64), [], []
    return rig_to_world(np.asarray(rows, dtype=np.float64),
                        ego.at(np.asarray(samp_t))), tids, clss


# ============================================================================
# the per-clip join (pure given loaded arrays; CPU-tested end to end)
# ============================================================================
def join_clip(clip_id: str, poses: np.ndarray, ego: EgoTrack, obs, *,
              tol_s: float = DEFAULT_TOL_S,
              hfov_deg: float = HFOV_DEG_DEFAULT) -> tuple[list[dict], dict]:
    """One episode -> its join records + stats.

    ``poses``: the episode's OWN ``[T, >=3]`` (x, y, yaw[, v]) in EPISODE index
    space (v2 manifests/payloads are already n_stack-trimmed, v2_dataset.py:356).
    Raises ``taniteval.lead_source.RegistrationError`` when the episode cannot be
    located on the clip's egomotion track (loud skip, never an approximate join).

    A frame is LABELLED iff its registered time lies within ``tol_s`` of the
    clip's obstacle span [min ts, max ts]. ⚠️ The guard is the MATCH tolerance on
    purpose, NOT lead_source's 0.5 s staleness guard: with a 0.5 s guard a frame
    past the span would emit ``agents: []`` — manufactured "road clear", the
    exact bias join doc §4 forbids. Inside the span, every live ~10 Hz track has
    a sample within 0.05 s <= tol of any frame time.
    """
    ls = _lead_source()
    poses = np.asarray(poses, dtype=np.float64)
    reg = ls.register_poses_to_time(poses[:, :2], ego.t, ego.x, ego.y)
    t_s = np.asarray(reg["t_s"], dtype=np.float64)
    tracks = clip_tracks(obs)
    ot = np.asarray(obs["timestamp_us"], dtype=np.float64) / 1e6
    if ot.size == 0:
        raise ValueError(f"{clip_id}: obstacle parquet has 0 rows")
    span = (float(ot.min()), float(ot.max()))
    records: list[dict] = []
    n_boxes = n_vis = 0
    for i in range(poses.shape[0]):
        ti = float(t_s[i])
        if not (span[0] - tol_s <= ti <= span[1] + tol_s):
            continue                                   # NO_LABEL: no line at all
        ag_w, tids, clss = world_agents_at(tracks, ti, ego, tol_s)
        ag_e = ego_frame_agents(ag_w, poses[i, :3])    # refb ego_frame convention
        occ = visibility_occ(ag_e, hfov_deg)
        agents = [{"cx": round(float(a[0]), 4), "cy": round(float(a[1]), 4),
                   "yaw": round(float(a[2]), 5),
                   "l": round(float(a[3]), 3), "w": round(float(a[4]), 3),
                   "occ": int(o), "track_id": t, "cls": c}
                  for a, o, t, c in zip(ag_e, occ, tids, clss)]
        records.append({"clip_id": str(clip_id), "frame_idx": int(i),
                        "t_s": round(ti, 4), "agents": agents})
        n_boxes += len(agents)
        n_vis += int((occ == 0).sum())
    stats = {"clip_id": str(clip_id), "n_frames": int(poses.shape[0]),
             "n_labelled": len(records), "n_agent_boxes": n_boxes,
             "n_visible_boxes": n_vis,
             "visible_frac": round(n_vis / n_boxes, 4) if n_boxes else None,
             "n_tracks": len(tracks),
             "label_span_s": [round(span[0], 3), round(span[1], 3)],
             "registration": {"a": round(float(reg["a"]), 6),
                              "b": round(float(reg["b"]), 6),
                              "residual_m": reg["residual_m"],
                              "n_inlier": reg["n_inlier"],
                              "n_probe": reg["n_probe"]}}
    return records, stats


# ============================================================================
# output writer + consumer-side verification
# ============================================================================
def open_out(path: str | Path):
    p = str(path)
    if p.endswith(".xz"):
        return lzma.open(p, "wt", encoding="utf-8", preset=6)
    return open(p, "w", encoding="utf-8")


def write_records(fh, records: list[dict]) -> None:
    for rec in records:
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


def verify_with_reader(path: str | Path) -> dict:
    """Re-read the written file with the REAL consumer
    (``train_p8_occupancy.JoinFileReader``) — the schema proof at build time.
    ``.xz`` outputs are decompressed to a sibling temp file first (the reader,
    like the trainer flag, takes PLAIN jsonl; the .xz is for the pod relay)."""
    from train_p8_occupancy import JoinFileReader
    p = Path(path)
    tmp = None
    try:
        if p.suffix == ".xz":
            tmp = p.with_name(p.name + ".verify.tmp")
            with lzma.open(p, "rt", encoding="utf-8") as src, \
                    open(tmp, "w", encoding="utf-8") as dst:
                for line in src:
                    dst.write(line)
            rd = JoinFileReader(tmp)
        else:
            rd = JoinFileReader(p)
        return {"n_records": rd.n_records, "n_clips": rd.n_clips,
                "has_occlusion_flags": bool(rd.has_occlusion_flags)}
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def md5_of(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# corpus + label-zip I/O (pod-side; every miss reported, never silent)
# ============================================================================
def corpus_first_clips(corpus_dir: str | Path, n: int
                       ) -> tuple[list[tuple[str, np.ndarray]], int]:
    """First ``n`` clips of the v2 corpus IN PROVIDER ORDER (sorted ``*.v2ep.pt``
    filenames — v2_dataset._list_clips:366-368; ``mini_eval`` selects
    ``e < episodes`` over exactly this order). Returns ``[(clip_id, poses)]``
    (poses already n_stack-trimmed = EPISODE index space) + the corpus size.

    Uses the ``_v2manifest.pt`` sidecar when fresh (version 3, same file set —
    v2_dataset.py:385), else scans payload metadata directly, mirroring
    ``v2_dataset._scan_meta`` (:354-363) without the module's torchvision
    import surface."""
    import torch
    corpus_dir = Path(corpus_dir)
    files = sorted(p.name for p in corpus_dir.glob("*.v2ep.pt"))
    if not files:
        raise SystemExit(f"[join] no *.v2ep.pt under {corpus_dir} — does "
                         f"--corpus point at the v2 split dir?")
    take = files[:max(int(n), 0)]
    man_p = corpus_dir / _MANIFEST_NAME
    if man_p.exists():
        try:
            man = torch.load(man_p, map_location="cpu", weights_only=False)
            if man.get("version") == _MANIFEST_VERSION \
                    and man.get("files") == files:
                return ([(str(man["clip_id"][i]),
                          man["poses"][i].numpy().astype(np.float64))
                         for i in range(len(take))], len(files))
            print(f"[join] manifest {man_p} stale/other-version -> direct scan",
                  flush=True)
        except Exception as e:                                 # noqa: BLE001
            print(f"[join] manifest {man_p} unreadable ({e!r}) -> direct scan",
                  flush=True)
    out = []
    for fn in take:
        d = torch.load(corpus_dir / fn, map_location="cpu",
                       weights_only=False, mmap=True)
        k = int(d["n_stack"]) - 1
        poses = d["poses"][k:].clone().float().numpy().astype(np.float64)
        cid = str(d.get("clip_id") or fn.split(".v2ep")[0])
        out.append((cid, poses))
    return out, len(files)


def find_label_zips(roots: list[str | Path], kind: str) -> list[Path]:
    """Every ``<kind>.chunk_*.zip`` under the given roots. Probes the local-dir
    layout (``<root>/labels/<kind>/`` — physicalai._calib_chunk_path's target),
    the HF hub-cache layout (``datasets--nvidia--…/snapshots/*/labels/<kind>/``),
    and — only when both structured probes miss — one recursive glob, because
    absence found at one location is not absence (operating standard §2)."""
    zips: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        hits = sorted((r / "labels" / kind).glob(f"{kind}.chunk_*.zip"))
        hits += sorted(r.glob("datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
                              f"/snapshots/*/labels/{kind}/{kind}.chunk_*.zip"))
        if not hits:
            hits = sorted(r.glob(f"**/{kind}.chunk_*.zip"))
        for h in hits:
            key = str(h.resolve())
            if key not in seen:
                seen.add(key)
                zips.append(h)
    return zips


def index_zip_members(zips: list[Path], label: str) -> dict[str, tuple[Path, str]]:
    """clip_id -> (zip path, member name), from the zips' central directories.
    Member names are ``{clip_id}.<feature>.parquet`` (lead_state_gate.py:331 keys
    them by ``name.split('.')[0]``; physicalai.load_egomotion matches by
    endswith — :481-482), possibly under a directory prefix."""
    out: dict[str, tuple[Path, str]] = {}
    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    if not n.endswith(".parquet"):
                        continue
                    cid = Path(n).name.split(".")[0]
                    out.setdefault(cid, (zp, n))
        except (zipfile.BadZipFile, OSError) as e:
            print(f"[join] WARNING: unreadable zip {zp}: {e!r} (skipped)",
                  flush=True)
    print(f"[join] {label}: {len(zips)} zips indexed, {len(out)} clips covered",
          flush=True)
    return out


def load_parquet_member(zp: Path, member: str):
    import pandas as pd
    with zipfile.ZipFile(zp) as z:
        return pd.read_parquet(io.BytesIO(z.read(member)))


def load_chunk_table(selection: str | None, hf_cache: Path) -> dict[str, int]:
    """clip_id -> chunk, for on-demand chunk downloads. Probed from --selection,
    then ``<hf-cache>/r0/{phase0_selection,r0_selection}.parquet``
    (lead_state_gate.py:295-297 / score_val40_lead.py's rule). ``{}`` when no
    table exists — downloads are then impossible and locals-only is stated."""
    import pandas as pd
    cands = ([Path(selection)] if selection else []) + \
        [hf_cache / "r0" / "phase0_selection.parquet",
         hf_cache / "r0" / "r0_selection.parquet"]
    for p in cands:
        if p.exists():
            sel = pd.read_parquet(p)
            print(f"[join] chunk table: {p} ({len(sel)} clips)", flush=True)
            return dict(zip(sel["clip_id"].astype(str),
                            sel["chunk"].astype(int)))
    print("[join] no selection parquet found — chunk downloads unavailable, "
          "using local zips only", flush=True)
    return {}


def try_download_chunk(hf_cache: Path, kind: str, chunk: int) -> Path | None:
    """One label chunk via the repo's committed HF path (tanitad.keys +
    hf_hub_download local_dir — physicalai.py:299-304 / pull_obs_chunks.py).
    Failure warns and returns None; the clip is then reported as a skip."""
    rel = f"labels/{kind}/{kind}.chunk_{int(chunk):04d}.zip"
    try:
        from tanitad.keys import enable_tls, load_keys
        enable_tls()
        load_keys()
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(HF_DATASET, rel, repo_type="dataset",
                            local_dir=str(hf_cache))
        print(f"[join] downloaded {rel} "
              f"({Path(p).stat().st_size / 1e6:.1f} MB)", flush=True)
        return Path(p)
    except Exception as e:                                     # noqa: BLE001
        print(f"[join] download {rel} FAILED: {e!r}", flush=True)
        return None


# ============================================================================
# main
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser(
        "build_obstacle_join", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-cache", default="/workspace/hf-cache",
                    help="root holding (or receiving) labels/{obstacle.offline,"
                         "egomotion} chunk zips; HF hub-cache layout also probed")
    ap.add_argument("--corpus", required=True,
                    help="v2 VAL split dir (*.v2ep.pt) — e.g. "
                         "physicalai-val-0c5f7dac3b11-w120-256x640cyl")
    ap.add_argument("--episodes", type=int, default=40,
                    help="first N clips in provider (sorted-file) order — the "
                         "same rule mini_eval's `e < episodes` uses")
    ap.add_argument("--out", required=True, help="output jsonl or jsonl.xz")
    ap.add_argument("--labels-root", nargs="*", default=[],
                    help="extra roots to probe for label zips")
    ap.add_argument("--selection", default=None,
                    help="phase0/r0 selection parquet (clip_id->chunk) for "
                         "downloads; auto-probed under <hf-cache>/r0/")
    ap.add_argument("--no-download", action="store_true",
                    help="never fetch missing chunks from HF")
    ap.add_argument("--tol-s", type=float, default=DEFAULT_TOL_S,
                    help="per-track nearest-sample tolerance (s) — "
                         "bev_raster.DEFAULT_TOL_S; ego-compensation makes "
                         "widening it unnecessary")
    ap.add_argument("--hfov-deg", type=float, default=HFOV_DEG_DEFAULT,
                    help="front-camera horizontal FOV for the P4 visibility "
                         "flag (the sensor field, camera_front_wide_120fov)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    t0 = time.time()
    hf_cache = Path(a.hf_cache)
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    clips, n_corpus = corpus_first_clips(a.corpus, a.episodes)
    print(f"[join] corpus {a.corpus}: {n_corpus} clips; joining first "
          f"{len(clips)} (provider order = sorted files)", flush=True)

    roots = [hf_cache] + list(a.labels_root)
    obs_idx = index_zip_members(find_label_zips(roots, OBSTACLE), OBSTACLE)
    ego_idx = index_zip_members(find_label_zips(roots, EGOMOTION), EGOMOTION)
    chunk_of = {} if a.no_download else load_chunk_table(a.selection, hf_cache)

    def resolve(idx: dict, kind: str, cid: str):
        if cid in idx:
            return idx[cid]
        if not a.no_download and cid in chunk_of:
            zp = try_download_chunk(hf_cache, kind, chunk_of[cid])
            if zp is not None:
                idx.update(index_zip_members([zp], f"{kind} (fetched)"))
        return idx.get(cid)

    fh = open_out(out_p)
    per_clip: list[dict] = []
    skipped: dict[str, list] = {"no_obstacle": [], "no_egomotion": [],
                                "registration_failed": [], "bad_clip": []}
    tot_frames = tot_boxes = tot_vis = n_joined = 0
    ls = _lead_source()
    try:
        for k, (cid, poses) in enumerate(clips):
            tag = f"[join] ep {k:03d} {cid}"
            hit_e = resolve(ego_idx, EGOMOTION, cid)
            if hit_e is None:
                skipped["no_egomotion"].append(cid)
                print(f"{tag}: NO egomotion parquet found — SKIP (cannot "
                      f"register; reported, not fatal)", flush=True)
                continue
            hit_o = resolve(obs_idx, OBSTACLE, cid)
            if hit_o is None:
                skipped["no_obstacle"].append(cid)
                print(f"{tag}: NO obstacle.offline for this clip — SKIP "
                      f"(2.5-3.1 % of the corpus has none, join doc §1; its "
                      f"frames stay NO_LABEL downstream)", flush=True)
                continue
            try:
                ego = EgoTrack(load_parquet_member(*hit_e))
                obs = load_parquet_member(*hit_o)
                recs, st = join_clip(cid, poses, ego, obs, tol_s=a.tol_s,
                                     hfov_deg=a.hfov_deg)
            except ls.RegistrationError as e:
                skipped["registration_failed"].append(cid)
                print(f"{tag}: REGISTRATION FAILED — SKIP ({e})", flush=True)
                continue
            except (ValueError, KeyError) as e:
                skipped["bad_clip"].append(cid)
                print(f"{tag}: BAD CLIP DATA — SKIP ({e!r})", flush=True)
                continue
            write_records(fh, recs)
            per_clip.append(st)
            n_joined += 1
            tot_frames += st["n_labelled"]
            tot_boxes += st["n_agent_boxes"]
            tot_vis += st["n_visible_boxes"]
            print(f"{tag}: frames {st['n_labelled']}/{st['n_frames']} labelled "
                  f"span {st['label_span_s'][0]:.2f}-{st['label_span_s'][1]:.2f}s"
                  f" tracks {st['n_tracks']} boxes {st['n_agent_boxes']} "
                  f"visible {st['visible_frac']} "
                  f"reg_res {st['registration']['residual_m']['median']}m "
                  f"b {st['registration']['b']:.6f}s", flush=True)
    finally:
        fh.close()

    if n_joined == 0:
        print(f"JOIN_FAILED no episode could be joined "
              f"({json.dumps({s: len(v) for s, v in skipped.items()})})",
              flush=True)
        return 1

    # consumer-side proof: the file parses with the EXACT reader P8 uses
    try:
        ver = verify_with_reader(out_p)
        ok = (ver["n_records"] == tot_frames and ver["n_clips"] == n_joined)
        print(f"[join] reader-verify (train_p8_occupancy.JoinFileReader): "
              f"{ver} vs built (records {tot_frames}, clips {n_joined}) -> "
              f"{'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            print("JOIN_FAILED reader/built count mismatch", flush=True)
            return 1
    except Exception as e:                                     # noqa: BLE001
        print(f"[join] WARNING: reader-verify unavailable here ({e!r}); the "
              f"consumer schema is still pinned by tests/test_obstacle_join.py",
              flush=True)
        ver = {"error": repr(e)}

    digest = md5_of(out_p)
    summary = {
        "n_episodes": n_joined,
        "n_episodes_requested": len(clips),
        "n_frames": tot_frames,
        "n_agent_boxes": tot_boxes,
        "visible_frac": round(tot_vis / tot_boxes, 4) if tot_boxes else None,
        "skipped": {s: v for s, v in skipped.items() if v},
        "out": str(out_p), "md5": digest,
        "wall_s": round(time.time() - t0, 1),
    }
    meta = {
        "task": "I1a obstacle.offline -> episode join "
                "(V58F_INTERP_PLAN.md; WM_PHYSICS_PROOF.md P4/P8)",
        "args": vars(a),
        "summary": summary,
        "reader_verify": ver,
        "per_clip": per_clip,
        "conventions": {
            "frame": "per-frame EGO frame, +x fwd +y LEFT (refb_labels."
                     "ego_frame, scripts/refb_labels.py:86-90, applied via "
                     "bev_raster.ego_frame_agents)",
            "frame_idx": "EPISODE index space (post-n_stack-trim); times "
                         "recovered per episode by lead_source."
                         "register_poses_to_time (grid ~0.1007 s, not 0.1)",
            "composition": "rig@sample -> world at the sample's OWN timestamp "
                           "(lead_source.py:308-316) -> ego@frame; the "
                           "bev_raster.py:69-77 ego-compensation",
            "occ": f"0 = center azimuth within +-{a.hfov_deg / 2:.0f} deg "
                   f"(front-camera frustum), 1 = out of view while the track "
                   f"continues. NOT object-object occlusion; no vertical bound "
                   f"(z dropped; rig-origin height not established).",
            "NO_LABEL": "absent (clip, frame) line — outside the ~20 s label "
                        "span or clip without obstacle.offline; NEVER emitted "
                        "as empty agents (that means labelled CLEAR)",
            "sample_rule": f"per-track NEAREST within {a.tol_s} s (GT label, "
                           f"not an arm input — causality not required)",
        },
        "_evidence_class": "MEASURED (ours; artifact = the jsonl + this meta)",
    }
    meta_p = Path(str(out_p) + ".meta.json")
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"[join] meta sidecar -> {meta_p}", flush=True)
    print(f"JOIN_DONE {json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
