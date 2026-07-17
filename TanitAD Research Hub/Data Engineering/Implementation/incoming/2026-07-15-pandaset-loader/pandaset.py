"""PandaSet (Hesai/Scale) -> TanitAD episode contract. CC-BY-4.0 (owned core).

WHY THIS EXISTS (OWN_DATASET_PLAN.md §7 ingest #2, 2026-07-13)
--------------------------------------------------------------
The real NVIDIA PhysicalAI-AV set is un-redistributable (D-002: internal-dev-only,
never shipped, even privately), which forces every pod to rebuild its cache and
blocks any public claim. The own-dataset plan removes that constraint by building
the shared corpus from license-CLEAN sources only. PandaSet is the cheapest clean
REAL-URBAN add in that plan: **CC-BY-4.0** (attribution + a privacy clause), the
"first AV dataset available for commercial use", San-Francisco + El-Camino urban/
suburban day driving with a real front pinhole camera and GPS/IMU ego-motion. It
reuses the Cosmos pose->signals geometry almost verbatim, so it is near-zero new
math on top of an already-tested path.

CORPUS (PandaSet, pandaset-devkit layout; HF mirror ``georghess/pandaset``)
---------------------------------------------------------------------------
Per sequence ``<seq>/`` the fields this loader needs:
    camera/front_camera/NN.jpg              real pinhole RGB, 10 Hz, sorted
    camera/front_camera/intrinsics.json     {"fx","fy","cx","cy"}   (real focal!)
    camera/front_camera/poses.json          [{"position":{x,y,z},
                                              "heading":{w,x,y,z}}, ...] world frame
    camera/front_camera/timestamps.json     per-frame epoch seconds
Camera rate is already 10 Hz == ``TARGET_HZ`` -> stride 1, dt = 0.1 s (no
resampling, unlike Cosmos' 30->10). Because PandaSet ships the REAL per-camera
``fx``, D-016 canonicalization uses the measured focal (``focal_crop_resize``)
rather than a nominal-HFOV estimate -> a tighter geometry match than the synthetic
corpora.

Signal derivation (no CAN on PandaSet -> geometry, exactly like Cosmos):
    x, y   = position[:2]                    world metres, re-origined to frame 0
    yaw    = atan2(dy, dx)                   MOTION heading (see HONEST LIMITATIONS)
    v, a, steer  <-  cosmos_drive.poses_to_signals on the assembled [N,4,4]
Frames D-015 3-frame/9-channel stacked, actions/poses aligned to the latest frame.

Splits: SEQUENCE-level (I3) -- each PandaSet sequence is an independent 8 s drive.

GEOMETRY BLOCKER (D-016 R1 dependency, measured 2026-07-15 -- FAIL-LOUD)
-----------------------------------------------------------------------
The front camera's REAL calibration (arXiv 2112.12610: fx=1970.01, 1920x1080,
distortion k1=-0.589) does NOT canonicalize to F_REF=266 by a centered SQUARE
crop: the ideal crop side (fx*size/F_REF = 1896 px) exceeds the 1080-px frame
height, so the crop clamps to 1080 and lands f_eff ~= 467 px (~1.75x more zoomed)
-- a real cross-corpus ACTION->PIXEL scale mismatch -- and the pinhole crop path
ignores the barrel distortion. General rule: any pinhole with fx>1122 px on a
1080-tall frame is height-bound. Therefore ``_canonicalize(strict=True)`` (the
default real decode path) RAISES ``GeometryError`` rather than emit silently
mis-scaled frames. PandaSet is BLOCKED until the D-016 R1 pad/letterbox crop +
undistort lands in ``stack/tanitad/data/calib.py`` (INTAKE risk section); the
pose/signal/contract code below is complete and unit-tested meanwhile.

HONEST LIMITATIONS (P8) -- verified on real bytes before any trained claim:
  * VEHICLE YAW is taken from the ego MOTION direction, not from the camera
    ``heading`` quaternion. PandaSet's ``heading`` is the front-camera-to-world
    rotation, which carries the (unknown-here) camera->vehicle mounting rotation;
    using it directly would bake a constant yaw offset into every steer label.
    Motion heading is offset-free and well-defined whenever the vehicle moves;
    the bicycle steer already zeroes at v<0.5 m/s (curvature guard), so standstill
    heading noise cannot emit a spurious hard-lock. ``verify_real_clip`` below is
    the deferred pod check: it compares motion heading vs quaternion heading to
    recover (and log) the constant camera-yaw offset -- if that offset is stable
    across sequences, a future revision may switch to quaternion + offset.
  * World-frame planarity / axis order (ENU vs the devkit's world convention) and
    the front_camera<->pose timestamp alignment are asserted on real bytes in
    ``verify_real_clip``; the pure signal/contract code is unit-tested on
    synthetic fixtures regardless (the Cosmos precedent, 2026-07-14).
  * Real DAY urban only -- no night/weather (that gap is ZOD's job, plan §7 #1).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tanitad.data._contract import EpisodeWindowDataset, assert_contract
from tanitad.data.calib import focal_crop_resize
from tanitad.data.comma2k19 import stack_frames
from tanitad.data.cosmos_drive import poses_to_signals
from tanitad.data.toy_driving import ToyEpisode

WHEELBASE = 3.0          # Chrysler Pacifica (PandaSet ego platform) ~3.0 m
TARGET_HZ = 10.0
SRC_HZ = 10.0            # PandaSet camera is already 10 Hz -> stride 1
FRONT_CAM = "front_camera"

DATA_TAG = "data:pandaset"       # every consuming experiment carries this tag
LICENSE = "CC-BY-4.0"            # permissive owned core (attribution + privacy)

# The REAL PandaSet front_camera calibration (arXiv 2112.12610 / devkit), grounded
# 2026-07-15 — NOT a guess. 1920x1080, fx~=1970 (HFOV ~52 deg), with NON-trivial
# barrel distortion. Two consequences for D-016 canonicalization (see GEOMETRY
# BLOCKER below):
PANDASET_FRONT = {
    "fx": 1970.0131, "fy": 1970.0091, "cx": 970.0002, "cy": 483.2988,
    "width": 1920, "height": 1080,
    "distortion": (-0.5894, 0.66, 0.0011, -0.001, -1.0088),   # k1,k2,p1,p2,k3
}
CANONICAL_FEFF = 266.0   # F_REF: the shared action->pixel scale (comma2k19)
FEFF_TOL = 0.05          # achieved f_eff must land within 5% of canonical


class GeometryError(ValueError):
    """Raised when PandaSet frames cannot be canonicalized to F_REF (fail-loud)."""


def front_camera_canonicalization(fx: float, h: int = 1080, w: int = 1920,
                                  size: int = 256) -> dict:
    """Does a CENTERED SQUARE crop of this pinhole reach the canonical f_eff=266?

    D-016 canonicalizes by cropping a square of side ``c = fx*size/F_REF`` then
    resizing to ``size`` (``focal_crop_resize``). For a 16:9 frame the square is
    bounded by the HEIGHT, so a camera whose required ``c`` exceeds ``h`` is
    height-clamped and lands MORE ZOOMED than canonical. PandaSet's front camera
    (fx~1970 on 1080-tall frames) needs c~1896 >> 1080 -> clamps to 1080 ->
    achieved f_eff ~= 467 px (~1.75x the canonical 266), i.e. a real cross-corpus
    ACTION->PIXEL SCALE mismatch. Pure + grounded on the real fx (unit-tested)."""
    from tanitad.data.calib import focal_crop_size
    c_ideal = int(round(fx * size / CANONICAL_FEFF))
    c = focal_crop_size(fx, h, w, size)                 # the clamped side actually used
    achieved = fx * size / c
    height_clamped = c_ideal > min(h, w)
    drop_in = abs(achieved - CANONICAL_FEFF) / CANONICAL_FEFF <= FEFF_TOL
    return {"fx": fx, "ideal_crop_px": c_ideal, "used_crop_px": c,
            "frame_min_dim": min(h, w), "achieved_feff_px": round(achieved, 1),
            "canonical_feff_px": CANONICAL_FEFF, "height_clamped": height_clamped,
            "drop_in": drop_in}


def _canonicalize(vid: Tensor, fx: float, size: int, strict: bool = True) -> Tensor:
    """focal_crop_resize with a FAIL-LOUD scale guard.

    If the achieved f_eff is off-canonical (height-bound frame; the PandaSet
    case), ``strict=True`` REFUSES to emit silently mis-scaled frames — PandaSet
    ingestion is BLOCKED until the D-016 R1 pad-crop lands (INTAKE §risk). With
    ``strict=False`` it emits the clamped frames and the caller owns the residual."""
    _, _, h, w = vid.shape
    rep = front_camera_canonicalization(fx, h, w, size)
    out = focal_crop_resize(vid, fx, size)
    if strict and not rep["drop_in"]:
        raise GeometryError(
            f"PandaSet front canonicalization off-scale: achieved f_eff "
            f"{rep['achieved_feff_px']}px vs canonical {CANONICAL_FEFF}px "
            f"(height_clamped={rep['height_clamped']}, frame {h}x{w}, fx={fx}). "
            f"Square-crop is height-bound on the 16:9 frame; needs the D-016 R1 "
            f"pad-crop (+undistort k1={PANDASET_FRONT['distortion'][0]}). "
            f"Pass strict=False to accept the residual.")
    return out

# I7 task-identity fingerprint (D-017) -- DELIBERATELY identical to comma2k19 /
# cosmos / physicalai so probes fit on one corpus are admissible on the others and
# MixedWindowDataset admits PandaSet into the owned real+sim mix.
CORPUS_META = {
    "channels": 9, "image_size": 256, "f_eff_px": 266.0, "hz": 10.0,
    "actions": ("steer_road_rad", "accel_mps2"),
    "poses": ("x_east_m", "y_north_m", "yaw_rad", "v_mps"),
}


# --------------------------------------------------------------------------- #
# Pose IO + assembly (pure, unit-tested without real data)                    #
# --------------------------------------------------------------------------- #
def load_camera_poses(pose_ref: str | Path) -> np.ndarray:
    """PandaSet ``poses.json`` -> world positions ``[N,3]`` (metres).

    The file is a JSON list of ``{"position": {x,y,z}, "heading": {w,x,y,z}}``.
    Only the position is used for signal derivation (heading is deliberately
    ignored; see module HONEST LIMITATIONS). Injectable via ``pose_fn`` for tests.
    """
    recs = json.loads(Path(pose_ref).read_text())
    pos = np.array([[r["position"]["x"], r["position"]["y"], r["position"]["z"]]
                    for r in recs], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"bad poses.json shape {pos.shape} from {pose_ref}")
    return pos


def load_intrinsics(intr_ref: str | Path) -> dict:
    """PandaSet ``intrinsics.json`` -> ``{fx, fy, cx, cy}`` (floats)."""
    d = json.loads(Path(intr_ref).read_text())
    return {k: float(d[k]) for k in ("fx", "fy", "cx", "cy")}


def positions_to_veh4x4(positions: np.ndarray, dt: float) -> np.ndarray:
    """World positions ``[N,3]`` -> ``[N,4,4]`` vehicle-to-world with yaw from the
    MOTION direction (offset-free heading; module HONEST LIMITATIONS).

    Yaw = ``atan2(dy, dx)`` of the (finite-difference) world velocity, unwrapped
    for continuity. Translation is the re-origined world (x, y). The result feeds
    the already-tested :func:`cosmos_drive.poses_to_signals`, so PandaSet reuses
    the exact steer/accel/speed derivation of the synthetic corpora.
    """
    p = np.asarray(positions, dtype=np.float64)
    x = p[:, 0] - p[0, 0]
    y = p[:, 1] - p[0, 1]
    vx = np.gradient(x, dt, edge_order=1)
    vy = np.gradient(y, dt, edge_order=1)
    yaw = np.unwrap(np.arctan2(vy, vx))                 # motion heading
    n = p.shape[0]
    M = np.zeros((n, 4, 4), dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 3] = c, -s, x
    M[:, 1, 0], M[:, 1, 1], M[:, 1, 3] = s, c, y
    M[:, 2, 2] = 1.0
    M[:, 3, 3] = 1.0
    return M


def pandaset_signals(positions: np.ndarray, dt: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """World positions ``[N,3]`` -> (actions ``[N,2]``, poses ``[N,4]``) via the
    shared Cosmos geometry. Thin wrapper: assemble a motion-heading 4x4, then reuse
    :func:`cosmos_drive.poses_to_signals`."""
    return poses_to_signals(positions_to_veh4x4(positions, dt), dt)


# --------------------------------------------------------------------------- #
# Discovery + video decode                                                     #
# --------------------------------------------------------------------------- #
def discover_sequences(root: str | Path, camera: str = FRONT_CAM) -> list[dict]:
    """Sequences that have a front-camera folder with images + poses + intrinsics.

    Each PandaSet sequence is ``<root>/<seq>/camera/<camera>/`` holding ``NN.jpg``,
    ``poses.json`` and ``intrinsics.json``. Returns one dict per usable sequence.
    """
    root = Path(root)
    out = []
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cam = seq_dir / "camera" / camera
        poses = cam / "poses.json"
        intr = cam / "intrinsics.json"
        imgs = sorted(cam.glob("*.jpg"))
        if imgs and poses.exists() and intr.exists():
            out.append({"seq_id": seq_dir.name, "cam_dir": cam,
                        "images": imgs, "poses": poses, "intrinsics": intr})
    return out


def _decode_frames(cam_dir: Path, images: list[Path], intr: dict,
                   size: int, strict: bool = True) -> Tensor:
    """Sorted front-camera jpgs -> uint8 ``[N,3,size,size]``, D-016 focal-canon
    with the fail-loud scale guard (:func:`_canonicalize`).

    Uses the REAL per-camera ``fx`` (PandaSet ships intrinsics). NOTE: for the
    front camera on 1920x1080 this canonicalization is height-bound and does NOT
    reach f_eff=266 (GeometryError under strict) — see the module GEOMETRY BLOCKER.
    """
    from PIL import Image                                 # lazy: .[real]
    frames = []
    for img in images:
        with Image.open(img) as im:
            arr = np.asarray(im.convert("RGB"))           # [H,W,3] u8
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))
    vid = torch.stack(frames)                             # [N,3,H,W] u8
    return _canonicalize(vid, float(intr["fx"]), size, strict=strict)


def _episode_id(seq_id: str, camera: str = FRONT_CAM) -> int:
    return int(hashlib.sha1(f"pandaset/{seq_id}/{camera}".encode())
               .hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# Build one episode                                                            #
# --------------------------------------------------------------------------- #
def build_episode(seq: dict, size: int = 256, n_stack: int = 3,
                  src_hz: float = SRC_HZ, decode_fn=_decode_frames,
                  pose_fn=load_camera_poses, intr_fn=load_intrinsics
                  ) -> ToyEpisode:
    """One PandaSet sequence -> contract episode at 10 Hz with D-015 stacking.

    Camera and poses share the sequence clock at ``src_hz`` (== TARGET_HZ, so
    stride 1, dt = 0.1 s). Signals are derived on that timeline, then ``n_stack``
    frames are channel-stacked with actions/poses aligned to the LATEST frame of
    each stack (drop the first k)."""
    stride = max(1, int(round(src_hz / TARGET_HZ)))
    dt = stride / src_hz

    intr = intr_fn(seq["intrinsics"])
    vid = decode_fn(seq["cam_dir"], seq["images"], intr, size)  # [M,3,S,S] u8
    positions = pose_fn(seq["poses"])                           # [N,3]
    n = min(vid.shape[0], positions.shape[0])
    if n < (n_stack + 2):
        raise ValueError(f"seq {seq.get('seq_id')} too short: {n} frames")
    vid = vid[:n][::stride]
    actions, poses = pandaset_signals(positions[:n][::stride], dt)

    m = min(vid.shape[0], actions.shape[0])
    stacked = stack_frames(vid[:m], n_stack)                    # [m-k,9,S,S] u8
    k = n_stack - 1
    return ToyEpisode(frames=stacked,
                      actions=torch.from_numpy(actions[k:m]),
                      poses=torch.from_numpy(poses[k:m]),
                      episode_id=_episode_id(seq["seq_id"]))


def build_episodes(seqs: list[dict], **kw) -> list[ToyEpisode]:
    """Build many sequences, skipping any that fail to decode (corrupt/partial)."""
    eps = []
    for i, seq in enumerate(seqs):
        if i % 10 == 0:
            print(f"[pandaset] building {i}/{len(seqs)}", flush=True)
        try:
            eps.append(build_episode(seq, **kw))
        except Exception as e:                                  # skip, keep going
            print(f"[pandaset] skip {seq.get('seq_id')}: {type(e).__name__}: {e}")
    return eps


def split_sequences(seqs: list[dict], val_frac: float = 0.2,
                    seed: int = 0) -> tuple[list[dict], list[dict]]:
    """SEQUENCE-level split (the I3 unit for independent PandaSet drives)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(seqs), generator=g).tolist()
    n_val = max(1, int(len(seqs) * val_frac))
    val_i = set(perm[:n_val])
    return ([s for i, s in enumerate(seqs) if i not in val_i],
            [s for i, s in enumerate(seqs) if i in val_i])


def PandaSetDataset(seqs: list[dict], window: int = 8, max_horizon: int = 16,
                    size: int = 256, n_stack: int = 3,
                    **build_kw) -> EpisodeWindowDataset:
    """Window dataset over PandaSet sequences -- item contract byte-identical to
    every other adapter (reuses the shared EpisodeWindowDataset). Pass an
    already-split sequence list (I3); never split one sequence across train/val."""
    eps = build_episodes(seqs, size=size, n_stack=n_stack, **build_kw)
    return EpisodeWindowDataset(eps, window=window, max_horizon=max_horizon)


# --------------------------------------------------------------------------- #
# Pod-only real-clip sanity (not run in CI -- documents the verification step) #
# --------------------------------------------------------------------------- #
def verify_real_clip(seq: dict, size: int = 256) -> dict:
    """Decode+derive ONE real PandaSet sequence and return sanity stats for the
    data card. Also recovers the constant camera-yaw offset by comparing the
    motion heading with the ``heading`` quaternion (the deferred check that would
    justify switching to quaternion+offset). Not run in CI (needs real bytes +
    Pillow); the pure code is unit-tested on synthetic fixtures."""
    intr = load_intrinsics(seq["intrinsics"])
    geom = front_camera_canonicalization(float(intr["fx"]),
                                         PANDASET_FRONT["height"],
                                         PANDASET_FRONT["width"], size)
    # strict=False so the residual is measured, not raised, in the sanity report
    ep = build_episode(seq, size=size,
                       decode_fn=lambda cd, im, it, s: _decode_frames(
                           cd, im, it, s, strict=False))
    assert_contract(ep, channels=9)
    v = ep.poses[:, 3]
    return {
        "seq_id": seq.get("seq_id"),
        "n_steps": int(ep.frames.shape[0]),
        "frames_shape": list(ep.frames.shape),
        "speed_mps": [float(v.min()), float(v.mean()), float(v.max())],
        "steer_p99_rad": float(torch.quantile(ep.actions[:, 0].abs(), 0.99)),
        "accel_p99_mps2": float(torch.quantile(ep.actions[:, 1].abs(), 0.99)),
        "frames_finite": bool(torch.isfinite(ep.frames.float()).all()),
        "actions_finite": bool(torch.isfinite(ep.actions).all()),
        "geometry": geom,                              # drop_in / achieved f_eff
        "distortion_k1": PANDASET_FRONT["distortion"][0],   # ignored by pinhole crop
    }
