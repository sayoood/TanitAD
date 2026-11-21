"""Zenseact Open Dataset (ZOD) -> TanitAD episode contract. CC-BY-SA-4.0.

WHY THIS EXISTS (OWN_DATASET_PLAN.md §7 ingest #1 -- the headline owned real-urban)
-----------------------------------------------------------------------------------
The real NVIDIA PhysicalAI-AV set is un-redistributable (D-002: internal-dev-only,
never shipped even privately), which forces every pod to rebuild its cache and
blocks every public claim. The own-dataset plan removes that by building the shared
corpus from license-CLEAN sources only. ZOD is the single best OWNED replacement
for PhysicalAI-AV's *real urban richness*: **CC-BY-SA-4.0** (ShareAlike -> a
SEPARATE public shard, never co-mingled with the permissive core), **14 European
countries**, highway+urban+suburban, **day / night / seasons / weather** -- exactly
the diversity the current 74%-straight, day-only mix lacks (FLEET_REVIEW 2026-07-17:
corpus diversity is the pending data-side gap; it is the enabling condition of the
ego-status shortcut). ZOD also ships **real CAN steering + OxTS RT3000 ego-motion**
(the comma-class action fidelity the synthetic corpora can only approximate).

CORPUS (ZOD Sequences/Drives; zod1-sdk layout; HF `Zenseact/ZOD`, access-gated)
-------------------------------------------------------------------------------
Grounded on the ZOD paper (arXiv 2305.02008, Table 2) + zod1-sdk:
    front camera : 8 MP, **3848 x 2168**, **HFOV 120 deg**, 10 Hz,
                   **Kannala-Brandt** fisheye (k1..k4), calibration.json per drive
    OxTS RT3000  : 100 Hz -- poses, velocities (X,Y,Z), accelerations (X,Y,Z),
                   angular rates, heading/pitch/roll, WGS84 lat/lon/alt
                   (pos accuracy 0.01 m, heading 0.1 deg)  -> ego-motion truth
    vehicle_data : 100 Hz -- steering angle (+rate, +wheel torque), accel/brake
                   pedal, turn indicator  -> real CAN steering (cross-check)
Front is the SAME 120-deg HFOV class as PhysicalAI/Cosmos front-wide, so D-016
canonicalization crops INWARD to the shared canonical half-angle (comma's ~51.4 deg)
-> the geometry falsifier (below) passes by construction with observed_frac=1.0.

Geometry -- the Kannala-Brandt <-> f-theta identity (the key reuse result)
--------------------------------------------------------------------------
The KB fisheye projects an incidence angle theta to a distorted angle
    theta_d = theta * (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)
and to native pixel radius  r(theta) = f_px * theta_d, i.e. a PURE ODD-power
polynomial in theta -- exactly the representation `calib.FThetaIntrinsics.poly`
already uses for the PhysicalAI f-theta fisheye:
    poly = (0, f_px, 0, f_px*k1, 0, f_px*k2, 0, f_px*k3, 0, f_px*k4)
So ZOD needs NO new geometry math: `kb_to_ftheta` builds an FThetaIntrinsics and
the proven `ftheta_crop_resize` / `ftheta_crop_size` canonicalize it to f_eff=266.
(This is why OWN_DATASET_PLAN said "fisheye -> existing ftheta_*" -- confirmed here.)

Signal derivation (REAL OxTS ego-motion -> the shared Cosmos geometry):
    x_east, y_north  = OxTS local-ENU position, re-origined to frame 0
    yaw              = OxTS heading (real vehicle heading, offset-free -- unlike
                       PandaSet, which had to fall back to motion heading because
                       its `heading` was the CAMERA-to-world rotation)
    v, accel, steer  <-  cosmos_drive.poses_to_signals on the assembled [N,4,4]
Camera 10 Hz == TARGET_HZ -> stride 1, dt = 0.1 s (OxTS/CAN 100 Hz are indexed to
the camera timestamps -- pinned on real bytes, see verify_real_clip).

Splits: SEQUENCE/DRIVE-level (I3) -- each ZOD sequence is an independent drive.

REPRESENTATIVE vs REAL calibration (honest scope, P8)
-----------------------------------------------------
ZOD ships the exact KB coefficients PER DRIVE in `calibration.json`; those are
behind the dataset's access gate (email request / gated HF repo). This module is
grounded on the PUBLISHED front-camera spec (3848x2168, 120 deg HFOV) via an
EQUIDISTANT KB approximation (`ZOD_FRONT_REPR`, k=0) so the geometry falsifier and
the contract are decided by the FOV alone -- which IS public. The real per-drive
(f_px, k1..k4, cx, cy) drop in unchanged at `verify_real_clip` time (the KB->poly
map is exact); a few-percent focal refinement only nudges the achieved f_eff, which
`ftheta_crop_resize` re-measures from the real poly. The byte-level Sequences/Drives
frame layout + OxTS<->camera timestamp alignment are likewise pinned on real bytes
(the Cosmos/PandaSet precedent); the pure signal/geometry/contract code below is
unit-tested on synthetic fixtures regardless.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tanitad.data._contract import EpisodeWindowDataset, assert_contract
from tanitad.data.calib import (F_REF, FThetaIntrinsics, canonical_halfangle_rad,
                                ftheta_crop_box, ftheta_crop_resize,
                                ftheta_crop_size)
from tanitad.data.comma2k19 import stack_frames
from tanitad.data.cosmos_drive import poses_to_signals
from tanitad.data.toy_driving import ToyEpisode

WHEELBASE = 2.94         # Volvo XC90 (ZOD ego platform, Zenseact/Volvo) ~2.94 m
TARGET_HZ = 10.0
SRC_HZ = 10.0            # ZOD front camera is 10 Hz -> stride 1
FRONT_CAM = "front"
STEER_RATIO_XC90 = 16.0  # Volvo XC90 nominal wheel:road ratio -- CALIBRATED on real
                         # bytes (verify_real_clip regresses CAN steer vs OxTS road-
                         # wheel); primary steer is OxTS-derived (ratio-free) below.

DATA_TAG = "data:zod"           # every consuming experiment carries this tag
LICENSE = "CC-BY-SA-4.0"        # COPYLEFT -> separate shard; never merged into core

CANONICAL_FEFF = 266.0          # F_REF: the shared action->pixel scale (comma2k19)
FEFF_TOL = 0.05                 # achieved f_eff must land within 5% of canonical
OBSERVED_FLOOR = 0.5            # ingest gate (D-016 R1 rule): >= ~50% real pixels


class GeometryError(ValueError):
    """Raised when ZOD frames cannot be canonicalized to F_REF (fail-loud)."""


# --------------------------------------------------------------------------- #
# Kannala-Brandt <-> f-theta (the exact identity; pure, unit-tested)          #
# --------------------------------------------------------------------------- #
def kb_to_ftheta(f_px: float, k: tuple[float, float, float, float],
                 cx: float, cy: float, width: int, height: int,
                 per_clip: bool = False) -> FThetaIntrinsics:
    """Kannala-Brandt (f_px, k1..k4) -> `calib.FThetaIntrinsics`.

    KB radius r(theta) = f_px * (theta + k1 th^3 + k2 th^5 + k3 th^7 + k4 th^9),
    which is exactly `FThetaIntrinsics`'s odd-power `poly` with
        poly = (0, f_px, 0, f_px*k1, 0, f_px*k2, 0, f_px*k3, 0, f_px*k4).
    So the proven f-theta crop/undistort path canonicalizes ZOD with zero new math.
    """
    k1, k2, k3, k4 = k
    poly = (0.0, f_px, 0.0, f_px * k1, 0.0, f_px * k2, 0.0, f_px * k3, 0.0,
            f_px * k4)
    return FThetaIntrinsics(poly=poly, cx=cx, cy=cy, width=width, height=height,
                            per_clip=per_clip)


# REPRESENTATIVE ZOD front intrinsics (grounded on the PUBLISHED spec: 3848x2168,
# HFOV 120 deg -> theta_max = 60 deg; equidistant focal f = (W/2)/theta_max). The
# real per-drive KB (f_px, k1..k4, cx, cy) from calibration.json drops in unchanged
# at verify_real_clip -- see the module docstring. k=0 (pure equidistant) is the
# conservative representative; real ZOD k are small and only refine f_eff a few %.
_ZOD_W, _ZOD_H, _ZOD_HFOV_DEG = 3848, 2168, 120.0
_ZOD_FPX_EQUI = (_ZOD_W / 2.0) / math.radians(_ZOD_HFOV_DEG / 2.0)   # ~1837 px
ZOD_FRONT_REPR = kb_to_ftheta(
    f_px=_ZOD_FPX_EQUI, k=(0.0, 0.0, 0.0, 0.0),
    cx=_ZOD_W / 2.0, cy=_ZOD_H / 2.0, width=_ZOD_W, height=_ZOD_H, per_clip=False)

# I7 task-identity fingerprint (D-017) -- DELIBERATELY identical to comma2k19 /
# cosmos / pandaset / physicalai so probes fit on one corpus are admissible on the
# others and MixedWindowDataset admits ZOD into the owned real+sim mix.
CORPUS_META = {
    "channels": 9, "image_size": 256, "f_eff_px": 266.0, "hz": 10.0,
    "actions": ("steer_road_rad", "accel_mps2"),
    "poses": ("x_east_m", "y_north_m", "yaw_rad", "v_mps"),
}


def front_camera_canonicalization(intr: FThetaIntrinsics, size: int = 256,
                                  f_ref: float = F_REF) -> dict:
    """Does a geometric-center crop of this KB fisheye reach f_eff=266 with a
    high enough OBSERVED fraction?  (The ZOD geometry falsifier -- pure + grounded.)

    D-016 crops the square that retains the shared canonical half-angle
    (`ftheta_crop_size`), then resizes to `size`. The IDEAL crop side is
    ``c_ideal = 2*r(theta_canon)`` (unclamped). ``observed_frac`` is the fraction
    of that ideal, geometric-centered box that lies WITHIN the native frame: a WIDE
    fisheye (ZOD, 120 deg) crops inward -> box fully in-frame -> 1.0; a NARROW
    camera whose native field is smaller than the canonical half-angle needs its
    periphery PADDED (unobserved) -> < 1.0. ``drop_in`` requires both f_eff within
    tol AND observed_frac >= OBSERVED_FLOOR (the D-016 R1 >=0.5 ingest rule --
    Udacity-like narrow FOV falsifies here, ZOD passes with margin)."""
    theta_canon = canonical_halfangle_rad(size, f_ref)
    r_ideal = float(intr.r_of_theta(theta_canon))            # native px, unclamped
    c_ideal = int(round(2.0 * r_ideal))
    # geometric-centered ideal box (may spill past the native edge for narrow cams)
    top = (intr.height - c_ideal) / 2.0
    left = (intr.width - c_ideal) / 2.0
    in_h = max(0.0, min(intr.height, top + c_ideal) - max(0.0, top))
    in_w = max(0.0, min(intr.width, left + c_ideal) - max(0.0, left))
    observed_frac = (in_h * in_w) / float(c_ideal * c_ideal) if c_ideal > 0 else 0.0
    # achieved f_eff of the ACTUAL (clamped) crop, measured through the real poly
    c_used = ftheta_crop_size(intr, size, f_ref)
    theta_edge = intr.theta_of_r(c_used / 2.0)
    achieved = (size / 2.0) / math.tan(theta_edge) if theta_edge > 0 else float("inf")
    feff_ok = abs(achieved - f_ref) / f_ref <= FEFF_TOL
    return {"f_px": round(intr.paraxial_focal, 2),
            "ideal_crop_px": c_ideal, "used_crop_px": c_used,
            "native_wh": [intr.width, intr.height],
            "achieved_feff_px": round(achieved, 1), "canonical_feff_px": f_ref,
            "observed_frac": round(observed_frac, 4),
            "feff_ok": feff_ok,
            "observed_ok": observed_frac >= OBSERVED_FLOOR,
            "drop_in": bool(feff_ok and observed_frac >= OBSERVED_FLOOR)}


def _canonicalize(vid: Tensor, intr: FThetaIntrinsics, size: int,
                  strict: bool = True) -> Tensor:
    """`ftheta_crop_resize` (geometric center -- single rig) with a fail-loud gate.

    ZOD is a SINGLE camera platform (not the PhysicalAI two-rig cy split), so a
    geometric-center crop is horizon-consistent and needs no per-clip cy; the
    principal-point path is unnecessary here. If the KB geometry cannot reach
    f_eff=266 at >= OBSERVED_FLOOR observed pixels, ``strict=True`` REFUSES rather
    than emit silently mis-scaled / mostly-padded frames."""
    rep = front_camera_canonicalization(intr, size)
    if strict and not rep["drop_in"]:
        raise GeometryError(
            f"ZOD front canonicalization off-contract: achieved f_eff "
            f"{rep['achieved_feff_px']}px vs {CANONICAL_FEFF}px (feff_ok="
            f"{rep['feff_ok']}), observed_frac {rep['observed_frac']} < "
            f"{OBSERVED_FLOOR} (observed_ok={rep['observed_ok']}). A narrow-FOV "
            f"source cannot fill the canonical canvas without padding the "
            f"unobserved periphery. Pass strict=False to accept the residual.")
    return ftheta_crop_resize(vid, intr, size, center="geometric")


# --------------------------------------------------------------------------- #
# OxTS ego-motion -> vehicle 4x4 -> shared Cosmos signal geometry (pure)       #
# --------------------------------------------------------------------------- #
def wgs84_to_enu(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
                 lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """WGS84 lat/lon/alt (deg, deg, m) -> local ENU metres about (lat0,lon0,alt0).

    Small-angle equirectangular projection about the origin -- exact to < a few cm
    over a ZOD sequence (<~1 km), which is well inside the 0.01 m OxTS accuracy at
    this scale. Only used when a sequence exposes raw WGS84 rather than a metric
    OxTS pose; `oxts_to_veh4x4` accepts either. Pure + unit-tested."""
    R = 6_378_137.0                                          # WGS84 equatorial radius
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)
    lat0r = math.radians(lat0)
    east = np.radians(lon - lon0) * R * math.cos(lat0r)
    north = np.radians(lat - lat0) * R
    up = alt - alt0
    return np.column_stack([east, north, up]).astype(np.float64)


def oxts_to_veh4x4(positions_enu: np.ndarray, headings: np.ndarray) -> np.ndarray:
    """OxTS local-ENU positions ``[N,3]`` + heading ``[N]`` (rad) -> ``[N,4,4]``
    vehicle-to-world, re-origined to frame 0.

    Unlike PandaSet (whose `heading` was the CAMERA-to-world rotation, forcing a
    motion-heading fallback), ZOD's OxTS heading IS the vehicle heading -- accurate
    to 0.1 deg -- so it drives yaw directly (offset-free, robust at standstill where
    motion heading is undefined). Feeds the tested `cosmos_drive.poses_to_signals`,
    so ZOD reuses the exact steer/accel/speed derivation of every other corpus."""
    p = np.asarray(positions_enu, dtype=np.float64)
    h = np.asarray(headings, dtype=np.float64)
    assert p.ndim == 2 and p.shape[1] == 3, p.shape
    assert h.shape == (p.shape[0],), (h.shape, p.shape)
    x = p[:, 0] - p[0, 0]
    y = p[:, 1] - p[0, 1]
    n = p.shape[0]
    M = np.zeros((n, 4, 4), dtype=np.float64)
    c, s = np.cos(h), np.sin(h)
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 3] = c, -s, x
    M[:, 1, 0], M[:, 1, 1], M[:, 1, 3] = s, c, y
    M[:, 2, 2] = 1.0
    M[:, 3, 3] = 1.0
    return M


def zod_signals(positions_enu: np.ndarray, headings: np.ndarray, dt: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """OxTS ENU positions ``[N,3]`` + heading ``[N]`` -> (actions ``[N,2]``,
    poses ``[N,4]``) via the shared Cosmos geometry (real OxTS heading for yaw)."""
    return poses_to_signals(oxts_to_veh4x4(positions_enu, headings), dt)


def can_steer_ratio(can_steer_wheel_rad: np.ndarray, oxts_road_wheel_rad: np.ndarray
                    ) -> float:
    """Least-squares wheel:road steering ratio from paired CAN vs OxTS-derived
    road-wheel angle (the calibration `verify_real_clip` recovers on real bytes).

    ratio = argmin_r || can/r - road ||  (closed form). Isolates ZOD's `steering
    angle` UNITS/ratio without trusting the platform nominal; a stable ratio across
    sequences would justify using CAN steer directly as a second action source."""
    a = np.asarray(can_steer_wheel_rad, dtype=np.float64)
    b = np.asarray(oxts_road_wheel_rad, dtype=np.float64)
    denom = float(np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else float("nan")


# --------------------------------------------------------------------------- #
# Discovery + decode (byte-level layout injectable; pinned on real bytes)      #
# --------------------------------------------------------------------------- #
def _episode_id(seq_id: str, camera: str = FRONT_CAM) -> int:
    return int(hashlib.sha1(f"zod/{seq_id}/{camera}".encode()).hexdigest()[:8], 16)


def _decode_frames(frame_paths: list[Path], intr: FThetaIntrinsics, size: int,
                   strict: bool = True) -> Tensor:
    """Sorted front-camera images -> uint8 ``[N,3,size,size]``, KB->f-theta canon
    with the fail-loud scale/observed gate (`_canonicalize`)."""
    from PIL import Image                                     # lazy: .[real]
    frames = []
    for img in frame_paths:
        with Image.open(img) as im:
            arr = np.asarray(im.convert("RGB"))               # [H,W,3] u8
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))
    vid = torch.stack(frames)                                 # [N,3,H,W] u8
    return _canonicalize(vid, intr, size, strict=strict)


# --------------------------------------------------------------------------- #
# Build one episode                                                            #
# --------------------------------------------------------------------------- #
def build_episode(seq: dict, size: int = 256, n_stack: int = 3,
                  src_hz: float = SRC_HZ, intr: FThetaIntrinsics | None = None,
                  decode_fn=_decode_frames, oxts_fn=None) -> ToyEpisode:
    """One ZOD sequence -> contract episode at 10 Hz with D-015 stacking.

    ``seq`` provides ``seq_id``, ``frames`` (sorted image paths) and an OxTS
    source; ``oxts_fn(seq)`` returns ``(positions_enu[N,3], headings[N])`` (default
    reads ``seq['positions_enu']`` / ``seq['headings']`` -- real HDF5 readers inject
    a real ``oxts_fn``). ``intr`` defaults to the representative ZOD front KB.
    Camera and OxTS share the 10 Hz timeline (stride 1, dt=0.1 s)."""
    stride = max(1, int(round(src_hz / TARGET_HZ)))
    dt = stride / src_hz
    intr = intr or ZOD_FRONT_REPR

    if oxts_fn is None:
        positions = np.asarray(seq["positions_enu"], dtype=np.float64)
        headings = np.asarray(seq["headings"], dtype=np.float64)
    else:
        positions, headings = oxts_fn(seq)
    vid = decode_fn(seq["frames"], intr, size)                # [M,3,S,S] u8

    n = min(vid.shape[0], positions.shape[0])
    if n < (n_stack + 2):
        raise ValueError(f"seq {seq.get('seq_id')} too short: {n} frames")
    vid = vid[:n][::stride]
    actions, poses = zod_signals(positions[:n][::stride], headings[:n][::stride], dt)

    m = min(vid.shape[0], actions.shape[0])
    stacked = stack_frames(vid[:m], n_stack)                  # [m-k,9,S,S] u8
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
            print(f"[zod] building {i}/{len(seqs)}", flush=True)
        try:
            eps.append(build_episode(seq, **kw))
        except Exception as e:                                # skip, keep going
            print(f"[zod] skip {seq.get('seq_id')}: {type(e).__name__}: {e}")
    return eps


def split_sequences(seqs: list[dict], val_frac: float = 0.2,
                    seed: int = 0) -> tuple[list[dict], list[dict]]:
    """SEQUENCE/DRIVE-level split (the I3 unit for independent ZOD drives)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(seqs), generator=g).tolist()
    n_val = max(1, int(len(seqs) * val_frac))
    val_i = set(perm[:n_val])
    return ([s for i, s in enumerate(seqs) if i not in val_i],
            [s for i, s in enumerate(seqs) if i in val_i])


def ZODDataset(seqs: list[dict], window: int = 8, max_horizon: int = 16,
               size: int = 256, n_stack: int = 3, **build_kw) -> EpisodeWindowDataset:
    """Window dataset over ZOD sequences -- item contract byte-identical to every
    other adapter (reuses the shared EpisodeWindowDataset). Pass an already-split
    sequence list (I3); never split one sequence across train/val."""
    eps = build_episodes(seqs, size=size, n_stack=n_stack, **build_kw)
    return EpisodeWindowDataset(eps, window=window, max_horizon=max_horizon)


# --------------------------------------------------------------------------- #
# Pod-only real-clip sanity (not run in CI -- documents the verification step) #
# --------------------------------------------------------------------------- #
def verify_real_clip(seq: dict, intr: FThetaIntrinsics | None = None,
                     size: int = 256) -> dict:
    """Decode+derive ONE real ZOD sequence and return sanity stats for the data
    card. Uses the REAL per-drive KB `intr` (from calibration.json) and reports the
    geometry drop-in, the OxTS speed/steer ranges, and -- if CAN steering is
    provided (``seq['can_steer_wheel_rad']``) -- the recovered wheel:road steering
    ratio. Not run in CI (needs real bytes + Pillow); the pure code is unit-tested
    on synthetic fixtures."""
    from tanitad.data._contract import frame_change_fraction, to_float_frames
    intr = intr or ZOD_FRONT_REPR
    geom = front_camera_canonicalization(intr, size)
    ep = build_episode(seq, size=size, intr=intr,
                       decode_fn=lambda fp, it, s: _decode_frames(fp, it, s,
                                                                  strict=False))
    assert_contract(ep, channels=9)
    v = ep.poses[:, 3]
    out = {
        "seq_id": seq.get("seq_id"),
        "n_steps": int(ep.frames.shape[0]),
        "frames_shape": list(ep.frames.shape),
        "speed_mps": [float(v.min()), float(v.mean()), float(v.max())],
        "steer_p99_rad": float(torch.quantile(ep.actions[:, 0].abs(), 0.99)),
        "accel_p99_mps2": float(torch.quantile(ep.actions[:, 1].abs(), 0.99)),
        "a8_frame_change_fraction": frame_change_fraction(
            to_float_frames(ep.frames)[:, -3:]),
        "frames_finite": bool(torch.isfinite(ep.frames.float()).all()),
        "actions_finite": bool(torch.isfinite(ep.actions).all()),
        "geometry": geom,                              # drop_in / observed_frac / f_eff
    }
    if "can_steer_wheel_rad" in seq:                   # recover the ratio on real data
        _, poses = zod_signals(np.asarray(seq["positions_enu"], np.float64),
                               np.asarray(seq["headings"], np.float64), 0.1)
        actions, _ = zod_signals(np.asarray(seq["positions_enu"], np.float64),
                                 np.asarray(seq["headings"], np.float64), 0.1)
        out["recovered_steer_ratio"] = can_steer_ratio(
            np.asarray(seq["can_steer_wheel_rad"], np.float64), actions[:, 0])
    return out
