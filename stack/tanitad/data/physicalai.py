"""PhysicalAI-AV R0 -> TanitAD episode contract (D-012, D-015; tag data:physicalai).

Consumes the Stage-R0 output of `scripts/physicalai_r0.py`:
    <root>/r0/r0_selection.parquet
    <root>/r0/camera_front_wide/**/{clip_id}.camera_front_wide_120fov.mp4
                                  + .timestamps.parquet
    <root>/labels/egomotion/egomotion.chunk_{chunk:04d}.zip   ({clip_id}.egomotion.parquet)

Signal derivation (egomotion schema: timestamp, q*, x/y/z, vx/vy/vz, ax/ay/az, curvature):
    v      = ||(vx, vy)||                       [m/s]
    steer  = atan(WHEELBASE * curvature)        road-wheel angle proxy [rad]
    accel  = finite difference of v             [m/s^2]  (contract semantics)
    pose   = (x, y, yaw=atan2(vy, vx), v)       clip-local frame
Video frames are resampled to 10 Hz by nearest-timestamp alignment against the
egomotion clock, then D-015 stacking (3 frames @100 ms -> 9 channels, actions/
poses aligned to the latest frame).

Splits: CLIP-level (each 20 s clip is an independent recording -> the I3 unit
here is the clip; never split windows of one clip across train/val).
"""

from __future__ import annotations

import io
import math
import os
import warnings
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.data._contract import finite_diff_accel
from tanitad.data.calib import (F_REF, PHYSICALAI_FRONT_WIDE_FTHETA,
                                FThetaIntrinsics, ftheta_crop_resize,
                                ftheta_horizon_row)
from tanitad.data.comma2k19 import stack_frames
from tanitad.data.toy_driving import ToyEpisode

WHEELBASE = 2.9          # Hyperion platform class, sedan/SUV proxy
TARGET_HZ = 10.0

# I7 task-identity fingerprint (D-017) — matches comma2k19's on purpose:
# D-016 canonicalization makes the corpora compatible; I7 PROVES it per run.
# f_eff_px is the shared canonical focal ACHIEVED by the f-theta-correct crop
# (calib.ftheta_crop_resize; verified ~F_REF in tests + build_pai_cache) — NOT
# the old nominal value the pipeline used to *assume* while actually delivering
# ~434 px. Sourced from calib.F_REF so the claim can't silently drift from the
# canonicalization again (the pre-fix failure GEOMETRY_INTEGRITY_AUDIT.md found).
CORPUS_META = {
    "channels": 9, "image_size": 256, "f_eff_px": F_REF, "hz": 10.0,
    "actions": ("steer_road_rad", "accel_mps2"),
    "poses": ("x_east_m", "y_north_m", "yaw_rad", "v_mps"),
}

# Per-clip f-theta intrinsics come from the dataset's OWN
# calibration/camera_intrinsics feature. PROVENANCE: PhysicalAI-AV is a GATED
# dataset, so per-clip calibration is NOT committed to this (public) repo — it is
# read (or fetched with the licensed HF token) on machines that hold the dataset.
# Two sources, in order: (1) a local CSV table
# (``<root>/calibration/physicalai_front_wide_intrinsics.csv`` or the path in
# ``$TANITAD_PAI_INTRINSICS``); (2) the dataset's per-chunk parquet
# ``calibration/camera_intrinsics/*.parquet`` (see `_intrinsics_from_parquet`).
# When BOTH are absent, the measured corpus-median ``PHYSICALAI_FRONT_WIDE_FTHETA``
# fallback (calib.py) still lands f_eff == F_REF (~266) via the near-constant
# focal (per-clip sigma 0.47%) — but its cy is a RIG-B value, so the crop then
# reverts to geometric-center (horizon NOT rig-corrected). The two-rig VERTICAL
# fix REQUIRES a per-clip cy (source 1 or 2), which drives the principal-point-
# centered crop in `calib.ftheta_crop_resize(center="principal")`.
_INTR_ENV = "TANITAD_PAI_INTRINSICS"
_INTR_BASENAME = "physicalai_front_wide_intrinsics.csv"
_warned_fallback: set[str] = set()


def _physicalai_root_of(mp4: Path) -> Path | None:
    """Recover the corpus root (the parent of ``r0/``) from a clip mp4 path."""
    for p in Path(mp4).parents:
        if p.name == "r0":
            return p.parent
    return None


@lru_cache(maxsize=8)
def _load_intrinsics_csv(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path, dtype={"clip_id": str})
    rows = []
    for r in df.itertuples(index=False):
        rows.append((str(r.clip_id), FThetaIntrinsics(
            poly=(float(r.fw_poly_0), float(r.fw_poly_1), float(r.fw_poly_2),
                  float(r.fw_poly_3), float(r.fw_poly_4)),
            cx=float(r.cx), cy=float(r.cy),
            width=int(r.width), height=int(r.height), per_clip=True)))
    return tuple(rows)


def _intrinsics_table(root: str | Path | None = None) -> dict[str, FThetaIntrinsics]:
    """clip_id -> FThetaIntrinsics from the LOCAL per-clip table, if present."""
    path = os.environ.get(_INTR_ENV)
    if not path and root is not None:
        cand = Path(root) / "calibration" / _INTR_BASENAME
        path = str(cand) if cand.exists() else None
    if not path or not Path(path).exists():
        return {}
    return dict(_load_intrinsics_csv(str(Path(path).resolve())))


def intrinsics_for_clip(clip_id: str, root: str | Path | None = None
                        ) -> FThetaIntrinsics:
    """Real per-clip f-theta intrinsics (PREFERRED); measured corpus-median
    fallback (warns once) when unavailable.

    Resolution order: (1) a local CSV table (``$TANITAD_PAI_INTRINSICS`` or
    ``<root>/calibration/physicalai_front_wide_intrinsics.csv``); (2) the
    dataset's own per-chunk ``calibration/camera_intrinsics`` parquet (downloaded
    on demand — see `_calib_chunk_path`); (3) the corpus-median fallback. Only
    (1) and (2) carry ``per_clip=True`` and can drive the principal-point crop;
    the fallback reverts that crop to geometric-center (its cy is a rig-B value).
    """
    intr = _intrinsics_table(root).get(str(clip_id))
    if intr is None and root is not None:
        intr = _intrinsics_from_parquet(str(clip_id), root)
    if intr is None:
        if clip_id not in _warned_fallback:
            _warned_fallback.add(clip_id)
            print(f"[physicalai] no per-clip intrinsics for {clip_id!r}"
                  f"{' under ' + str(root) if root else ''}; using measured "
                  f"corpus-median f-theta fallback (f_eff still ~266, but the crop "
                  f"reverts to geometric-center -> horizon NOT rig-corrected)",
                  flush=True)
        return PHYSICALAI_FRONT_WIDE_FTHETA
    return intr


# --------------------------------------------------------------------------- #
# Per-clip calibration from the dataset's OWN calibration/ feature (gated).     #
# Intrinsics (cx, cy, fw_poly) key the principal-point crop; extrinsics (mount  #
# pose) give the per-clip pitch that locates the horizon. Both live per CHUNK   #
# (chunk == the r0-selection `chunk`); downloaded on demand to <root>.          #
# --------------------------------------------------------------------------- #
_HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
_FRONT_WIDE_CAM = "camera_front_wide_120fov"
_CALIB_INTR = "camera_intrinsics"
_CALIB_EXTR = "sensor_extrinsics"
_warned_calib: set[str] = set()


@dataclass(frozen=True)
class FrontWideExtrinsics:
    """Front-wide camera mount pose in the vehicle frame (x fwd, y left, z up),
    from ``calibration/sensor_extrinsics``: rotation quaternion (cam->vehicle)
    and translation [m]. Supplies the per-clip pitch that places the horizon."""

    qx: float
    qy: float
    qz: float
    qw: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def rotation_cam_to_vehicle(self) -> np.ndarray:
        n = math.sqrt(self.qx**2 + self.qy**2 + self.qz**2 + self.qw**2) or 1.0
        qx, qy, qz, qw = self.qx / n, self.qy / n, self.qz / n, self.qw / n
        return np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])

    def vehicle_forward_in_cam(self) -> tuple[float, float, float]:
        """Vehicle-forward (the straight-ahead horizon ray) in the CAMERA frame
        (+x right, +y down, +z boresight) — feed to `calib.ftheta_project_ray`."""
        d = self.rotation_cam_to_vehicle().T @ np.array([1.0, 0.0, 0.0])
        return (float(d[0]), float(d[1]), float(d[2]))

    def optical_axis_pitch_rad(self) -> float:
        """Downward pitch of the optical axis below horizontal [rad], +down."""
        b = self.rotation_cam_to_vehicle() @ np.array([0.0, 0.0, 1.0])
        return math.asin(max(-1.0, min(1.0, -float(b[2]))))


@lru_cache(maxsize=8)
def _chunk_of_clip(root_str: str) -> dict:
    """clip_id -> chunk number, from the R0 selection (keys the calibration file)."""
    sel = pd.read_parquet(Path(root_str) / "r0" / "r0_selection.parquet")
    return dict(zip(sel["clip_id"].astype(str), sel["chunk"].astype(int)))


def _calib_chunk_path(root: str | Path, kind: str, chunk: int,
                      download: bool = True) -> Path | None:
    """Local path to ``calibration/<kind>/<kind>.chunk_{chunk:04d}.parquet``,
    fetching it from the gated HF dataset (via the repo TLS/token helpers) if it
    is not already cached under ``root``. Returns None (warns once) on failure."""
    rel = f"calibration/{kind}/{kind}.chunk_{chunk:04d}.parquet"
    local = Path(root) / rel
    if local.exists():
        return local
    if not download:
        return None
    key = f"{kind}:{chunk}"
    try:
        from tanitad.keys import enable_tls, load_keys
        enable_tls()
        load_keys()
        import huggingface_hub
        huggingface_hub.hf_hub_download(_HF_REPO, rel, repo_type="dataset",
                                        local_dir=str(root))
    except Exception as e:  # noqa: BLE001
        if key not in _warned_calib:
            _warned_calib.add(key)
            warnings.warn(f"[physicalai] could not obtain {rel}: {e!r}; "
                          f"per-clip calibration unavailable for chunk {chunk}",
                          RuntimeWarning, stacklevel=2)
        return None
    return local if local.exists() else None


def _front_wide_rows(parquet_path: str) -> pd.DataFrame:
    """front_wide rows of a per-chunk calibration parquet, clip_id as a column
    (the file is MultiIndexed by (clip_id, camera/sensor_name))."""
    df = pd.read_parquet(parquet_path).reset_index()
    name_col = "camera_name" if "camera_name" in df.columns else (
        "sensor_name" if "sensor_name" in df.columns else None)
    if name_col is None:                       # unnamed MultiIndex levels
        df = df.rename(columns={"level_0": "clip_id", "level_1": "name"})
        name_col = "name"
    return df[df[name_col].astype(str) == _FRONT_WIDE_CAM]


@lru_cache(maxsize=32)
def _load_chunk_intrinsics(parquet_path: str) -> dict:
    out = {}
    for r in _front_wide_rows(parquet_path).itertuples(index=False):
        out[str(r.clip_id)] = FThetaIntrinsics(
            poly=(float(r.fw_poly_0), float(r.fw_poly_1), float(r.fw_poly_2),
                  float(r.fw_poly_3), float(r.fw_poly_4)),
            cx=float(r.cx), cy=float(r.cy),
            width=int(r.width), height=int(r.height), per_clip=True)
    return out


@lru_cache(maxsize=32)
def _load_chunk_extrinsics(parquet_path: str) -> dict:
    out = {}
    for r in _front_wide_rows(parquet_path).itertuples(index=False):
        out[str(r.clip_id)] = FrontWideExtrinsics(
            qx=float(r.qx), qy=float(r.qy), qz=float(r.qz), qw=float(r.qw),
            x=float(r.x), y=float(r.y), z=float(r.z))
    return out


def _intrinsics_from_parquet(clip_id: str, root: str | Path
                             ) -> FThetaIntrinsics | None:
    chunk = _chunk_of_clip(str(root)).get(str(clip_id))
    if chunk is None:
        return None
    path = _calib_chunk_path(root, _CALIB_INTR, int(chunk))
    return _load_chunk_intrinsics(str(path)).get(str(clip_id)) if path else None


def extrinsics_for_clip(clip_id: str, root: str | Path | None = None
                        ) -> FrontWideExtrinsics | None:
    """Per-clip front-wide extrinsics from the dataset's ``sensor_extrinsics``
    calibration (downloaded on demand). None when unavailable — callers then
    treat the mount as level (optical axis == horizon)."""
    if root is None:
        return None
    chunk = _chunk_of_clip(str(root)).get(str(clip_id))
    if chunk is None:
        return None
    path = _calib_chunk_path(root, _CALIB_EXTR, int(chunk))
    return _load_chunk_extrinsics(str(path)).get(str(clip_id)) if path else None


def discover_r0_clips(root: str | Path) -> list[dict]:
    """Selected clips that have both camera mp4 and egomotion available."""
    root = Path(root)
    sel = pd.read_parquet(root / "r0" / "r0_selection.parquet")
    chunk_of = dict(zip(sel["clip_id"].astype(str), sel["chunk"].astype(int)))
    out = []
    for mp4 in sorted((root / "r0" / "camera_front_wide").rglob("*.mp4")):
        clip_id = mp4.name.split(".")[0]
        if clip_id not in chunk_of:
            continue
        ts = mp4.with_name(mp4.name.replace(".mp4", ".timestamps.parquet"))
        ego_zip = root / "labels" / "egomotion" / \
            f"egomotion.chunk_{chunk_of[clip_id]:04d}.zip"
        if ts.exists() and ego_zip.exists():
            out.append({"clip_id": clip_id, "mp4": mp4, "timestamps": ts,
                        "ego_zip": ego_zip})
    return out


def load_egomotion(ego_zip: Path, clip_id: str) -> pd.DataFrame:
    with zipfile.ZipFile(ego_zip) as z:
        name = next(n for n in z.namelist()
                    if n.endswith(f"{clip_id}.egomotion.parquet"))
        return pd.read_parquet(io.BytesIO(z.read(name)))


def _decode_mp4(mp4: Path, size: int) -> Tensor:
    """All frames of the clip -> uint8 [N, 3, size, size], f-theta-correct crop.

    D-016 fix (GEOMETRY_INTEGRITY_AUDIT.md): the front-wide is an f-theta
    FISHEYE. Resolve the clip's REAL per-clip intrinsics (keyed by the clip_id
    in the filename) and crop against the true radial map so the canonical
    f_eff == F_REF (comma-matched, ~266 px), instead of the old nominal-120-deg
    PINHOLE focal (554 px) that cropped a 533-px square retaining only ~16.4 deg
    -> f_eff ~434 px, i.e. 1.6x over-zoomed vs comma.

    D-016 R1 (two-rig fix): the crop is centered on the per-clip principal point
    (cx, cy) via `ftheta_crop_resize(center="principal")` — its default — so the
    horizon lands at the SAME output row for rig A (cy~543) and rig B (cy~755).
    Per-clip extrinsics are resolved too (the mount pitch that locates the
    horizon) and the achieved horizon output row is recorded in
    ``_decode_mp4.last_calib`` for the data card / build check. The sacrificed
    wide periphery still returns later as H2 side-view modalities.
    """
    import av
    frames = []
    with av.open(str(mp4)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for fr in c.decode(stream):
            rgb = torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1)
            frames.append(rgb)
    vid = torch.stack(frames)
    clip_id = Path(mp4).name.split(".")[0]
    root = _physicalai_root_of(mp4)
    intr = intrinsics_for_clip(clip_id, root)
    extr = extrinsics_for_clip(clip_id, root)
    # Provenance: with the per-clip (cx, cy) crop the horizon output row is
    # ~size/2 for both rigs; record it (and the legacy geometric-center row it
    # replaces) so a build check can confirm rig A and rig B now agree.
    h, w = int(vid.shape[-2]), int(vid.shape[-1])
    d_cam = extr.vehicle_forward_in_cam() if extr is not None else (0.0, 0.0, 1.0)
    applied = "principal" if intr.per_clip else "geometric"
    _decode_mp4.last_calib = {
        "clip_id": clip_id, "per_clip": intr.per_clip, "center": applied,
        "cx": round(intr.cx, 1), "cy": round(intr.cy, 1),
        "extrinsics": extr is not None,
        "horizon_row": round(ftheta_horizon_row(intr, d_cam, h, w, size,
                                                center=applied), 1),
        "horizon_row_legacy_geometric": round(
            ftheta_horizon_row(intr, d_cam, h, w, size, center="geometric"), 1),
    }
    return ftheta_crop_resize(vid, intr, size)


def signals_at(ego: pd.DataFrame, t_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate egomotion signals at query timestamps (same clock).

    Returns (actions [n,2], poses [n,4]) per the contract."""
    t = ego["timestamp"].to_numpy(np.float64)
    order = np.argsort(t)
    t = t[order]

    def col(c):
        return np.interp(t_query, t, ego[c].to_numpy(np.float64)[order])

    vx, vy = col("vx"), col("vy")
    v = np.hypot(vx, vy)
    curv = col("curvature")
    steer = np.arctan(WHEELBASE * curv)
    dt = float(np.median(np.diff(t_query))) if len(t_query) > 1 else 0.1
    # normalize dt to seconds if the clock is in micro/nanoseconds
    for scale in (1e9, 1e6, 1e3):
        if dt > 50.0:
            dt /= scale if dt / scale >= 1e-3 else 1.0
    dt = dt if 1e-3 < dt < 10.0 else 0.1
    accel = finite_diff_accel(v.astype(np.float32), dt)
    actions = np.column_stack([steer, accel]).astype(np.float32)
    yaw = np.arctan2(vy, vx)
    poses = np.column_stack([col("x"), col("y"), yaw, v]).astype(np.float32)
    return actions, poses


def build_episode(clip: dict, size: int = 256, n_stack: int = 3,
                  decode_fn=_decode_mp4) -> ToyEpisode:
    """One R0 clip -> contract episode at 10 Hz with D-015 stacking."""
    ts = pd.read_parquet(clip["timestamps"])
    tcol = next(c for c in ts.columns if "time" in c.lower())
    t_frames = ts[tcol].to_numpy(np.float64)
    ego = load_egomotion(clip["ego_zip"], clip["clip_id"])

    # resample the video timeline to 10 Hz on the shared clock
    span = t_frames[-1] - t_frames[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):                 # ns/us/ms clocks -> seconds
        if span / cand > 1.0:
            unit = cand
            break
    n_target = max(int(span / unit * TARGET_HZ), n_stack + 1)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    frame_idx = np.searchsorted(t_frames, t_query).clip(0, len(t_frames) - 1)

    vid = decode_fn(clip["mp4"], size)[frame_idx]              # [n,3,S,S] u8
    actions, poses = signals_at(ego, t_query)
    n = min(vid.shape[0], actions.shape[0])
    stacked = stack_frames(vid[:n], n_stack)
    k = n_stack - 1
    ep_id = int.from_bytes(clip["clip_id"].encode()[:4].ljust(4, b"\0"), "big")
    # uint8 in memory; datasets convert per window (500 float32 clips ~ 236 GB).
    return ToyEpisode(frames=stacked,
                      actions=torch.from_numpy(actions[k:n]),
                      poses=torch.from_numpy(poses[k:n]),
                      episode_id=ep_id)


def split_clips(clips: list[dict], val_frac: float = 0.2,
                seed: int = 0) -> tuple[list[dict], list[dict]]:
    """CLIP-level split (the I3 unit for independent 20 s recordings)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    val_i = set(perm[:n_val])
    return ([c for i, c in enumerate(clips) if i not in val_i],
            [c for i, c in enumerate(clips) if i in val_i])
