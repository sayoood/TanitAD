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
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.data._contract import finite_diff_accel
from tanitad.data.calib import (F_REF, PHYSICALAI_FRONT_WIDE_FTHETA,
                                FThetaIntrinsics, ftheta_crop_resize)
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
# dataset, so the per-clip table is NOT committed to this (public) repo — it is
# read from the LOCAL data dir on machines that already hold the licensed
# dataset (``<root>/calibration/physicalai_front_wide_intrinsics.csv``, or the
# path in ``$TANITAD_PAI_INTRINSICS``). When absent, the measured corpus-median
# ``PHYSICALAI_FRONT_WIDE_FTHETA`` fallback (an aggregate in calib.py) is used —
# and because the geometric-center crop depends only on the near-constant focal
# (per-clip sigma 0.47%), it lands f_eff == F_REF (~266) for EVERY clip either
# way, so the corrected cache is correct with or without the per-clip table.
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
            width=int(r.width), height=int(r.height))))
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
    """Real per-clip f-theta intrinsics (PREFERRED, from the local licensed data
    dir); measured corpus-median fallback (warns once) when unavailable."""
    intr = _intrinsics_table(root).get(str(clip_id))
    if intr is None:
        if clip_id not in _warned_fallback:
            _warned_fallback.add(clip_id)
            print(f"[physicalai] no per-clip intrinsics for {clip_id!r}"
                  f"{' under ' + str(root) if root else ''}; using measured "
                  f"corpus-median f-theta fallback (f_eff still ~266)", flush=True)
        return PHYSICALAI_FRONT_WIDE_FTHETA
    return intr


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
    -> f_eff ~434 px, i.e. 1.6x over-zoomed vs comma. The sacrificed wide
    periphery still returns later as H2 side-view modalities.
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
    intr = intrinsics_for_clip(Path(mp4).name.split(".")[0],
                               _physicalai_root_of(mp4))
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
