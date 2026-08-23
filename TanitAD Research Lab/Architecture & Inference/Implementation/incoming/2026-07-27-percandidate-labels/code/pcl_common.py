"""Shared loaders for the per-candidate label ladder (2026-07-27).

⚠️ HOST / CORPUS DISCLOSURE — read before quoting anything produced here.

The dev box does NOT hold the parity corpus. It holds
  * ``physicalai-train-14231cd29c74``  (400 episodes)  — NOT the parity key
    ``e438721ae894``; usable ONLY for corpus-property measurements, never for a
    cross-arm comparison.
  * ``physicalai-val-bb543bdf7836``    (100 episodes)  — NOT the canonical clean
    val ``physicalai-val-0c5f7dac3b11`` either.
  * a PhysicalAI-AV raw probe cache with ``obstacle.offline`` and ``egomotion``
    **chunk 0000 only** (96 clips carrying both).

Nothing here re-selects episodes for an arm comparison, so the parity invariant
is not touched; every number is a property of *labels*, not of a model.

🔒 PhysicalAI-AV is gated-confidential. Clip UUIDs never leave this module —
public artifacts carry an opaque index (``clip_00``…) only.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROBE = Path(r"C:/Users/Admin/AppData/Local/Temp/claude/pai_probe_cache")
EPCACHE = Path(r"C:/Users/Admin/tanitad-data/physicalai/_epcache")
TRAIN_DIR = EPCACHE / "physicalai-train-14231cd29c74"
VAL_DIR = EPCACHE / "physicalai-val-bb543bdf7836"

OBST_ZIP = PROBE / "labels/obstacle.offline/obstacle.offline.chunk_0000.zip"
EGO_ZIP = PROBE / "labels/egomotion/egomotion.chunk_0000.zip"
VEHDIM = PROBE / "calibration/vehicle_dimensions/vehicle_dimensions.chunk_0000.parquet"

DT = 0.1           # 10 Hz window grid — the epcache / trainer convention
N_STEPS = 20       # 2 s dense horizon, matching v4's [B, 256, 20, 2] fan
WP_STEPS = (5, 10, 15, 20)   # the 4-wp eval convention (taniteval)


def clip_alias(clip_id: str) -> str:
    """Opaque, stable, non-reversible alias for a gated clip UUID."""
    return "clip_" + hashlib.sha256(clip_id.encode()).hexdigest()[:8]


@dataclass
class ClipScene:
    alias: str
    ego: pd.DataFrame          # egomotion rows, timestamp in us, sorted
    obst: pd.DataFrame         # obstacle.offline rows, timestamp_us
    ego_len: float
    ego_wid: float


def _zip_clips(z: zipfile.ZipFile) -> dict[str, str]:
    return {n.split(".")[0]: n for n in z.namelist() if n.endswith(".parquet")}


def joint_clips() -> list[str]:
    """Clip ids present in BOTH obstacle.offline and egomotion chunk 0000."""
    with zipfile.ZipFile(OBST_ZIP) as zo, zipfile.ZipFile(EGO_ZIP) as ze:
        return sorted(set(_zip_clips(zo)) & set(_zip_clips(ze)))


def load_scenes(limit: int | None = None) -> list[ClipScene]:
    dims = pd.read_parquet(VEHDIM)
    out: list[ClipScene] = []
    with zipfile.ZipFile(OBST_ZIP) as zo, zipfile.ZipFile(EGO_ZIP) as ze:
        oc, ec = _zip_clips(zo), _zip_clips(ze)
        for cid in sorted(set(oc) & set(ec))[:limit]:
            ego = pd.read_parquet(io.BytesIO(ze.read(ec[cid])))
            ego = ego.sort_values("timestamp").reset_index(drop=True)
            obst = pd.read_parquet(io.BytesIO(zo.read(oc[cid])))
            row = dims.loc[cid] if cid in dims.index else None
            out.append(ClipScene(
                alias=clip_alias(cid), ego=ego, obst=obst,
                ego_len=float(row["length"]) if row is not None else 4.872,
                ego_wid=float(row["width"]) if row is not None else 2.121))
    return out


# --------------------------------------------------------------------------- #
# ego kinematics                                                              #
# --------------------------------------------------------------------------- #
def quaternion_yaw(qx, qy, qz, qw):
    """Same convention as tanitad.data.physicalai.quaternion_yaw."""
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def ego_at(ego: pd.DataFrame, t_us: np.ndarray) -> np.ndarray:
    """Interpolate (x, y, yaw, v) at query timestamps [us] -> [n, 4].

    Yaw is UNWRAPPED before interpolation then re-wrapped — the physicalai.py
    convention, so no +-pi seam is smeared across a query point.
    """
    t = ego["timestamp"].to_numpy(np.float64)
    x = np.interp(t_us, t, ego["x"].to_numpy(np.float64))
    y = np.interp(t_us, t, ego["y"].to_numpy(np.float64))
    v = np.interp(t_us, t, np.hypot(ego["vx"].to_numpy(np.float64),
                                    ego["vy"].to_numpy(np.float64)))
    yaw_native = np.unwrap(quaternion_yaw(*(ego[c].to_numpy(np.float64)
                                            for c in ("qx", "qy", "qz", "qw"))))
    yaw_u = np.interp(t_us, t, yaw_native)
    yaw = np.arctan2(np.sin(yaw_u), np.cos(yaw_u))
    return np.stack([x, y, yaw, v], axis=1)


def to_ego_frame(pts_world: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """World xy [..., 2] -> ego frame of pose ``origin`` (x, y, yaw, v)."""
    d = pts_world - origin[:2]
    c, s = np.cos(-origin[2]), np.sin(-origin[2])
    return np.stack([d[..., 0] * c - d[..., 1] * s,
                     d[..., 0] * s + d[..., 1] * c], axis=-1)


def from_ego_frame(pts_ego: np.ndarray, origin: np.ndarray) -> np.ndarray:
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return np.stack([pts_ego[..., 0] * c - pts_ego[..., 1] * s + origin[0],
                     pts_ego[..., 0] * s + pts_ego[..., 1] * c + origin[1]],
                    axis=-1)


# --------------------------------------------------------------------------- #
# oriented-box overlap (separating-axis theorem, exact for convex quads)       #
# --------------------------------------------------------------------------- #
def obb_corners(cx, cy, yaw, length, width):
    """[..., 4, 2] corners. cx/cy/yaw/length/width broadcast."""
    hl, hw = length / 2.0, width / 2.0
    ox = np.stack([hl, hl, -hl, -hl], axis=-1)
    oy = np.stack([hw, -hw, -hw, hw], axis=-1)
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    return np.stack([cx[..., None] + ox * c - oy * s,
                     cy[..., None] + ox * s + oy * c], axis=-1)


def obb_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SAT overlap between two batches of quads [..., 4, 2] -> bool [...].

    Two convex polygons intersect iff no edge normal of either separates them.
    """
    def axes(q):
        e = np.roll(q, -1, axis=-2) - q                     # [..., 4, 2]
        return np.stack([-e[..., 1], e[..., 0]], axis=-1)   # normals
    sep = np.zeros(a.shape[:-2], dtype=bool)
    for q in (a, b):
        ax = axes(q)                                        # [..., 4, 2]
        pa = np.einsum("...kd,...nd->...kn", ax, a)
        pb = np.einsum("...kd,...nd->...kn", ax, b)
        sep |= ((pa.min(-1) > pb.max(-1)) | (pb.min(-1) > pa.max(-1))).any(-1)
    return ~sep
