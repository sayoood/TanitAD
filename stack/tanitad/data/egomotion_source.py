"""Authoritative ego-pose source for the label pipeline, keyed by CLIP UUID.

⛔⛔ WHY THIS EXISTS — A 20.5 % JOIN ERROR THAT CORRUPTED AN ENTIRE VALIDATION.
MEASURED 2026-08-23. The episode cache is keyed by `episode_id`, the 16-bit
`episode_id_legacy` value, and the label↔frame join ran through it. The clip
index itself warns that this id COLLIDES. I refused ids claimed by more than one
LABELLED clip, and ids claimed by more than one CACHE episode — and that is
still not enough:

    A cache episode whose TRUE clip is NOT in the labelled set can collide with
    a labelled clip's legacy id. Neither ambiguity check can see it, because on
    each side the id looks unique.

**8 of 39 joined clips (20.5 %) were the WRONG EPISODE.** Verified by correlating
each cache episode's speed against the egomotion of the clip it claimed to be:
31/39 gave r > 0.999 (rmse < 0.15 m/s); the other 8 gave r = −0.96 … +0.87 with
rmse 2.6–18.4 m/s. Those are different vehicles at different moments.

The damage was not academic: the mismatched set included `5b4eef8f`, `5aef0388`,
`4d389996`, `00d05901` — the very clips a frame-by-frame validation had just
"confirmed". The frames belonged to other clips. See RETRACTION_LOG C140.

⇒ **THIS MODULE READS EGOMOTION BY TRUE CLIP UUID AND NOTHING ELSE.** There is
no legacy id, no ordering assumption, and no lookup that can silently return a
neighbour. Coverage is 801/801 labelled clips (18,987 clips available locally).

## What the source gives that the cache did not

* **100 Hz** (cache: 10 Hz) with `curvature` supplied directly;
* full quaternion attitude, velocity and acceleration vectors;
* ⭐ **the recording continues well past the 20 s clip** — spans of 20–140 s —
  so the strategic horizon (key + 8…30 s) is often OBSERVABLE, where the 20 s
  clip truncated 81 % of it. `horizon_available_s` reports what each clip has.

⚠️ The clip window sits at **offset 0** of the recording; verified on the 31
correctly-joined clips (best cross-correlation offset 0.0–0.1 s, r > 0.99).
"""
from __future__ import annotations

import glob
import io
import zipfile
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

EGO_GLOB = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion/*.zip"
HZ = 10.0                 # the label timeline
RAW_T0_S = 8.0            # s2 anchor on the raw clip timeline
CLIP_LEN_S = 20.0


@dataclass(frozen=True)
class EgoTrack:
    clip_id: str
    t: np.ndarray          # seconds from the recording start
    poses: np.ndarray      # [T, 4] = x, y, yaw, speed  at HZ
    curvature: np.ndarray  # [T] as supplied by the provider
    accel: np.ndarray      # [T] along-track acceleration
    span_s: float          # full recording span
    key_index: int         # index of the s2 anchor in `poses`

    @property
    def horizon_available_s(self) -> float:
        return (len(self.poses) - 1 - self.key_index) / HZ


@lru_cache(maxsize=1)
def _index() -> dict[str, tuple[str, str]]:
    """clip_uuid -> (zip path, member name). Built once, from filenames only."""
    out: dict[str, tuple[str, str]] = {}
    for p in sorted(glob.glob(EGO_GLOB)):
        with zipfile.ZipFile(p) as z:
            for e in z.namelist():
                out[e.split(".")[0]] = (p, e)
    return out


def available() -> set[str]:
    return set(_index())


def _yaw_from_quaternion(qx, qy, qz, qw):
    """Yaw about the z axis. Standard ZYX convention."""
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def load(clip_id: str, *, hz: float = HZ, max_s: float | None = None) -> EgoTrack:
    """Load one clip's ego track, resampled to ``hz``.

    ⚠️ Raises on an unknown clip rather than returning an approximation. A
    lookup that quietly substitutes a neighbouring clip is precisely the defect
    this module replaces.
    """
    import pandas as pd
    idx = _index()
    if clip_id not in idx:
        raise KeyError(f"no egomotion for clip {clip_id!r}")
    path, member = idx[clip_id]
    with zipfile.ZipFile(path) as z:
        df = pd.read_parquet(io.BytesIO(z.read(member)))

    ts = df["timestamp"].to_numpy(dtype=np.float64) / 1e6      # us -> s
    ts = ts - ts[0]
    v = np.linalg.norm(df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64), axis=1)
    yaw = _yaw_from_quaternion(*(df[c].to_numpy(dtype=np.float64)
                                 for c in ("qx", "qy", "qz", "qw")))
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    kappa = df["curvature"].to_numpy(dtype=np.float64)
    a_lon = np.gradient(v, ts, edge_order=1)

    end = ts[-1] if max_s is None else min(ts[-1], max_s)
    grid = np.arange(0.0, end + 1e-9, 1.0 / hz)
    # unwrap BEFORE interpolating: interpolating across a +-pi wrap invents a
    # full rotation, which would fabricate a turn out of a straight road.
    yaw_u = np.unwrap(yaw)
    poses = np.stack([np.interp(grid, ts, x), np.interp(grid, ts, y),
                      np.interp(grid, ts, yaw_u), np.interp(grid, ts, v)],
                     axis=1)
    return EgoTrack(clip_id=clip_id, t=grid, poses=poses,
                    curvature=np.interp(grid, ts, kappa),
                    accel=np.interp(grid, ts, a_lon),
                    span_s=float(ts[-1]),
                    key_index=int(round(RAW_T0_S * hz)))


def verify_against(clip_id: str, cache_poses, *, hz: float = HZ) -> dict:
    """Speed-correlate a cache episode against this clip's true egomotion.

    This is the check that exposed the 20.5 % join error, kept as a callable so
    any future consumer of the episode cache can run it. Returns the statistics
    and a boolean ``same_clip``; a caller that ignores it is choosing to trust a
    colliding 16-bit id.
    """
    tr = load(clip_id, hz=hz)
    cp = np.asarray(cache_poses, dtype=np.float64)
    n = min(len(cp), len(tr.poses))
    if n < 20:
        return {"same_clip": False, "reason": "too few samples", "n": n}
    a, b = tr.poses[:n, 3], cp[:n, 3]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return {"same_clip": False, "reason": "degenerate (constant speed)", "n": n}
    corr = float(np.corrcoef(a, b)[0, 1])
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return {"same_clip": bool(corr > 0.99 and rmse < 0.5),
            "corr": round(corr, 4), "rmse_ms": round(rmse, 3), "n": n}
