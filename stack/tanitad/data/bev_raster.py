"""P8 — ego-frame BEV occupancy rasteriser over `obstacle.offline` 3D agent cuboids.

WM_PHYSICS_PROOF.md P8: a frozen-latent decoder `z -> BEV occupancy raster` needs a GT
raster target. This module builds that raster from the corpus's own agent cuboids.

WHAT THE CORPUS ACTUALLY CARRIES (all MEASURED, none inherited)
---------------------------------------------------------------
* **Episodes do NOT carry agent tracks.** The episode contract is
  ``frames/actions/poses/episode_id/maneuvers`` only (``tanitad/data/_contract.py:8-12``,
  ``physicalai.build_episode`` -> ``ToyEpisode``, ``physicalai.py:742-746``; the v2 lazy
  provider mirrors the same surface, ``v2_dataset.LazyV2Episode.__slots__``). The corpus
  build reads 4 of `obstacle.offline`'s 36 sibling features and `obstacle.offline` is not
  among them.
* Therefore **the raster is built from the RAW `obstacle.offline` records, and the JOIN
  to episodes is a POD-SIDE step keyed by (clip_id, episode frame index)** — the dataset
  zips live only on hosts holding the gated corpus, and mapping an episode frame index to
  clip time needs the episode's own poses
  (``taniteval/taniteval/lead_source.register_poses_to_time``; the realised episode grid
  is ~0.1007 s, NOT 0.1 s — 2026-08-03-obstacle-offline-join §5). This module does not
  fake that join; it rasterises what the join hands it.
* Raw record layout, as ``scripts/lead_state_gate.py`` reads it and as MEASURED from
  bytes (…/incoming/2026-08-03-obstacle-offline-join/raw/obstacle_schema_probe.json):
  zips ``<root>/labels/obstacle.offline/obstacle.offline.chunk_{c:04d}.zip`` holding one
  parquet per clip with columns ``timestamp_us · source · track_id · center_{x,y,z} ·
  size_{x,y,z} · orientation_{x,y,z,w} · label_class · reference_frame ·
  reference_frame_timestamp_us``.
* **Frame:** ``reference_frame == "rig"`` on every row and
  ``reference_frame_timestamp_us == timestamp_us`` on every row — each cuboid is
  expressed in the EGO/RIG frame at ITS OWN timestamp. Axis convention **x forward,
  y left, z up**, MEASURED by the parked-car experiment (world-static under xf_yl for
  1,756/2,778 tracks, 7.4x over the nearest alternative — join doc §2). That is the SAME
  convention as ``refb_labels.ego_frame`` (see below), so a same-timestamp cuboid needs
  NO rotation before rasterising.
* **Sampling:** ~10 Hz per track, tracks staggered (1.000-1.005 rows per unique
  timestamp) — "agents at time t" is a per-track temporal lookup, never a frame index.
  Labels span ~20 s; egomotion runs 48-140 s, so frames past ~20 s are NO_LABEL (a
  state of their own, never "empty road" — join doc §4).

EGO-FRAME ROTATION CONVENTION (binding for any world->ego transform here)
-------------------------------------------------------------------------
Exactly ``scripts/refb_labels.py:86-90`` (``ego_frame``)::

    c, s = cos(-yaw), sin(-yaw)
    x_ego = dx * c - dy * s
    y_ego = dx * s + dy * c          # +x = forward, +y = LEFT

:func:`ego_frame_agents` implements those lines verbatim in numpy (pinned against the
torch original in ``tests/test_p8.py``).

GRID (P8 spec: 60 m forward, +-16 m lateral, 0.5 m cell -> [120, 64])
---------------------------------------------------------------------
``raster[i, j]``: row i covers forward x in [0.5*i, 0.5*(i+1)) — row 0 at the ego
origin, row 119 ending at 60 m; col j covers lateral y in [-16 + 0.5*j, -16 + 0.5*(j+1))
— col 0 is 16 m to the RIGHT (y = -16, +y is LEFT), col 63 ends at +16 m (left). A cell
is occupied (1.0) iff its CENTER lies inside (closed inequality) any agent's oriented
footprint rectangle; everything else 0.0.

Pure numpy — unit-testable with no corpus, no torch, no pandas. Duck-typed column
access (``obs["center_x"]``) accepts a pandas DataFrame or a plain dict of arrays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# Nearest-sample tolerance for `agents_at_time`. Track cadence is ~0.1 s (MEASURED
# median 0.0999-0.1001 s), so +-0.06 s admits at most one sample per track. NOTE the
# residual: a cuboid is expressed in the rig frame at ITS OWN timestamp, so a sample
# dt away from the query time is mis-registered by ~|v_ego - v_agent_along| * dt
# (<= ~1.6 m at 26 m/s closing for dt=0.06 s; typically ~1 cell). A pod-side join
# builder that wants exactness should ego-compensate via egomotion instead of widening
# this tolerance — a 0.5 s stale window (lead_state_gate's MAX_STALE_S, fine for gap
# FEATURES) would smear a 13 m/s ego by ~13 cells and is NOT admissible for a raster.
DEFAULT_TOL_S = 0.06

# The full measured `label_class` enum (schema probe class_counts_probed) — 10 classes,
# all dynamic agents; None below means "rasterise everything".
ALL_CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer",
               "person", "rider", "stroller", "animal", "protruding_object")


@dataclass(frozen=True)
class BEVGrid:
    """Ego-frame occupancy grid spec. Defaults = the P8 spec (60 m fwd, +-16 m, 0.5 m)."""

    x_fwd_m: float = 60.0
    y_half_m: float = 16.0
    cell_m: float = 0.5

    @property
    def shape(self) -> tuple[int, int]:
        return (int(round(self.x_fwd_m / self.cell_m)),
                int(round(2.0 * self.y_half_m / self.cell_m)))


GRID_DEFAULT = BEVGrid()                      # (120, 64)


@lru_cache(maxsize=8)
def _cell_centers(grid: BEVGrid) -> tuple[np.ndarray, np.ndarray]:
    """(xc [nx], yc [ny]) cell-center coordinates for ``grid`` (cached per spec)."""
    nx, ny = grid.shape
    xc = (np.arange(nx, dtype=np.float64) + 0.5) * grid.cell_m
    yc = -grid.y_half_m + (np.arange(ny, dtype=np.float64) + 0.5) * grid.cell_m
    return xc, yc


def wrap_to_pi(a: np.ndarray | float) -> np.ndarray | float:
    """Numpy twin of ``refb_labels.wrap_to_pi`` (same (-pi, pi] wrap)."""
    return a - (2.0 * math.pi) * np.floor((np.asarray(a) + math.pi)
                                          / (2.0 * math.pi))


def yaw_from_quaternion(qx, qy, qz, qw) -> np.ndarray:
    """Yaw about +z from a (qx, qy, qz, qw) quaternion.

    Same convention/formula as ``tanitad.data.physicalai.quaternion_yaw``
    (physicalai.py:583-593), duplicated here — with this citation, per the
    ``lead_state_gate.py:116-118`` precedent — so this module stays importable
    without the physicalai module's pandas/calib import surface."""
    qx, qy, qz, qw = (np.asarray(q, dtype=np.float64) for q in (qx, qy, qz, qw))
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def agents_to_array(agents) -> np.ndarray:
    """Normalise an agent set to float64 ``[A, 6] = (cx, cy, yaw, l, w, occ)``.

    Accepts a list of dicts with keys ``cx, cy, yaw, l, w`` (the join-file record
    schema; optional ``occ``/``occluded`` -> the occ column, else -1.0 = "no
    occlusion information"), or an ndarray ``[A, 5]``/``[A, 6]`` in that column
    order. ``[0, 6]`` for an empty set — "labelled and clear", which is NOT the
    same state as "no label" (that is a ``None`` from the join, join doc §4).
    """
    if isinstance(agents, np.ndarray):
        a = np.asarray(agents, dtype=np.float64)
        if a.size == 0:
            return np.zeros((0, 6), dtype=np.float64)
        if a.ndim != 2 or a.shape[1] not in (5, 6):
            raise ValueError(f"agents array must be [A, 5|6], got {a.shape}")
        if a.shape[1] == 5:
            a = np.concatenate([a, np.full((a.shape[0], 1), -1.0)], axis=1)
        return a
    rows = []
    for d in agents:
        occ = d.get("occ", d.get("occluded", -1.0))
        occ = -1.0 if occ is None else float(occ)
        rows.append((float(d["cx"]), float(d["cy"]), float(d["yaw"]),
                     float(d["l"]), float(d["w"]), occ))
    if not rows:
        return np.zeros((0, 6), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def rasterize(agents_frame, grid: BEVGrid = GRID_DEFAULT) -> np.ndarray:
    """Ego-frame agent footprints -> occupancy raster ``[nx, ny] float32 {0, 1}``.

    ``agents_frame``: agents ALREADY in the ego frame (+x fwd, +y left — the
    refb_labels.ego_frame convention, and natively what `obstacle.offline` rig-frame
    rows are at their own timestamp), any form :func:`agents_to_array` accepts.
    A cell is 1.0 iff its center lies inside (closed, <= half-extent) any agent's
    oriented ``l x w`` rectangle around ``(cx, cy)`` at heading ``yaw``.
    """
    ag = agents_to_array(agents_frame)
    nx, ny = grid.shape
    out = np.zeros((nx, ny), dtype=np.float32)
    if ag.shape[0] == 0:
        return out
    xc, yc = _cell_centers(grid)
    px = xc[:, None]                                     # [nx, 1]
    py = yc[None, :]                                     # [1, ny]
    for cx, cy, yaw, length, width, _occ in ag:
        # cheap reject: agent circumscribed circle entirely off-grid
        r = 0.5 * math.hypot(length, width)
        if (cx + r < 0.0 or cx - r > grid.x_fwd_m
                or cy + r < -grid.y_half_m or cy - r > grid.y_half_m):
            continue
        dx = px - cx
        dy = py - cy
        c, s = math.cos(-yaw), math.sin(-yaw)            # rotate world->agent frame:
        lon = dx * c - dy * s                            # the refb_labels.ego_frame
        lat = dx * s + dy * c                            # rotation (:88-90) about yaw
        out[(np.abs(lon) <= 0.5 * length) & (np.abs(lat) <= 0.5 * width)] = 1.0
    return out


def ego_frame_agents(agents_world, ego_pose) -> np.ndarray:
    """World-frame agents -> ego frame of ``ego_pose = (x, y, yaw[, v])``.

    The rotation is EXACTLY ``refb_labels.ego_frame`` (scripts/refb_labels.py:86-90):
    ``c, s = cos(-yaw), sin(-yaw); x' = dx*c - dy*s; y' = dx*s + dy*c`` — +x forward,
    +y LEFT (pinned against the torch original in tests/test_p8.py). Agent headings
    become ``wrap_to_pi(yaw_agent - yaw_ego)``; sizes and the occ column pass through.

    NOTE: `obstacle.offline` rows are ALREADY rig-frame at their own timestamp
    (join doc §1) and must NOT be passed through this — this exists for synthetic
    tests and for any future world-frame track source (e.g. AlpaSim actors).
    """
    ag = agents_to_array(agents_world).copy()
    ex, ey, eyaw = (float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2]))
    if ag.shape[0] == 0:
        return ag
    dx = ag[:, 0] - ex
    dy = ag[:, 1] - ey
    c, s = math.cos(-eyaw), math.sin(-eyaw)
    ag[:, 0] = dx * c - dy * s                           # refb_labels.py:89
    ag[:, 1] = dx * s + dy * c                           # refb_labels.py:90
    ag[:, 2] = wrap_to_pi(ag[:, 2] - eyaw)
    return ag


def agents_at_time(obs, t_s: float, tol_s: float = DEFAULT_TOL_S,
                   classes: tuple[str, ...] | None = None) -> np.ndarray:
    """Raw `obstacle.offline` rows of ONE clip -> rig-frame agents ``[A, 6]`` at ``t_s``.

    Per track (tracks are STAGGERED — a per-track temporal lookup is mandatory,
    join doc §1): take the single sample nearest to ``t_s`` if within ``tol_s``
    (see :data:`DEFAULT_TOL_S` for the mis-registration budget this bounds), else
    the track contributes nothing at ``t_s``. Emitted agents are in the rig frame
    at (approximately) ``t_s`` with ``l = size_x``, ``w = size_y`` and yaw from the
    orientation quaternion; ``occ = -1`` always — `obstacle.offline` carries NO
    occlusion flag, so visible/occluded splits must come from the pod-side P4 join.

    ``obs``: duck-typed columns (pandas DataFrame or dict of arrays) with
    ``timestamp_us, track_id, center_x, center_y, size_x, size_y,
    orientation_{x,y,z,w}`` and optionally ``label_class`` (required iff
    ``classes`` filters).
    """
    ts = np.asarray(obs["timestamp_us"], dtype=np.float64) / 1e6
    if ts.size == 0:
        return np.zeros((0, 6), dtype=np.float64)
    tid = np.asarray(obs["track_id"]).astype(str)
    cx = np.asarray(obs["center_x"], dtype=np.float64)
    cy = np.asarray(obs["center_y"], dtype=np.float64)
    sx = np.asarray(obs["size_x"], dtype=np.float64)
    sy = np.asarray(obs["size_y"], dtype=np.float64)
    yaw = yaw_from_quaternion(obs["orientation_x"], obs["orientation_y"],
                              obs["orientation_z"], obs["orientation_w"])
    cls = (np.asarray(obs["label_class"]).astype(str)
           if classes is not None else None)
    rows = []
    for track in np.unique(tid):
        m = tid == track
        if cls is not None and cls[m][0] not in classes:
            continue
        dt = np.abs(ts[m] - float(t_s))
        j = int(np.argmin(dt))
        if dt[j] > tol_s:
            continue
        rows.append((cx[m][j], cy[m][j], yaw[m][j], sx[m][j], sy[m][j], -1.0))
    if not rows:
        return np.zeros((0, 6), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def raster_at_time(obs, t_s: float, grid: BEVGrid = GRID_DEFAULT,
                   tol_s: float = DEFAULT_TOL_S,
                   classes: tuple[str, ...] | None = None) -> np.ndarray:
    """Convenience: :func:`agents_at_time` -> :func:`rasterize` (rig frame is
    already the ego frame at ~``t_s``, so no rotation happens in between)."""
    return rasterize(agents_at_time(obs, t_s, tol_s=tol_s, classes=classes),
                     grid=grid)
