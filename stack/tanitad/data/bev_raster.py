"""P8 — ego-frame BEV occupancy rasteriser over `obstacle.offline` 3D agent cuboids.

WM_PHYSICS_PROOF.md P8: a frozen-latent decoder `z -> BEV occupancy raster` needs a GT
raster target. This module builds that raster from the corpus's own agent cuboids.

WHAT THE CORPUS ACTUALLY CARRIES (all MEASURED, none inherited)
---------------------------------------------------------------
* **Episodes do NOT carry agent tracks.** The episode contract is
  ``frames/actions/poses/episode_id/maneuvers`` only (``tanitad/data/_contract.py:8-12``,
  ``physicalai.build_episode`` -> ``ToyEpisode``, ``physicalai.py:742-746``; the v2 lazy
  provider mirrors the same surface, ``v2_dataset.LazyV2Episode.__slots__``).
  **The EPISODE BUILD** (``tanitad/data/physicalai.py``) reads **5** of the corpus's
  36 features and `obstacle.offline` is **not** among them; **PROGRAM-WIDE** the count
  is **6**, and the sixth IS `obstacle.offline` — read by the pod-side side-car join
  (``scripts/build_obstacle_join.py``) that hands this module its rows.

  ⚠️ NAME THE LAYER, NEVER WRITE A BARE COUNT. This sentence previously said "4",
  and that number has now gone stale FOUR times (2 -> 4 -> 5 -> 6) in the programme's
  prose. The root cause was never carelessness: the subject *"our ingest"* was
  undefined, and three legitimate read-sets exist —
  ``scripts/physicalai_r0.py`` (clip selection) **2**, the episode build **5**,
  program-wide **6**. All three, their exact feature names and the 36 denominator are
  now pinned to source in ``stack/tests/test_physicalai_feature_readset.py``; read
  that, not this paragraph, if the numbers matter. A count that lives only in prose —
  or, as here, only in a docstring nobody re-reads — rots silently.
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


# ===========================================================================
# CAMERA-FIELD GEOMETRY — the seam the v6 port needs (added 2026-08-16)
# ===========================================================================
# ⛔ WHY THIS EXISTS. The P8 grid above is a CARTESIAN ego-frame raster
# (60 m x +-16 m). The thing that has to fill it is a VISION latent, and a
# camera does not see a rectangle — it sees an azimuth WEDGE. Scoring a
# vision-only readout on cells no camera in the rig can observe measures the
# grid's corners, not the world model: it is the C9/C14 family (an instrument
# structurally unable to report the answer it is cited for), and it costs the
# probe exactly the near-field cells where distance-keeping lives.
#
# MEASURED 2026-08-16 (…/incoming/2026-08-16-p8-v6-port/raw/p8_v6_geometry.json,
# reproduced by code/p8_geometry_census.py) at the v6 field of 120 deg:
#   * 590 of 7 680 cells (7.682 %) of the P8 grid lie OUTSIDE the horizontal
#     field entirely — ALL of them at x < 9.09 m, where they are 51.2 % of that
#     near band;
#   * rows 18..119 (x >= 9.25 m) are fully inside; rows 0..17 are partly out.
#
# Everything here is pure numpy and depends on the frame only through its
# HALF-ANGLE in radians, so it stays importable without `tanitad.data.calib`
# (the `yaw_from_quaternion` precedent above). Callers pass
# ``CanonicalFrame.half_angle_x_rad()`` — the frame the ENCODER is actually fed,
# which for a centred sub-frame is the SUB-frame's angle, not the cache's.

#: Output parametrisations :func:`readout_column_index` knows how to invert.
#: Same two names as ``calib.PROJECTIONS`` — duplicated with this citation so
#: this module keeps its no-corpus, no-torch import surface.
PROJECTIONS_SUPPORTED = ("pinhole", "cylindrical")


def cell_centers_xy(grid: BEVGrid = GRID_DEFAULT
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast cell-center coordinates ``(X, Y)``, each ``[nx, ny]`` metres.

    Same centers :func:`rasterize` tests against (:func:`_cell_centers`), just
    meshed — so a mask built here and a raster built there index identically.
    """
    xc, yc = _cell_centers(grid)
    nx, ny = grid.shape
    return (np.broadcast_to(xc[:, None], (nx, ny)).copy(),
            np.broadcast_to(yc[None, :], (nx, ny)).copy())


def cell_azimuth_rad(grid: BEVGrid = GRID_DEFAULT) -> np.ndarray:
    """Ego-frame azimuth ``atan2(y, x)`` of every cell center, ``[nx, ny]``.

    Positive = LEFT (the ``+x`` forward / ``+y`` left convention of the whole
    module). The ego origin itself is never a cell center (row 0 is at
    ``x = cell_m/2``), so the ``atan2(0, 0)`` degeneracy cannot arise.
    """
    x, y = cell_centers_xy(grid)
    return np.arctan2(y, x)


def fov_mask(grid: BEVGrid = GRID_DEFAULT,
             half_angle_rad: float = math.radians(60.0)) -> np.ndarray:
    """Cells inside the camera's HORIZONTAL field, bool ``[nx, ny]``.

    ``True`` == the cell center's azimuth is within ``+-half_angle_rad`` of
    straight ahead, i.e. **the camera could in principle observe it**.

    ⚠️ THIS IS A NECESSARY, NOT A SUFFICIENT, VISIBILITY CONDITION. It uses the
    horizontal field ONLY. It does NOT model (a) the vertical field / hood
    occlusion, which removes more of the near band; (b) occlusion by other
    agents; (c) the rig-B rectifier mask. A calibrated visibility mask needs
    ``camera_intrinsics`` + ``sensor_extrinsics``, which the corpus DOES carry
    (``physicalai.py:153-154``) but which live pod-side. So a cell marked
    ``True`` here may still be unobservable; a cell marked ``False`` is
    definitely unobservable. Reporting the loose bound is the point — it is the
    half that can be established without the corpus, and it is stated as a
    bound rather than sold as visibility.
    """
    if not 0.0 < float(half_angle_rad) < math.pi:
        raise ValueError(f"half_angle_rad must be in (0, pi), got "
                         f"{half_angle_rad}")
    return np.abs(cell_azimuth_rad(grid)) <= float(half_angle_rad)


def fov_row_floor(grid: BEVGrid = GRID_DEFAULT,
                  half_angle_rad: float = math.radians(60.0),
                  cols: np.ndarray | None = None) -> int | None:
    """First row from which EVERY cell of ``cols`` is inside the field.

    ``None`` when no such row exists. ``cols`` defaults to the whole width, in
    which case this is :func:`fov_census`'s ``first_fully_visible_row``.

    ⭐ WHY A COLUMN SUBSET. A consumer that scans a NARROW band — LF0 walks a
    ±1.0/1.5/2.0 m ego corridor, not the grid — has a far smaller exposure than
    the grid-wide census, because a cell at lateral ``|y|`` only leaves a
    half-angle ``th`` field below ``x = |y| / tan(th)``. Quoting the grid-wide
    fraction against a corridor consumer OVERSTATES it by more than an order of
    magnitude (MEASURED 2026-08-16: 8.151 % grid-wide vs 0.833 % over LF0's
    ±1.5 m corridor at the same 117° frame, and 0.000 % once LF0's own
    ``--min-row 2`` is applied). This returns the floor so a caller can state
    the exposure of the set it actually reads.

    ⚠️ Same NECESSARY-not-sufficient bound as :func:`fov_mask`: horizontal
    field only.
    """
    m = fov_mask(grid, half_angle_rad)
    if cols is not None:
        m = m[:, np.asarray(cols, dtype=np.int64)]
    for i in range(m.shape[0]):
        if bool(m[i:].all()):
            return int(i)
    return None


def readout_column_index(grid: BEVGrid = GRID_DEFAULT,
                         half_angle_rad: float = math.radians(60.0),
                         n_cols: int = 4,
                         projection: str = "cylindrical",
                         token_w: int | None = None) -> np.ndarray:
    """Which READOUT COLUMN each BEV cell falls in. ``int [nx, ny]``, ``-1``
    outside the field.

    The v6 readout (``models/readout.py:SpatialGridReadout``) average-pools the
    encoder's token grid down to ``grid x grid_w`` cells, so a readout COLUMN is
    a contiguous block of image columns. Under ``cylindrical`` the image column
    is LINEAR in azimuth (``x = f * phi``, ``calib.py:90``), so an equal-pixel
    column block is an equal-AZIMUTH WEDGE; under ``pinhole`` it is linear in
    ``tan(phi)`` (``calib.py:88``) and the wedges are unequal. Both are inverted
    here, so the mapping is the frame's, never an assumption.

    ⛔ REFUSES when the pool does not tile. Pass ``token_w`` (the encoder's token
    columns) and this checks ``token_w % n_cols == 0``. When it does not tile,
    ``SpatialGridReadout`` falls back to ``AdaptiveAvgPool2d``, whose bins
    OVERLAP (``_adaptive_avg_matrix``: bin i spans
    ``[floor(i*n/N), ceil((i+1)*n/N))``) — a BEV cell then belongs to two
    readout columns and a single index is not a fact. Returning one anyway is
    exactly the silent-reshape this port exists to avoid, so this raises and the
    caller records the refusal.
    """
    if projection not in PROJECTIONS_SUPPORTED:
        raise ValueError(f"projection must be one of {PROJECTIONS_SUPPORTED}, "
                         f"got {projection!r}")
    if int(n_cols) < 1:
        raise ValueError(f"n_cols must be >= 1, got {n_cols}")
    if projection == "pinhole" and not 0.0 < float(half_angle_rad) < 0.5 * math.pi:
        raise ValueError(f"a pinhole frame cannot retain half-angle "
                         f"{half_angle_rad} rad (>= 90 deg); tan diverges")
    if token_w is not None and int(token_w) % int(n_cols):
        raise ValueError(
            f"readout column mapping is NOT EXACT for token_w={token_w} onto "
            f"{n_cols} readout columns ({token_w} % {n_cols} != 0). "
            f"SpatialGridReadout then uses AdaptiveAvgPool2d, whose bins "
            f"OVERLAP, so a BEV cell belongs to more than one readout column "
            f"and a single index would be fiction. Report the refusal instead "
            f"of an index — the FOV mask itself is unaffected and stays valid.")
    n_cols = int(n_cols)
    half = float(half_angle_rad)
    az = cell_azimuth_rad(grid)
    inside = np.abs(az) <= half
    # normalised image abscissa u in [0, 1] across the retained field
    with np.errstate(invalid="ignore"):
        if projection == "cylindrical":
            u = 0.5 * (az / half + 1.0)
        else:
            u = 0.5 * (np.tan(np.clip(az, -half, half)) / math.tan(half) + 1.0)
    col = np.floor(np.clip(u, 0.0, 1.0 - 1e-12) * n_cols).astype(np.int64)
    col[~inside] = -1
    return col


def readout_row_ranges_m(grid_h: int, near_m: float = 3.0,
                         far_m: float = 80.0) -> np.ndarray:
    """Numpy twin of ``tanitad.models.v6.readout_grid_ranges``' row vector.

    ⚠️ ROW 0 IS THE **FAR** ROW. v6 lays the readout out in IMAGE order — row 0
    is the TOP of the image, hence the farthest ground — and spaces the rows
    GEOMETRICALLY because image row maps roughly to inverse depth
    (``v6.py:247-273``). The P8 BEV grid is the OPPOSITE: its row 0 is at the
    EGO ORIGIN and row ``nx-1`` is the farthest. Anything that aligns the two
    grids row-for-row without flipping maps far onto near.

    ⚠️ EVIDENCE CLASS: **ESTIMATED**, inherited verbatim from v6 — a declared
    monotone image-row prior, NOT calibrated depth. Pinned equal to the torch
    original in ``tests/test_p8_v6.py``.
    """
    if int(grid_h) < 1:
        raise ValueError(f"grid_h must be >= 1, got {grid_h}")
    if not 0.0 < float(near_m) < float(far_m):
        raise ValueError(f"need 0 < near_m < far_m, got {near_m}, {far_m}")
    if int(grid_h) == 1:
        return np.asarray([math.sqrt(near_m * far_m)], dtype=np.float64)
    frac = np.linspace(1.0, 0.0, int(grid_h), dtype=np.float64)
    return float(near_m) * (float(far_m) / float(near_m)) ** frac


def fov_census(grid: BEVGrid = GRID_DEFAULT,
               half_angle_rad: float = math.radians(60.0),
               n_cols: int = 4, projection: str = "cylindrical",
               token_w: int | None = None,
               readout_rows: int | None = None,
               near_m: float = 3.0, far_m: float = 80.0) -> dict:
    """The whole geometry statement as one JSON-able dict (MEASURED, pure).

    Everything a P8-on-v6 run must publish so a reader can tell what fraction of
    the pre-registered target the model could even see, and how the target grid
    lines up against the readout's own cells. Never decides anything — the
    caller reports it and, separately, chooses whether to gate on the masked
    metric.
    """
    m = fov_mask(grid, half_angle_rad)
    x, _y = cell_centers_xy(grid)
    nx, ny = grid.shape
    total = int(m.size)
    n_in = int(m.sum())
    full_rows = [int(i) for i in range(nx) if bool(m[i].all())]
    out = {
        "bev_grid": {"x_fwd_m": grid.x_fwd_m, "y_half_m": grid.y_half_m,
                     "cell_m": grid.cell_m, "shape": [nx, ny],
                     "row0_is": "ego origin (NEAREST)"},
        "camera_field": {"half_angle_deg": round(math.degrees(half_angle_rad), 6),
                         "hfov_deg": round(math.degrees(2.0 * half_angle_rad), 6),
                         "projection": projection},
        "in_fov_cells": n_in, "out_of_fov_cells": total - n_in,
        "total_cells": total,
        "in_fov_frac": round(n_in / total, 6),
        "out_of_fov_frac": round((total - n_in) / total, 6),
        "first_fully_visible_row": (full_rows[0] if full_rows else None),
        "first_fully_visible_row_x_m": (float(x[full_rows[0], 0])
                                        if full_rows else None),
        "out_of_fov_max_x_m": (float(x[~m].max()) if n_in < total else None),
        "_bound": "HORIZONTAL field only — a necessary, not sufficient, "
                  "visibility condition (see fov_mask)",
        "_evidence_class": "MEASURED (ours; pure geometry, no corpus needed)",
    }
    try:
        col = readout_column_index(grid, half_angle_rad, n_cols, projection,
                                   token_w=token_w)
    except ValueError as ex:
        out["readout_columns"] = {"exact": False, "reason": str(ex)}
    else:
        cols = {}
        for j in range(int(n_cols)):
            sel = col == j
            cols[str(j)] = {
                "cells": int(sel.sum()),
                "frac_of_grid": round(float(sel.sum()) / total, 6),
                "x_min_m": float(x[sel].min()) if bool(sel.any()) else None,
                "x_max_m": float(x[sel].max()) if bool(sel.any()) else None}
        far = x >= 0.5 * grid.x_fwd_m
        out["readout_columns"] = {
            "exact": True, "n_cols": int(n_cols), "token_w": token_w,
            "per_column": cols,
            "far_half_columns": sorted({int(c) for c in col[far].ravel()}),
            "_read": "a readout COLUMN is an image-column block; under "
                     "cylindrical that is an equal-azimuth wedge"}
    if readout_rows:
        rows = readout_row_ranges_m(int(readout_rows), near_m, far_m)
        out["readout_rows"] = {
            "n_rows": int(readout_rows),
            "nominal_range_m": [round(float(v), 4) for v in rows],
            "row0_is": "image TOP == FARTHEST (opposite of the BEV row order)",
            "cells_nearer_than_near_m": int((x < float(near_m)).sum()),
            "frac_nearer_than_near_m": round(float((x < float(near_m)).sum())
                                             / total, 6),
            "prior_far_m_beyond_grid": bool(float(far_m) > grid.x_fwd_m),
            "_evidence_class": "ESTIMATED (v6's declared image-row prior — "
                               "NOT calibrated depth; v6.py:247-273)"}
    return out
