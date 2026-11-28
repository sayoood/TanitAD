#!/usr/bin/env python3
"""LiDAR -> metric BEV ground truth, in the RIG frame, at camera timestamps.

LINEAGE (2026-09-13): copied byte-for-byte from
`2026-09-11-lidar-bev-gt/code/lidar_bev.py` (md5 204fe929f4af4b5f3d462cd893e776ef) with
EXACTLY ONE semantic change, in `deskew_rigid`, plus a schema bump to
`tanitad.lidar_bev_gt/2` in `artifact_meta` so an artifact built with the corrected
deskew can never be mistaken for one built with the old sign.

⛔⛔ THE 09-11 DESKEW ROTATION SIGN WAS INVERTED -- MEASURED, NOT ARGUED.
`p0_deskew_sign_check.py`: two consecutive spins deskewed to the same instant must place
static structure in the same place. Median xy nearest-neighbour distance, spin k -> k+1:
  * run 1 (`raw/p0_deskew_sign_check.json`), 12 high-|yaw| instants, 1 clip, right turns:
    rotation OFF 0.1080 m · 09-11 sign 0.1545 m · FLIPPED sign 0.0135 m; flipped wins 12/12
  * run 2 (`raw/p0_deskew_sign_check_replicate.json`), 12 instants, 2 more clips, 5 LEFT
    turns: OFF 0.0321 · 09-11 0.0499 · FLIPPED 0.0146; flipped wins 12/12
  * control, |yaw| <= 0.02 rad/s, n = 48 + 54: all three variants equal to ~1 mm
    (0.0126/0.0136/0.0122 and 0.0133/0.0146/0.0127), as they must be.
The yaw-rate estimate itself was cross-checked independently: its sign agrees with
`v * curvature` on 100 % of |r| > 0.05 samples and with the trajectory-heading rate on
82-100 %, over 4 clips. ⇒ the defect is the ROTATION in `deskew_rigid`, not the yaw rate.
Family: a sign nobody asserts (`R-2026-09-08-wpa-mirror`).

⛔ BINDING SCOPE (PI, 2026-08-03): labels may use ego / agents / maps / future poses /
LiDAR; **INFERENCE IS VISION-ONLY**. Everything in this module is a TRAIN-TIME TARGET
builder. Nothing here may be imported by an inference path, and the artifacts it writes
are labels, not inputs.

EVERY GEOMETRIC FACT BELOW IS MEASURED, AND EACH ONE IS NAMED WHERE IT WAS MEASURED
===================================================================================
* **The points live in the LIDAR SENSOR frame, not the rig frame.** The Draco POSITION
  attribute's near-range ground peak is ``z = -1.925 m``; ``sensor_extrinsics``
  puts ``lidar_top_360fov`` at ``z = +1.8897 m`` in the rig frame with a near-identity
  quaternion. ``-1.925 + 1.890 = -0.035 m`` -- i.e. after the extrinsic the road lands
  on rig ``z = 0`` to **3.5 cm**, independently reproducing ``D-V5A-GROUND1``
  (rig z = 0 IS the road plane, 0.05-0.13 m). ⭐ That agreement between two
  independently derived facts is the cross-check; without it the frame would be a guess.
* **Rig frame convention:** ``+x forward, +y LEFT, +z UP``, origin rear axle at the road
  plane -- identical to ``tanitad.data.bev_raster`` and ``tanitad.data.rig_projection``,
  so this target and the ``obstacle.offline`` raster are in the SAME space with no
  transform between them.
* **The Draco blob carries four attributes** (MEASURED on a real spin):
  ``0`` POSITION float32 [N,3]; ``1`` uint8 intensity 0-255; ``2`` uint32 **per-point
  timestamp in µs on the clip clock**; ``3`` uint32 [N,2] = (laser row 0-127, azimuth
  column 1-3599) -- matching ``lidar_intrinsics.offline``'s published
  ``row-offset-spinning`` 128 x 3600 model, which is an INDEPENDENT confirmation.
* **Clock:** ``spin_start_timestamp`` / ``spin_end_timestamp`` are µs on the SAME clock
  as ``<clip>.camera_front_wide_120fov.timestamps.parquet``. MEASURED on a real clip:
  LiDAR 199 spins, dt median 99,994 µs (**10.0006 Hz**), span 0.0995-19.899 s; camera
  605 frames at 30 Hz spanning -0.0142-20.119 s. ⇒ the LiDAR interval is CONTAINED in
  the camera interval and alignment is a nearest-timestamp lookup, never an assumption.

⛔ **A DECODE THAT RAISES INTO A PRE-ALLOCATED ARRAY LEAVES A FULL-SIZE FILE OF ZEROS.**
Every builder path here asserts on CONTENT (non-zero point counts, occupancy mean,
per-channel min/max) and refuses to write a frame whose cloud decoded to nothing.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

import numpy as np

# ---------------------------------------------------------------------------
# MEASURED constants. Each carries the artifact that measured it.
# ---------------------------------------------------------------------------
#: Road plane in the rig frame. `D-V5A-GROUND1` (SUPPORTED, MEASURED 2026-09-05) over
#: 87,481 `obstacle.offline` cuboids: ground-standing classes' bottom faces read
#: -0.05..-0.13 m, `protruding_object` (the control that must NOT touch the road) reads
#: +1.676 m. Reproduced here from LiDAR returns at -0.035 m.
RIG_GROUND_Z_M = 0.0

#: Draco attribute ids, MEASURED from a real spin (see module docstring).
DRACO_ATTR_INTENSITY = 1
DRACO_ATTR_POINT_TS_US = 2
DRACO_ATTR_ROW_COL = 3


@dataclass(frozen=True)
class CartesianBEVSpec:
    """The P8 grid, byte-for-byte the extent `tanitad.data.bev_raster` rasterises.

    ⭐ Same extent and same cell size ON PURPOSE: the LiDAR occupancy target and the
    `obstacle.offline` agent raster then share one address space, so "do the boxes land
    on the returns" is an elementwise question, not a resampling exercise.
    """

    x_max_m: float = 60.0          # forward extent, x in [0, x_max_m)
    y_half_m: float = 16.0         # lateral half-extent, y in [-y_half, +y_half)
    cell_m: float = 0.5
    frame: str = "rig"             # +x forward, +y LEFT, +z UP, origin rear axle @ road
    units: str = "m"

    @property
    def n_x(self) -> int:
        return int(round(self.x_max_m / self.cell_m))

    @property
    def n_y(self) -> int:
        return int(round(2.0 * self.y_half_m / self.cell_m))


@dataclass(frozen=True)
class PolarBEVSpec:
    """Range x azimuth, registered to the encoder's own columns.

    Defaults mirror `tanitad.data.bev_aux.PolarBEVSpec` (n_az=20, n_rng=24, r_max 60 m,
    hfov 120°) so a LiDAR target is a DROP-IN for the `obstacle.offline` one.

    ⛔ The corpus is 256x640 **CYLINDRICAL**, f_ref 305.5775, HFOV **120.000°** -- the
    image column is LINEAR IN AZIMUTH. The pinhole formula `2*atan((W/2)/f)` gives
    92.641° on the same data and looks entirely plausible (MEASURED; WP-A §2).
    """

    n_az: int = 20
    n_rng: int = 24
    r_max_m: float = 60.0
    r_min_m: float = 0.0
    hfov_deg: float = 120.0
    frame: str = "rig"
    units: str = "m"

    @property
    def cell_deg(self) -> float:
        return self.hfov_deg / self.n_az

    @property
    def cell_rng_m(self) -> float:
        return (self.r_max_m - self.r_min_m) / self.n_rng


@dataclass(frozen=True)
class ZBand:
    """What counts as an obstacle, in metres above the rig road plane.

    ⚠️ These are a DECLARED CHOICE, not a measurement, and they are written into the
    artifact so a consumer never has to guess. `ground_max_m` sits above the measured
    3.5 cm road residual with room for road camber and suspension; `obstacle_max_m`
    excludes bridges, gantries and foliage the ego drives under.

    ⛔ **A FLAT z THRESHOLD IS WRONG AT RANGE, AND IT FAILS BY OVER-REPORTING** -- which
    is the dangerous direction, because an over-occupied target manufactures a high base
    rate and every arm then looks registered. MEASURED on this rig: a pure
    `z in [0.30, 3.00]` rule marks **24.9-35.0 %** of the 60 x 32 m grid occupied, and
    the marginal occupancy of an OBSERVED cell reaches **0.544** -- a road with 0.5 %
    camber rises 0.30 m over 60 m, so the road itself crosses the threshold.
    ⇒ the default rule adds a PER-CELL VERTICAL SPREAD test (`spread_min_m`): a ground
    cell's returns are coplanar, a wall's or a vehicle's are not. Both rules are
    computed and BOTH are written into the artifact, so the choice is auditable rather
    than baked in.
    """

    ground_max_m: float = 0.30
    obstacle_min_m: float = 0.30
    obstacle_max_m: float = 3.00
    #: minimum (z_max - z_min) within a cell for it to count as a vertical structure
    spread_min_m: float = 0.30
    #: minimum returns in a cell before the spread test is meaningful
    spread_min_pts: int = 2


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Rotation matrix from a (x, y, z, w) quaternion. Sensor -> rig."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        raise ValueError("zero quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def lidar_to_rig(pts: np.ndarray, ext: dict) -> np.ndarray:
    """Apply `sensor_extrinsics` (sensor -> rig). `pts` is [N,3] float."""
    R = quat_to_R(ext["qx"], ext["qy"], ext["qz"], ext["qw"])
    t = np.array([ext["x"], ext["y"], ext["z"]], dtype=np.float64)
    return (pts.astype(np.float64) @ R.T) + t


def deskew_rigid(pts_rig: np.ndarray, pt_ts_us: np.ndarray, t_ref_us: int,
                 v_ms: float, yaw_rate_rps: float) -> np.ndarray:
    """Ego-motion compensation of a rolling 100 ms spin to a single reference instant.

    A spinning LiDAR's points are captured across the whole revolution, so at 13.6 m/s
    the first and last point of one spin are displaced by ~1.36 m -- ~2.7 cells of a
    0.5 m grid. The per-point timestamp (Draco attribute 2) makes the correction exact
    rather than approximate.

    Constant-velocity, constant-yaw-rate model over <= 100 ms. Returns points as they
    would have been seen from the ego pose at `t_ref_us`.

    ⛔ SIGN (corrected 2026-09-13). The ego frame at t_pt is rotated by +dth relative to
    the frame at t_ref, so p_ref = R(+dth) p_pt + (v dt, 0). A left-turning ego (+yaw)
    sees a static point dead ahead drift to the RIGHT in later points; R(+dth) moves it
    back. The 09-11 code used R(-dth), which DOUBLED the yaw smear (see module docstring).
    """
    dt = (pt_ts_us.astype(np.float64) - float(t_ref_us)) * 1e-6   # s, signed
    dth = yaw_rate_rps * dt
    dx = v_ms * dt
    c, s = np.cos(dth), np.sin(dth)
    x, y, z = pts_rig[:, 0], pts_rig[:, 1], pts_rig[:, 2]
    xs = x + dx                      # ego travelled +dx forward between t_ref and t_pt
    out = np.empty_like(pts_rig)
    out[:, 0] = xs * c - y * s
    out[:, 1] = xs * s + y * c
    out[:, 2] = z
    return out


# ---------------------------------------------------------------------------
# The rasterisers
# ---------------------------------------------------------------------------
def _visibility_polar(pts_rig: np.ndarray, obstacle: np.ndarray,
                      n_az_fine: int = 720, r_max_m: float = 80.0,
                      r_step_m: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Per-azimuth first-obstacle range over the FULL 360°, from real returns.

    ⭐ THIS IS THE THIRD STATE, AND IT IS MEASURED RATHER THAN DERIVED. WP-D had to
    DERIVE occlusion from `obstacle.offline` footprints and could only ever produce a
    LOWER BOUND: the join drops `z`, the shadow is quantised to a range bin, and -- the
    binding limit -- **only labelled agents occlude**, so buildings, walls, parked rows
    and vegetation cast no shadow at all in a derived mask. A LiDAR return is the
    sensor's own statement about what it could see, and it shadows everything opaque.

    Returns `(first_hit_r, az_edges)` with `first_hit_r[k] = inf` where the column has
    no obstacle return (nothing blocks the view out to r_max).
    """
    az = np.arctan2(pts_rig[:, 1], pts_rig[:, 0])            # [-pi, pi], +y LEFT
    r = np.hypot(pts_rig[:, 0], pts_rig[:, 1])
    k = np.floor((az + math.pi) / (2 * math.pi) * n_az_fine).astype(np.int64)
    np.clip(k, 0, n_az_fine - 1, out=k)
    first = np.full(n_az_fine, np.inf, dtype=np.float64)
    m = obstacle & (r > 0.5) & (r < r_max_m)
    if m.any():
        np.minimum.at(first, k[m], r[m])
    az_edges = -math.pi + np.arange(n_az_fine + 1) * (2 * math.pi / n_az_fine)
    return first, az_edges


def rasterise_cartesian(pts_rig: np.ndarray, intensity: np.ndarray,
                        spec: CartesianBEVSpec, zband: ZBand,
                        min_pts: int = 1) -> dict[str, np.ndarray]:
    """[n_x, n_y] channels in the rig frame. Row 0 at the ego origin (x=0)."""
    nx, ny = spec.n_x, spec.n_y
    x, y, z = pts_rig[:, 0], pts_rig[:, 1], pts_rig[:, 2]
    in_box = (x >= 0.0) & (x < spec.x_max_m) & (y >= -spec.y_half_m) & (y < spec.y_half_m)
    obstacle = (z >= zband.obstacle_min_m) & (z <= zband.obstacle_max_m)
    ground = z < zband.ground_max_m

    ix = np.floor(x / spec.cell_m).astype(np.int64)
    iy = np.floor((y + spec.y_half_m) / spec.cell_m).astype(np.int64)
    flat = ix * ny + iy

    def _count(mask: np.ndarray) -> np.ndarray:
        m = in_box & mask
        return np.bincount(flat[m], minlength=nx * ny).reshape(nx, ny)

    n_obs = _count(obstacle)
    n_gnd = _count(ground)
    n_all = _count(np.ones_like(in_box))

    z_max = np.full(nx * ny, -np.inf)
    m = in_box & obstacle
    if m.any():
        np.maximum.at(z_max, flat[m], z[m])
    z_max = z_max.reshape(nx, ny)
    z_max[~np.isfinite(z_max)] = 0.0

    # per-cell vertical spread over ALL returns (not just the z-band ones) - the test
    # that separates a cambered road from a wall. See ZBand's docstring.
    zc_hi = np.full(nx * ny, -np.inf)
    zc_lo = np.full(nx * ny, np.inf)
    if in_box.any():
        np.maximum.at(zc_hi, flat[in_box], z[in_box])
        np.minimum.at(zc_lo, flat[in_box], z[in_box])
    spread = (zc_hi - zc_lo).reshape(nx, ny)
    spread[~np.isfinite(spread)] = 0.0

    inten = np.zeros(nx * ny, dtype=np.float64)
    if m.any():
        np.maximum.at(inten, flat[m], intensity[m].astype(np.float64))
    inten = inten.reshape(nx, ny)

    # occupancy under BOTH rules; the default is the conjunction
    occ_zband = (n_obs >= min_pts)
    occ_spread = occ_zband & (n_all >= zband.spread_min_pts) & (spread >= zband.spread_min_m)

    # observed / shadow, from the sensor's own first-return range per azimuth.
    # ⚠️ the shadow is cast by the DEFAULT occupancy rule, so a cambered road does not
    # shadow the world behind it.
    obst_pts = obstacle & in_box & occ_spread.reshape(-1)[np.clip(flat, 0, nx * ny - 1)]
    first, _ = _visibility_polar(pts_rig, obst_pts)
    n_az_fine = first.shape[0]
    gx = (np.arange(nx) + 0.5) * spec.cell_m
    gy = (np.arange(ny) + 0.5) * spec.cell_m - spec.y_half_m
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    GR = np.hypot(GX, GY)
    GA = np.arctan2(GY, GX)
    GK = np.clip(np.floor((GA + math.pi) / (2 * math.pi) * n_az_fine).astype(np.int64),
                 0, n_az_fine - 1)
    first_at = first[GK]
    # a cell is SHADOWED when it lies beyond the nearest obstacle return in its bearing
    shadow = GR > (first_at + spec.cell_m)
    occ = occ_spread.astype(np.float32)
    observed = (~shadow) | (occ > 0) | (n_all > 0)

    return {
        "occ": occ,
        "occ_zband_only": occ_zband.astype(np.float32),
        "z_max": z_max.astype(np.float32),
        "z_spread": spread.astype(np.float32),
        "n_obstacle_pts": n_obs.astype(np.int32),
        "n_ground_pts": n_gnd.astype(np.int32),
        "n_pts": n_all.astype(np.int32),
        "intensity_max": inten.astype(np.float32),
        "observed": observed.astype(np.uint8),
        "range_m": GR.astype(np.float32),
    }


def rasterise_polar(pts_rig: np.ndarray, spec: PolarBEVSpec, zband: ZBand,
                    min_pts: int = 1) -> dict[str, np.ndarray]:
    """[n_rng, n_az] in the rig frame, column 0 = +hfov/2 (LEFT).

    ⛔ COLUMN SENSE IS ASSERTED, NEVER ASSUMED. `bev_aux.azimuth_column` and
    `psg_targets.azimuth_column` -- two independently authored modules -- agree that
    **column 0 = +60° (LEFT)**, and WP-A's `s5_indexed.py` used the opposite sense,
    which is `R-2026-09-08-wpa-mirror`: it inverted the programme's reading of whether
    the trunk localises agents at all. This function follows the two modules that agree.
    """
    x, y, z = pts_rig[:, 0], pts_rig[:, 1], pts_rig[:, 2]
    r = np.hypot(x, y)
    az = np.degrees(np.arctan2(y, x))                  # +y LEFT => +az LEFT
    half = spec.hfov_deg / 2.0
    in_fov = (az <= half) & (az >= -half) & (r >= spec.r_min_m) & (r < spec.r_max_m) & (x > 0)
    obstacle = (z >= zband.obstacle_min_m) & (z <= zband.obstacle_max_m)

    # column 0 = +half (LEFT), column n_az-1 = -half (RIGHT)
    col = np.floor((half - az) / spec.cell_deg).astype(np.int64)
    rng = np.floor((r - spec.r_min_m) / spec.cell_rng_m).astype(np.int64)
    np.clip(col, 0, spec.n_az - 1, out=col)
    np.clip(rng, 0, spec.n_rng - 1, out=rng)
    flat = rng * spec.n_az + col

    m = in_fov & obstacle
    n_obs = np.bincount(flat[m], minlength=spec.n_rng * spec.n_az).reshape(spec.n_rng, spec.n_az)
    n_all = np.bincount(flat[in_fov], minlength=spec.n_rng * spec.n_az).reshape(spec.n_rng, spec.n_az)

    # same vertical-spread rule as the Cartesian grid (ZBand docstring)
    n_cell = spec.n_rng * spec.n_az
    zhi = np.full(n_cell, -np.inf)
    zlo = np.full(n_cell, np.inf)
    if in_fov.any():
        np.maximum.at(zhi, flat[in_fov], z[in_fov])
        np.minimum.at(zlo, flat[in_fov], z[in_fov])
    spread = (zhi - zlo).reshape(spec.n_rng, spec.n_az)
    spread[~np.isfinite(spread)] = 0.0

    occ_zband = n_obs >= min_pts
    occ_b = occ_zband & (n_all >= zband.spread_min_pts) & (spread >= zband.spread_min_m)

    # shadow: beyond the first occupied cell in this column
    r_centres = spec.r_min_m + (np.arange(spec.n_rng) + 0.5) * spec.cell_rng_m
    first = np.full(spec.n_az, np.inf)
    for c in range(spec.n_az):
        hit = np.nonzero(occ_b[:, c])[0]
        if hit.size:
            first[c] = r_centres[hit[0]]
    shadow = r_centres[:, None] > (first[None, :] + spec.cell_rng_m)
    occ = occ_b.astype(np.float32)
    observed = (~shadow) | (occ > 0)
    # ⛔ 2026-09-13, ADDITIVE (the shipped `observed` above is unchanged). MEASURED by
    # `p2b_observed_rule_probe.py` on 3 clips x 24 spins: the shipped polar rule keeps an
    # OCCUPIED cell behind the first hit as observed but drops a FREE cell behind it even
    # when it holds LiDAR returns -- 78-91 % of its scored positives sit behind the first
    # hit, and 5,087-11,421 free-with-returns cells are dropped. The CARTESIAN rule of this
    # module already counts any return (`n_all > 0`); `observed_npts` ports it, and
    # `visible_first_hit` is the pure geometric shadow (no occupancy term at all).
    observed_npts = observed | (n_all > 0)
    return {
        "occ": occ,
        "occ_zband_only": occ_zband.astype(np.float32),
        "z_spread": spread.astype(np.float32),
        "n_obstacle_pts": n_obs.astype(np.int32),
        "n_pts": n_all.astype(np.int32),
        "observed": observed.astype(np.uint8),
        "observed_npts": observed_npts.astype(np.uint8),
        "visible_first_hit": (~shadow).astype(np.uint8),
    }


def artifact_meta(cart: CartesianBEVSpec, polar: PolarBEVSpec, zband: ZBand,
                  **extra) -> dict:
    """⭐ The metadata that MUST travel inside the artifact.

    An artifact is opened in isolation far more often than its run record. A `controls`
    column read as curvature instead of lateral acceleration produced a 396 g table that
    was arithmetically perfect and completely wrong, from a file that stated no units.
    """
    meta = {
        "schema": "tanitad.lidar_bev_gt/2",
        "deskew_rotation_sign": "+yaw_rate*dt (corrected 2026-09-13; schema /1 artifacts "
                                "used -yaw_rate*dt, measured wrong)",
        "frame": "rig",
        "frame_convention": "+x forward, +y LEFT, +z UP; origin rear axle on the road plane",
        "units": {
            "occ": "binary {0,1}",
            "z_max": "m above rig road plane",
            "n_pts": "count",
            "intensity_max": "uint8 0-255 sensor reflectance",
            "observed": "binary {0=shadowed/unobserved, 1=observed}",
            "range_m": "m",
            "timestamp": "microseconds on the clip clock (same clock as the camera timestamps parquet)",
        },
        "cartesian": {**asdict(cart), "shape": [cart.n_x, cart.n_y],
                      "extent_m": {"x": [0.0, cart.x_max_m],
                                   "y": [-cart.y_half_m, cart.y_half_m]},
                      "cell_m": cart.cell_m,
                      "row0": "x in [0, cell_m); col0 is y = -y_half (RIGHT), +y is LEFT"},
        "polar": {**asdict(polar), "shape": [polar.n_rng, polar.n_az],
                  "cell_deg": polar.cell_deg, "cell_rng_m": polar.cell_rng_m,
                  "col0": "+hfov/2 = LEFT (agrees with bev_aux.azimuth_column and "
                          "psg_targets.azimuth_column; the opposite sense is "
                          "R-2026-09-08-wpa-mirror)"},
        "z_band_m": asdict(zband),
        "ground_plane": "rig z = 0 (D-V5A-GROUND1; reproduced from LiDAR returns at -0.035 m)",
        "provenance": {
            "sensor": "lidar_top_360fov",
            "codec": "Draco (DracoPy); POSITION float32 in the LIDAR SENSOR frame",
            "sensor_to_rig": "calibration/sensor_extrinsics (per clip; quaternion + xyz)",
            "rate_hz": 10.0,
            "binding": "LABEL ONLY. Inference is VISION-ONLY (PI, 2026-08-03). "
                       "LiDAR must never reach an inference path.",
        },
    }
    meta.update(extra)
    return meta
