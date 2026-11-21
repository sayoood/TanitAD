#!/usr/bin/env python3
"""geodac.py -- "geoDAC": a geometric drivable-area-compliance check of a trajectory against a BEV raster.  numpy only.

WHAT IT IS.  NAVSIM v1's DAC rule, with the map polygons replaced by a raster:
  * a trajectory is a set of rear-axle poses (x, y, heading) in the EGO frame at t0 (x forward, y LEFT, radians CCW);
  * each pose is expanded to the 4 footprint corners of the nuPlan Pacifica (half-length 2.588, half-width 1.1485, the
    footprint centre sits 1.461 m ahead of the rear axle) -- `navsim_dac.py:35-37`, verified against
    `nuplan.common.actor_state.vehicle_parameters.get_pacifica_parameters()` (2.588 / 1.1485 / 1.461);
  * a STEP is off the drivable area when FEWER THAN 4 corners lie in the drivable set (`pdm_scorer.py:355`), and the
    trajectory FAILS if ANY step is (`pdm_scorer.py:427-428`).  The steps are the 0.1 s grid over 4 s INCLUDING t = 0
    (41 steps).  The default horizon below builds exactly that grid from NAVSIM's 8 poses at 0.5 s.

RASTER CONVENTION (the GT store, `aux_label_store.py:59-64`; the predicted heads use the same frame at 2 m cells):
  value[ix, iy], C-order, cell centre = (x0 + (ix + .5) * res_x,  y0 + (iy + .5) * res_y), POSITIVE = inside drivable,
  NEGATIVE = outside.  GT: x0 = -8, y0 = -32, res = 0.5, 160 x 128, int8 signed distance in 0.1 m units.
  Predicted (scene-tap probe heads): x0 = -8, y0 = -32, res = 2, 40 x 32, float `sdf` in metres (6 m * tanh).

SAMPLING.  `mode="bilinear"` interpolates the SIGNED FIELD between the 4 surrounding cell centres and reads its sign (the
SPEC's chosen mode: sub-cell boundary accuracy on a coarse grid, `SPEC_AUX_DRIVABLE_HEAD.md` section 2.2);
`mode="nearest"` reads the cell that contains the point (what a mask would give).  A point whose cell lies outside the raster
extent is UNKNOWN; `unknown` decides how it is scored ("drivable" = benefit of the doubt, "outside" = conservative).

This module computes NO model output and NO outcome.  It is an instrument, validated by test_geodac.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- footprint: nuPlan Pacifica, == refe/navsim_dac.py:35-37 ------------------------------------------------------------
HALF_LENGTH = (4.049 + 1.127) / 2.0
HALF_WIDTH = 2.297 / 2.0
REAR_AXLE_TO_CENTER = (4.049 - 1.127) / 2.0

# ---- NAVSIM's trajectory grid: 8 poses at 0.5 s, judged on 41 states at 0.1 s including t = 0 --------------------------
SRC_DT, SRC_N, DST_DT, DST_N = 0.5, 8, 0.1, 41


@dataclass(frozen=True)
class RasterGeom:
    """Where a raster sits.  value[ix, iy]; centre of cell (ix, iy) = (x0 + (ix + .5) * res_x, y0 + (iy + .5) * res_y)."""
    x0: float
    y0: float
    res_x: float
    res_y: float
    nx: int
    ny: int

    @property
    def shape(self):
        return (self.nx, self.ny)


GT_GEOM = RasterGeom(-8.0, -32.0, 0.5, 0.5, 160, 128)        # aux_label_store.py:59-60
GT_UNIT_M = 0.1                                                # int8 LSB in metres, aux_label_store.py:61
PRED_GEOM = RasterGeom(-8.0, -32.0, 2.0, 2.0, 40, 32)         # aux_drivable_head.py grid (SPEC section 2.2)


def gt_field_m(sdf_i8: np.ndarray) -> np.ndarray:
    """The GT store's int8 [160, 128] -> metres (float64), positive = inside."""
    return np.asarray(sdf_i8, dtype=np.float64) * GT_UNIT_M


# ---- geometry primitives (each is a separate function so a test can mutate it) --------------------------------------------
def densify(poses8: np.ndarray) -> np.ndarray:
    """[..., 8, 3] rear-axle poses at 0.5 .. 4.0 s (ego frame, t0 pose = origin) -> [..., 41, 3] at 0.0 .. 4.0 s.
    Linear in x, y and in the UNWRAPPED heading -- the same interpolation as `refe/navsim_dac.py:at_10hz` (asserted equal in the test)."""
    p = np.asarray(poses8, dtype=np.float64)
    lead = p.shape[:-2]
    full = np.concatenate([np.zeros(lead + (1, 3)), p], axis=-2)                       # t = 0, .5, ..., 4.0   [..., 9, 3]
    hd = np.unwrap(full[..., 2], axis=-1)
    t_q = np.arange(DST_N) * DST_DT
    j = np.minimum((t_q / SRC_DT + 1e-9).astype(int), SRC_N - 1)                       # left knot of each query
    w = (t_q - j * SRC_DT) / SRC_DT                                                    # [41]
    out = np.empty(lead + (DST_N, 3))
    for c in (0, 1):
        out[..., c] = full[..., j, c] * (1 - w) + full[..., j + 1, c] * w
    out[..., 2] = hd[..., j] * (1 - w) + hd[..., j + 1] * w
    return out


def corners(poses: np.ndarray) -> np.ndarray:
    """[..., T, 3] rear-axle poses -> [..., T, 4, 2] footprint corners (ego frame): centre = rear axle + 1.461 m along the heading,
    corners at +-half_length along it and +-half_width across it.  Corner order is irrelevant to the verdict."""
    p = np.asarray(poses, dtype=np.float64)
    ch, sh = np.cos(p[..., 2]), np.sin(p[..., 2])
    cx = p[..., 0] + REAR_AXLE_TO_CENTER * ch
    cy = p[..., 1] + REAR_AXLE_TO_CENTER * sh
    out = np.empty(p.shape[:-1] + (4, 2))
    for i, (lo, la) in enumerate(((HALF_LENGTH, HALF_WIDTH), (HALF_LENGTH, -HALF_WIDTH), (-HALF_LENGTH, HALF_WIDTH), (-HALF_LENGTH, -HALF_WIDTH))):
        out[..., i, 0] = cx + lo * ch - la * sh
        out[..., i, 1] = cy + lo * sh + la * ch
    return out


def cell_coords(geom: RasterGeom, pts: np.ndarray):
    """[..., 2] ego-frame points -> (fx, fy): FRACTIONAL cell index with cell centres at integers (so index 0.0 is the centre of cell 0)."""
    pts = np.asarray(pts, dtype=np.float64)
    return (pts[..., 0] - geom.x0) / geom.res_x - 0.5, (pts[..., 1] - geom.y0) / geom.res_y - 0.5


def in_extent(geom: RasterGeom, pts: np.ndarray) -> np.ndarray:
    """True where the point lies inside the raster's rectangle (cell edges, not centres)."""
    fx, fy = cell_coords(geom, pts)
    return (fx >= -0.5) & (fx < geom.nx - 0.5) & (fy >= -0.5) & (fy < geom.ny - 0.5)


def sample_field(field: np.ndarray, geom: RasterGeom, pts: np.ndarray, mode: str = "bilinear") -> np.ndarray:
    """Signed field (positive = inside) at ego-frame points -> [...] float.  Points outside the extent return NaN (UNKNOWN).
    bilinear: weights from the 4 surrounding cell centres, indices clamped to the grid (edge replicate) for points in the half-cell
    border; nearest: the cell that contains the point."""
    f = np.asarray(field, dtype=np.float64)
    if f.shape != geom.shape:
        raise ValueError(f"field shape {f.shape} != geometry {geom.shape}")
    fx, fy = cell_coords(geom, pts)
    ok = in_extent(geom, pts)
    if mode == "nearest":
        ix = np.clip(np.floor(fx + 0.5).astype(int), 0, geom.nx - 1)
        iy = np.clip(np.floor(fy + 0.5).astype(int), 0, geom.ny - 1)
        v = f[ix, iy]
    elif mode == "bilinear":
        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        tx, ty = fx - x0, fy - y0
        x0c, x1c = np.clip(x0, 0, geom.nx - 1), np.clip(x0 + 1, 0, geom.nx - 1)
        y0c, y1c = np.clip(y0, 0, geom.ny - 1), np.clip(y0 + 1, 0, geom.ny - 1)
        v = (f[x0c, y0c] * (1 - tx) * (1 - ty) + f[x1c, y0c] * tx * (1 - ty) + f[x0c, y1c] * (1 - tx) * ty + f[x1c, y1c] * tx * ty)
    else:
        raise ValueError("mode must be 'bilinear' or 'nearest'")
    return np.where(ok, v, np.nan)


def is_inside(values: np.ndarray) -> np.ndarray:
    """Sign convention: strictly positive = inside the drivable area (a value of exactly 0 is on the boundary = NOT inside)."""
    return values > 0.0


def _verdict(inside: np.ndarray, unknown_mask: np.ndarray, unknown: str, min_corners: int = 4):
    """inside/unknown_mask [..., T, 4] -> (step_ok [..., T])."""
    if unknown == "drivable":
        ins = inside | unknown_mask
    elif unknown == "outside":
        ins = inside & ~unknown_mask
    else:
        raise ValueError("unknown must be 'drivable' or 'outside'")
    return ins.sum(axis=-1) >= min_corners


# ---- the instrument ------------------------------------------------------------------------------------------------------
@dataclass
class GeoDAC:
    passed: np.ndarray            # [...] bool, True = compliant (NAVSIM DAC = 1)
    frac_outside: np.ndarray      # [...] share of the T x 4 footprint-corner samples that are NOT inside (UNKNOWN follows `unknown`)
    min_margin_m: np.ndarray      # [...] min over all corner samples of the signed field (m): <= 0 = some corner is outside; NaN-free (unknown -> +inf)
    first_bad_step: np.ndarray    # [...] int, index of the first off-road step (-1 = none)
    n_unknown: np.ndarray         # [...] corner samples outside the raster extent


def geodac_corners(field: np.ndarray, geom: RasterGeom, cs: np.ndarray, *, mode: str = "bilinear", unknown: str = "drivable") -> GeoDAC:
    """The verdict from FOOTPRINT CORNERS directly: cs [..., T, 4, 2] in the ego frame (the step grid is whatever the caller built).
    `geodac` below is `densify -> corners -> this`; a caller that already has the corners (e.g. a simulator's output) enters here."""
    v = sample_field(field, geom, cs, mode)                             # [..., T, 4]
    unk = np.isnan(v)
    inside = is_inside(np.where(unk, 0.0, v)) & ~unk
    step_ok = _verdict(inside, unk, unknown)
    ins_for_frac = (inside | unk) if unknown == "drivable" else inside
    bad = ~step_ok
    first = np.where(bad.any(-1), bad.argmax(-1), -1)
    return GeoDAC(passed=step_ok.all(-1), frac_outside=1.0 - ins_for_frac.mean(axis=(-2, -1)),
                  min_margin_m=np.where(unk, np.inf, v).min(axis=(-2, -1)), first_bad_step=first, n_unknown=unk.sum(axis=(-2, -1)))


def geodac(field: np.ndarray, geom: RasterGeom, poses: np.ndarray, *, horizon: str = "navsim41", mode: str = "bilinear",
           unknown: str = "drivable") -> GeoDAC:
    """poses: [..., 8, 3] (horizon="navsim41": densified to the 41-step 0.1 s grid) or [..., T, 3] already at the steps to judge
    (horizon="as_given").  field: ONE raster [nx, ny] shared by every trajectory in the batch (one scene)."""
    p = np.asarray(poses, dtype=np.float64)
    if horizon == "navsim41":
        if p.shape[-2:] != (SRC_N, 3):
            raise ValueError(f"navsim41 needs [..., 8, 3] poses, got {p.shape}")
        p = densify(p)
    elif horizon != "as_given":
        raise ValueError("horizon must be 'navsim41' or 'as_given'")
    return geodac_corners(field, geom, corners(p), mode=mode, unknown=unknown)
