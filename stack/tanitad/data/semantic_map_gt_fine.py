"""Reader for the SAM3 map GT at **10 cm** (``fine_codes``) -- refcv7 NEW-2.

``Project Steering/SPEC_REFCV7.md`` §6.2 (Amendment A1, 2026-09-26): the map head
predicts at 0.1 m. The label already exists: every ``tanitad.sam3_map_gt/2`` file
carries ``fine_codes`` ``[T, 600, 320]`` uint8 next to the 0.5 m ``cart_frac``
that :mod:`tanitad.data.semantic_map_gt` reads, and ``cart_frac`` is EXACTLY the
5 x 5 block fraction of ``fine_codes`` (``TanitAD Research Lab/Architecture &
Inference/Research/2026-09-26-refcv7-map-hires/RESULT.md`` §2: 0 differing cells;
the exporter's rule, MEASURED here on 3,663,360 cells: ``cart = rint(count *
255 / 25)`` -- no ties are possible, because ``count * 10.2`` never ends in .5).

⭐⭐ THE EXTENT IS A DECLARED PARAMETER (SPEC_REFCV7 §11.2, Amendment A6, PI
2026-09-27: *"assure that the range of the map is maximal and not only 20 m"*).
:class:`MapExtent` is ``x_max_m`` ahead and ``y_half_m`` to each side; every grid
in the 10 cm path -- this reader's shapes, the lift grid, the decoder's output, the
metric bands (every 20 m) and the pooling to the planner's grid -- is DERIVED from
it, never typed. The value comes from the GT-coverage census (§11.2 item 2), and the
census SELECTED 100 m ahead x +-30 m (§12, A7: :data:`EXTENT_REFCV7`, 1000 x 600 at
0.1 m); this module's default stays the ``/2`` grid, 60 m x +-16 m (:data:`EXTENT_V2`),
the only extent a ``/2`` file can carry.

⭐ ANCHORED COORDINATES (§12 item 3). The ``/3`` export writes every lateral cell centre
relative to the ``/2`` window's origin, ``y = -16 + (j_rel + 0.5) * cell`` with ``j_rel =
j - 140`` (0.1 m) / ``j - 28`` (0.5 m) at the A7 extent, so inside the old window the
floats -- and therefore the codes -- are the ``/2`` exporter's byte for byte
(:meth:`MapExtent.lateral_centres`, :meth:`MapExtent.anchor_offset`, and the control
:func:`compare_v2_window`, for ``fine_codes`` AND ``cart_frac``).

TWO SCHEMAS, ONE SET OF GUARDS
------------------------------
* ``tanitad.sam3_map_gt/2`` (every file today): 60 m x +-16 m only.
  :func:`open_path_fine` runs :func:`semantic_map_gt.open_path` FIRST (schema string,
  rig frame, grid spec, channel order, fraction scale, array names / shapes /
  dtypes, ``clip_sha12`` identity, a monotone time axis) and then adds only what is
  new at 10 cm. ⛔ A ``/2`` file opened under any OTHER declared extent is REFUSED
  (:class:`FineSpecMismatch`): it cannot carry the larger grid, and cropping or
  padding it silently would put "not seen" where the label was never exported.
* ``tanitad.sam3_map_gt/3`` (the re-export at the census extent, §11.2 item 3; SAM3
  is not re-run): the SAME identity / frame-axis / legend guards, spelled once here
  for a file that carries no 0.5 m ``cart_frac`` requirement (the 0.5 m map is
  removed in refcv7, §11.1). Its contract is :data:`V3_CONTRACT`. ⛔ A ``/3`` file
  whose ``meta.fine`` extent is not the DECLARED one is REFUSED. The re-export's own
  control -- the 10 cm codes inside the old 60 x 32 m window are byte-identical to
  ``/2`` -- is :func:`compare_v2_window`.

What is checked at 10 cm, on both schemas:

* ``fine_codes`` must exist, be ``uint8`` and be ``[T, *extent.fine_shape]`` with the
  SAME ``T`` as every other array (:class:`FineSpecMismatch`);
* ``meta_json['fine']`` must equal ``extent.fine_spec`` on every key it declares --
  x 0..x_max, y +-y_half, 0.1 m cells, the shape, the code legend
  (:class:`FineSpecMismatch`);
* ``/2`` only: ``meta_json['sub_samples_per_axis']`` must be 5 -- the statement that
  the 0.5 m and the 0.1 m arrays are the SAME map at two resolutions;
* ``meta_json['source']['world_map_res_m']`` must be 0.1: a "10 cm" array
  resampled from a coarser world map would be a 10 cm GRID carrying a coarser
  LABEL, and nothing downstream could tell;
* every code on a READ frame is in ``{0..7, 255}`` (:class:`FineCodeError`).

Frame index and time checks are the base reader's own methods
(``ClipMapGT._index`` / ``ClipMapGT.check_times``), called on the base object.

⛔ LABEL-ONLY. The file is non-causal (``meta_json['non_causal']``: *"labels use
every frame of the clip; inference must never read this file"*). Nothing on an
inference path may import this module; the model branch that consumes the target
(``tanitad.models.map_head_hires``) takes vision tensors only.

Grid (the file's own words, ``meta_json['fine']`` and ``['cartesian']['row0']``):
row 0 = x in [0, 0.1 m) (NEAREST), col 0 = y in [-y_half, -y_half + 0.1) m (RIGHT),
+y LEFT. The range bands are every 20 m = 200 fine rows from the nearest row
(:attr:`MapExtent.bands`; 0-20 / 20-40 / 40-60 m on the ``/2`` grid).

Clip ids never appear in this module's messages: errors carry sha12 only.
"""
from __future__ import annotations

import json
import math
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tanitad.data import semantic_map_gt as _G
from tanitad.data.bev_raster import BEVGrid
from tanitad.data.perception_targets import MapGTStore
from tanitad.data.semantic_map_gt import (
    CART_SHAPE, CHANNELS, FRACTION_SCALE, NOT_SEEN_CHANNEL, TIME_TOL_US,
    ClipIdentityMismatch, ClipMapGT, FrameIndexError, SchemaMismatch,
    SemanticMapGTError, TimeMisalignment, gt_path, raw_frame_index, sha12,
)

__all__ = [
    "FINE_SPEC", "FINE_SHAPE", "FINE_CELL_M", "N_FINE_CLASSES", "NOT_SEEN_CODE",
    "FINE_CLASSES", "BLOCK", "SUB_SAMPLES_PER_AXIS", "WORLD_MAP_RES_M",
    "X_BANDS_M", "BAND_ROWS", "BAND_M", "BAND_ROWS_FINE",
    "SCHEMA_V2", "SCHEMA_V3", "FINE_SCHEMAS", "V3_CONTRACT",
    "MapExtent", "EXTENT_V2", "EXTENT_REFCV7", "band_key", "extent_from_meta",
    "FineSpecMismatch", "FineCodeError",
    "SemanticMapGTError", "SchemaMismatch", "ClipIdentityMismatch",
    "FrameIndexError", "TimeMisalignment",
    "ClipFineMapGT", "FineMapFrames", "FineMapGTStore",
    "open_clip_fine", "open_path_fine", "block_fraction_u8", "band_of_row",
    "compare_v2_window", "raw_frame_index", "sha12",
]

#: ``meta_json['fine']`` of every ``/2`` file, compared KEY BY KEY (extra keys in the
#: file are tolerated, exactly as ``_check_meta`` tolerates ``cartesian.row0``).
#: MEASURED on the 137 eval-kit files 2026-09-26. == ``EXTENT_V2.fine_spec``.
FINE_SPEC: dict = {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.1,
                   "shape": [600, 320], "codes": "0-7 class, 255 not seen"}
FINE_SHAPE: tuple[int, int] = (600, 320)
FINE_CELL_M: float = 0.1
N_FINE_CLASSES: int = 8
NOT_SEEN_CODE: int = 255
#: the class order of the codes 0..7 == ``cart_frac``'s channels 0..7.
FINE_CLASSES: tuple[str, ...] = tuple(CHANNELS[:N_FINE_CLASSES])
#: fine cells per 0.5 m cell, per axis.
BLOCK: int = 5
SUB_SAMPLES_PER_AXIS: int = BLOCK
#: the SAM3 world map the exporter rasterised (``meta.source.world_map_res_m``).
WORLD_MAP_RES_M: float = 0.1
#: the range bands are every ``BAND_M`` metres from the nearest row (SPEC_REFCV7 §6.2,
#: and §11.2 item 1: "plus every further 20 m band the grid gains").
BAND_M: float = 20.0
BAND_ROWS_FINE: int = 200
#: the ``/2`` grid's bands, in metres and in fine rows (row 0 = nearest).
X_BANDS_M: tuple[tuple[float, float], ...] = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0))
BAND_ROWS: tuple[tuple[int, int], ...] = ((0, 200), (200, 400), (400, 600))

SCHEMA_V2: str = _G.SCHEMA                      # "tanitad.sam3_map_gt/2"
SCHEMA_V3: str = "tanitad.sam3_map_gt/3"
FINE_SCHEMAS: tuple[str, ...] = (SCHEMA_V2, SCHEMA_V3)
#: ⭐ What a ``tanitad.sam3_map_gt/3`` file must carry for THIS reader (SPEC_REFCV7
#: §11.2 item 3). Written down so the re-export and the reader cannot drift: it is
#: the ``/2`` contract minus the 0.5 m arrays, with the fine grid at the census extent.
V3_CONTRACT: dict = {
    "meta_json": {
        "schema": SCHEMA_V3, "frame": "rig", "channels": list(CHANNELS),
        "fine": "x_max_m, y_half_m, cell_m 0.1, shape [x_max/0.1, 2*y_half/0.1], "
                "codes '0-7 class, 255 not seen' -- equal to the DECLARED extent",
        "source": "clip_sha12 = sha256(clip_id)[:12]; world_map_res_m 0.1",
    },
    "arrays": {"fine_codes": "[T, x_max/0.1, 2*y_half/0.1] uint8",
               "t_img_us": "[T] int64", "t_query_us": "[T] float64",
               "cam_frame_idx": "[T] int32", "T_world_rig": "[T, 4, 4] float64"},
    "not_required": "cart_frac / polar arrays (tolerated, never read)",
    "control": "fine_codes inside the 60 m x +-16 m window byte-identical to /2 "
               "(compare_v2_window)",
}

# ⛔ self-consistency, at import: a future edit that moves one constant without the
# others fails HERE, not as a half-cell shift in a metric.
if (tuple(FINE_SPEC["shape"]) != FINE_SHAPE
        or FINE_SHAPE != (CART_SHAPE[0] * BLOCK, CART_SHAPE[1] * BLOCK)
        or abs(FINE_SPEC["cell_m"] * BLOCK - 0.5) > 1e-12
        or CHANNELS[NOT_SEEN_CHANNEL] != "not seen"
        or len(FINE_CLASSES) != N_FINE_CLASSES
        or abs(BAND_M / FINE_CELL_M - BAND_ROWS_FINE) > 1e-9
        or tuple(r for b in BAND_ROWS for r in b)
        != tuple(int(round(m / FINE_CELL_M)) for b in X_BANDS_M for m in b)):
    raise RuntimeError("semantic_map_gt_fine: the fine grid constants disagree "
                       "with each other or with semantic_map_gt")


class FineSpecMismatch(SchemaMismatch):
    """The file has no 10 cm map, or its 10 cm map is not the one this reader reads."""


class FineCodeError(SchemaMismatch):
    """A read frame carries a code outside ``{0..7, 255}``."""


# --------------------------------------------------------------------------- #
# the declared extent                                                          #
# --------------------------------------------------------------------------- #
#: the ``/2`` window's lateral half-width: the ANCHOR of every ``/3`` lateral index.
EXTENT_V2_Y_HALF_M: float = 16.0


def _fmt_m(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}".replace(".", "p")


def band_key(lo_m: float, hi_m: float) -> str:
    """``(0, 20) -> "0_20"``; a partial last band keeps its true edge, e.g. ``60_70``."""
    return f"{_fmt_m(lo_m)}_{_fmt_m(hi_m)}"


@dataclass(frozen=True)
class MapExtent:
    """The map's metric extent: ``x`` in ``[0, x_max_m)`` ahead, ``y`` in
    ``[-y_half_m, +y_half_m)`` (SPEC_REFCV7 §11.2). Frozen and validated.

    ⛔ Both are multiples of 0.5 m, so the 10 cm label grid, the 0.25 m lift grid and
    the 0.5 m planner grid all tile it exactly and share cell edges. ⛔ Never smaller
    than today's 60 m x +-16 m (§11.2 item 2's constraint): the planner's grid is a
    window INSIDE the map, never outside it."""

    x_max_m: float = 60.0
    y_half_m: float = 16.0

    def __post_init__(self) -> None:
        for n in ("x_max_m", "y_half_m"):
            v = float(getattr(self, n))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"MapExtent.{n} must be a finite positive value, got {v!r}")
            if abs(2.0 * v - round(2.0 * v)) > 1e-9:
                raise ValueError(f"MapExtent.{n} {v!r} is not a multiple of 0.5 m: the "
                                 f"0.1 / 0.25 / 0.5 m grids would not share cell edges")
        if float(self.x_max_m) < 60.0 - 1e-9 or float(self.y_half_m) < 16.0 - 1e-9:
            raise ValueError(
                f"MapExtent {self.x_max_m} m x +-{self.y_half_m} m is smaller than the "
                f"60 m x +-16 m minimum (SPEC_REFCV7 §11.2): the planner's 0.5 m grid "
                f"must lie inside the map")

    def grid(self, cell_m: float) -> BEVGrid:
        """The extent as a :class:`BEVGrid` of ``cell_m`` cells (exact tiling asserted)."""
        c = float(cell_m)
        for n, v in (("x_max_m", float(self.x_max_m)), ("2*y_half_m", 2.0 * float(self.y_half_m))):
            k = v / c
            if abs(k - round(k)) > 1e-9:
                raise ValueError(f"{n} {v} m is not a whole number of {c} m cells")
        return BEVGrid(x_fwd_m=float(self.x_max_m), y_half_m=float(self.y_half_m), cell_m=c)

    @property
    def fine_shape(self) -> tuple[int, int]:
        return tuple(self.grid(FINE_CELL_M).shape)

    @property
    def fine_spec(self) -> dict:
        """What ``meta_json['fine']`` must say for this extent (``FINE_SPEC`` on ``/2``)."""
        return {"x_max_m": float(self.x_max_m), "y_half_m": float(self.y_half_m),
                "cell_m": FINE_CELL_M, "shape": list(self.fine_shape),
                "codes": FINE_SPEC["codes"]}

    @property
    def bands(self) -> tuple[tuple[str, float, float, int, int], ...]:
        """``(key, lo_m, hi_m, row0, row1)`` for every 20 m band, nearest first; a
        partial last band (``x_max`` not a multiple of 20) keeps its true edge."""
        out, lo = [], 0.0
        x = float(self.x_max_m)
        while lo < x - 1e-9:
            hi = min(lo + BAND_M, x)
            out.append((band_key(lo, hi), lo, hi, int(round(lo / FINE_CELL_M)),
                        int(round(hi / FINE_CELL_M))))
            lo = hi
        return tuple(out)

    @property
    def band_keys(self) -> tuple[str, ...]:
        return tuple(b[0] for b in self.bands)

    def anchor_offset(self, cell_m: float) -> int:
        """Columns between this grid's column 0 and the ``/2`` window's column 0:
        ``(y_half - 16) / cell`` -- 140 on the 10 cm grid and 28 on the 0.5 m grid of the
        A7 extent (SPEC_REFCV7 §12 item 3)."""
        off = (float(self.y_half_m) - float(EXTENT_V2_Y_HALF_M)) / float(cell_m)
        if abs(off - round(off)) > 1e-9:
            raise ValueError(f"the /2 window is not aligned to {cell_m} m cells")
        return int(round(off))

    def lateral_centres(self, cell_m: float) -> np.ndarray:
        """Lateral cell centres in the ANCHORED form the ``/3`` export writes (SPEC_REFCV7
        §12 item 3): ``y = -16 + (j_rel + 0.5) * cell`` with ``j_rel = j - anchor_offset``.
        Inside the old window ``j_rel`` IS the ``/2`` index, so the float is the ``/2``
        exporter's, character for character (a ``floor`` at a world-map cell edge cannot
        flip a code). Mathematically ``-y_half + (j + 0.5) * cell``."""
        ny = int(self.grid(cell_m).shape[1])
        off = self.anchor_offset(cell_m)
        return -float(EXTENT_V2_Y_HALF_M) + (np.arange(ny) - off + 0.5) * float(cell_m)

    def window_in(self, outer: "MapExtent") -> tuple[int, int, int, int]:
        """``(row0, row1, col0, col1)`` of THIS extent's 10 cm cells inside ``outer``'s
        fine grid (rows from x = 0, columns centred). Refuses when it does not fit."""
        if float(self.x_max_m) > float(outer.x_max_m) + 1e-9 \
                or float(self.y_half_m) > float(outer.y_half_m) + 1e-9:
            raise ValueError(f"{self} does not fit inside {outer}")
        c0 = int(round((float(outer.y_half_m) - float(self.y_half_m)) / FINE_CELL_M))
        h, w = self.fine_shape
        return 0, h, c0, c0 + w

    def as_dict(self) -> dict:
        return {"x_max_m": float(self.x_max_m), "y_half_m": float(self.y_half_m),
                "fine_shape": list(self.fine_shape), "band_keys": list(self.band_keys)}


#: the ``/2`` grid: 60 m ahead, +-16 m.
EXTENT_V2 = MapExtent(60.0, EXTENT_V2_Y_HALF_M)
#: ⭐ the refcv7 extent the pre-registered rule SELECTED (SPEC_REFCV7 §12, Amendment A7,
#: 56ae4eb; census MEASURED on 4,369 TRAIN clips / 78,321 frames): 100 m ahead x +-30 m
#: -> 1000 x 600 at 0.1 m, 400 x 240 at 0.25 m, bands 0-20 ... 80-100 m.
EXTENT_REFCV7 = MapExtent(100.0, 30.0)
if EXTENT_V2.fine_spec != FINE_SPEC or EXTENT_V2.fine_shape != FINE_SHAPE \
        or tuple((b[3], b[4]) for b in EXTENT_V2.bands) != BAND_ROWS \
        or EXTENT_REFCV7.fine_shape != (1000, 600) \
        or EXTENT_REFCV7.anchor_offset(FINE_CELL_M) != 140 \
        or EXTENT_REFCV7.anchor_offset(0.5) != 28:
    raise RuntimeError("semantic_map_gt_fine: EXTENT_V2 / EXTENT_REFCV7 disagree with "
                       "FINE_SPEC or with SPEC_REFCV7 §12")


# --------------------------------------------------------------------------- #
# the fine-specific checks (everything else is the base reader's)              #
# --------------------------------------------------------------------------- #
def _check_fine_header(headers: dict, n_frames: int, s12: str,
                       extent: MapExtent = EXTENT_V2) -> None:
    if "fine_codes" not in headers:
        raise FineSpecMismatch(
            f"[{s12}] no fine_codes array: this file carries no 10 cm map "
            f"(refcv7 NEW-2 needs it; the 0.5 m cart_frac is not a substitute)")
    shape, dtype = headers["fine_codes"]
    if dtype != np.dtype(np.uint8):
        raise FineSpecMismatch(f"[{s12}] fine_codes dtype {dtype} != uint8")
    want = (int(n_frames),) + tuple(extent.fine_shape)
    if tuple(shape) != want:
        raise FineSpecMismatch(
            f"[{s12}] fine_codes shape {tuple(shape)} != {want} (T from the frame "
            f"axis; rows along +x, columns along y, for the DECLARED extent "
            f"{float(extent.x_max_m)} m x +-{float(extent.y_half_m)} m)")


def extent_from_meta(meta: dict, s12: str) -> MapExtent:
    """The extent a file DECLARES: ``meta.fine.{x_max_m, y_half_m}`` (the ``/2``
    convention), or a top-level ``meta.extent`` -- both must agree when both exist.
    A file that declares none is refused, never assumed."""
    fine = meta.get("fine") if isinstance(meta.get("fine"), dict) else {}
    top = meta.get("extent") if isinstance(meta.get("extent"), dict) else {}
    got = []
    for src in (fine, top):
        if "x_max_m" in src and "y_half_m" in src:
            try:
                got.append(MapExtent(float(src["x_max_m"]), float(src["y_half_m"])))
            except (TypeError, ValueError) as e:
                raise FineSpecMismatch(f"[{s12}] meta extent {src!r} is not a valid "
                                       f"MapExtent: {e}") from None
    if not got:
        raise FineSpecMismatch(f"[{s12}] meta_json declares no extent (fine.x_max_m / "
                               f"fine.y_half_m, or extent.*)")
    if len(got) == 2 and got[0] != got[1]:
        raise FineSpecMismatch(f"[{s12}] meta.fine says {got[0]}, meta.extent says "
                               f"{got[1]}: the file contradicts itself")
    return got[0]


def _check_fine_meta(meta: dict, s12: str, extent: MapExtent = EXTENT_V2,
                     *, v2: bool = True) -> None:
    fine = meta.get("fine")
    if not isinstance(fine, dict):
        raise FineSpecMismatch(f"[{s12}] meta_json has no 'fine' spec")
    for k, v in extent.fine_spec.items():
        if not v2 and k == "codes" and k not in fine:
            continue            # /3: the legend is `channels`, checked in _open_v3
        if fine.get(k) != v:
            raise FineSpecMismatch(
                f"[{s12}] fine.{k} {fine.get(k)!r} != {v!r}: the file's 10 cm grid is "
                f"not the DECLARED extent's, and this reader's cells would not be the "
                f"file's cells")
    if v2:
        ssp = meta.get("sub_samples_per_axis")
        if ssp != SUB_SAMPLES_PER_AXIS:
            raise FineSpecMismatch(
                f"[{s12}] sub_samples_per_axis {ssp!r} != {SUB_SAMPLES_PER_AXIS}: the "
                f"0.5 m and 0.1 m arrays would not be the same map at two resolutions")
    wres = (meta.get("source") or {}).get("world_map_res_m")
    if wres != WORLD_MAP_RES_M:
        raise FineSpecMismatch(
            f"[{s12}] source.world_map_res_m {wres!r} != {WORLD_MAP_RES_M}: a "
            f"10 cm grid resampled from a coarser world map is not a 10 cm label")


def _check_codes(u8: np.ndarray, s12: str, idx: np.ndarray) -> None:
    bad = (u8 > N_FINE_CLASSES - 1) & (u8 != NOT_SEEN_CODE)
    if bad.any():
        f = int(np.argwhere(bad.reshape(bad.shape[0], -1).any(1))[0, 0])
        vals = sorted(int(v) for v in np.unique(u8[bad]))[:8]
        raise FineCodeError(
            f"[{s12}] frame {int(idx[f])} carries fine code(s) {vals} outside "
            f"{{0..{N_FINE_CLASSES - 1}, {NOT_SEEN_CODE}}} ({int(bad.sum())} cells "
            f"over the {idx.size} read frames)")


#: the ``/3`` frame-axis arrays (the ``/2`` table without ``cart_frac``).
_V3_REQUIRED = {
    "t_img_us": (np.int64, (None,)),
    "t_query_us": (np.float64, (None,)),
    "cam_frame_idx": (np.int32, (None,)),
    "T_world_rig": (np.float64, (None, 4, 4)),
}


def _read_meta(path: Path, s12: str) -> dict:
    try:
        with np.load(path, allow_pickle=False) as z:
            if "meta_json" not in z.files:
                raise SchemaMismatch(f"[{s12}] no meta_json in the file")
            meta = json.loads(str(z["meta_json"]))
    except zipfile.BadZipFile as e:
        raise SchemaMismatch(f"[{s12}] not a readable .npz archive: {e}") from None
    except (ValueError, TypeError) as e:
        raise SchemaMismatch(f"[{s12}] meta_json does not parse: {e}") from None
    if not isinstance(meta, dict):
        raise SchemaMismatch(f"[{s12}] meta_json is not an object")
    return meta


def _open_v3(path: Path, s12: str, headers: dict, meta: dict) -> ClipMapGT:
    """The ``/2`` identity, legend and frame-axis guards for a ``/3`` file."""
    T = None
    for name, (dtype, shape) in _V3_REQUIRED.items():
        if name not in headers:
            raise SchemaMismatch(f"[{s12}] required array {name!r} is missing")
        got_shape, got_dtype = headers[name]
        if got_dtype != np.dtype(dtype):
            raise SchemaMismatch(f"[{s12}] {name} dtype {got_dtype} != {np.dtype(dtype)}")
        if len(got_shape) != len(shape) or any(
                e is not None and g != e for g, e in zip(got_shape, shape)):
            raise SchemaMismatch(f"[{s12}] {name} shape {got_shape} != "
                                 f"{tuple('T' if e is None else e for e in shape)}")
        if T is None:
            T = got_shape[0]
        elif got_shape[0] != T:
            raise SchemaMismatch(f"[{s12}] {name} has {got_shape[0]} frames, "
                                 f"t_img_us has {T}")
    if not T:
        raise SchemaMismatch(f"[{s12}] the file holds zero frames")
    if meta.get("frame") != "rig":
        raise SchemaMismatch(f"[{s12}] frame {meta.get('frame')!r} != 'rig'")
    if tuple(meta.get("channels") or ()) != CHANNELS:
        raise SchemaMismatch(f"[{s12}] channel list/order differs from the reader's: "
                             f"{meta.get('channels')!r}")
    stored = (meta.get("source") or {}).get("clip_sha12")
    if stored != s12:
        raise ClipIdentityMismatch(
            f"GT file requested as clip sha12 {s12} stores clip sha12 {stored!r}: it "
            f"belongs to another clip (renamed or mis-copied)")
    with np.load(path, allow_pickle=False) as z:
        t_img = z["t_img_us"]
        t_query = z["t_query_us"]
        cam_idx = z["cam_frame_idx"]
        pose = z["T_world_rig"]
    if np.any(np.diff(t_img) < 0) or not np.all(np.isfinite(t_query)):
        raise SchemaMismatch(f"[{s12}] t_img_us is not non-decreasing / t_query_us "
                             f"not finite: the frame axis is corrupt")
    return ClipMapGT(path=path, clip_sha12=s12, meta=meta, n_frames=int(T),
                     t_img_us=t_img, t_query_us=t_query, cam_frame_idx=cam_idx,
                     T_world_rig=pose)


# --------------------------------------------------------------------------- #
# the reader                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FineMapFrames:
    """10 cm labels for N requested raw v2ep frames."""

    codes: np.ndarray         # [N, *extent.fine_shape] uint8, 0..7 class, 255 not seen
    seen: np.ndarray          # [N, *extent.fine_shape] bool, codes != 255
    t_img_us: np.ndarray      # [N] int64
    frame_idx: np.ndarray     # [N] int64


@dataclass
class ClipFineMapGT:
    """One clip's validated 10 cm GT. Construct with :func:`open_path_fine`."""

    base: ClipMapGT
    extent: MapExtent = EXTENT_V2
    _fine_u8: np.ndarray | None = field(default=None, repr=False)

    # the identity and the frame axis ARE the base reader's -- never copies
    @property
    def path(self) -> Path:
        return self.base.path

    @property
    def clip_sha12(self) -> str:
        return self.base.clip_sha12

    @property
    def meta(self) -> dict:
        return self.base.meta

    @property
    def schema(self) -> str:
        return str(self.base.meta.get("schema"))

    @property
    def n_frames(self) -> int:
        return self.base.n_frames

    @property
    def t_img_us(self) -> np.ndarray:
        return self.base.t_img_us

    @property
    def t_query_us(self) -> np.ndarray:
        return self.base.t_query_us

    @property
    def cam_frame_idx(self) -> np.ndarray:
        return self.base.cam_frame_idx

    @property
    def T_world_rig(self) -> np.ndarray:
        return self.base.T_world_rig

    def fine_u8(self) -> np.ndarray:
        """The full ``fine_codes`` array (loaded once; ~38.6 MB for 201 frames on the
        ``/2`` grid, MEASURED 0.07-0.18 s to decompress on the dev box)."""
        if self._fine_u8 is None:
            with np.load(self.path, allow_pickle=False) as z:
                a = z["fine_codes"]
            if a.shape != (self.n_frames,) + tuple(self.extent.fine_shape) \
                    or a.dtype != np.uint8:
                raise FineSpecMismatch(f"[{self.clip_sha12}] fine_codes decoded as "
                                       f"{a.shape}/{a.dtype}, header said otherwise")
            self._fine_u8 = a
        return self._fine_u8

    def check_times(self, frame_idx, frame_t_us, tol_us: float = TIME_TOL_US) -> float:
        return self.base.check_times(frame_idx, frame_t_us, tol_us)

    def check_pose_alignment(self, v2ep_poses, **kw) -> dict:
        return self.base.check_pose_alignment(v2ep_poses, **kw)

    def read(self, frame_idx, frame_t_us=None, *, tol_us: float = TIME_TOL_US
             ) -> FineMapFrames:
        """Codes ``[N, H, W]`` uint8 and seen mask ``[N, H, W]`` bool (``(H, W) =
        extent.fine_shape``).

        ``frame_idx`` are RAW v2ep frame indices (the base reader's refusals:
        negative, out-of-range, non-integer and boolean indices). When
        ``frame_t_us`` is given the 1 ms time check runs first."""
        idx = self.base._index(frame_idx)
        if frame_t_us is not None:
            self.base.check_times(idx, frame_t_us, tol_us)
        u8 = self.fine_u8()[idx]
        _check_codes(u8, self.clip_sha12, idx)
        return FineMapFrames(codes=u8, seen=u8 != NOT_SEEN_CODE,
                             t_img_us=self.t_img_us[idx].astype(np.int64),
                             frame_idx=idx)


def open_path_fine(path, clip_id: str, extent: MapExtent = EXTENT_V2) -> ClipFineMapGT:
    """Open ``path`` as ``clip_id``'s 10 cm GT at the DECLARED ``extent``.

    ``/2``: through :func:`semantic_map_gt.open_path` (its every guard), refused under
    any extent but :data:`EXTENT_V2`. ``/3``: the same identity / legend / frame-axis
    guards (:func:`_open_v3`). Then, on both, the 10 cm header and meta against the
    extent. Raises ``FileNotFoundError``, :class:`SchemaMismatch`,
    :class:`ClipIdentityMismatch`, and :class:`FineSpecMismatch` for anything
    10 cm- or extent-specific."""
    if not isinstance(extent, MapExtent):
        raise TypeError(f"extent must be a MapExtent, got {type(extent).__name__}")
    s12 = sha12(clip_id)
    path = Path(path)
    if not path.is_file():
        os.stat(path)          # raises FileNotFoundError / the real OSError
        raise SchemaMismatch(f"[{s12}] GT path exists but is not a regular file")
    try:
        headers = _G._npz_headers(path)
    except zipfile.BadZipFile as e:
        raise SchemaMismatch(f"[{s12}] not a readable .npz archive: {e}") from None
    meta = _read_meta(path, s12)
    schema = meta.get("schema")
    if schema == SCHEMA_V2:
        base = _G.open_path(path, clip_id)            # every /2 guard, one spelling
        if extent != EXTENT_V2:
            raise FineSpecMismatch(
                f"[{s12}] a {SCHEMA_V2} file carries the 60 m x +-16 m grid only; the "
                f"DECLARED extent is {float(extent.x_max_m)} m x "
                f"+-{float(extent.y_half_m)} m. Re-export it as {SCHEMA_V3} "
                f"(SPEC_REFCV7 §11.2 item 3) -- a crop or pad here would invent "
                f"'not seen' where the label was never exported.")
        _check_fine_header(headers, base.n_frames, s12, extent)
        _check_fine_meta(base.meta, s12, extent, v2=True)
        return ClipFineMapGT(base=base, extent=extent)
    if schema == SCHEMA_V3:
        base = _open_v3(path, s12, headers, meta)
        declared_by_file = extent_from_meta(meta, s12)
        if declared_by_file != extent:
            raise FineSpecMismatch(
                f"[{s12}] the file's extent is {float(declared_by_file.x_max_m)} m x "
                f"+-{float(declared_by_file.y_half_m)} m, the DECLARED extent is "
                f"{float(extent.x_max_m)} m x +-{float(extent.y_half_m)} m "
                f"(SPEC_REFCV7 §11.2 item 4: refuse, never crop or pad)")
        _check_fine_header(headers, base.n_frames, s12, extent)
        _check_fine_meta(meta, s12, extent, v2=False)
        return ClipFineMapGT(base=base, extent=extent)
    raise SchemaMismatch(f"[{s12}] schema {schema!r} not in {FINE_SCHEMAS}")


def open_clip_fine(root, clip_id: str, extent: MapExtent = EXTENT_V2) -> ClipFineMapGT:
    """:func:`open_path_fine` at the canonical ``<root>/semantic_maps/gt/`` path."""
    return open_path_fine(gt_path(root, clip_id), clip_id, extent)


# --------------------------------------------------------------------------- #
# the store (the SAME three layouts as MapGTStore, one resolver)               #
# --------------------------------------------------------------------------- #
@dataclass
class FineMapGTStore(MapGTStore):
    """:class:`perception_targets.MapGTStore` for the 10 cm codes, at a DECLARED
    extent.

    ⭐ Resolution (canonical / flat clip id / flat sha12) and the LRU are the
    parent's; only the OPEN is different, so a file the 0.5 m store resolves is
    the file this store resolves. ``max_open`` counts decompressed ``fine_codes``
    arrays (~38.6 MB each on the ``/2`` grid; x 3.1 at 100 m x +-30 m).
    """

    extent: MapExtent = field(default_factory=lambda: EXTENT_V2)

    def open(self, clip_id: str) -> ClipFineMapGT:          # noqa: A003
        s12 = sha12(clip_id)
        if s12 in self._open:
            self._order.remove(s12)
            self._order.append(s12)
            return self._open[s12]
        path = self.resolve(clip_id)
        if path is None:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, f"no SAM3 map GT for clip sha12 {s12} under "
                              f"{self.root} (fine store; the same three layouts as "
                              f"MapGTStore)", str(self.root))
        gt = open_path_fine(path, clip_id, self.extent)
        self._open[s12] = gt
        self._order.append(s12)
        while len(self._order) > int(self.max_open):
            self._open.pop(self._order.pop(0), None)
        return gt

    def frames_for_windows(self, clip_id: str, window_idx, *, n_stack: int = 3,
                           frame_t_us=None, tol_us: float = TIME_TOL_US
                           ) -> FineMapFrames:
        """10 cm labels for WINDOW (stacked-row) indices; ``+ n_stack - 1`` is
        applied by ``MapGTStore.raw_frames``, the one place a trainer may do it."""
        gt = self.open(clip_id)
        raw = self.raw_frames(window_idx, n_stack)
        return gt.read(raw, frame_t_us, tol_us=tol_us)


# --------------------------------------------------------------------------- #
# the 0.5 m target, derived -- the analytic control                            #
# --------------------------------------------------------------------------- #
def block_fraction_u8(codes: np.ndarray) -> np.ndarray:
    """``[N, 600, 320]`` codes -> ``[N, 9, 120, 64]`` uint8 = ``cart_frac``'s rule.

    Channel ``k < 8`` = ``rint(#cells of class k in the 5 x 5 block * 255 / 25)``,
    channel 8 = the same for code 255 ("not seen"). The exporter's rule, MEASURED:
    ``rint`` matched all 3,663,360 compared cells (floor missed 50,161, ceil
    54,249). ⚠️ Used ONLY as a control on the ``/2`` grid.
    """
    c = np.asarray(codes)
    if c.ndim == 2:
        c = c[None]
    if c.ndim != 3 or tuple(c.shape[1:]) != FINE_SHAPE:
        raise ValueError(f"codes must be [N, {FINE_SHAPE[0]}, {FINE_SHAPE[1]}], "
                         f"got {c.shape}")
    n = c.shape[0]
    blk = c.reshape(n, CART_SHAPE[0], BLOCK, CART_SHAPE[1], BLOCK)
    out = np.empty((n, len(CHANNELS)) + CART_SHAPE, np.uint8)
    for k in range(N_FINE_CLASSES):
        cnt = (blk == k).sum(axis=(2, 4))
        out[:, k] = np.rint(cnt * (FRACTION_SCALE / BLOCK ** 2)).astype(np.uint8)
    cnt = (blk == NOT_SEEN_CODE).sum(axis=(2, 4))
    out[:, NOT_SEEN_CHANNEL] = np.rint(cnt * (FRACTION_SCALE / BLOCK ** 2)
                                       ).astype(np.uint8)
    return out


def band_of_row(row, extent: MapExtent = EXTENT_V2) -> np.ndarray:
    """Fine row index -> band index (every 20 m from the nearest row); refuses rows
    outside the extent's grid rather than clipping them into a band."""
    r = np.asarray(row)
    h = int(extent.fine_shape[0])
    if np.any(r < 0) or np.any(r >= h):
        raise ValueError(f"fine rows must be in [0, {h - 1}]")
    return (r // BAND_ROWS_FINE).astype(np.int64)


def compare_v2_window(codes_v3: np.ndarray, extent_v3: MapExtent,
                      codes_v2: np.ndarray, cart_v3: np.ndarray | None = None,
                      cart_v2: np.ndarray | None = None) -> dict:
    """SPEC_REFCV7 §12 item 3's CONTROL: inside the old 60 m x +-16 m window the
    ``/3`` arrays must be byte-identical to the ``/2`` arrays of the same frames --
    ``fine_codes`` always, ``cart_frac`` when both are given.

    ``codes_v3`` ``[N, *extent_v3.fine_shape]``, ``codes_v2`` ``[N, 600, 320]``;
    ``cart_v3`` ``[N, 9, x/0.5, 2y/0.5]``, ``cart_v2`` ``[N, 9, 120, 64]``. The window
    is the ANCHORED one (rows from x = 0, columns ``anchor_offset`` in). Returns
    ``{"n_cells", "n_differ", "window", ...cart}``; the caller refuses any non-zero
    ``n_differ``."""
    a = np.asarray(codes_v3)
    b = np.asarray(codes_v2)
    if a.ndim == 2:
        a = a[None]
    if b.ndim == 2:
        b = b[None]
    if tuple(a.shape[1:]) != tuple(extent_v3.fine_shape):
        raise ValueError(f"codes_v3 {a.shape} is not [N, {extent_v3.fine_shape}]")
    if tuple(b.shape[1:]) != FINE_SHAPE or a.shape[0] != b.shape[0]:
        raise ValueError(f"codes_v2 {b.shape} is not [{a.shape[0]}, 600, 320]")
    r0, r1, c0, c1 = EXTENT_V2.window_in(extent_v3)
    if c0 != extent_v3.anchor_offset(FINE_CELL_M):
        raise RuntimeError("the fine window and the anchor disagree")
    w = a[:, r0:r1, c0:c1]
    out = {"n_cells": int(w.size), "n_differ": int((w != b).sum()),
           "window": [r0, r1, c0, c1]}
    if cart_v3 is not None or cart_v2 is not None:
        cv3, cv2 = np.asarray(cart_v3), np.asarray(cart_v2)
        cg = extent_v3.grid(0.5).shape
        if cv3.ndim != 4 or tuple(cv3.shape[1:]) != (len(CHANNELS),) + tuple(cg):
            raise ValueError(f"cart_v3 {cv3.shape} is not [N, 9, {cg[0]}, {cg[1]}]")
        if tuple(cv2.shape[1:]) != (len(CHANNELS),) + CART_SHAPE \
                or cv2.shape[0] != cv3.shape[0]:
            raise ValueError(f"cart_v2 {cv2.shape} is not [N, 9, 120, 64]")
        k0 = extent_v3.anchor_offset(0.5)
        cw = cv3[:, :, :CART_SHAPE[0], k0:k0 + CART_SHAPE[1]]
        out.update(n_cart_cells=int(cw.size), n_cart_differ=int((cw != cv2).sum()),
                   cart_window=[0, CART_SHAPE[0], k0, k0 + CART_SHAPE[1]])
    return out
