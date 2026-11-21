"""Reader for the SAM3 semantic-map ground truth (schema ``tanitad.sam3_map_gt/2``).

refcv6-grounded (``Project Steering/REFCV6_DESIGN_GROUNDED.md`` §2.3, §4, gate C7):
the map head is supervised by SAM3's per-cell soft class fractions on the
120 x 64 @ 0.5 m rig grid, only on cells that are at least half seen. This module
opens one clip's GT file, refuses anything that is not provably that clip's
labels on that clip's v2ep frame axis, and returns the fractions + seen mask for
requested frames. It is LABEL-ONLY: the files are non-causal (built from every
frame of a clip) and must never be read on an inference path.

File contract (``semantic_maps/gt/<clip_id>.sam3mapgt.npz``, MEASURED on the three
published clips, sha256 equal to ``SEMANTIC_MAPS_MANIFEST.json``):

    meta_json      0-d unicode  schema, grid spec, channel list, fraction_scale 255,
                                 source.clip_sha12 = sha256(clip_id)[:12]
    cart_frac      [T,9,120,64] uint8   cell fraction x 255; row 0 = x in [0, 0.5 m),
                                        col 0 = y in [-16, -15.5) m (RIGHT); +y LEFT
    t_query_us     [T] float64  linspace(t_cam[0], t_cam[-1], int(span_s * 10))
    t_img_us       [T] int64    exposure time of the camera frame the pixels came from
    cam_frame_idx  [T] int32    that camera frame's index
    T_world_rig    [T,4,4] float64 ego pose at t_img
    (polar_frac, polar48_frac, fine_codes are not read here)

Time axis = the v2ep EPISODE grid of ``stack/scripts/v2_compressed.py:_resampled``
(lines 115-121): axis 0 is the RAW v2ep frame index. A trainer's stacked row ``j``
reads raw frames ``j .. j+n_stack-1`` with the current frame LAST
(``tanitad/data/v2_dataset.py:115-117``), so its label is raw frame
``j + n_stack - 1`` -- see :func:`raw_frame_index`.

What is REFUSED (each is a subclass of :class:`SemanticMapGTError`):

* :class:`SchemaMismatch` -- wrong schema string, grid spec, channel order,
  fraction scale, or array names/shapes/dtypes;
* :class:`ClipIdentityMismatch` -- ``meta.source.clip_sha12`` is not
  ``sha256(clip_id)[:12]`` (e.g. a file renamed to another clip);
* :class:`FrameIndexError` -- a negative, out-of-range, non-integer or boolean
  frame index (numpy would silently wrap ``-1``);
* :class:`TimeMisalignment` -- caller-supplied frame timestamps differ from
  ``t_img_us`` by more than 1 ms, or v2ep poses do not line up with the GT poses.

⚠️ ``meta.source.split`` is the UPSTREAM PhysicalAI split, NOT the TanitAD eval
split: all three published clips read ``"train"`` and all three are v7.2 EVAL
clips (``tanitad/data/v72_eval_clip_digests.json``; MEASURED 2026-09-15). Eval
exclusion must never be read from this field; this reader does not expose it as
a split.

Clip ids never appear in this module's messages: errors carry sha12 only.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "SCHEMA", "CHANNELS", "N_CHANNELS", "NOT_SEEN_CHANNEL", "CART_SHAPE",
    "SEEN_SHARE_MIN", "TIME_TOL_US", "GT_SUBDIR",
    "SemanticMapGTError", "SchemaMismatch", "ClipIdentityMismatch",
    "FrameIndexError", "TimeMisalignment",
    "sha12", "gt_path", "open_clip", "open_path", "ClipMapGT", "MapFrames",
    "coverage", "episode_frame_times_us", "raw_frame_index",
]

SCHEMA = "tanitad.sam3_map_gt/2"
#: channel order of ``cart_frac`` (meta_json ``channels``); compared EXACTLY.
CHANNELS = ("seen, no map class", "drivable", "lane / road line", "crosswalk",
            "arrow / text", "non-drivable edge", "hatched area",
            "sidewalk / verge", "not seen")
N_CHANNELS = len(CHANNELS)
NOT_SEEN_CHANNEL = 8
#: the rig grid of the GT == ``tanitad.data.bev_raster.GRID_DEFAULT``
CART_SHAPE = (120, 64)
CART_SPEC = {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.5, "shape": [120, 64]}
SEEN_SHARE_MIN = 0.5
TIME_TOL_US = 1000.0
GT_SUBDIR = ("semantic_maps", "gt")
GT_SUFFIX = ".sam3mapgt.npz"
FRACTION_SCALE = 255

#: arrays this reader needs: name -> (dtype, shape with None = T)
_REQUIRED = {
    "cart_frac": (np.uint8, (None, N_CHANNELS) + CART_SHAPE),
    "t_img_us": (np.int64, (None,)),
    "t_query_us": (np.float64, (None,)),
    "cam_frame_idx": (np.int32, (None,)),
    "T_world_rig": (np.float64, (None, 4, 4)),
}


class SemanticMapGTError(ValueError):
    """Base class: the GT cannot be used as this clip's labels."""


class SchemaMismatch(SemanticMapGTError):
    """The file is not a ``tanitad.sam3_map_gt/2`` file this reader understands."""


class ClipIdentityMismatch(SemanticMapGTError):
    """The file's stored clip sha12 is not the requested clip's."""


class FrameIndexError(SemanticMapGTError, IndexError):
    """A requested frame index is not a valid raw v2ep frame of this clip."""


class TimeMisalignment(SemanticMapGTError):
    """The GT frame axis does not match the caller's episode frames."""


def sha12(clip_id: str) -> str:
    """``sha256(clip_id)[:12]`` -- the identity every GT file stores."""
    return hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:12]


def gt_path(root, clip_id: str) -> Path:
    """``<root>/semantic_maps/gt/<clip_id>.sam3mapgt.npz`` (validated tier only;
    the ``gt_flagged`` tier is excluded from training by the design, §2.3)."""
    return Path(root).joinpath(*GT_SUBDIR, f"{clip_id}{GT_SUFFIX}")


def raw_frame_index(stacked_row, n_stack: int = 3):
    """Raw v2ep frame index whose label a stacked training row carries.

    Stacked row ``j`` = raw frames ``j .. j+n_stack-1``, current frame LAST
    (``v2_dataset._decode_stacked``), so the label frame is ``j + n_stack - 1``."""
    if int(n_stack) < 1:
        raise ValueError(f"n_stack must be >= 1, got {n_stack}")
    return np.asarray(stacked_row, dtype=np.int64) + (int(n_stack) - 1)


def episode_frame_times_us(t_cam_us) -> dict:
    """The v2ep episode grid from a clip's camera timestamps [us].

    A restatement of ``stack/scripts/v2_compressed.py:_resampled`` lines 115-121
    (that function also decodes video, so it cannot be called for times alone):
    ``unit`` from the span, ``n = max(int(span / unit * 10), 4)``,
    ``t_query = linspace(t[0], t[-1], n)``, ``idx = searchsorted(t, t_query)``.
    Returns ``{t_query_us, cam_frame_idx, t_img_us}``. Refuses a clock that is not
    microseconds, because the GT's 1 ms tolerance is stated in microseconds.
    """
    t = np.asarray(t_cam_us, dtype=np.float64).reshape(-1)
    if t.size < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("camera timestamps must be >= 2 finite, strictly "
                         "increasing values")
    span = t[-1] - t[0]
    unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand
            break
    if unit != 1e6:
        raise ValueError(f"camera timestamps are not microseconds (span "
                         f"{span:.0f} resolves to unit {unit:g})")
    n = max(int(span / unit * 10.0), 4)
    t_query = np.linspace(t[0], t[-1], n)
    idx = np.searchsorted(t, t_query).clip(0, len(t) - 1)
    return {"t_query_us": t_query, "cam_frame_idx": idx.astype(np.int64),
            "t_img_us": t[idx]}


# --------------------------------------------------------------------------- #
# low-level: array headers without decompressing the payload                   #
# --------------------------------------------------------------------------- #
def _npz_headers(path: Path) -> dict:
    """name -> (shape, dtype) of every ``.npy`` member, read from the headers only."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info) as fh:
                version = np.lib.format.read_magic(fh)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fh)
                elif version == (2, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fh)
                else:
                    raise SchemaMismatch(f"{info.filename}: unsupported .npy "
                                         f"format version {version}")
            out[info.filename[:-4]] = (tuple(int(s) for s in shape), np.dtype(dtype))
    return out


def _check_arrays(headers: dict, s12: str) -> int:
    """Refuse missing / mis-shaped / mis-typed arrays; return T."""
    if "meta_json" not in headers:
        raise SchemaMismatch(f"[{s12}] no meta_json in the file")
    T = None
    for name, (dtype, shape) in _REQUIRED.items():
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
                                 f"cart_frac has {T}")
    if not T:
        raise SchemaMismatch(f"[{s12}] the file holds zero frames")
    return int(T)


def _check_meta(meta: dict, s12_requested: str) -> None:
    if not isinstance(meta, dict):
        raise SchemaMismatch(f"[{s12_requested}] meta_json is not an object")
    if meta.get("schema") != SCHEMA:
        raise SchemaMismatch(f"[{s12_requested}] schema {meta.get('schema')!r} "
                             f"!= {SCHEMA!r}")
    if meta.get("frame") != "rig":
        raise SchemaMismatch(f"[{s12_requested}] frame {meta.get('frame')!r} != 'rig'")
    cart = meta.get("cartesian") or {}
    for k, v in CART_SPEC.items():
        if cart.get(k) != v:
            raise SchemaMismatch(f"[{s12_requested}] cartesian.{k} {cart.get(k)!r} != {v!r}")
    if tuple(meta.get("channels") or ()) != CHANNELS:
        raise SchemaMismatch(f"[{s12_requested}] channel list/order differs from "
                             f"the reader's: {meta.get('channels')!r}")
    if meta.get("fraction_scale") != FRACTION_SCALE:
        raise SchemaMismatch(f"[{s12_requested}] fraction_scale "
                             f"{meta.get('fraction_scale')!r} != {FRACTION_SCALE}")
    stored = (meta.get("source") or {}).get("clip_sha12")
    if stored != s12_requested:
        raise ClipIdentityMismatch(
            f"GT file requested as clip sha12 {s12_requested} stores clip sha12 "
            f"{stored!r}: it belongs to another clip (renamed or mis-copied)")


# --------------------------------------------------------------------------- #
# the reader                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MapFrames:
    """Labels for N requested raw v2ep frames."""

    cart: np.ndarray          # [N, 9, 120, 64] float32 in [0, 1]
    seen: np.ndarray          # [N, 120, 64] bool, seen share >= 0.5
    t_img_us: np.ndarray      # [N] int64
    frame_idx: np.ndarray     # [N] int64


@dataclass
class ClipMapGT:
    """One clip's validated GT. Construct with :func:`open_clip`."""

    path: Path
    clip_sha12: str
    meta: dict
    n_frames: int
    t_img_us: np.ndarray
    t_query_us: np.ndarray
    cam_frame_idx: np.ndarray
    T_world_rig: np.ndarray
    _cart_u8: np.ndarray | None = field(default=None, repr=False)

    def cart_u8(self) -> np.ndarray:
        """The full ``cart_frac`` array (loaded once, ~14 MB for 200 frames)."""
        if self._cart_u8 is None:
            with np.load(self.path, allow_pickle=False) as z:
                a = z["cart_frac"]
            if a.shape != (self.n_frames, N_CHANNELS) + CART_SHAPE or a.dtype != np.uint8:
                raise SchemaMismatch(f"[{self.clip_sha12}] cart_frac decoded as "
                                     f"{a.shape}/{a.dtype}, header said otherwise")
            self._cart_u8 = a
        return self._cart_u8

    def _index(self, frame_idx) -> np.ndarray:
        a = np.asarray(frame_idx)
        if a.dtype == np.bool_ or not np.issubdtype(a.dtype, np.integer):
            raise FrameIndexError(f"[{self.clip_sha12}] frame indices must be "
                                  f"integers, got dtype {a.dtype}")
        a = np.atleast_1d(a).astype(np.int64)
        if a.ndim != 1:
            raise FrameIndexError(f"[{self.clip_sha12}] frame indices must be 1-D, "
                                  f"got shape {a.shape}")
        bad = (a < 0) | (a >= self.n_frames)
        if bad.any():
            raise FrameIndexError(
                f"[{self.clip_sha12}] frame index {int(a[bad][0])} outside "
                f"[0, {self.n_frames - 1}] ({int(bad.sum())} of {a.size} requested "
                f"indices out of range; negative indices are refused, not wrapped)")
        return a

    def check_times(self, frame_idx, frame_t_us, tol_us: float = TIME_TOL_US) -> float:
        """Refuse unless ``t_img_us[frame_idx]`` equals ``frame_t_us`` within
        ``tol_us``. Returns the worst absolute difference [us]."""
        idx = self._index(frame_idx)
        t = np.atleast_1d(np.asarray(frame_t_us, dtype=np.float64))
        if t.shape != idx.shape:
            raise TimeMisalignment(f"[{self.clip_sha12}] {t.size} timestamps for "
                                   f"{idx.size} frame indices")
        if not np.all(np.isfinite(t)):
            raise TimeMisalignment(f"[{self.clip_sha12}] non-finite frame timestamp")
        dt = self.t_img_us[idx].astype(np.float64) - t
        worst = int(np.argmax(np.abs(dt))) if dt.size else 0
        if dt.size and abs(dt[worst]) > float(tol_us):
            raise TimeMisalignment(
                f"[{self.clip_sha12}] GT frame {int(idx[worst])} was exposed at "
                f"t_img {int(self.t_img_us[idx[worst]])} us but the episode frame "
                f"is at {t[worst]:.0f} us: |dt| {abs(dt[worst]) / 1e3:.3f} ms > "
                f"{float(tol_us) / 1e3:.3f} ms ({int((np.abs(dt) > tol_us).sum())} "
                f"of {dt.size} frames misaligned)")
        return float(np.abs(dt).max()) if dt.size else 0.0

    def read(self, frame_idx, frame_t_us=None, *, tol_us: float = TIME_TOL_US
             ) -> MapFrames:
        """Soft fractions ``[N,9,120,64]`` float32 and seen mask ``[N,120,64]``.

        ``frame_idx`` are RAW v2ep frame indices (a scalar gives N = 1). When
        ``frame_t_us`` (the episode's frame timestamps for those indices) is
        given, the time axis is verified first (1 ms)."""
        idx = self._index(frame_idx)
        if frame_t_us is not None:
            self.check_times(idx, frame_t_us, tol_us)
        u8 = self.cart_u8()[idx]
        cart = u8.astype(np.float32) / np.float32(FRACTION_SCALE)
        # seen share = 1 - not_seen/255 >= 0.5, in exact integer arithmetic
        ns = u8[:, NOT_SEEN_CHANNEL].astype(np.int32)
        seen = 2 * (FRACTION_SCALE - ns) >= FRACTION_SCALE
        return MapFrames(cart=cart, seen=seen,
                         t_img_us=self.t_img_us[idx].astype(np.int64), frame_idx=idx)

    def check_pose_alignment(self, v2ep_poses, *, tol_m: float = 0.05,
                             shifts=(-2, -1, 1, 2)) -> dict:
        """Content check of the frame axis that needs no timestamps.

        ``v2ep_poses`` = the payload's RAW ``poses`` ``[T, 4]`` (x, y, yaw, v) --
        ``v2_compressed._resampled`` evaluates them at ``t_query``; the GT pose is
        at ``t_img``. Moving each v2ep pose by ``v * (t_img - t_query)`` along its
        heading must land on the GT translation within ``tol_m`` on every frame.
        MEASURED on the three published clips: aligned residual <= 2.3 mm (max over
        all frames), a one-frame shift >= 0.57 m (median).

        Raises :class:`TimeMisalignment` when the aligned pairing is off. Returns
        the residual report with ``verdict`` ``"ALIGNED"`` when every shifted
        pairing is clearly worse (median >= 2 * ``tol_m``), else
        ``"INCONCLUSIVE"`` -- an ego that barely moves cannot identify the axis,
        and that is reported, not passed."""
        P = np.asarray(v2ep_poses, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] < 4:
            raise TimeMisalignment(f"[{self.clip_sha12}] v2ep poses must be [T, 4], "
                                   f"got {P.shape}")
        if P.shape[0] != self.n_frames:
            raise TimeMisalignment(f"[{self.clip_sha12}] v2ep episode has "
                                   f"{P.shape[0]} frames, GT has {self.n_frames}: "
                                   f"different episode grids")
        xy_gt = self.T_world_rig[:, :2, 3]
        dt_s = (self.t_img_us.astype(np.float64) - self.t_query_us) / 1e6

        def resid(s: int) -> np.ndarray:
            i = np.arange(max(0, -s), min(self.n_frames, self.n_frames - s))
            j = i + s
            pred = P[j, :2] + (P[j, 3] * dt_s[i])[:, None] * np.stack(
                [np.cos(P[j, 2]), np.sin(P[j, 2])], axis=1)
            return np.linalg.norm(xy_gt[i] - pred, axis=1)

        r0 = resid(0)
        rep = {"n_frames": self.n_frames, "aligned_max_m": float(r0.max()),
               "aligned_median_m": float(np.median(r0)),
               "shift_median_m": {int(s): float(np.median(resid(s))) for s in shifts}}
        if not np.all(np.isfinite(r0)) or r0.max() > tol_m:
            raise TimeMisalignment(
                f"[{self.clip_sha12}] GT poses do not follow the v2ep poses on the "
                f"same frame index: max residual {r0.max():.3f} m > {tol_m} m "
                f"(median {np.median(r0):.3f} m)")
        weak = [s for s, m in rep["shift_median_m"].items() if m < 2.0 * tol_m]
        rep["verdict"] = "INCONCLUSIVE" if weak else "ALIGNED"
        rep["undiscriminated_shifts"] = weak
        return rep


def open_clip(root, clip_id: str) -> ClipMapGT:
    """Open and validate ``clip_id``'s GT under ``root``.

    Raises ``FileNotFoundError`` when there is no file, :class:`SchemaMismatch` /
    :class:`ClipIdentityMismatch` when there is one that is not this clip's
    ``tanitad.sam3_map_gt/2`` labels, and lets I/O errors propagate unchanged."""
    return open_path(gt_path(root, clip_id), clip_id)


def open_path(path, clip_id: str) -> ClipMapGT:
    """:func:`open_clip` for an EXPLICIT file path.

    Added 2026-09-16 (refcv6 perception) so that a non-canonical layout -- the
    dev-box copy of the eval maps is a FLAT directory named by ``sha12``, not
    ``<root>/semantic_maps/gt/<clip_id>...`` -- reaches exactly this validation
    and no second copy of it. :func:`open_clip` is now this function plus
    :func:`gt_path`; the schema, identity and frame-axis checks are unchanged
    and have one spelling.
    """
    s12 = sha12(clip_id)
    path = Path(path)
    if not path.is_file():
        os.stat(path)          # raises FileNotFoundError / the real OSError
        raise SchemaMismatch(f"[{s12}] GT path exists but is not a regular file")
    try:
        headers = _npz_headers(path)
    except zipfile.BadZipFile as e:
        raise SchemaMismatch(f"[{s12}] not a readable .npz archive: {e}") from None
    T = _check_arrays(headers, s12)
    with np.load(path, allow_pickle=False) as z:
        try:
            meta = json.loads(str(z["meta_json"]))
        except (ValueError, TypeError) as e:
            raise SchemaMismatch(f"[{s12}] meta_json does not parse: {e}") from None
        _check_meta(meta, s12)
        t_img = z["t_img_us"]
        t_query = z["t_query_us"]
        cam_idx = z["cam_frame_idx"]
        pose = z["T_world_rig"]
    if np.any(np.diff(t_img) < 0) or not np.all(np.isfinite(t_query)):
        raise SchemaMismatch(f"[{s12}] t_img_us is not non-decreasing / t_query_us "
                             f"not finite: the frame axis is corrupt")
    return ClipMapGT(path=path, clip_sha12=s12, meta=meta, n_frames=T,
                     t_img_us=t_img, t_query_us=t_query, cam_frame_idx=cam_idx,
                     T_world_rig=pose)


def coverage(clip_ids, root) -> dict:
    """Fraction of ``clip_ids`` with a readable GT file under ``root``.

    Three states per clip, never two:

    * ``readable``     -- :func:`open_clip` validates it (headers + meta + identity;
      the fraction arrays are not decompressed);
    * ``absent``       -- the GT directory is a readable directory AND the clip's
      file stat reports ``ENOENT`` (two probes: a dead mount or a wrong root can
      not masquerade as a missing map);
    * ``inconclusive`` -- anything else: any I/O error, a corrupt archive, a wrong
      schema, a file that belongs to another clip, or an unreadable GT directory.

    ``frac_readable`` is exact only when ``n_inconclusive == 0``; otherwise it is a
    lower bound and ``frac_readable_upper`` the upper bound, and ``verdict`` is
    ``"INCONCLUSIVE"``. Per-clip states are keyed by sha12 (clip ids stay out of
    run records).
    """
    ids = [str(c) for c in clip_ids]
    uniq = list(dict.fromkeys(ids))
    gt_dir = Path(root).joinpath(*GT_SUBDIR)
    dir_ok, dir_reason = True, ""
    try:
        if not os.path.isdir(gt_dir):
            os.stat(gt_dir)
            dir_ok, dir_reason = False, "GT path is not a directory"
        else:
            with os.scandir(gt_dir) as it:      # probe that it is listable
                next(it, None)
    except OSError as e:
        dir_ok, dir_reason = False, f"GT directory unreadable: {type(e).__name__} errno {e.errno}"
    states: dict = {}
    for cid in uniq:
        s12 = sha12(cid)
        if not dir_ok:
            states[s12] = ("inconclusive", dir_reason)
            continue
        try:
            open_clip(root, cid)
            states[s12] = ("readable", "")
        except FileNotFoundError as e:
            if e.errno == errno.ENOENT:
                states[s12] = ("absent", "no GT file")
            else:
                states[s12] = ("inconclusive", f"FileNotFoundError errno {e.errno}")
        except Exception as e:  # noqa: BLE001 -- every other failure is INCONCLUSIVE
            msg = str(e).replace(cid, "<clip>")
            states[s12] = ("inconclusive", f"{type(e).__name__}: {msg[:200]}")
    n = len(uniq)
    cnt = {k: sum(1 for s, _ in states.values() if s == k)
           for k in ("readable", "absent", "inconclusive")}
    return {
        "n": n, "n_duplicates_ignored": len(ids) - n,
        "n_readable": cnt["readable"], "n_absent": cnt["absent"],
        "n_inconclusive": cnt["inconclusive"],
        "frac_readable": cnt["readable"] / n if n else math.nan,
        "frac_readable_upper": (cnt["readable"] + cnt["inconclusive"]) / n if n else math.nan,
        "verdict": "INCONCLUSIVE" if cnt["inconclusive"] or not n else "COMPLETE",
        "states": {s: {"state": st, "reason": r} for s, (st, r) in states.items()},
    }
