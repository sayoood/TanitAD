"""refcv6 §6: the two columns the agent join dropped -- ``center_z`` and ``size_z``.

``SPEC_REFCV6_V2.md`` §6 asks the BOX head for ``(x, y, z, l, w, h, yaw)`` from
``obstacle.offline`` cuboids. **MEASURED 2026-09-16**, the built eval join
(``TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-06-b1-agent-join/``)
emits per agent exactly::

    {"cx", "cy", "yaw", "l", "w", "occ", "track_id", "cls"}

-- no ``z``, no ``h``. The cuboids themselves do carry both: the parquet schema
of ``labels/obstacle.offline`` (and its ``obstacle_offline_b1eval`` /
``_b1train`` mirrors) is ``timestamp_us, source, track_id, center_x, center_y,
center_z, size_x, size_y, size_z, orientation_{x,y,z,w}, label_class,
reference_frame, reference_frame_timestamp_us``. This module recovers the two
missing columns WITHOUT rebuilding the join.

## Why recovering them is exact, and not a second geometry

``build_b1_agent_join.py:410-415`` takes the rig-frame agents at a frame time
and applies ``bev_raster.ego_frame_agents``, which is a **planar SE(2)**
transform: it writes columns 0 (x), 1 (y) and 2 (yaw) and *"sizes and the occ
column pass through"* (``bev_raster.py:202-224``). ``center_z`` and ``size_z``
are therefore **invariant** under every step between the parquet and the join
line, so joining them back by ``(clip, frame time, track_id)`` re-derives no
geometry at all -- it recovers columns that were only ever dropped.

## The duplication, and its proof

:func:`cuboids_at_time` repeats the per-track nearest-sample selection of
``bev_raster.agents_at_time`` (``bev_raster.py:227-268``) because that function
returns a ``[A, 6]`` array and no indices, so the extra columns cannot be
attached to its output. The repetition is admissible only WITH the equivalence
proof -- the rule ``agent_slots.hungarian`` follows against scipy:
``tests/test_refcv6_cuboid_gt.py::test_first_six_columns_are_agents_at_time``
asserts columns 0..5 are **bit-identical** to ``agents_at_time``'s on real
``obstacle_offline_b1eval`` parquets, and FAILS if either implementation drifts.

## What is refused

* a parquet whose ``reference_frame`` is not ``rig`` on every row -- the whole
  invariance argument above is a statement about the rig frame
  (:class:`CuboidFrameMismatch`);
* a missing ``center_z``/``size_z`` column (:class:`CuboidSchemaMismatch`) --
  a 2-D parquet must be reported, never silently ground-standing-imputed;
* a non-positive ``size_z`` (:class:`CuboidSchemaMismatch`).

Clip ids never appear in this module's messages: sha12 only
(``semantic_map_gt.sha12``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tanitad.data.bev_raster import DEFAULT_TOL_S, agents_at_time, yaw_from_quaternion
from tanitad.data.semantic_map_gt import sha12

__all__ = [
    "CUBOID_COLUMNS", "Z_COLUMNS", "CuboidGTError", "CuboidSchemaMismatch",
    "CuboidFrameMismatch", "cuboids_at_time", "read_clip_cuboids",
    "zh_by_track", "ground_bottom_stats",
]

#: the columns :func:`cuboids_at_time` needs, in the parquet's own spelling.
CUBOID_COLUMNS: tuple[str, ...] = (
    "timestamp_us", "track_id", "center_x", "center_y", "center_z",
    "size_x", "size_y", "size_z",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
)
#: the two the join dropped.
Z_COLUMNS: tuple[str, ...] = ("center_z", "size_z")
#: emitted column order of :func:`cuboids_at_time`: the join's six, then z, h.
OUT_COLUMNS: tuple[str, ...] = ("cx", "cy", "yaw", "l", "w", "occ", "cz", "h")


class CuboidGTError(ValueError):
    """Base class: these cuboids cannot be used as 3-D targets."""


class CuboidSchemaMismatch(CuboidGTError):
    """The parquet is not a 3-D ``obstacle.offline`` table this reader trusts."""


class CuboidFrameMismatch(CuboidGTError):
    """``reference_frame`` is not ``rig``, so z is not a rig-frame height."""


def _col(obs, name: str):
    try:
        return obs[name]
    except (KeyError, IndexError, TypeError):
        return None


def check_cuboid_schema(obs, *, clip_id: str | None = None) -> None:
    """Refuse a table that cannot carry 3-D targets. ``obs``: a pandas
    DataFrame or a dict of arrays with :data:`CUBOID_COLUMNS`."""
    tag = f"[{sha12(clip_id)}] " if clip_id is not None else ""
    for name in CUBOID_COLUMNS:
        if _col(obs, name) is None:
            raise CuboidSchemaMismatch(
                f"{tag}column {name!r} is missing. ⛔ A 2-D obstacle table must "
                f"be REPORTED, not imputed: a ground-standing z is a "
                f"label-derived constant (box3d_head.ground_standing_z), not a "
                f"target.")
    ref = _col(obs, "reference_frame")
    if ref is not None:
        u = set(np.asarray(ref).astype(str).tolist())
        if u and u != {"rig"}:
            raise CuboidFrameMismatch(
                f"{tag}reference_frame = {sorted(u)} -- this reader's whole "
                f"argument that z survives the join is a statement about the "
                f"RIG frame (module docstring)")
    sz = np.asarray(_col(obs, "size_z"), dtype=np.float64)
    if sz.size and (not np.all(np.isfinite(sz)) or np.any(sz <= 0)):
        n = int((~np.isfinite(sz)).sum() + (sz <= 0).sum())
        raise CuboidSchemaMismatch(
            f"{tag}{n} of {sz.size} cuboids have a non-positive or non-finite "
            f"size_z: a height is positive by construction")


def cuboids_at_time(obs, t_s: float, tol_s: float = DEFAULT_TOL_S,
                    classes: tuple[str, ...] | None = None, *,
                    clip_id: str | None = None,
                    check_schema: bool = True) -> np.ndarray:
    """Rig-frame cuboids ``[A, 8]`` = ``(cx, cy, yaw, l, w, occ, cz, h)`` at ``t_s``.

    ⭐ Columns 0..5 are ``bev_raster.agents_at_time``'s, value for value, with
    the SAME per-track nearest-sample policy and the same ``tol_s``; columns 6
    and 7 are ``center_z`` and ``size_z`` of that same chosen sample. ``occ`` is
    ``-1`` throughout, because ``obstacle.offline`` carries no occlusion flag
    (``agents_at_time``'s own docstring) -- the visible/occluded split comes
    from the join, and this reader does not invent one.
    """
    if check_schema:
        check_cuboid_schema(obs, clip_id=clip_id)
    ts = np.asarray(obs["timestamp_us"], dtype=np.float64) / 1e6
    if ts.size == 0:
        return np.zeros((0, 8), dtype=np.float64)
    tid = np.asarray(obs["track_id"]).astype(str)
    cx = np.asarray(obs["center_x"], dtype=np.float64)
    cy = np.asarray(obs["center_y"], dtype=np.float64)
    cz = np.asarray(obs["center_z"], dtype=np.float64)
    sx = np.asarray(obs["size_x"], dtype=np.float64)
    sy = np.asarray(obs["size_y"], dtype=np.float64)
    sz = np.asarray(obs["size_z"], dtype=np.float64)
    yaw = yaw_from_quaternion(obs["orientation_x"], obs["orientation_y"],
                              obs["orientation_z"], obs["orientation_w"])
    cls = (np.asarray(obs["label_class"]).astype(str)
           if classes is not None else None)
    rows = []
    for track in np.unique(tid):                  # bev_raster.py:256 -- sorted
        m = tid == track
        if cls is not None and cls[m][0] not in classes:
            continue
        dt = np.abs(ts[m] - float(t_s))
        j = int(np.argmin(dt))
        if dt[j] > tol_s:
            continue
        rows.append((cx[m][j], cy[m][j], yaw[m][j], sx[m][j], sy[m][j], -1.0,
                     cz[m][j], sz[m][j]))
    if not rows:
        return np.zeros((0, 8), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def track_ids_at_time(obs, t_s: float, tol_s: float = DEFAULT_TOL_S,
                      classes: tuple[str, ...] | None = None) -> list[str]:
    """The track ids :func:`cuboids_at_time` emits, in its row order -- the key
    a caller joins z/h onto the join's own agents by."""
    ts = np.asarray(obs["timestamp_us"], dtype=np.float64) / 1e6
    if ts.size == 0:
        return []
    tid = np.asarray(obs["track_id"]).astype(str)
    cls = (np.asarray(obs["label_class"]).astype(str)
           if classes is not None else None)
    out = []
    for track in np.unique(tid):
        m = tid == track
        if cls is not None and cls[m][0] not in classes:
            continue
        dt = np.abs(ts[m] - float(t_s))
        if dt[int(np.argmin(dt))] <= tol_s:
            out.append(str(track))
    return out


def zh_by_track(obs, t_s: float, tol_s: float = DEFAULT_TOL_S) -> dict:
    """``{track_id: (cz, h)}`` at ``t_s`` -- the lookup a loader uses to attach
    3-D targets to the join's agents, which carry ``track_id``."""
    cub = cuboids_at_time(obs, t_s, tol_s=tol_s, check_schema=False)
    ids = track_ids_at_time(obs, t_s, tol_s=tol_s)
    if len(ids) != cub.shape[0]:                   # unreachable by construction
        raise CuboidGTError(f"track-id list ({len(ids)}) and cuboid rows "
                            f"({cub.shape[0]}) disagree")
    return {t: (float(c[6]), float(c[7])) for t, c in zip(ids, cub)}


def read_clip_cuboids(path) -> dict:
    """One clip's obstacle parquet as a dict of numpy arrays (the duck-typed
    shape ``agents_at_time`` and :func:`cuboids_at_time` both take).

    ``pyarrow`` only -- pandas is not needed to read columns, and the join
    builder already depends on pyarrow through ``read_parquet``.
    """
    import pyarrow.parquet as pq
    p = Path(path)
    want = list(CUBOID_COLUMNS) + ["label_class", "reference_frame"]
    have = set(pq.ParquetFile(p).schema.names)
    cols = [c for c in want if c in have]
    tbl = pq.read_table(p, columns=cols)
    return {c: np.asarray(tbl[c].to_pylist()) if c in ("track_id", "label_class",
                                                       "reference_frame")
            else np.asarray(tbl[c], dtype=np.float64) for c in cols}


def ground_bottom_stats(obs) -> dict:
    """Bottom-face height statistics ``center_z - size_z/2`` per class.

    The evidence behind :func:`box3d_head.ground_standing_z`'s prior and behind
    :data:`box3d_head.Z_RANGE_M`; reported per class because ``protruding_object``
    is airborne by definition and pooling it with cars would move the median.
    """
    cz = np.asarray(obs["center_z"], dtype=np.float64)
    sz = np.asarray(obs["size_z"], dtype=np.float64)
    bottom = cz - sz / 2.0
    cls = np.asarray(obs.get("label_class", np.full(cz.shape, "?"))).astype(str)
    out = {"n": int(cz.size), "per_class": {}}
    if cz.size == 0:
        return out
    for c in sorted(set(cls.tolist())):
        m = cls == c
        out["per_class"][c] = {
            "n": int(m.sum()),
            "bottom_median_m": float(np.median(bottom[m])),
            "bottom_p10_m": float(np.percentile(bottom[m], 10)),
            "bottom_p90_m": float(np.percentile(bottom[m], 90)),
            "h_median_m": float(np.median(sz[m])),
        }
    out["bottom_median_m"] = float(np.median(bottom))
    return out


def _equivalence_probe(obs, t_s: float, tol_s: float = DEFAULT_TOL_S) -> dict:
    """The equivalence the module docstring promises, as data: max |delta| of
    columns 0..5 against ``bev_raster.agents_at_time`` at the same ``t_s``."""
    a = agents_at_time(obs, t_s, tol_s=tol_s)
    c = cuboids_at_time(obs, t_s, tol_s=tol_s, check_schema=False)
    if a.shape[0] != c.shape[0]:
        return {"same_rows": False, "n_a": int(a.shape[0]), "n_c": int(c.shape[0]),
                "max_abs_delta": float("inf")}
    d = float(np.abs(a - c[:, :6]).max()) if a.size else 0.0
    return {"same_rows": True, "n_a": int(a.shape[0]), "n_c": int(c.shape[0]),
            "max_abs_delta": d, "bit_identical": bool(a.size == 0 or
                                                      np.array_equal(a, c[:, :6]))}
