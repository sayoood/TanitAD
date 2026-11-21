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
    # the 3-D join seam (2026-09-17) -- see the section at the end of the file
    "JOIN3D_ENV", "CZ_KEY", "H_KEY", "AgentJoin3D", "open_join3d",
    "zh_for_frame", "base_from_centre",
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


# ===========================================================================
# THE 3-D JOIN SEAM (2026-09-17) -- prefer the join, fall back to the mask
# ===========================================================================
# ⛔ WHAT CHANGED, AND WHAT DID NOT.
#
# Until 2026-09-17 the eval join carried no z and no h, so ``box3d_head``
# masked both terms: ``zh_mask`` all-False, ``n_z == 0``, never zero-filled.
# That was the CORRECT behaviour for a missing label and it is still the
# behaviour here whenever the 3-D join is absent. What is new is that the join
# can now be present:
#
#   ``stack/scripts/build_b1_agent_join_3d.py`` annotates the banked 2-D join
#   with ``cz`` (cuboid CENTRE height, rig frame, z = 0 the road plane) and
#   ``h`` (``size_z``) per agent, keeping the key ``(clip_id, frame)``, the RAW
#   v2ep frame index space, the agent order and every existing byte.
#
# ⚠️ ``cz`` IS THE CENTRE. The base is ``cz - h/2``. MEASURED 2026-09-17 over
# 905,512 annotated agents, the ground-standing vehicle base median is
# -0.0146 m against a LiDAR-measured road plane -- while the rival reading
# ("cz is already the base") puts those same vehicles 0.789 m in the air. A
# consumer that subtracts ``h/2`` from an already-base z lands 0.77 m UNDER the
# road, which is the defect ``RETRACTION_LOG.md:15278`` records.
#
# ⛔ ABSENCE IS A STATE, NOT A ZERO. :func:`zh_for_frame` returns an all-False
# mask and zero VALUES; the values are meaningless and the mask says so. They
# are zeros only because an array needs a fill -- never because 0 m is a
# plausible height.

#: env var naming the 3-D join, so no absolute path is ever written in code.
JOIN3D_ENV = "TANITAD_AGENT_JOIN3D"

#: the two keys the 3-D builder appends to each agent dict.
CZ_KEY = "cz"
H_KEY = "h"


class AgentJoin3D:
    """``(clip_id, frame) -> {track_id: (cz, h)}`` read from a 3-D agent join.

    Keyed on the join's OWN key. The 3-D artifact is the 2-D artifact plus two
    fields per agent (byte-identical once stripped), so a reader that resolves
    ``(clip_id, frame)`` here and in the 2-D join resolves the same rows by
    construction -- there is no second index space to get wrong.

    Memory: the eval join is 26,394 lines / 905,512 agents and the track-id
    strings dominate. They are ``sys.intern``-ed, so the cost is one string per
    TRACK, not one per agent. Pass ``clips=`` to index a subset.
    """

    __slots__ = ("path", "_idx", "n_lines", "n_agents", "n_clips")

    def __init__(self, path, index, n_lines: int, n_agents: int):
        self.path = str(path)
        self._idx = index
        self.n_lines = int(n_lines)
        self.n_agents = int(n_agents)
        self.n_clips = len({k[0] for k in index})

    @classmethod
    def open(cls, path, clips=None) -> "AgentJoin3D":
        """Index a ``.jsonl`` / ``.jsonl.xz`` 3-D join.

        Refuses a file whose agents carry no ``cz``/``h`` at all -- that is the
        2-D join under a 3-D name, and silently indexing it would hand every
        consumer an all-False mask while REPORTING that a 3-D join was found.
        """
        import json as _json
        import lzma as _lzma
        import sys as _sys

        p = Path(path)
        want = None if clips is None else set(clips)
        opener = (_lzma.open if str(p).endswith(".xz") else open)
        idx: dict = {}
        n_lines = n_agents = n_zh = 0
        with opener(str(p), "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = _json.loads(line)
                cid = rec["clip_id"]
                if want is not None and cid not in want:
                    continue
                n_lines += 1
                per: dict = {}
                for a in rec["agents"]:
                    n_agents += 1
                    if CZ_KEY in a and H_KEY in a:
                        per[_sys.intern(str(a["track_id"]))] = (
                            float(a[CZ_KEY]), float(a[H_KEY]))
                        n_zh += 1
                idx[(_sys.intern(str(cid)), int(rec["frame"]))] = per
        if n_lines and n_zh == 0:
            raise CuboidGTError(
                "%s indexed %d lines and %d agents but NOT ONE carries %r/%r "
                "-- that is a 2-D join. Refusing rather than reporting a 3-D "
                "join that would mask every target it was opened to supply."
                % (p.name, n_lines, n_agents, CZ_KEY, H_KEY))
        return cls(p, idx, n_lines, n_agents)

    def has(self, clip_id: str, frame: int) -> bool:
        return (str(clip_id), int(frame)) in self._idx

    def zh(self, clip_id: str, frame: int, track_ids):
        """``(cz, h, mask)`` as float64/bool arrays aligned to ``track_ids``.

        A track absent from the line is masked OFF, not zeroed into the loss.
        """
        per = self._idx.get((str(clip_id), int(frame)))
        ids = [str(t) for t in track_ids]
        cz = np.zeros(len(ids), dtype=np.float64)
        h = np.zeros(len(ids), dtype=np.float64)
        m = np.zeros(len(ids), dtype=bool)
        if per is None:
            return cz, h, m
        for k, t in enumerate(ids):
            hit = per.get(t)
            if hit is not None:
                cz[k], h[k] = hit
                m[k] = True
        return cz, h, m

    def __len__(self) -> int:
        return len(self._idx)

    def __repr__(self) -> str:                               # pragma: no cover
        return ("AgentJoin3D(%r, clips=%d, lines=%d, agents=%d)"
                % (Path(self.path).name, self.n_clips, self.n_lines,
                   self.n_agents))


def open_join3d(path=None, clips=None):
    """The 3-D join, or ``None`` -- the "prefer it when present" half.

    ``path`` None reads :data:`JOIN3D_ENV`. An UNSET env var and a path that
    does not exist both return ``None`` (the caller then runs masked); a path
    that exists but cannot be read RAISES, because a configured-and-broken
    label store is a defect, not an absence.
    """
    import os

    p = path if path is not None else os.environ.get(JOIN3D_ENV)
    if not p:
        return None
    if not Path(p).exists():
        return None
    return AgentJoin3D.open(p, clips=clips)


def zh_for_frame(clip_id: str, frame: int, track_ids, join3d=None):
    """``(cz, h, mask)`` for one frame's agents -- the ONE call a loader makes.

    * ``join3d`` present AND carrying this ``(clip_id, frame)`` -> the join's
      ``cz``/``h``, masked per track;
    * otherwise -> zeros with an ALL-FALSE mask, i.e. exactly the behaviour the
      perception branch has had since the 3-D head was built.

    The two paths differ ONLY in the mask, so a consumer that honours the mask
    (``box3d_head.zh_targets`` and ``box3d_set_loss`` do) needs no branch of its
    own and cannot accidentally train on the fallback's zeros.
    """
    ids = list(track_ids)
    if join3d is None:
        z = np.zeros(len(ids), dtype=np.float64)
        return z, z.copy(), np.zeros(len(ids), dtype=bool)
    return join3d.zh(clip_id, frame, ids)


def base_from_centre(cz, h):
    """The bottom face: ``cz - h/2``. One spelling, so no call site chooses.

    MEASURED 2026-09-17 (905,512 agents): applying this to an ALREADY-base z
    lands 0.77 m under the road -- ``RETRACTION_LOG.md:15278``.
    """
    return (np.asarray(cz, dtype=np.float64)
            - np.asarray(h, dtype=np.float64) / 2.0)
