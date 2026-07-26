"""Argoverse 2 adapter — LANE-GRAPH-FIRST, image-optional (2026-07-26).

Sibling of :mod:`tanitad.data.nuscenes`. Built after the PI-approved metadata pull
of **all 1 000 AV2 *sensor* lane graphs** (train 700 / val 150 / test 150),
verified 1000/1000 byte-exact and MD5-exact against the bucket's own ETags.

WHY THIS EXISTS
---------------
AV2 is the program's only **credential-free** corpus with a real routable lane
graph: ``s3://argoverse`` serves unsigned anonymous GETs, and the licensor's own
instruction is ``s5cmd --no-sign-request``. There is no account to create, no
checkbox for a human to tick, no token. That is why the strategic-brain ground
truth (branch selection S1, lane selection S2, junction topology HP-4) can be
built here without waiting on anybody.

⚠️ LICENSE — READ BEFORE USING THIS MODULE
------------------------------------------
AV2 data is **CC BY-NC-SA 4.0** (``nc-research``, ``share_alike=True`` in
``lake.schema.SOURCE_REGISTRY``) — the *same* class as nuScenes. So:

* it can **NEVER** enter TanitDataSet-C or any commercial artifact;
* it is **copyleft** — records route to the segregated shard
  ``shards/nc-research/sharealike/argoverse2/…`` and must never co-mingle;
* **derivatives inherit it**: labels or weights built on AV2 are themselves NC+SA;
* we ship **pointers + derived features, never source bytes**.

⚠️ The registry entry is a **FLOOR, not a ceiling.** Argoverse's Terms of Use add
obligations that no CC short name carries: a trademark/marketing restriction, a
ban on re-identifying individuals (incl. by combining with another dataset), and
a clause letting the licensor decline or terminate a licence. Read
``argoverse.org/about.html``, not the short name.

⚠️ **There is deliberately NO terms-gate guard in this module.** ``nuscenes.py``
raises ``NuScenesTermsError`` because a human must register and accept before any
byte is served. AV2 has **no access-control gate** — downloading accepts its
browse-wrap terms in the ordinary way. Copying that guard here would be a **lie in
the code**. What AV2 needs instead is an honest *"the maps are not on disk"*
error, which is what :class:`Argoverse2MapError` is.

THE THREE TRAPS — each MEASURED over the full 1 000-log pull, each costs a day
-----------------------------------------------------------------------------
1. **SENSOR maps carry NO ``centerline`` field.** MEASURED: **0 / 163 698** lane
   segments across all 1 000 sensor logs have one; the motion-forecasting split
   has it on 100 %. An adapter that does ``seg["centerline"]`` works perfectly on
   motion-forecasting and **fails on every log that has images**. This module
   therefore *always* goes through :meth:`LaneGraph.centerline`, which uses the
   explicit polyline when present and otherwise **derives** the midpoint line —
   and reports which of the two it did via ``centerline_source``.
2. **Left and right boundaries have DIFFERENT lengths on 49.0 %** of segments
   (MEASURED, n=19 713). A naive elementwise mean of the two boundaries is
   therefore wrong roughly half the time. :func:`midpoint_line` arc-length
   **resamples** both polylines onto a common parameterisation first.
3. **8.4 % of successor ids point OUTSIDE the local map** (MEASURED: 16 141 /
   191 770 edges). This is the log-local crop boundary, not corruption. Traversal
   treats a dangling successor as **terminal**, never as an error — and the
   dangling ids stay inspectable via :meth:`LaneGraph.dangling_successors`.

Plus one naming rule the devkit states verbatim — lane ids are *"guaranteed to be
unique only within this local map"*. Never build a cross-log index on a raw id;
use :func:`global_lane_key`.

MEASURED corpus facts (all 1 000 sensor logs, 2026-07-26; raw JSON in
``…/incoming/2026-07-26-av2-zod-ingest/evidence/av2_pull_summary_1000.json``)
---------------------------------------------------------------------------
  lane segments            163 698        successor edges          191 770
  branch points             22 606        intersection segments     57 415
  logs with >=1 branch     998 / 1000     logs with >=1 intersection 997 / 1000
  left / right neighbour   104 924 / 44 788
  cities                   MIA 354 · PIT 350 · WDC 126 · DTW 117 · ATX 31 · PAO 22
  total bytes              161 255 215 (153.8 MiB)

NO NEW DEPENDENCY
-----------------
Map archives are plain JSON, read with the stdlib + numpy. The per-log pose and
calibration tables are Apache **feather**, read through pandas/pyarrow, which the
stack already carries — and only when a caller actually asks for them, so the
lane-graph half of this module imports nothing beyond numpy.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from tanitad.data._contract import finite_diff_accel

# Geometry helpers are IDENTICAL between the two corpora — both store rotations as
# [w, x, y, z] unit quaternions in a right-handed frame — so they are imported,
# not re-derived. Divergent copies of a quaternion convention is exactly how a
# silent sign error gets into a pose pipeline.
from tanitad.data.nuscenes import quat_to_rotmat, quat_to_yaw, wrap_pi

__all__ = [
    "Argoverse2MapError", "LaneSegment", "LaneGraph", "load_lane_graph",
    "lane_graph_from_dict", "interp_arc", "midpoint_line", "global_lane_key",
    "ego_track", "actions_from_track", "camera_intrinsics_of", "discover_logs",
    "split_unit_of", "map_archive_path", "lane_graph_stats",
    "quat_to_rotmat", "quat_to_yaw", "wrap_pi",
    "RING_CAMERAS", "STEREO_CAMERAS", "CAMERAS", "EGO_CAMERA", "CITIES",
    "SENSOR_HZ", "NS",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
#: The 7-camera ring. A *superset* of nuScenes' 6 — strictly better for the
#: camera-attention workstream. Front-center is our ego view.
RING_CAMERAS = ("ring_front_center", "ring_front_left", "ring_front_right",
                "ring_side_left", "ring_side_right",
                "ring_rear_left", "ring_rear_right")
STEREO_CAMERAS = ("stereo_front_left", "stereo_front_right")
CAMERAS = RING_CAMERAS + STEREO_CAMERAS
EGO_CAMERA = "ring_front_center"

#: MEASURED over all 1 000 sensor logs (not a published list).
CITIES = ("ATX", "DTW", "MIA", "PAO", "PIT", "WDC")

#: AV2 sensor imagery is 20 Hz (vs nuScenes' 2 Hz keyframes) — this removes the
#: keyframe-interpolation hack entirely. Used ONLY as a degenerate-dt fallback.
SENSOR_HZ = 20.0

#: AV2 stores timestamps in NANOseconds (nuScenes uses microseconds).
NS = 1e-9

#: The complete set of top-level layers, MEASURED on 1000/1000 archives.
MAP_LAYERS = ("drivable_areas", "lane_segments", "pedestrian_crossings")

#: `log_map_archive_<logid>____<CITY>_city_<n>.json`
_ARCHIVE_RE = re.compile(r"log_map_archive_(?P<log>[0-9a-f\-]+)"
                         r"(?:____(?P<city>[A-Z]+)_city_(?P<n>\d+))?", re.I)


class Argoverse2MapError(RuntimeError):
    """Raised when an AV2 map archive is absent or structurally unusable.

    Deliberately **not** a terms/credential error: AV2 has no access gate. The
    message says how to fetch the bytes, because an agent legitimately can.
    """


_ACQUIRE_MSG = """\
Argoverse 2 map archive not found at {path!r}.

Unlike nuScenes, AV2 needs NO account, NO token and NO Terms click — the bucket
serves unsigned anonymous GETs and the licensor's own documented command is:

    s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/*" <TARGET>

For the LANE GRAPHS ONLY (~154 MiB for all 1 000 sensor logs, no imagery), the
archives are individually addressable and need no tar:

    datasets/av2/sensor/{{train,val,test}}/<log_id>/map/log_map_archive_*.json

Reproducible puller (anonymous, verifies size + MD5 + parse on every file):
    TanitAD Research Hub/Data Engineering/Implementation/incoming/
      2026-07-26-av2-zod-ingest/evidence/av2_pull_sensor_lane_graphs.py

DEV-BOX TRAP: bare curl here fails with CRYPT_E_NO_REVOCATION_CHECK and reports
HTTP=000, which is indistinguishable from an outage. Pass --ssl-no-revoke.

License once acquired: CC BY-NC-SA 4.0 -> nc-research + share_alike -> NEVER in
TanitDataSet-C, segregated copyleft shard only, derivatives inherit NC+SA.
"""


# --------------------------------------------------------------------------- #
# Polyline geometry — TRAP 2 lives here                                        #
# --------------------------------------------------------------------------- #
def _pts_to_array(pts: Sequence[Any]) -> np.ndarray:
    """AV2 ships polylines as ``[{"x":..,"y":..,"z":..}, ...]`` -> ``[N,3]``.

    Also accepts an already-arrayed ``[N,2]``/``[N,3]`` so fixtures and derived
    polylines round-trip through the same code path.
    """
    if isinstance(pts, np.ndarray):
        a = np.asarray(pts, dtype=np.float64)
    elif len(pts) and isinstance(pts[0], dict):
        a = np.array([[float(p.get("x", 0.0)), float(p.get("y", 0.0)),
                       float(p.get("z", 0.0))] for p in pts], dtype=np.float64)
    else:
        a = np.asarray(pts, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if a.shape[1] == 2:                      # pad a 2-D polyline to 3-D
        a = np.concatenate([a, np.zeros((a.shape[0], 1))], axis=1)
    return a[:, :3]


def interp_arc(pts: Sequence[Any], n: int) -> np.ndarray:
    """Resample a polyline to exactly ``n`` points, uniform in ARC LENGTH.

    Arc length, not index: AV2 boundary vertices are not equally spaced, so index
    interpolation would bunch the derived centerline toward the denser boundary
    and bias every downstream heading.
    """
    a = _pts_to_array(pts)
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if a.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float64)
    if a.shape[0] == 1:
        return np.repeat(a, n, axis=0)
    seg = np.linalg.norm(np.diff(a, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:                            # degenerate: all points coincident
        return np.repeat(a[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(target, s, a[:, d]) for d in range(3)], axis=1)


def midpoint_line(left: Sequence[Any], right: Sequence[Any],
                  n: int | None = None) -> np.ndarray:
    """DERIVE a lane centerline from its two boundaries -> ``[n,3]``.

    ⚠️ TRAP 2: the two boundaries have **different lengths on 49.0 %** of AV2
    segments (MEASURED, n=19 713), so both are arc-length resampled onto a common
    parameterisation *before* averaging. ``n`` defaults to the longer boundary, so
    no resolution is thrown away.

    This is what makes the SENSOR split usable at all — see TRAP 1.
    """
    l = _pts_to_array(left)
    r = _pts_to_array(right)
    if l.shape[0] == 0 and r.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if l.shape[0] == 0:
        return interp_arc(r, n or r.shape[0])
    if r.shape[0] == 0:
        return interp_arc(l, n or l.shape[0])
    n = int(n) if n else max(l.shape[0], r.shape[0])
    return 0.5 * (interp_arc(l, n) + interp_arc(r, n))


def global_lane_key(log_id: str, lane_id: int | str) -> str:
    """Namespaced lane key.

    The devkit states verbatim that a lane id is *"guaranteed to be unique only
    within this local map"*. Any cross-log index MUST be built on this, never on
    the raw integer.
    """
    return f"{log_id}:{int(lane_id)}"


# --------------------------------------------------------------------------- #
# Lane segment + graph                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class LaneSegment:
    """One AV2 lane segment. Field set MEASURED on 163 698/163 698 segments."""

    id: int
    is_intersection: bool = False
    lane_type: str = "VEHICLE"                       # VEHICLE | BIKE | BUS
    left_lane_boundary: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    right_lane_boundary: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    left_lane_mark_type: str = "NONE"
    right_lane_mark_type: str = "NONE"
    successors: tuple[int, ...] = ()
    predecessors: tuple[int, ...] = ()
    left_neighbor_id: int | None = None
    right_neighbor_id: int | None = None
    #: Present ONLY in the motion-forecasting split (0/163 698 in sensor).
    explicit_centerline: np.ndarray | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "LaneSegment":
        def _ids(key):
            return tuple(int(v) for v in (d.get(key) or []))

        def _opt(key):
            v = d.get(key)
            return None if v is None else int(v)

        cl = d.get("centerline")
        return cls(
            id=int(d["id"]),
            is_intersection=bool(d.get("is_intersection", False)),
            lane_type=str(d.get("lane_type", "VEHICLE")),
            left_lane_boundary=_pts_to_array(d.get("left_lane_boundary") or []),
            right_lane_boundary=_pts_to_array(d.get("right_lane_boundary") or []),
            left_lane_mark_type=str(d.get("left_lane_mark_type", "NONE")),
            right_lane_mark_type=str(d.get("right_lane_mark_type", "NONE")),
            successors=_ids("successors"),
            predecessors=_ids("predecessors"),
            left_neighbor_id=_opt("left_neighbor_id"),
            right_neighbor_id=_opt("right_neighbor_id"),
            explicit_centerline=(_pts_to_array(cl) if cl else None),
        )

    @property
    def has_explicit_centerline(self) -> bool:
        return (self.explicit_centerline is not None
                and self.explicit_centerline.shape[0] > 0)


@dataclass
class LaneGraph:
    """A log-local routable lane graph.

    "Log-local" is load-bearing: AV2 ships one map per log (sensor) or per
    scenario (motion-forecasting), cropped to that region. Edges leaving the crop
    dangle by construction — see :meth:`dangling_successors`.
    """

    segments: dict[int, LaneSegment]
    log_id: str = ""
    city: str = ""
    source_path: str = ""

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def from_dict(cls, d: dict, log_id: str = "", city: str = "",
                  source_path: str = "") -> "LaneGraph":
        if not isinstance(d, dict) or "lane_segments" not in d:
            raise Argoverse2MapError(
                f"map archive {source_path or '<dict>'} has no 'lane_segments' "
                f"layer (top-level keys: {sorted(d) if isinstance(d, dict) else type(d)}). "
                f"Expected the AV2 layers {MAP_LAYERS}.")
        segs = {}
        for raw in (d.get("lane_segments") or {}).values():
            try:
                s = LaneSegment.from_dict(raw)
            except (KeyError, TypeError, ValueError) as e:
                raise Argoverse2MapError(
                    f"malformed lane segment in {source_path or '<dict>'}: {e}") from e
            segs[s.id] = s
        return cls(segments=segs, log_id=log_id, city=city, source_path=source_path)

    # ---- basic views ------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.segments)

    def __contains__(self, lane_id: object) -> bool:
        return int(lane_id) in self.segments if isinstance(lane_id, (int, np.integer)) \
            else False

    def __iter__(self) -> Iterator[LaneSegment]:
        return iter(self.segments.values())

    def ids(self) -> list[int]:
        return sorted(self.segments)

    def get(self, lane_id: int) -> LaneSegment:
        try:
            return self.segments[int(lane_id)]
        except KeyError:
            raise Argoverse2MapError(
                f"lane {lane_id} not in local map {self.log_id or self.source_path!r} "
                f"({len(self.segments)} segments). Lane ids are unique only WITHIN "
                f"one local map — did you index across logs on a raw id?") from None

    def global_key(self, lane_id: int) -> str:
        return global_lane_key(self.log_id, lane_id)

    # ---- TRAP 1: the centerline ------------------------------------------ #
    def centerline(self, lane_id: int, n: int | None = None) -> np.ndarray:
        """Lane centerline ``[N,3]`` — explicit if the archive had one, else DERIVED.

        ⚠️ This is the single most important method in the module. The SENSOR
        split — the one with imagery — has **no** ``centerline`` field on any of
        its 163 698 segments, so every consumer must come through here. Never
        read ``seg["centerline"]``.
        """
        s = self.get(lane_id)
        if s.has_explicit_centerline:
            cl = s.explicit_centerline
            return interp_arc(cl, n) if n else np.asarray(cl, dtype=np.float64)
        return midpoint_line(s.left_lane_boundary, s.right_lane_boundary, n)

    def centerline_source(self, lane_id: int) -> str:
        """``"explicit"`` (motion-forecasting) or ``"derived"`` (sensor).

        Provenance is reported, never assumed — a derived centerline is a
        midpoint estimate and downstream numbers must be able to say so.
        """
        return "explicit" if self.get(lane_id).has_explicit_centerline else "derived"

    def centerline_source_counts(self) -> dict[str, int]:
        c = Counter(self.centerline_source(i) for i in self.segments)
        return {"explicit": c.get("explicit", 0), "derived": c.get("derived", 0)}

    # ---- TRAP 3: edges, resolved vs dangling ------------------------------ #
    def successors(self, lane_id: int, resolved_only: bool = True) -> tuple[int, ...]:
        su = self.get(lane_id).successors
        return tuple(s for s in su if s in self.segments) if resolved_only else su

    def predecessors(self, lane_id: int, resolved_only: bool = True) -> tuple[int, ...]:
        pr = self.get(lane_id).predecessors
        return tuple(p for p in pr if p in self.segments) if resolved_only else pr

    def dangling_successors(self, lane_id: int) -> tuple[int, ...]:
        """Successor ids that leave the local crop. NOT an error — a map edge."""
        return tuple(s for s in self.get(lane_id).successors if s not in self.segments)

    def neighbors(self, lane_id: int, resolved_only: bool = True
                  ) -> dict[str, int | None]:
        """Lateral adjacency — the S2 (lane-selection) signal."""
        s = self.get(lane_id)
        def _f(v):
            if v is None:
                return None
            return v if (v in self.segments or not resolved_only) else None
        return {"left": _f(s.left_neighbor_id), "right": _f(s.right_neighbor_id)}

    def successor_edges(self, resolved_only: bool = True
                        ) -> list[tuple[int, int]]:
        return [(i, s) for i in self.segments
                for s in self.successors(i, resolved_only)]

    def in_degree(self, resolved_only: bool = True) -> dict[int, int]:
        d: dict[int, int] = defaultdict(int)
        for _, t in self.successor_edges(resolved_only):
            d[t] += 1
        return dict(d)

    # ---- the strategic-brain signals -------------------------------------- #
    def branch_points(self, min_out_degree: int = 2,
                      resolved_only: bool = True) -> list[int]:
        """Lanes with a genuine CHOICE of successor — the S1 decision points.

        ``resolved_only=True`` (default) counts only options that exist inside
        this map, because an option we cannot represent cannot be a label. The
        raw count is available with ``resolved_only=False`` and is larger.
        """
        return sorted(i for i in self.segments
                      if len(self.successors(i, resolved_only)) >= min_out_degree)

    def merge_points(self, min_in_degree: int = 2,
                     resolved_only: bool = True) -> list[int]:
        return sorted(i for i, d in self.in_degree(resolved_only).items()
                      if d >= min_in_degree)

    def intersection_ids(self) -> list[int]:
        """The HP-4 (junction topology) signal."""
        return sorted(i for i, s in self.segments.items() if s.is_intersection)

    def branch_options(self, lane_id: int) -> list[dict]:
        """Per-successor descriptor at a branch — one row per S1 action option."""
        out = []
        for s in self.successors(lane_id):
            seg = self.segments[s]
            cl = self.centerline(s)
            out.append({
                "lane_id": s,
                "global_key": self.global_key(s),
                "is_intersection": seg.is_intersection,
                "lane_type": seg.lane_type,
                "heading_rad": self.heading(s),
                "entry_xyz": cl[0].tolist() if cl.shape[0] else None,
                "exit_xyz": cl[-1].tolist() if cl.shape[0] else None,
            })
        return out

    def heading(self, lane_id: int) -> float | None:
        """Entry->exit heading (rad) of a lane's centerline, or None if degenerate."""
        cl = self.centerline(lane_id)
        if cl.shape[0] < 2:
            return None
        d = cl[-1, :2] - cl[0, :2]
        if float(np.hypot(d[0], d[1])) < 1e-9:
            return None
        return float(math.atan2(d[1], d[0]))

    def routes_from(self, lane_id: int, max_depth: int = 4,
                    max_routes: int = 256) -> list[tuple[int, ...]]:
        """Enumerate successor routes up to ``max_depth`` lanes.

        Two things this must survive, both MEASURED in real AV2 data:
          * a **dangling** successor -> the route TERMINATES there, no exception;
          * a **directed cycle** (36 % of sensor logs contain one — the roundabout
            proxy) -> a lane already on the current path is not revisited, so this
            terminates instead of hanging.
        """
        start = int(lane_id)
        self.get(start)                                  # validate membership
        routes: list[tuple[int, ...]] = []

        def walk(path: tuple[int, ...]):
            if len(routes) >= max_routes:
                return
            su = [s for s in self.successors(path[-1]) if s not in path]
            if len(path) >= max_depth or not su:
                routes.append(path)
                return
            for s in su:
                walk(path + (s,))
                if len(routes) >= max_routes:
                    return

        walk((start,))
        return routes

    def has_cycle(self) -> bool:
        """Directed cycle in the successor graph — the roundabout/loop proxy."""
        colour: dict[int, int] = {}
        for root in self.segments:
            if colour.get(root, 0):
                continue
            stack = [(root, iter(self.successors(root)))]
            colour[root] = 1
            while stack:
                node, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    colour[node] = 2
                    stack.pop()
                    continue
                c = colour.get(nxt, 0)
                if c == 1:
                    return True
                if c == 0:
                    colour[nxt] = 1
                    stack.append((nxt, iter(self.successors(nxt))))
        return False

    # ---- summary ---------------------------------------------------------- #
    def stats(self) -> dict:
        """Derived per-map statistics — the shape the corpus report aggregates."""
        raw_edges = sum(len(s.successors) for s in self.segments.values())
        res_edges = len(self.successor_edges(resolved_only=True))
        cs = self.centerline_source_counts()
        return {
            "log_id": self.log_id, "city": self.city,
            "n_lane_segments": len(self.segments),
            "n_successor_edges_raw": raw_edges,
            "n_successor_edges_resolved": res_edges,
            "n_successor_refs_dangling": raw_edges - res_edges,
            "n_branch_points": len(self.branch_points()),
            "n_merge_points": len(self.merge_points()),
            "n_is_intersection": len(self.intersection_ids()),
            "n_left_neighbor": sum(1 for s in self.segments.values()
                                   if s.left_neighbor_id is not None),
            "n_right_neighbor": sum(1 for s in self.segments.values()
                                    if s.right_neighbor_id is not None),
            "lane_types": dict(Counter(s.lane_type for s in self.segments.values())),
            "centerline_source": cs,
            "has_cycle": self.has_cycle(),
        }


def lane_graph_from_dict(d: dict, log_id: str = "", city: str = "") -> LaneGraph:
    return LaneGraph.from_dict(d, log_id=log_id, city=city)


def load_lane_graph(path: str | Path, log_id: str = "",
                    city: str = "") -> LaneGraph:
    """Read one ``log_map_archive_*.json`` -> :class:`LaneGraph`.

    ``log_id``/``city`` are parsed out of the AV2 filename when not given. Our
    own puller renames archives to ``<log_id>.json``, so the stem is used as the
    fallback log id — both layouts work.
    """
    p = Path(path)
    if not p.exists():
        raise Argoverse2MapError(_ACQUIRE_MSG.format(path=str(p)))
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise Argoverse2MapError(
            f"map archive {p} did not parse ({type(e).__name__}: {e}). A short or "
            f"truncated file looks exactly like this — verify the byte count "
            f"against the bucket's ListObjectsV2 <Size> before blaming the schema."
        ) from e
    if not log_id or not city:
        m = _ARCHIVE_RE.search(p.name)
        if m:
            log_id = log_id or (m.group("log") or "")
            city = city or (m.group("city") or "")
        log_id = log_id or p.stem
    return LaneGraph.from_dict(d, log_id=log_id, city=city, source_path=str(p))


def map_archive_path(log_dir: str | Path) -> Path:
    """Locate the map archive inside an AV2 log directory."""
    d = Path(log_dir)
    for cand in (d / "map", d):
        if cand.is_dir():
            hits = sorted(cand.glob("log_map_archive_*.json"))
            if hits:
                return hits[0]
    raise Argoverse2MapError(_ACQUIRE_MSG.format(path=str(d / "map")))


def lane_graph_stats(graphs: Iterable[LaneGraph]) -> dict:
    """Aggregate :meth:`LaneGraph.stats` over a corpus."""
    rows = [g.stats() for g in graphs]
    agg: dict[str, Any] = {"n_maps": len(rows)}
    for k in ("n_lane_segments", "n_successor_edges_raw",
              "n_successor_edges_resolved", "n_successor_refs_dangling",
              "n_branch_points", "n_merge_points", "n_is_intersection",
              "n_left_neighbor", "n_right_neighbor"):
        agg[k] = int(sum(r[k] for r in rows))
    agg["maps_with_a_branch"] = sum(1 for r in rows if r["n_branch_points"] > 0)
    agg["maps_with_an_intersection"] = sum(1 for r in rows if r["n_is_intersection"] > 0)
    agg["maps_with_cycle"] = sum(1 for r in rows if r["has_cycle"])
    agg["cities"] = dict(Counter(r["city"] for r in rows if r["city"]))
    lt: Counter = Counter()
    cs: Counter = Counter()
    for r in rows:
        lt.update(r["lane_types"])
        cs.update(r["centerline_source"])
    agg["lane_types"] = dict(lt)
    agg["centerline_source"] = dict(cs)
    return agg


# --------------------------------------------------------------------------- #
# Ego track — `city_SE3_egovehicle.feather`                                    #
# --------------------------------------------------------------------------- #
#: Columns of AV2's per-log pose table (documented schema).
POSE_COLUMNS = ("timestamp_ns", "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m")


def _read_table(path: str | Path):
    """Feather -> DataFrame. pandas/pyarrow imported lazily so the lane-graph
    half of this module never pays for them."""
    p = Path(path)
    if not p.exists():
        raise Argoverse2MapError(
            f"AV2 table not found: {p}\nThe LANE-GRAPH pull (~154 MiB) does not "
            f"include per-log pose/calibration tables — those live in the sensor "
            f"tars (1 051 GiB). Pull the log directory before calling this.")
    try:
        import pandas as pd
    except ImportError as e:                                  # pragma: no cover
        raise Argoverse2MapError(
            "reading AV2 feather tables needs pandas+pyarrow") from e
    return pd.read_feather(p)


def ego_track(log_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Ego poses ``[T,4] (x, y, yaw, v)`` + timestamps ``[T]`` (seconds).

    Same contract as :func:`tanitad.data.nuscenes.ego_track`; speed is the central
    finite difference against the REAL (non-uniform) timestamps, never a constant
    dt. AV2 timestamps are NANOseconds.
    """
    df = _read_table(Path(log_dir) / "city_SE3_egovehicle.feather")
    missing = [c for c in POSE_COLUMNS if c not in df.columns]
    if missing:
        raise Argoverse2MapError(
            f"{log_dir}: city_SE3_egovehicle.feather missing columns {missing} "
            f"(has {list(df.columns)})")
    df = df.sort_values("timestamp_ns")
    t = df["timestamp_ns"].to_numpy(dtype=np.float64) * NS
    x = df["tx_m"].to_numpy(dtype=np.float64)
    y = df["ty_m"].to_numpy(dtype=np.float64)
    if len(t) < 2:
        raise Argoverse2MapError(f"{log_dir}: {len(t)} poses — too short")
    yaw = np.unwrap(np.array([quat_to_yaw(q) for q in
                              df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)]))
    dt = np.gradient(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if (dt > 0).any() else 1.0 / SENSOR_HZ
    v = np.hypot(np.gradient(x) / dt, np.gradient(y) / dt)
    poses = np.stack([x, y, wrap_pi(yaw), v], axis=1).astype(np.float32)
    return poses, t.astype(np.float32)


def actions_from_track(poses: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``[T,2] (steer rad, accel m/s^2)`` — POSE-DERIVED (AV2 ships no CAN).

    Identical formulation to the nuScenes adapter (bicycle-model steering proxy +
    the contract's forward finite-difference accel); only the degenerate-dt
    fallback differs, because AV2 runs at 20 Hz rather than 2 Hz keyframes.
    """
    L = 2.588
    yaw = np.unwrap(np.asarray(poses)[:, 2].astype(np.float64))
    v = np.asarray(poses)[:, 3].astype(np.float64)
    dt = np.gradient(np.asarray(t, dtype=np.float64))
    dt[dt <= 0] = 1.0 / SENSOR_HZ
    yawrate = np.gradient(yaw) / dt
    with np.errstate(divide="ignore", invalid="ignore"):
        steer = np.arctan(np.where(v > 0.5, L * yawrate / np.maximum(v, 1e-6), 0.0))
    steer = np.nan_to_num(steer, nan=0.0, posinf=0.0, neginf=0.0)
    accel = finite_diff_accel(v, float(np.median(dt)))
    return np.stack([steer, accel], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #
def camera_intrinsics_of(log_dir: str | Path, camera: str = EGO_CAMERA):
    """Per-log camera intrinsics -> ``calib.PinholeIntrinsics``.

    AV2 ships intrinsics **per sensor per log**, so — exactly as the PhysicalAI
    two-rig ``cy`` lesson demands — nothing here asserts a nominal constant.
    """
    from tanitad.data.calib import PinholeIntrinsics

    df = _read_table(Path(log_dir) / "calibration" / "intrinsics.feather")
    row = df[df["sensor_name"] == camera]
    if row.empty:
        raise Argoverse2MapError(
            f"{log_dir}: no intrinsics row for camera {camera!r} "
            f"(available: {sorted(df['sensor_name'].astype(str).unique())})")
    r = row.iloc[0]
    # AV2 ships RADIAL distortion (k1,k2,k3) and no tangential terms, so they map
    # into the OpenCV-order tuple as (k1, k2, p1=0, p2=0, k3). This is a real gain
    # over nuScenes, whose `camera_intrinsic` is a bare 3x3 with no lens model at
    # all — `pinhole_rectify` can genuinely undistort here instead of degrading to
    # its pad-crop half.
    def _k(name):
        return float(r[name]) if name in row.columns else 0.0
    return PinholeIntrinsics(
        fx=float(r["fx_px"]), fy=float(r["fy_px"]),
        cx=float(r["cx_px"]), cy=float(r["cy_px"]),
        width=int(r["width_px"]), height=int(r["height_px"]),
        dist=(_k("k1"), _k("k2"), 0.0, 0.0, _k("k3")))


# --------------------------------------------------------------------------- #
# Discovery + the I3 split unit                                                #
# --------------------------------------------------------------------------- #
def discover_logs(root: str | Path, split: str | None = "val") -> list[str]:
    """Ingest units = LOGS. AV2's splits are directories — no ``splits.json``.

    ``split=None``/``""`` means *root is already the split directory*.

    ⚠️ A missing split RAISES rather than falling back to ``root``. The first
    draft of this function did fall back, and the test suite caught it returning
    ``['train','val','test']`` as if the three splits were three logs — a silent
    corpus-wide mis-ingest. Fail loud instead.
    """
    base = Path(root)
    d = base / split if split else base
    if not d.is_dir():
        avail = (sorted(p.name for p in base.iterdir() if p.is_dir())
                 if base.is_dir() else [])
        raise Argoverse2MapError(
            f"AV2 split directory not found: {d}\n"
            f"Available under {base}: {avail or '(root does not exist)'}\n"
            f"Pass split=None if {base} is already the split directory.")
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def split_unit_of(log_id: str) -> str:
    """I3 split unit = the LOG itself.

    AV2 sensor is already one-drive-per-log, so the log IS the drive-disjoint
    unit — unlike nuScenes, where scenes must be grouped back up to their log.
    """
    return str(log_id)
