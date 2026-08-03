"""LAN — Lane-Anchored Navigation: a dense, leak-guarded route/goal signal.

WHY THIS EXISTS — the defect it replaces, from primary sources
--------------------------------------------------------------
REF-C (and REF-B, and the flagship strategic brain) receive their route/goal
input as a **4-way one-hot** ``nav_cmd`` ∈ {follow, left, right, straight}
(``tanitad.refs.refc.NAV_COMMANDS``), concatenated with ``v0`` into a 2-layer
``measurement`` MLP. Three MEASURED facts make that input degenerate:

1. **The label is a time-thresholded net heading change.** ``scripts/refb_labels
   .nav_command`` fires left/right only when |Δyaw| over 15–25 s of future poses
   exceeds 45°, and returns ``(NAV_FOLLOW, valid=False)`` whenever fewer than
   ``NAV_MIN_STEPS`` future poses exist. On a 74 %-straight corpus the emitted
   command is ``follow`` for the overwhelming majority of windows, and
   ``nav_valid_frac`` is **0.21–0.25 in all four arms including the deployed v1**
   (RETRACTION_LOG 2026-07-21).
2. **The route head learned nothing.** ``route_skill_vs_chance = 0.0`` — a pure
   command echo; the gate metric ``nonav_route_beats_majority`` FAILS
   (240/240 straight). See ``tests/test_refb_labels_v2.py`` and
   ``tanitad/config.py`` (the comment at the ``route`` loss weight).
3. **Evaluation never exercises it.** Every published REF-C number is decoded
   with ``nav_cmd=None`` → ``refc.py`` substitutes index 0 (``follow``) for the
   whole batch, so the route pathway is a CONSTANT at eval. This is the C6
   confound logged **twice** (2026-07-21 "nearly designed the hierarchy away";
   2026-07-25 hierarchy-proof pre-condition #1).

LAN replaces the 4-way scalar with a **geometric route corridor**: K points
sampled at fixed ARC-LENGTH ahead along the route, expressed in the ego frame,
encoded as scale-free LATERAL TOPOLOGY. Arc-length (not time) sampling is the
point: net-heading-over-time conflates κ·v·t, which is precisely the defect
``refb_labels`` v2 documents in its own docstring.

WHAT THIS DELIBERATELY DOES **NOT** SUPPLY — and why
----------------------------------------------------
``GOAL_INPUT.md`` (2026-07-27, paired episode-cluster bootstrap, B=2000, 40 ep /
881 win) MEASURED the split of the oracle-goal advantage on the 2 s selection
surface:

    oracle ALONG-track + learned cross-track → +83.7 % recovery (separated)
    oracle CROSS-track + learned along-track → **+2.9 %**   (NOT separated)

⇒ the **along-track** coordinate is the valuable one, and it is also the one a
route signal must NOT contain, because "how far will the ego travel in the next
2 s" IS the answer to the prediction task. The encoding here therefore carries
**bearing and normalised lateral offset only** — never an along-track distance —
and every anchor closer than :func:`horizon_lead_m` is masked out. A LAN feature
vector cannot be decoded back into the 2 s waypoints it is meant to condition.

TWO SUPPLIERS, ONE CONTRACT
---------------------------
* **S1 ``ego_future``** — resample the ego's own future path by arc-length.
  Works on ANY corpus with egomotion, including the parity corpus
  ``physicalai-train-e438721ae894`` (which has **no map** — settled at five
  probes, CLAUDE.md).
* **S2 ``lane_graph``** — snap to lane centrelines and read the route off the
  lane graph (:class:`LaneCorridor`). Available for NuRec (``map.xodr``, 356
  driving lanes / 340 edges) and Argoverse 2 (``tanitad.data.argoverse2``).

S1 is only admissible as a stand-in for a real map route if the two AGREE. That
is a measurement, not an assumption: :func:`route_agreement` computes it, and
``scripts/lan_probe.py`` runs it on the banked NuRec artifacts.

⚠️ **`nav_cmd` is NOT removed.** LAN is an ADDITIVE, gated seam (REF-C
``graft_lan``), so a LAN-off model is byte-identical to one that never had it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Per-anchor feature layout — pinned by tests/test_lan.py.
#   0: cos(bearing)      bearing of the route point in the ego frame
#   1: sin(bearing)      (sign carries left/right — the lateral topology)
#   2: lat_norm          signed lateral offset / arc-length, clipped
#   3: valid             1.0 if this anchor is real AND past the leak guard
LAN_FEATS_PER_ANCHOR = 4
LAN_FEAT_NAMES = ("cos_bearing", "sin_bearing", "lat_norm", "valid")

# Route-scale arc-lengths. The shortest (20 m) is ~2 s at 10 m/s, so the leak
# guard masks it at highway speed and keeps it at urban speed — the guard, not
# the constant, is what makes this safe.
LAN_ARCLENGTHS_M = (20.0, 40.0, 80.0, 160.0)

_EPS = 1e-9


@dataclass(frozen=True)
class LanConfig:
    """Shape + safety parameters of the LAN encoding.

    ``min_lead_m`` is EXTRA margin added on top of the measured 2 s path length
    by :func:`horizon_lead_m`; it is the knob that trades signal for leak safety.
    """

    arclengths_m: tuple[float, ...] = LAN_ARCLENGTHS_M
    min_lead_m: float = 5.0
    lat_clip: float = 1.0

    def __post_init__(self) -> None:
        if not self.arclengths_m:
            raise ValueError("LanConfig needs at least one arc-length")
        if any(s <= 0 for s in self.arclengths_m):
            raise ValueError(f"arc-lengths must be positive: {self.arclengths_m}")
        if list(self.arclengths_m) != sorted(self.arclengths_m):
            raise ValueError(f"arc-lengths must be ascending: {self.arclengths_m}")
        if self.lat_clip <= 0:
            raise ValueError("lat_clip must be positive")

    @property
    def k(self) -> int:
        return len(self.arclengths_m)

    @property
    def dim(self) -> int:
        """Flat feature width the model consumes."""
        return self.k * LAN_FEATS_PER_ANCHOR


@dataclass(frozen=True)
class LanRoute:
    """One window's route corridor.

    ``points_ego`` [K, 2] are ego-frame (x forward, y left) metres; masked
    anchors are zeroed. ``features`` [K*4] is what the model sees.
    """

    points_ego: np.ndarray
    valid: np.ndarray
    features: np.ndarray
    lead_m: float
    source: str

    @property
    def any_valid(self) -> bool:
        return bool(self.valid.any())

    @property
    def valid_frac(self) -> float:
        return float(self.valid.mean())


# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------

def cumulative_arclength(xy: np.ndarray) -> np.ndarray:
    """[T, 2] polyline -> [T] cumulative arc-length, ``s[0] == 0``."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"expected [T, 2] polyline, got {xy.shape}")
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def resample_arclength(xy: np.ndarray, s_query) -> tuple[np.ndarray, np.ndarray]:
    """Linear resample of a polyline at cumulative arc-lengths ``s_query``.

    Returns ``(points [K, 2], valid [K] bool)``. ``valid`` is False where the
    polyline is SHORTER than the requested arc-length — extrapolating a route
    past the end of the evidence is exactly the silent failure this mask exists
    to prevent. Out-of-range points are returned zeroed, never clamped to the
    last vertex (a clamped point looks like a real route hop).
    """
    xy = np.asarray(xy, dtype=np.float64)
    s_query = np.asarray(s_query, dtype=np.float64).reshape(-1)
    k = s_query.shape[0]
    out = np.zeros((k, 2), dtype=np.float64)
    valid = np.zeros((k,), dtype=bool)
    if xy.shape[0] < 2:
        return out, valid
    s = cumulative_arclength(xy)
    total = float(s[-1])
    if total <= _EPS:                       # degenerate (stationary) polyline
        return out, valid
    inside = (s_query >= 0.0) & (s_query <= total)
    if not inside.any():
        return out, valid
    q = s_query[inside]
    out[inside, 0] = np.interp(q, s, xy[:, 0])
    out[inside, 1] = np.interp(q, s, xy[:, 1])
    valid[inside] = True
    return out, valid


def to_ego_frame(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    """World [N, 2] -> ego frame (x forward, y LEFT), CCW-positive yaw.

    Matches the repo's ``_ego`` convention (``refb_labels``: CCW == left ==
    NAV_LEFT) and the anchor vocabulary in ``refc.synth_anchor_pool`` (positive
    yaw-rate integrates to positive y).
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    d = xy - np.asarray(origin, dtype=np.float64).reshape(1, 2)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.stack([d[:, 0] * c + d[:, 1] * s,
                     -d[:, 0] * s + d[:, 1] * c], axis=1)


def yaw_from_path(xy: np.ndarray, i: int = 0, lookahead: int = 5) -> float:
    """Heading at index ``i`` estimated from the next ``lookahead`` samples.

    Used when a corpus supplies positions but no yaw. Falls back to the previous
    sample, then to 0.0 — a stationary window has no defined heading and must
    not silently inherit one.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = xy.shape[0]
    if n < 2:
        return 0.0
    j = min(i + max(1, lookahead), n - 1)
    d = xy[j] - xy[i]
    if float(np.hypot(d[0], d[1])) < 1e-6 and i > 0:
        d = xy[i] - xy[i - 1]
    if float(np.hypot(d[0], d[1])) < 1e-6:
        return 0.0
    return float(math.atan2(d[1], d[0]))


# ---------------------------------------------------------------------------
# the leak guard
# ---------------------------------------------------------------------------

def horizon_lead_m(gt_path_ego: np.ndarray | None = None,
                   v0: float | None = None,
                   t_pred_s: float = 2.0,
                   cfg: LanConfig | None = None) -> float:
    """Minimum arc-length a LAN anchor may sit at, in metres.

    The route corridor must start BEYOND anything the model is asked to predict,
    or it is an oracle. The guard is the **maximum** of

      * the measured arc length of the ground-truth path over the prediction
        horizon (what the model must actually output), and
      * ``v0 * t_pred_s`` (an upper bound available at inference, when the GT
        path is not),

    plus ``cfg.min_lead_m``. Taking the max makes the guard CONSERVATIVE: it can
    only mask more anchors, never fewer. Both inputs are optional; supplying
    neither yields ``cfg.min_lead_m`` and is only correct for a stationary ego.
    """
    cfg = cfg or LanConfig()
    lead = 0.0
    if gt_path_ego is not None:
        p = np.asarray(gt_path_ego, dtype=np.float64).reshape(-1, 2)
        if p.shape[0] >= 1:
            # The GT path is ego-frame and starts at the ego origin, which is
            # NOT necessarily its first sample — prepend the origin so the arc
            # length is measured from the car, not from waypoint 0.
            p = np.concatenate([np.zeros((1, 2)), p], axis=0)
            lead = max(lead, float(cumulative_arclength(p)[-1]))
    if v0 is not None:
        lead = max(lead, float(v0) * float(t_pred_s))
    return lead + float(cfg.min_lead_m)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def encode_route(points_ego: np.ndarray, valid: np.ndarray,
                 cfg: LanConfig, lead_m: float = 0.0) -> LanRoute:
    """Ego-frame route points -> the flat [K*4] LAN feature vector.

    Anchors whose arc-length is inside ``lead_m`` are masked (the leak guard),
    as are anchors the supplier could not resolve. A masked anchor contributes
    an all-zero 4-tuple — including ``valid = 0`` — so the model can tell
    "no route here" from "route straight ahead" (which is ``cos=1, sin=0,
    lat=0, valid=1``).
    """
    points_ego = np.asarray(points_ego, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    k = cfg.k
    if points_ego.shape[0] != k or valid.shape[0] != k:
        raise ValueError(f"expected {k} route points/flags, got "
                         f"{points_ego.shape[0]}/{valid.shape[0]}")
    arc = np.asarray(cfg.arclengths_m, dtype=np.float64)
    keep = valid & (arc >= float(lead_m))
    feats = np.zeros((k, LAN_FEATS_PER_ANCHOR), dtype=np.float64)
    pts = points_ego.copy()
    pts[~keep] = 0.0
    if keep.any():
        x, y = pts[keep, 0], pts[keep, 1]
        r = np.hypot(x, y)
        safe = r > _EPS
        cos_b = np.ones_like(x)
        sin_b = np.zeros_like(x)
        cos_b[safe] = x[safe] / r[safe]
        sin_b[safe] = y[safe] / r[safe]
        feats[keep, 0] = cos_b
        feats[keep, 1] = sin_b
        feats[keep, 2] = np.clip(y / arc[keep], -cfg.lat_clip, cfg.lat_clip)
        feats[keep, 3] = 1.0
    return LanRoute(points_ego=pts.astype(np.float32),
                    valid=keep,
                    features=feats.reshape(-1).astype(np.float32),
                    lead_m=float(lead_m), source="")


def lan_from_future_path(future_xy: np.ndarray, origin: np.ndarray, yaw: float,
                         cfg: LanConfig | None = None,
                         lead_m: float = 0.0) -> LanRoute:
    """S1 supplier — the ego's own future path, resampled by arc-length.

    ``future_xy`` is the WORLD-frame ego track from the current sample forward
    (the current position may be included or not; arc-length is measured from
    ``origin`` either way because the origin is prepended).
    """
    cfg = cfg or LanConfig()
    fut = np.asarray(future_xy, dtype=np.float64).reshape(-1, 2)
    origin = np.asarray(origin, dtype=np.float64).reshape(2)
    path = np.concatenate([origin.reshape(1, 2), fut], axis=0)
    pts_w, valid = resample_arclength(path, cfg.arclengths_m)
    pts_e = to_ego_frame(pts_w, origin, yaw)
    out = encode_route(pts_e, valid, cfg, lead_m)
    return LanRoute(out.points_ego, out.valid, out.features, out.lead_m,
                    "ego_future")


def lan_window_features(poses, t: int, cfg: LanConfig | None = None,
                        t_pred_s: float = 2.0) -> np.ndarray:
    """Trainer/eval bridge: episode poses ``[T, 4]`` (x, y, yaw, v) -> [K*4].

    ``t`` is the window's LAST index — the same anchor timestep
    ``refb_labels.nav_command`` / ``route_from_future_v21`` use, so a LAN run
    differs from a baseline run in the ROUTE INPUT alone and in nothing else.
    Accepts numpy or torch (anything ``np.asarray`` can take). The leak guard is
    driven by ``v0`` from the pose itself, so it is available at inference too.
    """
    cfg = cfg or LanConfig()
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError(f"poses must be [T, 4] (x, y, yaw, v), got {p.shape}")
    t = int(t)
    if not 0 <= t < p.shape[0]:
        raise IndexError(f"window index {t} outside episode of {p.shape[0]}")
    lead = horizon_lead_m(v0=float(p[t, 3]), t_pred_s=t_pred_s, cfg=cfg)
    return lan_from_future_path(p[t:, :2], p[t, :2], float(p[t, 2]),
                                cfg, lead).features


def lan_from_polyline(route_xy: np.ndarray, origin: np.ndarray, yaw: float,
                      cfg: LanConfig | None = None,
                      lead_m: float = 0.0,
                      source: str = "lane_graph") -> LanRoute:
    """S2 supplier — an already-built route polyline (e.g. from a lane graph).

    The polyline is re-anchored at ``origin``: arc-length 0 is the point on the
    polyline nearest the ego, so a map route and an ego-future route are
    measured from the same place and are directly comparable.
    """
    cfg = cfg or LanConfig()
    poly = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    origin = np.asarray(origin, dtype=np.float64).reshape(2)
    if poly.shape[0] >= 2:
        j, foot = _project_onto_polyline(poly, origin)
        poly = np.concatenate([foot.reshape(1, 2), poly[j + 1:]], axis=0)
    pts_w, valid = resample_arclength(poly, cfg.arclengths_m)
    pts_e = to_ego_frame(pts_w, origin, yaw)
    out = encode_route(pts_e, valid, cfg, lead_m)
    return LanRoute(out.points_ego, out.valid, out.features, out.lead_m, source)


def _project_onto_polyline(poly: np.ndarray, p: np.ndarray
                           ) -> tuple[int, np.ndarray]:
    """Nearest point on a polyline -> (segment index, foot point)."""
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    denom = (ab ** 2).sum(axis=1)
    denom = np.where(denom < _EPS, 1.0, denom)
    t = (((p.reshape(1, 2) - a) * ab).sum(axis=1) / denom).clip(0.0, 1.0)
    feet = a + t[:, None] * ab
    d = np.linalg.norm(feet - p.reshape(1, 2), axis=1)
    j = int(np.argmin(d))
    return j, feet[j]


# ---------------------------------------------------------------------------
# counterfactuals — the instrument's treatment arms
# ---------------------------------------------------------------------------

def mirror_route(route: LanRoute) -> LanRoute:
    """Left<->right mirrored route: the SAME topology commanded the other way.

    This is the treatment arm of the route-counterfactual test. It changes only
    the lateral channel, so a model that responds to it is responding to route
    TOPOLOGY and not to a change in signal energy (‖features‖ is preserved).
    """
    f = route.features.reshape(-1, LAN_FEATS_PER_ANCHOR).copy()
    f[:, 1] *= -1.0                                    # sin(bearing)
    f[:, 2] *= -1.0                                    # lat_norm
    pts = route.points_ego.copy()
    pts[:, 1] *= -1.0
    return LanRoute(pts, route.valid.copy(), f.reshape(-1),
                    route.lead_m, route.source + "|mirror")


def inert_route(cfg: LanConfig) -> LanRoute:
    """The all-masked route — "no route supplied". The reference condition."""
    k = cfg.k
    return LanRoute(np.zeros((k, 2), dtype=np.float32),
                    np.zeros((k,), dtype=bool),
                    np.zeros((k * LAN_FEATS_PER_ANCHOR,), dtype=np.float32),
                    0.0, "inert")


def straight_route(cfg: LanConfig, lead_m: float = 0.0) -> LanRoute:
    """A synthetic dead-ahead route — the majority-class control.

    A route instrument that cannot separate the true route from "straight" on a
    74 %-straight corpus is reporting the base rate, which is exactly how
    ``route_skill_vs_chance = 0.0`` was missed.
    """
    arc = np.asarray(cfg.arclengths_m, dtype=np.float64)
    pts = np.stack([arc, np.zeros_like(arc)], axis=1)
    out = encode_route(pts, np.ones((cfg.k,), dtype=bool), cfg, lead_m)
    return LanRoute(out.points_ego, out.valid, out.features, out.lead_m,
                    "straight")


# ---------------------------------------------------------------------------
# S1 vs S2 agreement — the measurement that licenses the map-free supplier
# ---------------------------------------------------------------------------

def route_agreement(a: LanRoute, b: LanRoute) -> dict:
    """Per-anchor agreement between two suppliers of the SAME window.

    Only anchors valid in BOTH are compared — an anchor one supplier could not
    resolve is a coverage fact, reported separately as ``n_compared`` /
    ``both_valid_frac``, never averaged in as a zero.
    """
    va = np.asarray(a.valid, dtype=bool)
    vb = np.asarray(b.valid, dtype=bool)
    both = va & vb
    k = va.shape[0]
    out = {"n_anchors": int(k), "n_compared": int(both.sum()),
           "both_valid_frac": float(both.mean()) if k else 0.0,
           "a_valid_frac": float(va.mean()) if k else 0.0,
           "b_valid_frac": float(vb.mean()) if k else 0.0}
    if not both.any():
        out.update(pos_l2_m=float("nan"), bearing_deg=float("nan"),
                   lat_delta_m=float("nan"), side_agree=float("nan"))
        return out
    pa = np.asarray(a.points_ego, dtype=np.float64)[both]
    pb = np.asarray(b.points_ego, dtype=np.float64)[both]
    out["pos_l2_m"] = float(np.linalg.norm(pa - pb, axis=1).mean())
    ba = np.arctan2(pa[:, 1], pa[:, 0])
    bb = np.arctan2(pb[:, 1], pb[:, 0])
    dd = np.abs(np.arctan2(np.sin(ba - bb), np.cos(ba - bb)))
    out["bearing_deg"] = float(np.degrees(dd).mean())
    out["lat_delta_m"] = float(np.abs(pa[:, 1] - pb[:, 1]).mean())
    # "Same side" is the decision the strategic brain actually makes. A ±0.5 m
    # dead band keeps a straight route from being scored as a coin flip.
    band = 0.5
    sa = np.where(np.abs(pa[:, 1]) < band, 0.0, np.sign(pa[:, 1]))
    sb = np.where(np.abs(pb[:, 1]) < band, 0.0, np.sign(pb[:, 1]))
    out["side_agree"] = float((sa == sb).mean())
    return out


# ---------------------------------------------------------------------------
# S2 supplier — a lane corridor (centrelines + directed edges)
# ---------------------------------------------------------------------------

class LaneCorridor:
    """Lane centrelines + directed successor edges — the S2 route supplier.

    Deliberately format-agnostic: built from ``{lane_id: [[x, y], ...]}`` plus
    ``[(from_id, to_id), ...]``. That is the shape the NuRec ``map.xodr`` probe
    banks (``lane_centerlines.json`` / ``lane_graph_edges.json``) and the shape
    ``tanitad.data.argoverse2.LaneGraph`` can emit, so one implementation serves
    both corpora and neither dictates the schema.

    ⚠️ Lane ids are strings here ON PURPOSE. NuRec ids are ``"road:lane"``
    composites and AV2 ids are "unique only within this local map" — an int key
    invites a cross-log collision.
    """

    def __init__(self, centerlines: dict, edges=()):
        self.centerlines: dict[str, np.ndarray] = {}
        for lid, poly in centerlines.items():
            arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            if arr.shape[0] >= 2:
                self.centerlines[str(lid)] = arr
        self.edges: list[tuple[str, str]] = [
            (str(u), str(v)) for u, v in edges
            if str(u) in self.centerlines and str(v) in self.centerlines]
        self._succ: dict[str, list[str]] = {}
        for u, v in self.edges:
            self._succ.setdefault(u, []).append(v)
        if not self.centerlines:
            raise ValueError("LaneCorridor needs at least one 2+-vertex lane")

    @classmethod
    def from_json(cls, centerlines_path, edges_path=None) -> "LaneCorridor":
        cl = json.loads(Path(centerlines_path).read_text(encoding="utf-8"))
        edges = ()
        if edges_path is not None:
            e = json.loads(Path(edges_path).read_text(encoding="utf-8"))
            edges = e["edges"] if isinstance(e, dict) else e
        return cls(cl, edges)

    @property
    def n_lanes(self) -> int:
        return len(self.centerlines)

    def successors(self, lane_id: str) -> list[str]:
        return list(self._succ.get(str(lane_id), ()))

    def _tangent(self, poly: np.ndarray, j: int) -> np.ndarray:
        """Unit tangent of segment ``j`` of a centreline (driving direction)."""
        d = poly[j + 1] - poly[j]
        n = float(np.linalg.norm(d))
        return d / n if n > _EPS else np.array([1.0, 0.0])

    def snap(self, p, heading: float | None = None,
             max_heading_dev_rad: float | None = None) -> tuple[str, float]:
        """Nearest lane to a world point -> ``(lane_id, distance_m)``.

        ⚠️ **Bare nearest-neighbour is NOT good enough, and that is MEASURED,
        not assumed.** The NuRec probe found 3 of 14 route hops mis-snapped
        among near-coincident junction roads (mean road length 19 m, junction
        roads overlap) and named heading gating as the fix. Reconstructing a
        route corridor with the ungated version on the banked scene gave a
        median S1-vs-S2 position error of **4.68 m / mean 19.80 m** — the
        detours dominate the arc-length. Passing ``heading`` (the ego yaw, rad)
        with ``max_heading_dev_rad`` rejects lanes whose local tangent runs the
        wrong way, which is what a route consumer needs.
        """
        p = np.asarray(p, dtype=np.float64).reshape(2)
        gate = (heading is not None and max_heading_dev_rad is not None)
        hv = (np.array([math.cos(heading), math.sin(heading)])
              if heading is not None else None)
        cos_min = (math.cos(float(max_heading_dev_rad)) if gate else -2.0)
        best_id, best_d = "", float("inf")
        fallback_id, fallback_d = "", float("inf")
        for lid, poly in self.centerlines.items():
            j, foot = _project_onto_polyline(poly, p)
            d = float(np.linalg.norm(foot - p))
            if d < fallback_d:
                fallback_id, fallback_d = lid, d
            if gate and float(self._tangent(poly, j) @ hv) < cos_min:
                continue
            if d < best_d:
                best_id, best_d = lid, d
        if not best_id:                       # every lane failed the gate
            return fallback_id, fallback_d
        return best_id, best_d

    def route_polyline(self, track_xy, max_snap_m: float = 5.0,
                       heading=None, max_heading_dev_deg: float | None = 60.0,
                       hysteresis_m: float = 1.0
                       ) -> tuple[np.ndarray, list[str], dict]:
        """The MAP route the given track traverses.

        The track selects WHICH lanes (that is the route decision a strategic
        brain would issue); the returned geometry comes from the MAP. Returns
        ``(polyline [N, 2], lane_sequence, stats)``.

        Two guards, both earned by measurement on the banked NuRec scene:

        * **heading gate** (``max_heading_dev_deg``) — a lane whose tangent
          opposes travel is not a route option, however near it is;
        * **hysteresis** (``hysteresis_m``) — keep the current lane unless a
          rival is nearer by this margin, which stops the flip-flopping among
          near-coincident junction roads that the probe reported.

        ``stats`` carries the snap distances and how many consecutive lane hops
        are real graph edges — the honest coverage number, never a claim of
        routability.
        """
        track = np.asarray(track_xy, dtype=np.float64).reshape(-1, 2)
        if heading is None:
            heading = [yaw_from_path(track, i) for i in range(track.shape[0])]
        heading = np.asarray(heading, dtype=np.float64).reshape(-1)
        dev = (None if max_heading_dev_deg is None
               else math.radians(float(max_heading_dev_deg)))
        seq: list[str] = []
        dists: list[float] = []
        for i, p in enumerate(track):
            lid, d = self.snap(p, float(heading[i]), dev)
            if seq and seq[-1] != lid:
                cur = self.centerlines[seq[-1]]
                _, foot = _project_onto_polyline(cur, p)
                if float(np.linalg.norm(foot - p)) <= d + hysteresis_m:
                    lid, d = seq[-1], float(np.linalg.norm(foot - p))
            dists.append(d)
            if d > max_snap_m:
                continue
            if not seq or seq[-1] != lid:
                seq.append(lid)
        hops = list(zip(seq[:-1], seq[1:]))
        on_graph = sum(1 for h in hops if h[1] in self._succ.get(h[0], ()))
        stats = {"n_lanes": len(seq), "n_hops": len(hops),
                 "hops_on_graph": on_graph,
                 "hops_on_graph_frac": (on_graph / len(hops)) if hops else 0.0,
                 "snap_median_m": float(np.median(dists)) if dists else float("nan"),
                 "snap_max_m": float(np.max(dists)) if dists else float("nan")}
        if not seq:
            return np.zeros((0, 2)), [], stats
        parts = [self.centerlines[seq[0]]]
        for lid in seq[1:]:
            nxt = self.centerlines[lid]
            # Orient each lane so it continues forward from the previous one —
            # OpenDRIVE left lanes are stored against the driving direction.
            if (np.linalg.norm(nxt[-1] - parts[-1][-1])
                    < np.linalg.norm(nxt[0] - parts[-1][-1])):
                nxt = nxt[::-1]
            parts.append(nxt)
        return np.concatenate(parts, axis=0), seq, stats
