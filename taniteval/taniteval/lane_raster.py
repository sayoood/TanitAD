"""EGO-FRAME LANE-GRAPH RASTER — the strategic conditioning surface, from `map.xodr`.

WHY THIS MODULE EXISTS
----------------------
The programme's standing conclusion was *"no map, lane graph, junction annotation, traffic-light
feature or route/goal signal — the strategic brain's topology must come from AlpaSim or an external
corpus"*, settled at five probes. That conclusion is **correct about PhysicalAI-AV** and **false
about the NuRec scene bundles**: they ship a georeferenced OpenDRIVE HD map (`map.xodr`, 219 roads /
356 driving lanes / 26 junctions in the probed scene), from which a sibling stream already extracted
lane centrelines and a 356-node / 340-edge lane graph
(`Research/2026-08-02-nurec-xodr-map/`, MEASURED).

PUBLISHED, [arXiv 2606.03159](https://arxiv.org/html/2606.03159v2) (NVIDIA Cosmos-Dreams /
OmniDreams): their real-time generative closed-loop simulator conditions on *a single frame + text +
**per-frame coarse HD-map image** + trajectory poses*. So an ego-frame raster of a lane graph is
simultaneously (a) the strategic-topology input our hierarchy thesis has never had and (b) the exact
conditioning format the open-sourced generative simulator consumes. This module produces it.

⛔ WHAT THIS MODULE IS NOT
--------------------------
It is a **renderer**, not a claim. Rendering a raster says nothing about whether our model would use
it; that is a trained probe and a separate spend. And the raster carries **no speed target** — the
`<speed max>` field in this xodr is MEASURED-mislabelled (values are km/h under a `mph` unit tag) and
is additionally inconsistent with the observed driving, so it must not become a LONGITUDINAL label.

FRAME CONVENTION
----------------
Input polylines are in the **map frame** (metres, the xodr's own projected frame). The ego pose is
``(x, y, yaw_rad)`` in that same frame. Output is an ego-frame image with

    +x forward (image ROWS, increasing UPWARD),  +y left (image COLS, increasing LEFTWARD)

matching ``four_families``' waypoint convention, so a raster pixel and a waypoint metre mean the
same thing without a second transform. Row 0 of the returned array is the FAR end of the forward
range, so ``imshow`` renders it the way a driver sees a BEV: ego at the bottom, forward up.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["RasterSpec", "render_ego_raster", "render_track"]


class RasterSpec:
    """Geometry of the raster. Kept as an object so every number travels with the image.

    ``forward_m`` is ``(behind, ahead)``: a strategic input needs mostly the road ahead, but a small
    behind-margin keeps the ego off the image edge, where a convolution has no context.
    """

    def __init__(self, forward_m=(-8.0, 56.0), lateral_m=32.0, res_m=0.5, lane_width_px=1):
        self.behind, self.ahead = float(forward_m[0]), float(forward_m[1])
        self.lateral = float(lateral_m)
        self.res = float(res_m)
        self.lane_width_px = int(lane_width_px)
        self.H = int(round((self.ahead - self.behind) / self.res))
        self.W = int(round(2 * self.lateral / self.res))

    def to_dict(self):
        return {"forward_m": [self.behind, self.ahead], "lateral_m": self.lateral,
                "res_m": self.res, "shape": [3, self.H, self.W],
                "channels": ["lane_presence", "lane_dir_cos", "lane_dir_sin"],
                "frame": "ego: +x forward (rows, row 0 = far), +y left (cols, col 0 = far left)"}

    def rc(self, x, y):
        """ego metres -> (row, col) float arrays. No rounding, no clipping — the caller masks."""
        row = (self.ahead - np.asarray(x, float)) / self.res
        col = (self.lateral - np.asarray(y, float)) / self.res
        return row, col


def _resample(poly: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline to <= ``step`` spacing so rasterisation cannot leave gaps.

    ⛔ Without this a 16-point centreline over 15 m draws a dotted line at 0.5 m/px and a downstream
    readout sees texture that is an artefact of the source's sampling, not of the road.
    """
    if len(poly) < 2:
        return poly
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    n = np.maximum(1, np.ceil(seg / step).astype(int))
    out = [poly[:1]]
    for i, k in enumerate(n):
        t = np.linspace(0, 1, k + 1)[1:][:, None]
        out.append(poly[i] + t * (poly[i + 1] - poly[i]))
    return np.concatenate(out, axis=0)


def render_ego_raster(lanes: dict, pose, spec: RasterSpec | None = None) -> np.ndarray:
    """-> ``[3, H, W]`` float32 raster for ONE ego pose.

    ``lanes`` maps ``lane_id -> [[x, y], ...]`` polylines **in the map frame**.
    ``pose`` is ``(x, y, yaw_rad)`` in the map frame.

    Channels: ``lane_presence`` (0/1), and the lane's local heading **relative to the ego** encoded
    as ``cos``/``sin`` — written only where a lane is present, 0 elsewhere. The heading is what makes
    the raster a *directed* graph rather than a picture of tarmac: a lane that goes the other way is
    not a route option, and a presence-only raster cannot express that.

    ⚠️ Overlapping lanes at a junction OVERWRITE rather than average. Averaging two opposed headings
    yields ~0, i.e. the raster would claim "no direction" exactly at the junctions the strategic
    level exists for. Last-write-wins is wrong too, but it is wrong in a way that stays a valid unit
    vector; a multi-hypothesis encoding is the real fix and is named, not silently approximated.
    """
    spec = spec or RasterSpec()
    px, py, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(-yaw), math.sin(-yaw)
    img = np.zeros((3, spec.H, spec.W), dtype=np.float32)
    for poly in lanes.values():
        p = np.asarray(poly, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 2:
            continue
        # map -> ego
        dx, dy = p[:, 0] - px, p[:, 1] - py
        ex = c * dx - s * dy
        ey = s * dx + c * dy
        # cheap reject before the expensive resample
        if (ex.max() < spec.behind or ex.min() > spec.ahead
                or ey.max() < -spec.lateral or ey.min() > spec.lateral):
            continue
        q = _resample(np.stack([ex, ey], axis=1), spec.res * 0.5)
        d = np.diff(q, axis=0, append=q[-1:][None].reshape(1, 2))
        d[-1] = d[-2] if len(d) > 1 else d[-1]
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.divide(d, nrm, out=np.zeros_like(d), where=nrm > 1e-9)
        row, col = spec.rc(q[:, 0], q[:, 1])
        r = np.round(row).astype(int)
        cc = np.round(col).astype(int)
        ok = (r >= 0) & (r < spec.H) & (cc >= 0) & (cc < spec.W)
        if not ok.any():
            continue
        r, cc, d = r[ok], cc[ok], d[ok]
        for w in range(-(spec.lane_width_px // 2), spec.lane_width_px // 2 + 1):
            rr = np.clip(r + w, 0, spec.H - 1)
            img[0, rr, cc] = 1.0
            img[1, rr, cc] = d[:, 0]      # cos of the lane heading in the EGO frame
            img[2, rr, cc] = d[:, 1]      # sin
    return img


def render_track(lanes: dict, poses, spec: RasterSpec | None = None) -> np.ndarray:
    """-> ``[n, 3, H, W]`` for a sequence of poses ``[(x, y, yaw_rad), ...]``."""
    spec = spec or RasterSpec()
    return np.stack([render_ego_raster(lanes, p, spec) for p in poses], axis=0)
