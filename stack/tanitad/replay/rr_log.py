"""rerun logging schema for the replay app (viz backbone, README verdict).

Entity map (one consistent color per arm everywhere, ARM_COLORS):

    /camera                     latest frame of the anchor stack (image)
    /camera/traj/gt|<arm>       trajectory fan projected onto the road
                                (approx pinhole, viz_trajectory_fan constants)
    /bev/gt|<arm>               ego-frame waypoint paths (m, forward = up)
    /bev/<arm>/imagination      MainArm/RefA imagination fan (A3 decode rays)
    /actions/steer|accel/...    per-arm action readout vs gt (time series)
    /error/<arm>                per-step waypoint ADE — the error-strip
                                timeline: scrub to the spikes
    /heads/conf|ood|sigma/<arm> fallback/monitor scalars
    /heads/imag_rel/<arm>/k<k>  imagination self-monitor per horizon (A9)
    /maneuver/<class>, /maneuver/gt_class   REF-B tactical distribution
    /ego/speed, /ego/yaw_rate   replayed ego kinematics
    /meta                       episode boundaries (corpus, episode id)

Timelines: ``step`` (global window counter — the scrub axis) and ``episode``
(episode counter, for jumping between routes). A default blueprint is sent on
init so the first open is already the screenshot-worthy 3-column layout.

rerun is an optional dependency: importing this module without ``rerun-sdk``
is fine; constructing :class:`RerunLogger` then raises with the install hint
(no silent no-op logger — viz mode without rerun is a config error).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tanitad.replay.engine import TimestepRecord
from tanitad.replay.arms import ARM_COLORS

try:
    import rerun as rr
    import rerun.blueprint as rrb
    HAVE_RERUN = True
except ImportError:                                    # pragma: no cover
    rr = rrb = None
    HAVE_RERUN = False

# Approximate pinhole ground-plane projection (scripts/viz_trajectory_fan.py,
# D-016): f_eff 266 px at 256 px frames, camera height ~1.22 m. Labeled
# approximate — for intuition, not measurement.
F_EFF_256, CAM_H, X_CLIP = 266.0, 1.22, 2.0


@dataclass(frozen=True)
class CamProjection:
    """Per-corpus ground-plane pinhole for the overlay fan (D-016 follow-up).

    The fan is drawn with ``v = horizon_row + f * cam_h / x`` (forward x, metres)
    and ``u = w/2 - f * y / x``. The three parameters are the corpus's *canonical*
    camera geometry — the geometry of the 256-px frame ``calib.py`` actually
    produces, NOT the raw sensor:

    f_eff_256 : effective focal [px] at a 256-px frame.
    cam_h     : camera height above the road plane [m].
    pitch_deg : camera pitch below horizontal [deg]; +down raises the horizon.

    comma2k19 (EON road cam, visually verified) is the default. physicalai's
    values come from the dataset's OWN calibration (nvidia/PhysicalAI-AV
    ``calibration/``): the front-wide is an f-theta lens whose near-centre focal
    is 925 px @1920 -> 444 px @256 after the central crop (NOT the 266 the
    nominal-rectilinear crop in ``calib.py`` assumes); sensor extrinsics give
    height 1.43 m and pitch 0.31 deg +/- 0.78 (horizon 128.0 +/- 6.1 px, i.e.
    == h/2 — the horizon is NOT offset). The overlay defect was the ~2x
    under-estimate of ``f * cam_h`` (266*1.22=325 vs 444*1.43=635), which drew
    the fan ~2x too close to the horizon ("pointing at the sky"), not a pitch.
    """
    f_eff_256: float = F_EFF_256
    cam_h: float = CAM_H
    pitch_deg: float = 0.0

    def focal(self, h: int) -> float:
        return self.f_eff_256 * (h / 256.0)

    def horizon_row(self, h: int) -> float:
        """Image row of the horizon (x->inf). Pitched-down cameras (+deg) put
        the horizon above the principal row h/2."""
        return h / 2.0 - self.focal(h) * math.tan(math.radians(self.pitch_deg))


# comma2k19 is the historical hardcoded reference (unchanged); physicalai from
# real calibration (see Benchmarks & Eval/FLAGSHIP_FIX_PLAN.md). pitch_deg=0.0:
# the measured 0.31 deg is negligible (<3 px) and the full calibration (incl.
# principal point) lands the horizon on h/2, so the material corrections are the
# focal (266->444) and height (1.22->1.43), not the horizon.
COMMA_CAM = CamProjection(f_eff_256=266.0, cam_h=1.22, pitch_deg=0.0)
PHYSICALAI_CAM = CamProjection(f_eff_256=444.0, cam_h=1.43, pitch_deg=0.0)
_CAM_BY_PREFIX = (("physicalai", PHYSICALAI_CAM), ("comma", COMMA_CAM))


def cam_for_corpus(corpus: str | None) -> CamProjection:
    """Resolve a corpus tag (engine tag == cache-dir name, e.g.
    ``physicalai-val-8c0d3047924e``) to its overlay camera model. Unknown tags
    fall back to the comma2k19 reference (fail-safe: the verified geometry)."""
    if corpus:
        low = corpus.lower()
        for pre, cam in _CAM_BY_PREFIX:
            if pre in low:
                return cam
    return COMMA_CAM


GT_COLOR = ARM_COLORS["gt"]
EGO_COLORS = {"speed": (120, 120, 120), "yaw_rate": (170, 170, 170)}
MANEUVER_PALETTE = ((31, 119, 180), (214, 39, 40), (148, 103, 189),
                    (140, 86, 75), (227, 119, 194), (127, 127, 127))

# BEV reference frame for the labelled metric grid + scale bar (metres, ego).
BEV_GRID_FWD = (0.0, 10.0, 20.0, 30.0)     # forward gridlines (m)
BEV_GRID_LAT = (-10.0, 0.0, 10.0)          # lateral gridlines (m, +y = left)
BEV_LAT_HALF = 15.0                         # lateral half-extent drawn
BEV_FWD_MAX = 30.0                          # forward extent drawn
BEV_AXIS_COLOR = (70, 80, 95)
BEV_LABEL_COLOR = (150, 160, 175)


def _hex(c: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2]))


def arm_color(name: str) -> tuple[int, int, int]:
    """Consistent arm color; deterministic fallback for unknown arm names."""
    if name in ARM_COLORS:
        return ARM_COLORS[name]
    h = abs(hash(name))
    return (64 + h % 160, 64 + (h // 7) % 160, 64 + (h // 49) % 160)


def to_image_plane(xy: np.ndarray, h: int, w: int,
                   cam: CamProjection | None = None) -> np.ndarray:
    """Ego ground points [N, 2] (x fwd, y left, m) -> approx pixel coords.

    ``cam`` is the per-corpus camera model (:func:`cam_for_corpus`); ``None``
    keeps the comma2k19 reference geometry, so callers that predate the
    per-corpus split are byte-identical to the old hardcoded projection."""
    cam = cam or COMMA_CAM
    x = np.clip(xy[:, 0], X_CLIP, None)
    f = cam.focal(h)
    u = w / 2 - f * (xy[:, 1] / x)
    v = cam.horizon_row(h) + f * (cam.cam_h / x)
    return np.stack([u, v], axis=1)


def to_bev(xy: np.ndarray) -> np.ndarray:
    """Ego (x fwd, y left) -> 2D view coords with forward = up, right = right
    (rerun 2D y grows downward)."""
    return np.stack([-xy[..., 1], -xy[..., 0]], axis=-1)


class RerunLogger:
    """Streams TimestepRecords into a rerun recording (.rrd and/or live).

    Parameters
    ----------
    rrd : path — save the recording to a ``.rrd`` artifact (offline viewing:
        ``rerun file.rrd`` or drag into https://app.rerun.io).
    serve : web-viewer port — serve a live viewer (``rr.serve_grpc`` +
        ``rr.serve_web_viewer``); see the README for RunPod proxy access.
    connect_url : override the viewer's data-stream URL (needed behind an
        HTTP proxy where localhost gRPC URLs are unreachable from the
        browser).
    maneuver_classes : names for /maneuver entities (default: the REF-B
        vocabulary).
    """

    def __init__(self, rrd: str | None = None, serve: int | None = None,
                 connect_url: str | None = None,
                 app_id: str = "tanitad_replay",
                 maneuver_classes: tuple[str, ...] | None = None,
                 jpeg_quality: int | None = 85,
                 grpc_port: int | None = None,
                 grpc_only: bool = False):
        if not HAVE_RERUN:
            raise RuntimeError(
                "rerun-sdk is not installed — `pip install rerun-sdk` "
                "(>=0.34) or run --mode test without --rrd/--serve")
        if rrd is None and serve is None:
            raise ValueError("RerunLogger needs an --rrd path and/or a "
                             "--serve port; a sink-less logger logs nowhere")
        if maneuver_classes is None:
            from tanitad.refs.refb import MANEUVER_CLASSES
            maneuver_classes = MANEUVER_CLASSES
        self.maneuver_classes = tuple(maneuver_classes)
        self.jpeg_quality = jpeg_quality
        self.serve_url: str | None = None

        rr.init(app_id)
        if rrd is not None:
            rr.save(str(rrd))
        if serve is not None:
            # Single-proxied-port pods (e.g. only 8888 exposed on RunPod):
            # grpc_only serves ONLY the data stream on `serve`, and the user
            # opens the HOSTED viewer app.rerun.io/?url=<proxied stream> —
            # no second port needed.
            kw = {"server_memory_limit": "25%"}
            if grpc_port is not None or grpc_only:
                kw["grpc_port"] = int(grpc_port if grpc_port is not None
                                      else serve)
            grpc_url = rr.serve_grpc(**kw)
            if not grpc_only:
                rr.serve_web_viewer(web_port=int(serve), open_browser=False,
                                    connect_to=connect_url or grpc_url)
            self.serve_url = connect_url or grpc_url
        rr.send_blueprint(self._blueprint())
        self._styled: set[str] = set()
        self._cur_ep: tuple[str, int] | None = None
        self._legend_logged = False
        self._log_bev_axes()

    # -- layout -----------------------------------------------------------
    @staticmethod
    def _blueprint():
        """Default 3-column layout: (camera + BEV + episode log) | (actions +
        error strip) | (monitor heads + maneuver + ego)."""
        return rrb.Blueprint(rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/camera", name="Camera + fans"),
                rrb.Spatial2DView(origin="/bev", name="BEV ego (m, fwd=up)"),
                rrb.Horizontal(
                    rrb.TextLogView(origin="/meta", name="Episodes"),
                    rrb.TextDocumentView(origin="/legend", name="Legend")),
                row_shares=[3, 3, 1]),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/actions/steer", name="Steer (rad)"),
                rrb.TimeSeriesView(origin="/actions/accel",
                                   name="Accel (m/s^2)"),
                rrb.TimeSeriesView(origin="/error",
                                   name="Waypoint ADE (m) — error strip")),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/heads", name="Monitor heads"),
                rrb.TimeSeriesView(origin="/maneuver",
                                   name="REF-B maneuver distribution"),
                rrb.TimeSeriesView(origin="/ego", name="Ego kinematics")),
            column_shares=[3, 3, 2]))

    # -- static legends / axes ---------------------------------------------
    def _log_bev_axes(self) -> None:
        """Static labelled metric grid + 10 m scale bar under ``/bev/axes`` so
        the BEV panel reads in metres without the viewer guessing the scale."""
        grid = [to_bev(np.array([[d, -BEV_LAT_HALF], [d, BEV_LAT_HALF]]))
                for d in BEV_GRID_FWD]
        grid += [to_bev(np.array([[0.0, lat], [BEV_FWD_MAX, lat]]))
                 for lat in BEV_GRID_LAT]
        rr.log("/bev/axes/grid",
               rr.LineStrips2D(grid, colors=[BEV_AXIS_COLOR] * len(grid),
                               radii=0.02, draw_order=-10.0), static=True)
        marks = [d for d in BEV_GRID_FWD if d > 0]
        pts = np.stack([to_bev(np.array([[d, BEV_LAT_HALF * 0.9]]))[0]
                        for d in marks])
        self._safe_labeled_points("/bev/axes/fwd_labels", pts,
                                  [f"{int(d)} m" for d in marks],
                                  BEV_LABEL_COLOR)
        bar = to_bev(np.array([[1.0, -BEV_LAT_HALF + 1.0],
                               [11.0, -BEV_LAT_HALF + 1.0]]))
        self._safe_labeled_strip("/bev/axes/scale_bar", [bar], "10 m",
                                 (230, 235, 240), radii=0.08)

    def _safe_labeled_points(self, path, positions, labels, color) -> None:
        try:
            rr.log(path, rr.Points2D(positions, labels=labels,
                                     colors=[color] * len(labels), radii=0.01,
                                     show_labels=True), static=True)
        except TypeError:                              # older show_labels API
            rr.log(path, rr.Points2D(positions, labels=labels,
                                     colors=[color] * len(labels), radii=0.01),
                   static=True)

    def _safe_labeled_strip(self, path, strips, label, color, radii) -> None:
        try:
            rr.log(path, rr.LineStrips2D(strips, colors=[color], radii=radii,
                                         labels=[label], show_labels=True),
                   static=True)
        except TypeError:
            rr.log(path, rr.LineStrips2D(strips, colors=[color], radii=radii,
                                         labels=[label]), static=True)

    def _log_legend(self, arm_names) -> None:
        """One `/legend` TextDocument mapping every arm (and GT) to its color
        and entity paths — logged once, on the first record, when the actual
        arm set is known."""
        rows = ["# TanitAD replay — legend", "",
                "| arm | color | entities |", "|---|---|---|",
                f"| **GT** (ground truth) | `{_hex(GT_COLOR)}` | "
                f"`/bev/gt`, `/camera/traj/gt`, `/actions/*/gt` |"]
        for n in arm_names:
            c = arm_color(n)
            rows.append(f"| **{n}** | `{_hex(c)}` | `/bev/{n}`, "
                        f"`/camera/traj/{n}`, `/error/{n}` |")
        rows += ["", "BEV: forward = up, +y = left; grid every 10 m; "
                 "scale bar = 10 m. Camera fans use a per-corpus pinhole "
                 "ground projection (calibrated focal/height/pitch; D-016)."]
        rr.log("/legend", rr.TextDocument(
            "\n".join(rows), media_type=rr.MediaType.MARKDOWN), static=True)

    # -- primitives ---------------------------------------------------------
    def _scalar(self, path: str, value: float,
                color: tuple[int, int, int], label: str) -> None:
        if path not in self._styled:
            rr.log(path, rr.SeriesLines(colors=[color], names=[label]),
                   static=True)
            self._styled.add(path)
        rr.log(path, rr.Scalars(float(value)))

    def _image(self, frame: np.ndarray) -> None:
        img = rr.Image(frame)
        if self.jpeg_quality is not None and frame.ndim == 3:
            try:                     # presentation-layer only; raw is correct
                img = img.compress(jpeg_quality=self.jpeg_quality)
            except Exception:
                pass
        rr.log("/camera", img)

    def _traj(self, name: str, wps: np.ndarray,
              color: tuple[int, int, int],
              frame_hw: tuple[int, int] | None,
              cam: CamProjection | None = None) -> None:
        """One arm's (or GT's) waypoint path into BEV + camera overlay.

        Both strips carry the arm ``name`` as a label so the entity tree and
        hover tooltips read cleanly (no anonymous line strips). ``cam`` is the
        per-corpus overlay camera (defaults to the comma2k19 reference)."""
        label = "GT" if name == "gt" else name
        path = np.vstack([[0.0, 0.0], wps])                 # from ego origin
        rr.log(f"/bev/{name}",
               rr.LineStrips2D([to_bev(path)], colors=[color], radii=0.12,
                               labels=[label]))
        if frame_hw is not None:
            h, w = frame_hw
            px = to_image_plane(path, h, w, cam=cam)
            rr.log(f"/camera/traj/{name}",
                   rr.LineStrips2D([px], colors=[color], radii=1.5,
                                   labels=[label]))

    # -- the record ---------------------------------------------------------
    def log_record(self, rec: TimestepRecord) -> None:
        rr.set_time("step", sequence=rec.step)
        rr.set_time("episode", sequence=rec.ep_index)

        if not self._legend_logged:                   # arm set known now
            self._log_legend(list(rec.arms.keys()))
            self._legend_logged = True

        if (rec.corpus, rec.episode_id) != self._cur_ep:
            self._cur_ep = (rec.corpus, rec.episode_id)
            rr.log("/meta", rr.TextLog(
                f"episode {rec.episode_id} [{rec.corpus}] enters at "
                f"step {rec.step} (t={rec.t})"))

        frame_hw: tuple[int, int] | None = None
        if rec.frame is not None:
            self._image(rec.frame)
            if rec.frame.ndim == 3:      # camera stack -> overlay projection
                frame_hw = rec.frame.shape[0], rec.frame.shape[1]
        cam = cam_for_corpus(rec.corpus)   # per-corpus overlay geometry (D-016)

        self._scalar("/ego/speed", rec.speed, EGO_COLORS["speed"],
                     "speed (m/s)")
        self._scalar("/ego/yaw_rate", rec.yaw_rate, EGO_COLORS["yaw_rate"],
                     "yaw rate (rad/s)")

        self._traj("gt", rec.gt_waypoints, GT_COLOR, frame_hw, cam)
        self._scalar("/actions/steer/gt", rec.gt_action[0], GT_COLOR,
                     "gt steer")
        self._scalar("/actions/accel/gt", rec.gt_action[1], GT_COLOR,
                     "gt accel")

        for name, out in rec.arms.items():
            color = arm_color(name)
            if out.waypoints is not None:
                self._traj(name, np.asarray(out.waypoints), color, frame_hw,
                           cam)
                ade = float(np.linalg.norm(
                    np.asarray(out.waypoints) - rec.gt_waypoints,
                    axis=-1).mean())
                self._scalar(f"/error/{name}", ade, color, f"{name} ADE")
            if out.imag_traj:
                faded = (*color, 130)
                rays = [to_bev(np.vstack([[0.0, 0.0], pt[None]]))
                        for pt in out.imag_traj.values()]
                rr.log(f"/bev/{name}/imagination",
                       rr.LineStrips2D(rays, colors=[faded] * len(rays),
                                       radii=0.06))
            if out.action is not None:
                self._scalar(f"/actions/steer/{name}", out.action[0], color,
                             f"{name} steer")
                self._scalar(f"/actions/accel/{name}", out.action[1], color,
                             f"{name} accel")
            for head in ("conf", "ood", "sigma"):
                v = getattr(out, head)
                if v is not None:
                    self._scalar(f"/heads/{head}/{name}", v, color,
                                 f"{name} {head}")
            if out.imag_rel:
                for k, v in out.imag_rel.items():
                    self._scalar(f"/heads/imag_rel/{name}/k{k}", v, color,
                                 f"{name} imag_rel@{k}")
            if out.maneuver_probs is not None:
                for i, cls in enumerate(self.maneuver_classes):
                    self._scalar(
                        f"/maneuver/{cls}", float(out.maneuver_probs[i]),
                        MANEUVER_PALETTE[i % len(MANEUVER_PALETTE)], cls)
                if out.maneuver_gt is not None:
                    self._scalar("/maneuver/gt_class", float(out.maneuver_gt),
                                 GT_COLOR, "gt class idx")

    def close(self) -> None:
        """Flush the recording (rrd files are complete after this)."""
        rec = rr.get_global_data_recording()
        if rec is not None:
            rec.flush()
