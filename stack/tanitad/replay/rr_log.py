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

GT_COLOR = ARM_COLORS["gt"]
EGO_COLORS = {"speed": (120, 120, 120), "yaw_rate": (170, 170, 170)}
MANEUVER_PALETTE = ((31, 119, 180), (214, 39, 40), (148, 103, 189),
                    (140, 86, 75), (227, 119, 194), (127, 127, 127))


def arm_color(name: str) -> tuple[int, int, int]:
    """Consistent arm color; deterministic fallback for unknown arm names."""
    if name in ARM_COLORS:
        return ARM_COLORS[name]
    h = abs(hash(name))
    return (64 + h % 160, 64 + (h // 7) % 160, 64 + (h // 49) % 160)


def to_image_plane(xy: np.ndarray, h: int, w: int) -> np.ndarray:
    """Ego ground points [N, 2] (x fwd, y left, m) -> approx pixel coords."""
    x = np.clip(xy[:, 0], X_CLIP, None)
    f = F_EFF_256 * (h / 256.0)
    u = w / 2 - f * (xy[:, 1] / x)
    v = h / 2 + f * (CAM_H / x)
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
                 jpeg_quality: int | None = 85):
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
            grpc_url = rr.serve_grpc(server_memory_limit="25%")
            rr.serve_web_viewer(web_port=int(serve), open_browser=False,
                                connect_to=connect_url or grpc_url)
            self.serve_url = connect_url or grpc_url
        rr.send_blueprint(self._blueprint())
        self._styled: set[str] = set()
        self._cur_ep: tuple[str, int] | None = None

    # -- layout -----------------------------------------------------------
    @staticmethod
    def _blueprint():
        """Default 3-column layout: (camera + BEV + episode log) | (actions +
        error strip) | (monitor heads + maneuver + ego)."""
        return rrb.Blueprint(rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/camera", name="Camera + fans"),
                rrb.Spatial2DView(origin="/bev", name="BEV ego (m)"),
                rrb.TextLogView(origin="/meta", name="Episodes"),
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
              frame_hw: tuple[int, int] | None) -> None:
        """One arm's (or GT's) waypoint path into BEV + camera overlay."""
        path = np.vstack([[0.0, 0.0], wps])                 # from ego origin
        rr.log(f"/bev/{name}",
               rr.LineStrips2D([to_bev(path)], colors=[color], radii=0.12))
        if frame_hw is not None:
            h, w = frame_hw
            px = to_image_plane(path, h, w)
            rr.log(f"/camera/traj/{name}",
                   rr.LineStrips2D([px], colors=[color], radii=1.5))

    # -- the record ---------------------------------------------------------
    def log_record(self, rec: TimestepRecord) -> None:
        rr.set_time("step", sequence=rec.step)
        rr.set_time("episode", sequence=rec.ep_index)

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

        self._scalar("/ego/speed", rec.speed, EGO_COLORS["speed"],
                     "speed (m/s)")
        self._scalar("/ego/yaw_rate", rec.yaw_rate, EGO_COLORS["yaw_rate"],
                     "yaw rate (rad/s)")

        self._traj("gt", rec.gt_waypoints, GT_COLOR, frame_hw)
        self._scalar("/actions/steer/gt", rec.gt_action[0], GT_COLOR,
                     "gt steer")
        self._scalar("/actions/accel/gt", rec.gt_action[1], GT_COLOR,
                     "gt accel")

        for name, out in rec.arms.items():
            color = arm_color(name)
            if out.waypoints is not None:
                self._traj(name, np.asarray(out.waypoints), color, frame_hw)
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
