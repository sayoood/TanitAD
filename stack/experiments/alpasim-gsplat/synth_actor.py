#!/usr/bin/env python3
"""A CONSTRUCTED close-following / cut-in lead vehicle for the NuRec scene.

⛔ READ THIS FIRST — PROVENANCE, so this is never mistaken for something we found.
`scene_geometry.py` established, at TWO independent probes that agree to 2 cm, that the
scene on Thor contains NO close-following and NO cut-in geometry:

    probe 1  extracted/sequence_tracks.json (USDZ annotation, the renderer's own poses)
             -> min in-lane headway 46.246 m, 0 rows under 25 m / 2 s, 0 cut-in events
    probe 2  extracted/clipgt/obstacle.parquet (autolabels v2, read with pyarrow)
             -> min in-lane headway 46.224 m, 0 rows under 25 m / 2 s

So the configuration is **CONSTRUCTED, NOT FOUND**. What is real, and what is not:

  REAL   the gaussians. The lead vehicle is one of the scene's own reconstructed
         `dynamic_rigids` cuboids — a real car, with its real appearance.
  REAL   the road, the lane, the lighting, the camera model, the ego's logged route.
  REAL   the pose CONVENTION. `track_frame_convention` MEASURED it rather than assuming:
         a track's stored yaw equals its direction of travel to within a 0.5 deg median
         (IQR 3.7 deg, 92.8 % within 10 deg, n=1524), and a track's centre sits
         +0.662 m above the logged rig origin (n=1469). Both are used below.
  SYNTH  the lead's TRAJECTORY. It is the ego's own logged path replayed at a time
         offset, so the lead is exactly where the ego itself drove `lead_dt` later —
         guaranteed on-road, in-lane, correctly banked, and at a realistic speed.

WHY A TIME OFFSET AND NOT A RIGID "always N metres ahead"
A rigidly-attached lead makes headway a constant by construction, so distance-keeping
becomes unmeasurable and the policy's own actions feed back into its observation. With a
time offset the lead is anchored to the WORLD: if the policy brakes the gap opens, if it
accelerates the gap closes. That is the quantity we want to score.

CONDITIONS (`--condition`)
  lead25 / lead15 / lead8   lead_dt 1.6 / 1.0 / 0.55 s -> ~25 / ~15 / ~8 m headway.
                            A DOSE-RESPONSE, not a single point: a real reaction must be
                            monotone in headway, an artefact need not be.
  cutin                     lead_dt 0.9 s, lateral offset ramping +3.5 m -> 0 m over
                            1.5 s (from the LEFT, which is where this clip's real
                            adjacent-lane traffic actually passes).
  behind                    lead_dt -1.5 s. ⚠️ THE NEGATIVE CONTROL, and it is the
                            reason any of the above is quotable: the SAME gaussians are
                            uploaded and posed every frame, but behind a forward-facing
                            camera. Any metric that also moves under `behind` is not
                            caused by seeing a lead vehicle.
"""
from __future__ import annotations

import json
import math

import numpy as np

# MEASURED by scene_geometry.track_frame_convention on this scene — not assumed.
YAW_TRACK_MINUS_MOTION_RAD = math.radians(0.5)
Z_CENTRE_ABOVE_RIG_M = 0.662

CONDITIONS = {
    #             lead_dt_s  lat_from  lat_to  ramp_s  ramp_start_s
    "lead25":  dict(lead_dt=1.60, lat_from=0.0, lat_to=0.0, ramp_s=0.0, ramp_t0=0.0),
    "lead15":  dict(lead_dt=1.00, lat_from=0.0, lat_to=0.0, ramp_s=0.0, ramp_t0=0.0),
    "lead8":   dict(lead_dt=0.55, lat_from=0.0, lat_to=0.0, ramp_s=0.0, ramp_t0=0.0),
    "cutin":   dict(lead_dt=0.90, lat_from=3.5, lat_to=0.0, ramp_s=1.5, ramp_t0=2.0),
    "behind":  dict(lead_dt=-1.50, lat_from=0.0, lat_to=0.0, ramp_s=0.0, ramp_t0=0.0),
}
SYNTH_CONDITIONS = tuple(CONDITIONS)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    R = np.eye(3)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def _yaw(T):
    return math.atan2(T[1, 0], T[0, 0])


class LoggedPath:
    """The rig's logged path, queryable at an arbitrary wall time.

    Uses the NATIVE 30 Hz poses (not the 10 Hz control subsample) so the interpolation
    is smooth. Outside the logged range it extrapolates at constant velocity along the
    terminal heading, and COUNTS every such call — an extrapolated lead is a weaker
    stimulus and the count travels with the result instead of being hidden.
    """

    def __init__(self, renderer):
        n = renderer.n_frames()
        self.T = [renderer.gt_rig_to_world(f) for f in range(n)]
        self.t = np.array([float(renderer.frame_timestamps_us(f)[1]) for f in range(n)])
        self.xy = np.stack([T[:2, 3] for T in self.T])
        self.z = np.array([T[2, 3] for T in self.T])
        self.yaw = np.unwrap(np.array([_yaw(T) for T in self.T]))
        # roll/pitch of each logged pose, yaw removed (the road surface attitude)
        self.R_rp = [_rz(-_yaw(T)) @ T[:3, :3] for T in self.T]
        d = np.diff(self.xy, axis=0)
        dt = np.diff(self.t) / 1e6
        self.v = np.concatenate([np.linalg.norm(d, axis=1) / np.maximum(dt, 1e-9),
                                 [np.linalg.norm(d[-1]) / max(dt[-1], 1e-9)]])
        self.n_extrap = 0

    def at(self, t_us: float):
        """(xy[2], z, yaw, R_rp) at wall time t_us."""
        t = float(t_us)
        if t < self.t[0] or t > self.t[-1]:
            self.n_extrap += 1
            i = 0 if t < self.t[0] else len(self.t) - 1
            dt = (t - self.t[i]) / 1e6
            yaw = self.yaw[i]
            xy = self.xy[i] + self.v[i] * dt * np.array([math.cos(yaw), math.sin(yaw)])
            return xy, float(self.z[i]), float(yaw), self.R_rp[i]
        j = int(np.searchsorted(self.t, t))
        j = max(1, min(j, len(self.t) - 1))
        a = (t - self.t[j - 1]) / max(self.t[j] - self.t[j - 1], 1e-9)
        xy = (1 - a) * self.xy[j - 1] + a * self.xy[j]
        z = (1 - a) * self.z[j - 1] + a * self.z[j]
        yaw = (1 - a) * self.yaw[j - 1] + a * self.yaw[j]
        return xy, float(z), float(yaw), self.R_rp[j if a > 0.5 else j - 1]


class SynthLeadTracks:
    """ActorTracks-compatible stand-in holding exactly ONE synthetic track."""

    def __init__(self, path: LoggedPath, cond: str, t0_us: float | None = None):
        if cond not in CONDITIONS:
            raise ValueError(f"unknown synthetic condition {cond!r}")
        self.cfg = dict(CONDITIONS[cond])
        self.cond = cond
        self.path = path
        self.ids = ["SYNTH_LEAD"]
        self.labels = ["automobile"]
        self.t0_us = float(t0_us if t0_us is not None else path.t[0])
        self.n_calls = 0

    def __len__(self):
        return 1

    def lat_at(self, t_us: float) -> float:
        c = self.cfg
        if c["ramp_s"] <= 0:
            return float(c["lat_from"])
        s = (float(t_us) - self.t0_us) / 1e6 - c["ramp_t0"]
        a = float(np.clip(s / c["ramp_s"], 0.0, 1.0))
        return float((1 - a) * c["lat_from"] + a * c["lat_to"])

    def pose_at_time(self, i: int, t_us: float, tol_us: float = 1.5e5):
        self.n_calls += 1
        t = float(t_us) + self.cfg["lead_dt"] * 1e6
        xy, z, yaw, R_rp = self.path.at(t)
        lat = self.lat_at(t_us)
        # +lat is LEFT of the path heading
        xy = xy + lat * np.array([-math.sin(yaw), math.cos(yaw)])
        T = np.eye(4)
        T[:3, :3] = _rz(yaw + YAW_TRACK_MINUS_MOTION_RAD) @ R_rp
        T[:3, 3] = (xy[0], xy[1], z + Z_CENTRE_ABOVE_RIG_M)
        return T

    def pose_at(self, i: int, frac: float):
        return self.pose_at_time(i, self.path.t[0]
                                 + frac * (self.path.t[-1] - self.path.t[0]))


# ------------------------------------------------------------------------------ #
def pick_cuboid(renderer, layer="dynamic_rigids"):
    """Choose the actor cuboid with the MOST gaussians — the best-reconstructed car.

    Also returns its RENDERED extent, measured from the 5th-95th percentile of its own
    gaussian means (robust to the few stray splats every reconstruction carries). The
    half-length feeds the bumper-to-bumper headway, so the distance-keeping metric is
    referenced to the car we actually draw, not to a nominal 4.1 m.
    """
    cid = np.frombuffer(
        renderer.scene.sd[f".gaussians_nodes.{layer}.gaussian_cuboid_ids"],
        dtype=np.int32).copy()
    raw = renderer.scene.raw(layer)
    keep = renderer._finite_mask(raw)
    cid = cid[keep]
    u, c = np.unique(cid, return_counts=True)
    order = np.argsort(-c)
    best = int(u[order[0]])
    g = renderer.scene.gaussians(layer, time_basis=renderer._basis(layer, 0.0))
    m = g.means[cid == best]
    lo, hi = np.percentile(m, 5, axis=0), np.percentile(m, 95, axis=0)
    extent = (hi - lo).astype(float).tolist()
    return (best, {int(u[k]): int(c[k]) for k in order[:8]}, int(cid.size),
            [round(v, 3) for v in extent])


def attach_synth_lead(renderer, cond: str, layer="dynamic_rigids"):
    """Attach the constructed lead. Returns a provenance dict for the run payload."""
    path = LoggedPath(renderer)
    cub, counts, n_tot, extent = pick_cuboid(renderer, layer)
    tracks = SynthLeadTracks(path, cond)
    renderer.attach_actors(tracks, {cub: 0}, layer)
    return {"CONSTRUCTED": True, "condition": cond, "cfg": tracks.cfg,
            "cuboid_used": cub, "gaussians_in_cuboid": counts.get(cub),
            "top_cuboid_gaussian_counts": counts, "layer_gaussians_total": n_tot,
            "cuboid_extent_m_p5p95": extent,
            "yaw_offset_rad_measured": YAW_TRACK_MINUS_MOTION_RAD,
            "z_above_rig_m_measured": Z_CENTRE_ABOVE_RIG_M,
            "provenance": "gaussians REAL (scene's own cuboid); trajectory SYNTHESISED "
                          "as the ego's logged path replayed at a time offset. Two "
                          "probes agree no such geometry exists in the scene."}


# ------------------------------------------------------------------------------ #
# lead geometry, computed on EVERY run regardless of what is rendered              #
# ------------------------------------------------------------------------------ #
EGO_FRONT_OVERHANG_M = 3.7


class LeadGeometry:
    """Ego-frame geometry of EVERY synthetic lead, evaluated at every control tick.

    ⚠️ This is deliberately computed even in the `empty` and `behind` conditions, where
    nothing is drawn. It is a purely geometric quantity, so the empty control carries a
    perfectly matched COUNTERFACTUAL headway: "if a lead had been there, how close would
    you have come?". Without it, the LONGITUDINAL family would be uncomputable on the
    control arm and the with-vs-without comparison would have nothing to pair against.
    """

    def __init__(self, renderer, half_len_m: float | None = None,
                 layer: str = "dynamic_rigids"):
        self.path = LoggedPath(renderer)
        self.tracks = {c: SynthLeadTracks(self.path, c) for c in CONDITIONS}
        # ⚠️ The half-length MUST NOT depend on the condition. It once did — it was
        # taken from the attach info, which only exists when an actor is attached, so
        # the `empty` control silently used a 2.0 m default while every lead condition
        # used the measured 1.542 m. The `behind` negative control caught it as a
        # +0.458 m headway delta that could not exist (behind and empty render
        # identically and drive identically). Deriving it from the renderer here makes
        # it condition-independent BY CONSTRUCTION.
        if half_len_m is None:
            _, _, _, extent = pick_cuboid(renderer, layer)
            half_len_m = 0.5 * float(max(extent[:2]))
        self.half_len = float(half_len_m)

    def set_t0(self, t_us: float):
        for tr in self.tracks.values():
            tr.t0_us = float(t_us)

    def at(self, T_ego: np.ndarray, t_us: float, v_ego: float):
        Ti = np.linalg.inv(T_ego)
        out = {}
        for c, tr in self.tracks.items():
            T = tr.pose_at_time(0, t_us)
            T2 = tr.pose_at_time(0, t_us + 1.0e5)
            p = Ti[:3, :3] @ T[:3, 3] + Ti[:3, 3]
            x, y = float(p[0]), float(p[1])
            v_lead = float(np.linalg.norm(T2[:2, 3] - T[:2, 3]) / 0.1)
            hw = x - EGO_FRONT_OVERHANG_M - self.half_len
            closing = v_ego - v_lead
            out[c] = dict(
                x=round(x, 3), y=round(y, 3), headway_m=round(hw, 3),
                v_lead=round(v_lead, 3),
                time_gap_s=(round(hw / v_ego, 4) if v_ego > 0.5 else None),
                ttc_s=(round(hw / closing, 3) if (closing > 0.1 and hw > 0) else None))
        return out


# ------------------------------------------------------------------------------ #
# THE FALSIFIER — run before any metric is quoted                                  #
# ------------------------------------------------------------------------------ #
def verify_synth(renderer, scene_dir, frames=(30, 90, 150),
                 conds=("lead25", "lead15", "lead8", "cutin", "behind")):
    """Can the manipulation even be SEEN, and does it behave like a lead vehicle?

    Three things must hold, and all three are reported whether they hold or not:
      1. `behind` changes essentially no pixels (the negative control).
      2. the visible conditions change pixels, and the count is MONOTONE in closeness
         (lead8 > lead15 > lead25).
      3. the changed-pixel centroid lands near the ANALYTIC projection of the actor
         centre, computed from the f-theta polynomial — i.e. the car is where the
         geometry says it should be, not merely somewhere.
    """
    from gsplat_renderer import ActorTracks  # noqa: F401  (kept: API parity)
    out = {"frames": [], "per_cond": {}}
    base = {}
    for f in frames:
        c2n = renderer.gt_cam_to_nre(f)
        renderer._actor = None
        img, _, _ = renderer.render(c2n)
        base[f] = img
    W, H = renderer.width, renderer.height
    poly = renderer.cam.angle_to_pixeldist_poly
    cx, cy = float(renderer.cam.cx), float(renderer.cam.cy)

    for cond in conds:
        info = attach_synth_lead(renderer, cond)
        rows = []
        for f in frames:
            ts = renderer.frame_timestamps_us(f)[1]
            c2n = renderer.gt_cam_to_nre(f)
            img, _, _ = renderer.render(c2n, actor_time_us=float(ts))
            d = (np.abs(img.astype(np.int32) - base[f].astype(np.int32)) > 8).any(-1)
            n = int(d.sum())
            cen = ([float(x) for x in np.argwhere(d).mean(0)[::-1]] if n else None)
            # analytic projection of the actor centre through the f-theta polynomial
            T_w = renderer._actor["tracks"].pose_at_time(0, float(ts))
            p_nre = renderer.rig.world_to_nre @ T_w
            p_c = np.linalg.inv(c2n) @ np.array([p_nre[0, 3], p_nre[1, 3],
                                                 p_nre[2, 3], 1.0])
            pred = None
            if p_c[2] > 0.1:
                th = math.atan2(math.hypot(p_c[0], p_c[1]), p_c[2])
                rr = sum(poly[i] * th ** i for i in range(len(poly)))
                nrm = math.hypot(p_c[0], p_c[1]) or 1e-9
                pred = [cx + rr * p_c[0] / nrm, cy + rr * p_c[1] / nrm]
            ego_T = renderer.gt_rig_to_world(f)
            dxy = np.linalg.inv(ego_T)[:3, :3] @ T_w[:3, 3] + np.linalg.inv(ego_T)[:3, 3]
            rows.append(dict(frame=f, pixels_changed=n,
                             frac_frame=round(n / (W * H), 5),
                             centroid=(None if cen is None else [round(v, 1) for v in cen]),
                             predicted_centroid=(None if pred is None
                                                 else [round(v, 1) for v in pred]),
                             centroid_err_px=(None if (cen is None or pred is None) else
                                              round(float(math.dist(cen, pred)), 1)),
                             ego_frame_x_m=round(float(dxy[0]), 2),
                             ego_frame_y_m=round(float(dxy[1]), 2)))
        out["per_cond"][cond] = {"attach": info, "rows": rows,
                                 "mean_pixels_changed": float(np.mean(
                                     [r["pixels_changed"] for r in rows]))}
    px = {c: out["per_cond"][c]["mean_pixels_changed"] for c in conds}
    out["negative_control_behind_pixels"] = px.get("behind")
    out["monotone_in_closeness"] = bool(
        px.get("lead8", 0) > px.get("lead15", 0) > px.get("lead25", 0))
    out["behind_is_invisible"] = bool(px.get("behind", 1e9) < 0.02 * px.get("lead15", 1))
    errs = [r["centroid_err_px"] for c in ("lead25", "lead15", "lead8")
            for r in out["per_cond"].get(c, {}).get("rows", [])
            if r["centroid_err_px"] is not None]
    out["centroid_err_px_median"] = (round(float(np.median(errs)), 1) if errs else None)
    out["verdict"] = ("ACCEPTED" if (out["monotone_in_closeness"]
                                     and out["behind_is_invisible"]
                                     and (out["centroid_err_px_median"] or 1e9) < 120)
                      else "REFUSED")
    return out


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", default="/tmp/synth_verify.json")
    ap.add_argument("--loader-dir", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gsplat_renderer import NuRecGsplatRenderer
    sd = Path(a.scene_dir).expanduser()
    r = NuRecGsplatRenderer(sd, loader_dir=a.loader_dir)
    v = verify_synth(r, sd)
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(json.dumps({k: v[k] for k in ("negative_control_behind_pixels",
                                        "monotone_in_closeness", "behind_is_invisible",
                                        "centroid_err_px_median", "verdict")}, indent=2))
    for c, d in v["per_cond"].items():
        print(f"{c:8s} mean_px={d['mean_pixels_changed']:9.0f}  "
              + "  ".join(f"f{r['frame']}: x={r['ego_frame_x_m']:6.1f} "
                          f"n={r['pixels_changed']:7d} err={r['centroid_err_px']}"
                          for r in d["rows"]))
    print("wrote", a.out)
