#!/usr/bin/env python3
"""Does a CLOSE-FOLLOWING / CUT-IN geometry exist in the NuRec scene material?

WHY THIS EXISTS
---------------
The with-objects vs empty-road closed-loop panel came back NULL (19 of 20 paired deltas
have CIs containing zero). That result only tested DISTANT traffic: the actors were
40-45 m ahead, ~2.8 s of gap, 0.02-0.4 % of the frame. The named follow-up is a
CUT-IN / CLOSE-FOLLOWING scene. Before running anything we have to answer, with the
scene's own annotations, whether that geometry is present at all.

This probe is deliberately CHEAP: it reads only `rig_trajectories.json` (ego rig poses,
30 Hz) and the USDZ's `sequence_tracks.json` (78 annotated agent tracks with world poses,
timestamps and cuboid dimensions). No GPU, no `volume.nurec`, no gsplat.

WHAT IT COMPUTES, per 10 Hz ego tick x per alive track
  * ego-frame position (x forward, y left) from inv(T_rig_world) @ p_world
  * range, bearing, in-front-camera-FOV test
  * BUMPER-TO-BUMPER headway  = range_along_x - (ego_front_overhang + actor_half_length)
  * time gap  = headway / v_ego            (the distance-keeping metric)
  * TTC       = headway / (v_ego - v_act)  (only when closing)
  * projected frame-area fraction from the f-theta linear pixel-per-radian coefficient

SELF-CONSISTENCY CONTROL (mandatory, and it has teeth): the ego's OWN pose 1 s in the
future, expressed in the current ego frame, must land at x > 0 with |y| small while the
rig is moving forward. If the frame convention were transposed or inverted this control
fails and every "lead vehicle" below would be an artefact of the transform. It is
evaluated FIRST and its result is carried in the output.

TWO INDEPENDENT PROBES of the same fact (the operating standard's absence rule):
  probe 1 = `sequence_tracks.json` (the USDZ annotation the renderer itself is posed from)
  probe 2 = `extracted/clipgt/obstacle.parquet` (the clip's own obstacle table)
Both are run when available; a disagreement is reported, never averaged away.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

CAM = "camera_front_wide_120fov"
DT = 0.1

# Ego geometry. The rig origin on this platform is the rear axle (the same assumption
# `closedloop_drive.WHEELBASE=2.7` encodes); a passenger car's front bumper is ~3.7 m
# ahead of it. Stated as a CONSTANT, not measured here, so a reader can re-run with
# another value: it shifts every headway by a fixed offset and cannot create or destroy
# a close-following event by itself.
EGO_FRONT_OVERHANG_M = 3.7
LANE_HALF_W = 1.8          # in-lane test, |y| <= this
ADJACENT_MAX = 5.4         # a track beyond this is not a plausible cut-in source
FOV_HALF_DEG = 60.0        # front-wide 120 deg


# ------------------------------------------------------------------------------ #
def quat_xyzw_to_R(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


class Ego:
    """Rig poses + timestamps, subsampled to the 10 Hz control rate."""

    def __init__(self, rig_json: Path, cam=CAM):
        d = json.load(open(rig_json))
        e = d["rig_trajectories"][0]
        key = f"{cam}@{e['sequence_id']}"
        Ts = e["cameras_frame_T_rig_worlds"][key]
        ts = e["cameras_frame_timestamps_us"][key]
        self.cam_calib = d["camera_calibrations"][key]
        n_native = len(Ts)
        dt_us = ts[1][1] - ts[0][1]
        self.stride = int(round(1e5 / dt_us))
        idx = list(range(0, n_native, self.stride))
        self.T = [np.array(Ts[i][1], np.float64) for i in idx]
        self.ts = [float(ts[i][1]) for i in idx]
        self.native_dt_us = dt_us
        self.n = len(self.T)
        xy = np.stack([T[:2, 3] for T in self.T])
        v = np.zeros(self.n)
        if self.n > 1:
            v[:-1] = np.linalg.norm(np.diff(xy, axis=0), axis=1) / DT
            v[-1] = v[-2]
        self.v = v
        self.xy = xy

    def to_ego(self, k: int, p_world) -> np.ndarray:
        """World point -> ego frame of tick k. Full 4x4 inverse (not yaw-only)."""
        Ti = np.linalg.inv(self.T[k])
        p = np.asarray(p_world, np.float64)
        return (Ti[:3, :3] @ p + Ti[:3, 3])

    def self_check(self):
        """The mandatory control: my own future must be AHEAD of me."""
        rows = []
        for k in range(0, self.n - 10):
            if self.v[k] < 1.0:
                continue
            p = self.to_ego(k, self.T[k + 10][:3, 3])
            rows.append((float(p[0]), float(p[1])))
        if not rows:
            return {"n": 0, "verdict": "NO_MOVING_TICKS"}
        a = np.array(rows)
        ok = bool((a[:, 0] > 0).all() and np.median(np.abs(a[:, 1])) < 3.0)
        return {"n": len(rows),
                "future_x_min_m": round(float(a[:, 0].min()), 3),
                "future_x_median_m": round(float(np.median(a[:, 0])), 3),
                "future_absy_median_m": round(float(np.median(np.abs(a[:, 1]))), 3),
                "frac_x_positive": round(float((a[:, 0] > 0).mean()), 4),
                "verdict": "PASS" if ok else "FAIL"}


class Tracks:
    def __init__(self, path: Path):
        d = json.load(open(path))
        ch = d[next(iter(d))]
        td = ch["tracks_data"]
        self.ids = [str(x) for x in td["tracks_id"]]
        self.labels = [str(x) for x in td["tracks_label_class"]]
        self.poses = [np.asarray(p, np.float64) for p in td["tracks_poses"]]
        self.ts = [np.asarray(t, np.float64) for t in td["tracks_timestamps_us"]]
        cd = ch.get("cuboidtracks_data", {})
        self.dims = ([np.asarray(x, np.float64) for x in cd["cuboids_dims"]]
                     if "cuboids_dims" in cd else None)

    def __len__(self):
        return len(self.ids)

    def at(self, i, t_us, tol_us=1.5e5):
        ts = self.ts[i]
        if ts.size == 0 or t_us < ts[0] - tol_us or t_us > ts[-1] + tol_us:
            return None
        k = int(np.argmin(np.abs(ts - t_us)))
        if abs(ts[k] - t_us) > tol_us:
            return None
        r = self.poses[i][k]
        T = np.eye(4)
        T[:3, :3] = quat_xyzw_to_R(r[3], r[4], r[5], r[6])
        T[:3, 3] = r[:3]
        return T

    def speed_at(self, i, t_us):
        ts, p = self.ts[i], self.poses[i]
        if ts.size < 2:
            return 0.0
        k = int(np.argmin(np.abs(ts - t_us)))
        a, b = max(0, k - 1), min(ts.size - 1, k + 1)
        dt = (ts[b] - ts[a]) / 1e6
        if dt <= 0:
            return 0.0
        return float(np.linalg.norm(p[b, :2] - p[a, :2]) / dt)


# ------------------------------------------------------------------------------ #
def build_table(ego: Ego, tr: Tracks, f_px: float, W: int, H: int,
                renderable: set | None = None):
    rows = []
    for k in range(ego.n):
        t_us = ego.ts[k]
        for i in range(len(tr)):
            T = tr.at(i, t_us)
            if T is None:
                continue
            p = ego.to_ego(k, T[:3, 3])
            x, y = float(p[0]), float(p[1])
            rng = math.hypot(x, y)
            if rng < 1e-3:
                continue
            bearing = math.degrees(math.atan2(y, x))
            dim = tr.dims[i] if tr.dims is not None else np.array([4.1, 1.85, 1.58])
            L, Wd, Ht = float(dim[0]), float(dim[1]), float(dim[2])
            headway = x - EGO_FRONT_OVERHANG_M - 0.5 * L
            v_e = float(ego.v[k])
            v_a = tr.speed_at(i, t_us)
            closing = v_e - v_a
            tg = headway / v_e if v_e > 0.5 else float("inf")
            ttc = headway / closing if (closing > 0.1 and headway > 0) else float("inf")
            # solid-angle -> pixels via the f-theta LINEAR coefficient (small-angle;
            # exact at the image centre, conservative off-axis).
            area_frac = (Wd * Ht) / max(rng * rng, 1e-6) * (f_px ** 2) / (W * H)
            rows.append(dict(
                k=k, t_us=t_us, track=i, track_id=tr.ids[i], label=tr.labels[i],
                x=round(x, 3), y=round(y, 3), rng=round(rng, 3),
                bearing_deg=round(bearing, 2),
                headway_m=round(headway, 3), time_gap_s=round(tg, 3),
                ttc_s=(round(ttc, 3) if math.isfinite(ttc) else None),
                v_ego=round(v_e, 3), v_act=round(v_a, 3),
                len_m=round(L, 2), wid_m=round(Wd, 2),
                area_frac=round(float(area_frac), 6),
                in_fov=bool(x > 0 and abs(bearing) <= FOV_HALF_DEG),
                in_lane=bool(abs(y) <= LANE_HALF_W),
                renderable=(None if renderable is None else bool(i in renderable)),
            ))
    return rows


def find_lead_events(rows, max_headway=25.0, max_time_gap=2.0):
    """In-lane, in-FOV, ahead, closer than a threshold. The close-following test."""
    out = [r for r in rows
           if r["in_fov"] and r["in_lane"] and r["headway_m"] > 0
           and (r["headway_m"] <= max_headway or r["time_gap_s"] <= max_time_gap)]
    return sorted(out, key=lambda r: r["headway_m"])


def find_cutins(rows, enter_lane=LANE_HALF_W, from_out=2.6, max_x=45.0, win=25):
    """A track whose |y| crosses INTO the ego lane while ahead of the ego.

    `win` ticks (2.5 s) is the maximum time allowed for the crossing, so a track that
    drifts across over 10 s is not counted as a cut-in.
    """
    by_track = {}
    for r in rows:
        by_track.setdefault(r["track"], []).append(r)
    events = []
    for tid, rs in by_track.items():
        rs.sort(key=lambda r: r["k"])
        for a in range(len(rs)):
            ra = rs[a]
            if abs(ra["y"]) < from_out or not (0 < ra["x"] <= max_x):
                continue
            for b in range(a + 1, len(rs)):
                rb = rs[b]
                if rb["k"] - ra["k"] > win:
                    break
                if abs(rb["y"]) <= enter_lane and 0 < rb["x"] <= max_x:
                    events.append({
                        "track": tid, "track_id": ra["track_id"], "label": ra["label"],
                        "k_start": ra["k"], "k_end": rb["k"],
                        "dt_s": round((rb["k"] - ra["k"]) * DT, 2),
                        "y_start": ra["y"], "y_end": rb["y"],
                        "x_start": ra["x"], "x_end": rb["x"],
                        "headway_end_m": rb["headway_m"],
                        "time_gap_end_s": rb["time_gap_s"],
                        "renderable": rb["renderable"],
                        "area_frac_end": rb["area_frac"]})
                    break
    return events


# ------------------------------------------------------------------------------ #
def probe2_obstacle_parquet(clipgt_dir: Path, ego: "Ego"):
    """SECOND, INDEPENDENT probe of the SAME fact — different file, different parser.

    `extracted/clipgt/obstacle.parquet` is the clip's own autolabelled obstacle table
    (`scene:obstacles:autolabels:v2`), read with pyarrow. It is transformed into the ego
    frame with the SAME rig poses and re-scored with the SAME thresholds, so a
    disagreement with probe 1 would be visible rather than averaged away.
    """
    p = clipgt_dir / "obstacle.parquet"
    if not p.exists():
        return {"available": False, "reason": f"{p} missing"}
    try:
        import pyarrow.parquet as pq
    except Exception as e:                                   # noqa: BLE001
        return {"available": False, "reason": f"pyarrow unavailable: {e!r}"}
    d = pq.read_table(p).to_pydict()
    ob, ky = d["obstacle"], d["key"]
    ts = np.array([float(k["timestamp_micros"]) for k in ky])
    cen = np.array([[o["center"]["x"], o["center"]["y"], o["center"]["z"]] for o in ob])
    siz = np.array([[o["size"]["x"], o["size"]["y"], o["size"]["z"]] for o in ob])
    cat = [str(o["category"]) for o in ob]
    tid = [str(o["trackline_id"]) for o in ob]

    inlane, close, rows = [], [], []
    for k in range(ego.n):
        m = np.abs(ts - ego.ts[k]) <= 5.0e4          # +-50 ms of this 10 Hz tick
        if not m.any():
            continue
        for j in np.flatnonzero(m):
            p_e = ego.to_ego(k, cen[j])
            x, y = float(p_e[0]), float(p_e[1])
            if x <= 0:
                continue
            bearing = math.degrees(math.atan2(y, x))
            if abs(bearing) > FOV_HALF_DEG:
                continue
            hw = x - EGO_FRONT_OVERHANG_M - 0.5 * float(siz[j, 0])
            v_e = float(ego.v[k])
            tg = hw / v_e if v_e > 0.5 else float("inf")
            r = dict(k=k, track_id=tid[j], label=cat[j], x=round(x, 3), y=round(y, 3),
                     headway_m=round(hw, 3), time_gap_s=round(tg, 3))
            rows.append(r)
            if abs(y) <= LANE_HALF_W and hw > 0:
                inlane.append(r)
                if hw <= 25.0 or tg <= 2.0:
                    close.append(r)
    inlane.sort(key=lambda r: r["headway_m"])
    return {"available": True, "path": str(p), "n_rows": int(len(ts)),
            "n_infov_ahead_rows": len(rows),
            "n_inlane_ahead_rows": len(inlane),
            "min_inlane_headway_m": (inlane[0]["headway_m"] if inlane else None),
            "n_close_following_rows": len(close),
            "closest_5_inlane": inlane[:5],
            "agrees_with_probe1": None}


def track_frame_convention(ego: Ego, tr: Tracks):
    """MEASURED: how a track's stored orientation relates to its own direction of travel.

    Needed before any pose can be SYNTHESISED — if the cuboid's local +x is not the
    direction of motion, a constructed lead vehicle would be rendered sideways. Also
    measures the height of a track centre above the logged rig origin, which is what a
    synthetic actor must reproduce to sit on the road rather than float.
    """
    dyaw, dz = [], []
    for i in range(len(tr)):
        ts, p = tr.ts[i], tr.poses[i]
        if ts.size < 3:
            continue
        for k in range(1, ts.size - 1):
            d = p[k + 1, :2] - p[k - 1, :2]
            if np.linalg.norm(d) < 0.5:              # parked / noise
                continue
            R = quat_xyzw_to_R(p[k, 3], p[k, 4], p[k, 5], p[k, 6])
            a = math.atan2(R[1, 0], R[0, 0]) - math.atan2(d[1], d[0])
            dyaw.append((a + math.pi) % (2 * math.pi) - math.pi)
            j = int(np.argmin(np.linalg.norm(ego.xy - p[k, :2][None, :], axis=1)))
            if np.linalg.norm(ego.xy[j] - p[k, :2]) < 60.0:
                dz.append(float(p[k, 2] - ego.T[j][2, 3]))
    if not dyaw:
        return {"n": 0, "verdict": "NO_MOVING_TRACKS"}
    a = np.array(dyaw)
    med = float(np.median(a))
    return {"n": len(a),
            "yaw_track_minus_yaw_motion_median_deg": round(math.degrees(med), 3),
            "yaw_iqr_deg": round(math.degrees(float(np.percentile(a, 75)
                                                    - np.percentile(a, 25))), 3),
            "frac_within_10deg_of_median": round(
                float((np.abs(a - med) < math.radians(10)).mean()), 4),
            "z_centre_minus_rig_origin_median_m": (round(float(np.median(dz)), 3)
                                                   if dz else None),
            "z_n": len(dz),
            "verdict": ("FORWARD_IS_PLUS_X" if abs(med) < math.radians(10)
                        else "NOT_PLUS_X")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--actor-map", default=None,
                    help="results/actor_map.json — restricts to RENDERABLE tracks")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sd = Path(a.scene_dir).expanduser()
    ego = Ego(sd / "rig_trajectories.json")
    tr = Tracks(sd / "extracted" / "sequence_tracks.json")

    ctrl = ego.self_check()
    print("SELF-CONSISTENCY (own future is ahead):", json.dumps(ctrl))
    if ctrl["verdict"] != "PASS":
        raise SystemExit("ego-frame control FAILED — refusing to report geometry from a "
                         "transform that cannot place my own future in front of me.")

    cm = ego.cam_calib["camera_model"]["parameters"]
    poly = [float(x) for x in cm["angle_to_pixeldist_poly"]]
    f_px = poly[1]
    W, H = int(cm["resolution"][0]), int(cm["resolution"][1])

    renderable = None
    if a.actor_map and Path(a.actor_map).exists():
        am = json.load(open(a.actor_map))
        renderable = {int(x["best_track"]) for x in am["per_track"] if x["accepted"]}

    rows = build_table(ego, tr, f_px, W, H, renderable)
    lead_all = find_lead_events(rows)
    lead_rend = [r for r in lead_all if r["renderable"]] if renderable is not None else []
    cut_all = find_cutins(rows)
    cut_rend = [c for c in cut_all if c["renderable"]] if renderable is not None else []

    inlane_ahead = [r for r in rows if r["in_fov"] and r["in_lane"] and r["headway_m"] > 0]
    min_headway = min((r["headway_m"] for r in inlane_ahead), default=None)

    summary = {
        "scene": sd.name,
        "ego_ticks_10hz": ego.n, "native_dt_us": ego.native_dt_us, "stride": ego.stride,
        "ego_speed_ms": {"min": round(float(ego.v.min()), 3),
                         "max": round(float(ego.v.max()), 3),
                         "mean": round(float(ego.v.mean()), 3)},
        "camera": {"W": W, "H": H, "f_px_per_rad": round(f_px, 2)},
        "self_consistency_control": ctrl,
        "n_tracks": len(tr), "n_renderable_tracks": (len(renderable) if renderable else None),
        "n_rows_alive": len(rows),
        "n_inlane_ahead_infov_rows": len(inlane_ahead),
        "min_inlane_headway_m": min_headway,
        "n_close_following_rows_ALL": len(lead_all),
        "n_close_following_rows_RENDERABLE": len(lead_rend),
        "n_cutin_events_ALL": len(cut_all),
        "n_cutin_events_RENDERABLE": len(cut_rend),
        "closest_10_inlane": lead_all[:10] if lead_all else
                             sorted(inlane_ahead, key=lambda r: r["headway_m"])[:10],
        "cutins_ALL": cut_all[:40],
        "track_frame_convention": track_frame_convention(ego, tr),
        "probe2_obstacle_parquet": probe2_obstacle_parquet(sd / "extracted" / "clipgt", ego),
    }
    p2 = summary["probe2_obstacle_parquet"]
    if p2.get("available"):
        p2["agrees_with_probe1"] = bool(
            (p2["n_close_following_rows"] == 0) == (len(lead_all) == 0))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("closest_10_inlane", "cutins_ALL")}, indent=2))
    print("\nCLOSEST IN-LANE AHEAD (headway m, time-gap s, area%, renderable):")
    for r in summary["closest_10_inlane"]:
        print(f"  k={r['k']:3d} trk={r['track']:2d} {r['label']:10s} "
              f"x={r['x']:7.2f} y={r['y']:6.2f} headway={r['headway_m']:7.2f} "
              f"tg={r['time_gap_s']:6.2f} area={100*r['area_frac']:6.3f}% "
              f"rend={r['renderable']}")
    print(f"\nCUT-IN EVENTS: all={len(cut_all)} renderable={len(cut_rend)}")
    for c in cut_all[:15]:
        print("  ", json.dumps(c))
    print("wrote", out)


if __name__ == "__main__":
    main()
