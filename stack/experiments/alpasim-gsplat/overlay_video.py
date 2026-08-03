#!/usr/bin/env python3
"""Closed-loop video in the TanitEval house style: camera + metric BEV + decision HUD.

The programme's standing visualisation standard (memory: "TanitEval viz standard") is
camera projection AND a metric BEV inset TOGETHER, with a text overlay of the model's
decoded tactical manoeuvre and its strategic route/goal. That is what this renders:

  * LEFT  — the f-theta camera frame the policy actually saw, with its own plan
            projected through the **real f-theta forward polynomial**
            (`tanitad.data.calib.ftheta_project_ray`), not a pinhole approximation,
            plus the logged route for reference.
  * RIGHT — a metric BEV, ego at bottom-centre, rotated to ego: logged route (green),
            the closed-loop path actually driven (cyan), the current plan (orange),
            annotated agents (yellow, with the lead agent called out), range rings.
  * HUD   — arm, tick, speed vs the plan's target speed, the decoded TACTICAL
            manoeuvre (planned / executed / logged) and the STRATEGIC route command,
            cross-track error and the running plan-ADE.

Written with cv2's VideoWriter because Thor has no ffmpeg (probed). The output is
verified by DECODING it back, never by exit code.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

MAN_NAMES = ("lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop")
NAV_NAMES = ("follow", "left", "right", "straight")
ROUTE_NAMES = ("left", "straight", "right", "unknown")

C_PLAN = (40, 165, 255)      # orange  (BGR)
C_ROUTE = (90, 220, 90)      # green
C_DRIVEN = (255, 220, 60)    # cyan
C_EGO = (255, 255, 255)
C_ACTOR = (60, 230, 250)     # yellow
C_LEAD = (80, 80, 255)       # red


def densify(traj, n=20):
    knots = np.vstack([[0.0, 0.0], np.asarray(traj, float)])
    ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    tq = np.linspace(0.0, 2.0, n)
    return np.stack([np.interp(tq, ts, knots[:, 0]), np.interp(tq, ts, knots[:, 1])], 1)


def rig_to_world(pt_rig, ego):
    x, y, z, yaw = ego
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * pt_rig[0] - s * pt_rig[1], y + s * pt_rig[0] + c * pt_rig[1]])


def world_to_rig(p, ego):
    x, y, z, yaw = ego
    d = np.asarray(p, float)[:2] - np.array([x, y])
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])


def project_rig_points(pts_rig, intr, T_sensor_rig, w, h, z_rig=0.0):
    """Rig-frame ground points -> native f-theta pixels, via the real polynomial."""
    from tanitad.data.calib import ftheta_project_ray
    inv = np.linalg.inv(T_sensor_rig)
    out = []
    for p in pts_rig:
        pc = inv @ np.array([p[0], p[1], z_rig, 1.0])
        if pc[2] <= 0.3:                      # behind / at the camera: not projectable
            out.append(None)
            continue
        u, v = ftheta_project_ray(intr, (float(pc[0]), float(pc[1]), float(pc[2])))
        out.append((int(round(u)), int(round(v))) if (-w < u < 2 * w and -h < v < 2 * h)
                   else None)
    return out


def polyline(img, pts, color, thick=3):
    import cv2
    prev = None
    for p in pts:
        if p is not None and prev is not None:
            cv2.line(img, prev, p, color, thick, cv2.LINE_AA)
        prev = p


def draw_bev(size, scale, ego, plan, gt_xy, driven, actors, lead_idx):
    """Metric BEV, ego at bottom-centre, rotated to ego (the house layout)."""
    import cv2
    bev = np.full((size, size, 3), 24, np.uint8)
    cx, cy = size // 2, int(size * 0.82)

    def P(xy_rig):
        return (int(cx - xy_rig[1] * scale), int(cy - xy_rig[0] * scale))

    for rng in (10, 20, 30, 40):
        cv2.circle(bev, (cx, cy), int(rng * scale), (52, 52, 52), 1, cv2.LINE_AA)
        cv2.putText(bev, f"{rng}m", (cx + 4, cy - int(rng * scale) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (95, 95, 95), 1, cv2.LINE_AA)
    cv2.line(bev, (cx, 0), (cx, size), (44, 44, 44), 1)

    polyline(bev, [P(world_to_rig(p, ego)) for p in gt_xy], C_ROUTE, 2)
    if len(driven) > 1:
        polyline(bev, [P(world_to_rig(p, ego)) for p in driven], C_DRIVEN, 2)
    polyline(bev, [P(p) for p in densify(plan, 24)], C_PLAN, 2)
    for h, k in zip((0.5, 1.0, 1.5, 2.0), range(4)):
        cv2.circle(bev, P(plan[k]), 4, C_PLAN, -1, cv2.LINE_AA)

    for j, a in enumerate(actors):
        r = world_to_rig(a["xy"], ego)
        col = C_LEAD if j == lead_idx else C_ACTOR
        yaw_r = a["yaw"] - ego[3]
        L, W = a.get("l", 4.5) / 2, a.get("w", 1.9) / 2
        corners = [(L, W), (L, -W), (-L, -W), (-L, W)]
        c, s = math.cos(yaw_r), math.sin(yaw_r)
        pts = np.array([P((r[0] + c * dx - s * dy, r[1] + s * dx + c * dy))
                        for dx, dy in corners], np.int32)
        cv2.polylines(bev, [pts], True, col, 2, cv2.LINE_AA)
        if j == lead_idx:
            cv2.putText(bev, "LEAD %.0fm" % r[0], (pts[:, 0].min(), pts[:, 1].min() - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    car = np.array([P((2.2, 0.9)), P((2.2, -0.9)), P((-2.2, -0.9)), P((-2.2, 0.9))], np.int32)
    cv2.polylines(bev, [car], True, C_EGO, 2, cv2.LINE_AA)
    cv2.putText(bev, "BEV (metric, ego-centred)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return bev


def hud_lines(rec, st, m, arm, cond):
    v = st["v"]
    lines = [
        f"{arm}  |  scene 00040136 (NuRec recon)  |  {cond.upper()} ROAD  |  closed loop @10Hz",
        f"t = {st['k'] * 0.1:5.1f}s   speed {v:5.2f} m/s   plan target {st['v_target']:5.2f} m/s"
        f"   steer {st['steer']:+.3f} rad   accel {st['accel']:+.2f} m/s2",
        f"TACTICAL   planned: {MAN_NAMES[m['man_plan']]:<11}"
        + (f" head: {MAN_NAMES[m['man_head']]:<11}" if m.get("man_head") is not None else " " * 18)
        + f" executed: {MAN_NAMES[m['man_exec']] if m.get('man_exec') is not None else '-':<11}"
        + f" logged: {MAN_NAMES[m['man_gt']]}",
        f"STRATEGIC  nav cmd fed: {NAV_NAMES[st['nav']]:<9}"
        + (f" route head: {ROUTE_NAMES[m['route_head']]:<9}" if m.get("route_head") is not None
           else " " * 21)
        + f" route logged: {ROUTE_NAMES[m['route_gt']]:<9} corridor: "
        + ("OFF-ROUTE" if abs(m["cross_track"]) > 2.0 else "on-route"),
        f"LATERAL cross-track {m['cross_track']:+5.2f} m   heading err {m['heading_err']:+.3f} rad"
        f"   LONGITUDINAL speed err {m['speed_err']:+5.2f} m/s   plan ADE@2s {m['de2s']:5.2f} m",
    ]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--cam-w", type=int, default=1280)
    ap.add_argument("--bev", type=int, default=520)
    ap.add_argument("--tracks", default=None)
    args = ap.parse_args()

    import cv2

    from cl_metrics import per_step_metrics, load_rollouts
    from gsplat_renderer import ActorTracks
    from nurec_loader import RigTrajectories

    d = json.loads(Path(args.rollouts).read_text())
    rec = d["rollouts"][0]
    gt = d["gt"]
    gt_xy = np.array([[g["x"], g["y"]] for g in gt])
    sd = Path(args.scene_dir).expanduser()
    rig = RigTrajectories(sd / "rig_trajectories.json")
    cam = rig.camera("camera_front_wide_120fov")
    Ts = np.asarray(cam.T_sensor_rig, np.float64)
    from tanitad.data.calib import FThetaIntrinsics
    intr = FThetaIntrinsics(poly=tuple(cam.angle_to_pixeldist_poly), cx=float(cam.cx),
                            cy=float(cam.cy), width=int(cam.width), height=int(cam.height),
                            per_clip=True)

    tracks = None
    if args.tracks and Path(args.tracks).exists():
        tracks = ActorTracks(args.tracks)

    mets = per_step_metrics(rec, gt, tracks=tracks)
    fdir = Path(args.frames)
    files = sorted(fdir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no frames in {fdir}")

    cw = args.cam_w
    ch = int(round(cw * cam.height / cam.width))
    bev_s = args.bev
    H = max(ch, bev_s) + 130
    W = cw + bev_s
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("cv2.VideoWriter refused to open — no usable mp4 encoder")

    driven = []
    for k, f in enumerate(files):
        if k >= len(rec["steps"]):
            break
        st = rec["steps"][k]
        m = mets[k]
        img = cv2.imread(str(f))
        if img is None:
            continue
        ego = st["ego"]
        driven.append([ego[0], ego[1]])
        plan = np.array(st["plan"], float)

        # --- camera overlay, in NATIVE pixels, then resized once ------------------
        i0 = max(0, st["i_gt"])
        route_rig = [world_to_rig(p, ego) for p in gt_xy[i0:i0 + 60]]
        polyline(img, project_rig_points(route_rig, intr, Ts, cam.width, cam.height),
                 C_ROUTE, 3)
        polyline(img, project_rig_points(densify(plan, 24), intr, Ts, cam.width, cam.height),
                 C_PLAN, 4)
        for k4 in range(4):
            p = project_rig_points([plan[k4]], intr, Ts, cam.width, cam.height)[0]
            if p:
                cv2.circle(img, p, 7, C_PLAN, -1, cv2.LINE_AA)
                cv2.putText(img, f"{(k4 + 1) * 0.5:.1f}s", (p[0] + 9, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_PLAN, 2, cv2.LINE_AA)
        cam_img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        cv2.putText(cam_img, "front-wide f-theta (gsplat render of the NuRec scene)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

        actors = m.get("actors", [])
        bev = draw_bev(bev_s, bev_s / 110.0 * 2.2, ego, plan, gt_xy[i0:i0 + 80], driven,
                       actors, m.get("lead_idx", -1))

        canvas = np.full((H, W, 3), 18, np.uint8)
        canvas[0:ch, 0:cw] = cam_img
        canvas[0:bev_s, cw:cw + bev_s] = bev
        y = max(ch, bev_s) + 24
        for i, line in enumerate(hud_lines(rec, st, m, d["arm"], d["condition"])):
            cv2.putText(canvas, line, (14, y + i * 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (235, 235, 235) if i else (120, 220, 255), 1, cv2.LINE_AA)
        vw.write(canvas)
    vw.release()

    # --- VERIFY BY DECODING, not by exit code ---------------------------------
    cap = cv2.VideoCapture(args.out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, fr = cap.read()
    cap.release()
    info = {"out": args.out, "frames_written": min(len(files), len(rec["steps"])),
            "frames_decoded": n, "first_frame_ok": bool(ok),
            "size": [W, H], "bytes": Path(args.out).stat().st_size,
            "mean_pixel": float(fr.mean()) if ok else None}
    print(json.dumps(info, indent=2))
    if not ok or n < 5:
        raise SystemExit("VIDEO VERIFICATION FAILED — the file does not decode")


if __name__ == "__main__":
    main()
