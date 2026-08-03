#!/usr/bin/env python3
"""Closed- OR open-loop video in the TanitEval house style: camera + metric BEV + HUD.

⭐ `--mode` decides which experiment the frame is FROM, and it is stamped on the frame:
  * `closed_loop` (default, unchanged behaviour) — the model drives; the cyan path is
    where the car actually went under its own plans.
  * `open_loop` — the ego is pinned to the LOGGED trajectory and the model only predicts.
    There is no driven path distinct from ground truth, so the cyan layer is dropped and
    the badge says OPEN-LOOP. ⛔ The two must never be confusable after the fact, which
    is why the mode is burned into the pixels and not only into the filename.

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


def draw_bev(size, scale, ego, plan, gt_xy, driven, actors, lead_idx, mode="closed_loop"):
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

    # GT is drawn FIRST and THICKER so it survives as a halo where the plan agrees with
    # it. Equal widths made the green vanish under the orange wherever the arm was right,
    # i.e. exactly where the viewer most needs to see that the two coincide.
    polyline(bev, [P(world_to_rig(p, ego)) for p in gt_xy], C_ROUTE, 5)
    # ⛔ In OPEN loop the ego IS the logged path, so a "driven" layer would be a second
    # copy of the green one and would read as agreement the run never demonstrated.
    if mode != "open_loop" and len(driven) > 1:
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


def draw_legend(canvas, x0, y0, w, mode):
    """Legend, in the dead space under the BEV.

    ⛔ "Draw GT and prediction distinguishably" is not satisfied by using two colours —
    the viewer has to be TOLD which is which, on the frame, or a screenshot of it is
    unreadable a week later.
    """
    import cv2
    leg = [("GROUND TRUTH — the logged path", C_ROUTE),
           ("MODEL PREDICTION — the plan, 0.5-2.0 s", C_PLAN)]
    if mode != "open_loop":
        leg.append(("DRIVEN — where the model actually went", C_DRIVEN))
    leg += [("annotated agent", C_ACTOR), ("LEAD agent", C_LEAD),
            ("ego", C_EGO)]
    cv2.putText(canvas, "LEGEND", (x0 + 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (150, 150, 150), 1, cv2.LINE_AA)
    for i, (txt, col) in enumerate(leg):
        yy = y0 + 40 + i * 20
        cv2.line(canvas, (x0 + 12, yy - 4), (x0 + 40, yy - 4), col, 4, cv2.LINE_AA)
        cv2.putText(canvas, txt, (x0 + 48, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def draw_mode_badge(img, mode):
    """Burn OPEN-LOOP / CLOSED-LOOP into the camera panel.

    The two experiments answer different questions and have been confused before; a
    filename is not a defence once the file has been copied somewhere else.
    """
    import cv2
    openl = mode == "open_loop"
    txt = "OPEN-LOOP" if openl else "CLOSED-LOOP"
    sub = ("ego on GROUND TRUTH - model predicts, does NOT drive"
           if openl else "model DRIVES - each frame rendered from where it went")
    col = (90, 220, 90) if openl else (40, 165, 255)
    h, w = img.shape[:2]
    x0, y0 = w - 470, 12
    cv2.rectangle(img, (x0, y0), (w - 12, y0 + 62), (12, 12, 12), -1)
    cv2.rectangle(img, (x0, y0), (w - 12, y0 + 62), col, 2)
    cv2.putText(img, txt, (x0 + 14, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2,
                cv2.LINE_AA)
    cv2.putText(img, sub, (x0 + 14, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (205, 205, 205), 1, cv2.LINE_AA)
    return img


def hud_lines(rec, st, m, arm, cond, mode="closed_loop", scene="00040136"):
    v = st["v"]
    openl = mode == "open_loop"
    head = (f"{arm}  |  scene {scene} (NuRec recon)  |  "
            + ("EGO ON LOGGED PATH  |  OPEN LOOP @10Hz - model predicts only"
               if openl else f"{cond.upper()} ROAD  |  CLOSED LOOP @10Hz - model drives"))
    ctrl = ("   steer {:+.3f} rad   accel {:+.2f} m/s2 (COMMANDED, NOT APPLIED)"
            if openl else "   steer {:+.3f} rad   accel {:+.2f} m/s2")
    lines = [
        head,
        f"t = {st['k'] * 0.1:5.1f}s   speed {v:5.2f} m/s   plan target {st['v_target']:5.2f} m/s"
        + ctrl.format(st["steer"], st["accel"]),
        f"TACTICAL   planned: {MAN_NAMES[m['man_plan']]:<11}"
        + (f" head: {MAN_NAMES[m['man_head']]:<11}" if m.get("man_head") is not None else " " * 18)
        + f" executed: {MAN_NAMES[m['man_exec']] if m.get('man_exec') is not None else '-':<11}"
        + f" logged: {MAN_NAMES[m['man_gt']]}",
        f"STRATEGIC  nav cmd fed: {NAV_NAMES[st['nav']]:<9}"
        + (f" route head: {ROUTE_NAMES[m['route_head']]:<9}" if m.get("route_head") is not None
           else " " * 21)
        + f" route logged: {ROUTE_NAMES[m['route_gt']]:<9} corridor: "
        + ("n/a (ego on GT)" if openl else
           ("OFF-ROUTE" if abs(m["cross_track"]) > 2.0 else "on-route")),
        (f"LATERAL lat-ADE {m['lat_ade']:5.2f} m   heading err {m['heading_err']:+.3f} rad"
         f"   LONGITUDINAL along-track {m['lon_ade']:5.2f} m  speed err {m['speed_err']:+5.2f} m/s"
         f"   plan ADE@2s {m['de2s']:5.2f} m" if openl else
         f"LATERAL cross-track {m['cross_track']:+5.2f} m   heading err {m['heading_err']:+.3f} rad"
         f"   LONGITUDINAL speed err {m['speed_err']:+5.2f} m/s   plan ADE@2s {m['de2s']:5.2f} m"),
    ]
    if m.get("synth_headway") is not None:
        hw, tg = m["synth_headway"], m.get("synth_time_gap")
        lines.append(
            f"CONSTRUCTED LEAD  headway {hw:+6.2f} m"
            + (f"   time-gap {tg:5.2f} s" if tg is not None else "")
            + f"   lateral {m['synth_y']:+5.2f} m"
            + ("   *** COLLISION (headway <= 0) ***" if hw <= 0 else ""))
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
    ap.add_argument("--mode", default=None, choices=["closed_loop", "open_loop"],
                    help="default: read `mode` from the rollout file, else closed_loop")
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

    cond = d["condition"]
    # ⚠️ The mode is taken from the PRODUCER's own record by default. Passing it only on
    # the command line would let a mislabelled invocation stamp CLOSED-LOOP on an
    # open-loop run, which is exactly the confusion the badge exists to prevent.
    mode = args.mode or d.get("mode") or "closed_loop"
    # In a CONSTRUCTED condition the drawn vehicle is not in `tracks` at all, so the BEV
    # would be empty and the HUD silent about the very thing under test. Score and draw
    # the synthetic lead from the geometry the rollout recorded.
    lead_ref = cond if cond not in ("empty", "objects", "logged") else None
    mets = per_step_metrics(rec, gt, tracks=tracks, lead_ref=lead_ref)
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
                 C_ROUTE, 9)
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
        draw_mode_badge(cam_img, mode)

        actors = list(m.get("actors", []))
        lead_idx = m.get("lead_idx", -1)
        if m.get("synth_x") is not None:
            sx, sy = float(m["synth_x"]), float(m["synth_y"])
            actors.append({"id": "SYNTH_LEAD", "xy": list(rig_to_world((sx, sy), ego)),
                           "yaw": ego[3], "rig": [sx, sy], "l": 3.08, "w": 1.63})
            lead_idx = len(actors) - 1
        bev = draw_bev(bev_s, bev_s / 110.0 * 2.2, ego, plan, gt_xy[i0:i0 + 80], driven,
                       actors, lead_idx, mode=mode)

        canvas = np.full((H, W, 3), 18, np.uint8)
        canvas[0:ch, 0:cw] = cam_img
        canvas[0:bev_s, cw:cw + bev_s] = bev
        if H > bev_s + 40:
            draw_legend(canvas, cw, bev_s + 6, bev_s, mode)
        y = max(ch, bev_s) + 24
        for i, line in enumerate(hud_lines(rec, st, m, d["arm"], d["condition"], mode,
                                           str(d.get("scene", "00040136"))[:8])):
            cv2.putText(canvas, line, (14, y + i * 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (235, 235, 235) if i else (120, 220, 255), 1, cv2.LINE_AA)
        vw.write(canvas)
    vw.release()

    # --- VERIFY BY DECODING, not by exit code ---------------------------------
    cap = cv2.VideoCapture(args.out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, fr = cap.read()
    cap.release()
    info = {"out": args.out, "mode": mode,
            "frames_written": min(len(files), len(rec["steps"])),
            "frames_decoded": n, "first_frame_ok": bool(ok),
            "duration_s": round(n / max(args.fps, 1), 2),
            "size": [W, H], "bytes": Path(args.out).stat().st_size,
            "mean_pixel": float(fr.mean()) if ok else None}
    print(json.dumps(info, indent=2))
    if not ok or n < 5:
        raise SystemExit("VIDEO VERIFICATION FAILED — the file does not decode")


if __name__ == "__main__":
    main()
