"""End-to-end demo: sync -> vehicle frame -> trajectory -> validation -> visualisation.

Runs on a *subset* of the data so it finishes in a couple of minutes.

    python run_demo.py --session 0811 --frames 600 --render 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trajlib import quality as Q
from trajlib import timesync as TS
from trajlib import viz
from trajlib.camera import calibrate_camera
from trajlib.frame_folder import load_frame_folder
from trajlib.io_sensorlogger import load_reassembled_session, load_session
from trajlib.trajectory import ego_trajectory, estimate_trajectory
from trajlib.validate import validate
from trajlib.vehicle_frame import estimate_vehicle_frame

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_0814 = os.path.join(ROOT, "Liebst_ckelweg-2025-08-14_15-15-02-android")
SESS_0811 = os.path.join(ROOT, "work", "session_2025-08-11")
FRAMES_0811 = os.path.join(ROOT, "synchronized_output")
OUT = os.path.join(ROOT, "work", "demo_out")


def banner(msg):
    print("\n" + "=" * 74)
    print(msg)
    print("=" * 74, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="0811", choices=["0811", "0814"])
    ap.add_argument("--frames", type=int, default=600, help="frames used for sync/calibration")
    ap.add_argument("--render", type=int, default=24, help="how many overlay figures to render")
    ap.add_argument("--cam-height", type=float, default=1.25)
    ap.add_argument("--vehicle-width", type=float, default=1.8)
    ap.add_argument("--t-past", type=float, default=3.0)
    ap.add_argument("--t-future", type=float, default=5.0)
    ap.add_argument("--video", action="store_true",
                    help="also write overlay.mp4 (image-only overlay, every usable frame)")
    ap.add_argument("--video-fps", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ------------------------------------------------------------------ load
    banner(f"1. LOAD SESSION [{args.session}]")
    if args.session == "0811":
        session = load_reassembled_session(SESS_0811)
        video = load_frame_folder(FRAMES_0811, limit=args.frames)
        print(f"frame folder: {video.nb_frames} frames, {video.width}x{video.height}, "
              f"{video.duration:.2f}s @ {video.avg_fps:.3f} fps")
    else:
        session = load_session(RAW_0814)
        video = TS.probe_video(session.video_path)
        print(f"video: {video.width}x{video.height} {video.duration:.2f}s "
              f"{video.nb_frames} frames @ {video.avg_fps:.3f} fps")
    print(session.summary())

    banner("2. DATA QUALITY GATES")
    rep = Q.merge(Q.check_gps(session.get("gps")), Q.check_imu(session),
                  Q.check_motion(session.get("gps")))
    print(rep)
    if not rep.ok:
        print("\n*** QC FAILED - this recording cannot produce trustworthy ground truth. ***")
        print("*** Continuing anyway for demonstration; do NOT ship such a session.  ***")

    # ------------------------------------------------------------------ sync
    banner("3. VIDEO <-> SENSOR SYNCHRONISATION")
    t0 = time.time()
    cam_flow = TS.image_angular_rate(video)
    print(f"optical flow on {len(cam_flow[0])} frame pairs in {time.time() - t0:.1f}s")
    sync = TS.synchronise(session, video, cam_flow=cam_flow)
    print(f"  anchor container      : {sync.anchor_container}")
    print(f"  anchor creation-end   : {sync.anchor_creation_end}")
    print(f"  xcorr shift           : {sync.xcorr_shift}")
    print(f"  score                 : {sync.xcorr_peak}")
    print(f"  => t_video_start      : {sync.t_video_start:.4f} s   [{sync.source}]")

    # ------------------------------------------------------- vehicle frame
    banner("4. PHONE -> VEHICLE FRAME")
    vframe = estimate_vehicle_frame(session)
    print(vframe.summary())

    # ------------------------------------------------------------ trajectory
    banner("5. SESSION TRAJECTORY (EKF forward + RTS backward)")
    t0 = time.time()
    traj = estimate_trajectory(session, vframe)
    print(f"solved {traj.meta['n']} steps in {time.time() - t0:.1f}s")
    print(f"  accel source : tier {traj.meta['accel_tier']} - {traj.meta['accel_source']}")
    print(f"  speed        : {traj.speed.min():.2f} .. {traj.speed.max():.2f} m/s")
    print(f"  path length  : {np.hypot(np.diff(traj.E), np.diff(traj.N)).sum():.1f} m")
    print(f"  pos 1-sigma  : median {np.median(traj.pos_std()):.2f} m")

    banner("6. VALIDATION")
    print(validate(session, vframe, traj, k=5))

    # ------------------------------------------------------------- camera
    banner("7. CAMERA CALIBRATION")
    cam = calibrate_camera(video, session, sync, traj=traj, cam_flow=cam_flow,
                           vframe=vframe, height_m=args.cam_height)
    print(cam.summary())

    # ----------------------------------------------------------- rendering
    banner("8. RENDER BEV + IMAGE PROJECTION")
    import cv2

    frame_times = sync.frame_times(video.pts)
    usable = np.nonzero((frame_times >= traj.t[0] + args.t_past) &
                        (frame_times <= traj.t[-1] - args.t_future))[0]
    if len(usable) == 0:
        print("no frame has both a full past and a full future window - nothing to render")
        return
    pick = usable[np.linspace(0, len(usable) - 1, min(args.render, len(usable))).astype(int)]
    print(f"{len(usable)} frames have a complete [-{args.t_past}, +{args.t_future}] s window; "
          f"rendering {len(pick)}")

    frames_dir = os.path.join(OUT, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    records = []
    for n, fi in enumerate(pick):
        t_ref = float(frame_times[fi])
        ego = ego_trajectory(traj, t_ref, t_past=args.t_past, t_future=args.t_future, dt_out=0.1)
        if ego is None:
            continue
        img = (video.read_bgr(int(fi)) if hasattr(video, "read_bgr")
               else _grab_bgr(video.path, float(video.pts[fi])))
        if img is None:
            continue
        over = viz.draw_trajectory_on_image(img, ego, cam, vehicle_width=args.vehicle_width)
        over = viz.draw_hud(over, ego, {"frame": f"{fi}  pts={video.pts[fi]:.3f}s"})

        fig = plt.figure(figsize=(16, 6.2))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"projected trajectory  |  t={t_ref:.2f}s")
        ax1.axis("off")
        ax2 = fig.add_subplot(1, 2, 2)
        viz.draw_bev(ax2, ego, vehicle_width=args.vehicle_width)
        ax2.set_title(f"bird's-eye  |  past {args.t_past:.0f}s  future {args.t_future:.0f}s")
        fig.tight_layout()
        p = os.path.join(frames_dir, f"overlay_{n:04d}.png")
        fig.savefig(p, dpi=100)
        plt.close(fig)

        records.append(dict(frame_index=int(fi), pts=float(video.pts[fi]),
                            t_session=t_ref,
                            t_utc=float(session.elapsed_to_utc(t_ref)),
                            speed=float(ego["speed_ref"]),
                            trajectory=[dict(t=float(a), x=float(b), y=float(c),
                                             speed=float(d), yaw=float(e))
                                        for a, b, c, d, e in zip(ego["t"], ego["x"], ego["y"],
                                                                 ego["speed"], ego["yaw"])]))
        if n % 5 == 0:
            print(f"  rendered {n + 1}/{len(pick)}", flush=True)

    if args.video:
        step = max(1, int(round(video.avg_fps / args.video_fps)))
        vid_idx = usable[::step]
        vp = os.path.join(OUT, "overlay.mp4")
        writer = None
        print(f"writing {vp} ({len(vid_idx)} frames @ {args.video_fps} fps)")
        for n, fi in enumerate(vid_idx):
            ego = ego_trajectory(traj, float(frame_times[fi]), t_past=args.t_past,
                                 t_future=args.t_future, dt_out=0.1)
            img = (video.read_bgr(int(fi)) if hasattr(video, "read_bgr")
                   else _grab_bgr(video.path, float(video.pts[fi])))
            if ego is None or img is None:
                continue
            fr = viz.draw_hud(viz.draw_trajectory_on_image(
                img, ego, cam, vehicle_width=args.vehicle_width), ego)
            if writer is None:
                writer = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.video_fps, (fr.shape[1], fr.shape[0]))
            writer.write(fr)
            if n % 25 == 0:
                print(f"  {n}/{len(vid_idx)}", flush=True)
        if writer is not None:
            writer.release()

    with open(os.path.join(OUT, "ego_trajectories.json"), "w") as f:
        json.dump(dict(session=os.path.basename(session.path),
                       t_video_start=sync.t_video_start, sync_source=sync.source,
                       camera=dict(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
                                   yaw=cam.yaw, pitch=cam.pitch, roll=cam.roll,
                                   height_m=cam.height_m, source=cam.source),
                       frames=records), f, indent=1)

    # summary figure of the whole session
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].plot(traj.E, traj.N, lw=1.4)
    if session.has("gps"):
        g = session["gps"]
        p = traj.enu.forward(g.latitude.to_numpy(), g.longitude.to_numpy(),
                             g.altitude.to_numpy() if "altitude" in g.columns else None)
        axs[0].plot(p[:, 0], p[:, 1], ".", ms=4, alpha=.7, label="GNSS fixes")
        axs[0].legend()
    axs[0].set_aspect("equal"); axs[0].grid(ls=":"); axs[0].set_title("session path (ENU)")
    axs[0].set_xlabel("East [m]"); axs[0].set_ylabel("North [m]")

    axs[1].plot(traj.t, traj.speed, lw=1.2, label="smoothed")
    if session.has("gps"):
        axs[1].plot(session["gps"].seconds_elapsed, session["gps"].speed, ".", ms=4, label="GNSS")
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("speed [m/s]"); axs[1].grid(ls=":")
    axs[1].legend(); axs[1].set_title("speed")

    if sync.xcorr_curve is not None:
        lags, sc = sync.xcorr_curve
        axs[2].plot(lags, sc, lw=1.2)
        axs[2].axvline(sync.xcorr_shift or 0, color="r", ls="--",
                       label=f"shift={sync.xcorr_shift:.3f}s")
        axs[2].legend()
    axs[2].set_xlabel("lag [s]"); axs[2].set_ylabel("explained variance")
    axs[2].grid(ls=":"); axs[2].set_title("camera/gyro time-shift scan")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "session_summary.png"), dpi=110)
    plt.close(fig)

    print(f"\nwrote {len(records)} overlays + ego_trajectories.json + session_summary.png")
    print(f"output dir: {OUT}")


def _grab_bgr(path, t):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


if __name__ == "__main__":
    main()
