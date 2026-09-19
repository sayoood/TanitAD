"""Render a full-session validation video: image projection + bird's-eye, side by side.

    python render_video.py --session 0811
    python render_video.py --session 0811 --stride 2 --out work/quick.mp4

Sync and camera calibration are run on a short slice (they do not need the whole
clip), then every usable frame is rendered.  Frames are piped straight into
ffmpeg/libx264 rather than written out and re-encoded: OpenCV's ``mp4v`` writer
produced a 126 MB file for 400 frames, x264 at crf 20 is roughly 20x smaller for
the same content.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trajlib import quality as Q
from trajlib import timesync as TS
from trajlib import viz
from trajlib.camera import calibrate_camera
from trajlib.frame_folder import load_frame_folder
from trajlib.io_sensorlogger import load_reassembled_session, load_session
from trajlib.trajectory import (ego_trajectory, estimate_trajectory,
                                to_vehicle_reference)
from trajlib.steering import AUDI_A6_ETRON, apply_drivability, estimate_steering
from trajlib.vehicle_frame import estimate_vehicle_frame

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_0814 = os.path.join(ROOT, "Liebst_ckelweg-2025-08-14_15-15-02-android")
SESS_0811 = os.path.join(ROOT, "work", "session_2025-08-11")
FRAMES_0811 = os.path.join(ROOT, "synchronized_output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="0811", choices=["0811", "0814"])
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "demo_out", "scene_validation.mp4"))
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--calib-frames", type=int, default=900,
                    help="frames used for sync + camera calibration")
    ap.add_argument("--height", type=int, default=720, help="output panel height")
    # 1.17 m is measured, not guessed: road-plane homography decomposition over 93
    # frame pairs gives 1.167 m with a bootstrap 95% CI of [1.100, 1.223] and a
    # split-half difference of only 0.025 m.  Re-measure per mount with --plane-calib.
    ap.add_argument("--cam-height", type=float, default=1.17)
    ap.add_argument("--vehicle-width", type=float, default=1.8)
    ap.add_argument("--lateral-offset", type=float, default=0.0,
                    help="camera offset from the vehicle centreline, +left (metres). "
                         "Unobservable from motion - measure it once per mount.")
    ap.add_argument("--plane-calib", action="store_true",
                    help="measure camera height (and cross-check pitch) by decomposing "
                         "the road-plane homography; adds ~15 min of frame reading")
    ap.add_argument("--ground-calib", action="store_true",
                    help="EXPERIMENTAL: fit extrinsics from road features. Ill-conditioned "
                         "on the sample data and measurably worse than the FOE alone - "
                         "see the warning in trajlib/ground_calib.py before using it.")
    ap.add_argument("--mount-longitudinal", type=float, default=2.1,
                    help="phone ahead of the vehicle reference point (rear axle), m")
    ap.add_argument("--lane-width", type=float, default=3.50,
                    help="true lane width in metres (DE 3.50, US 12 ft = 3.66)")
    ap.add_argument("--t-past", type=float, default=3.0)
    ap.add_argument("--t-future", type=float, default=5.0)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--max-wheel-rate", type=float, default=180.0,
                    help="steering-rate limit for the drivability constraint (deg/s)")
    ap.add_argument("--drivability", action="store_true",
                    help="rewrite the PATH with rate-limited curvature; off by default "
                         "because it displaces the ego window by up to 2.5 m at sharp turns")
    ap.add_argument("--wheelbase", type=float, default=AUDI_A6_ETRON["wheelbase_m"])
    ap.add_argument("--steering-ratio", type=float, default=AUDI_A6_ETRON["steering_ratio"])
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    import cv2

    # ---------------------------------------------------------------- load
    if args.session == "0811":
        session = load_reassembled_session(SESS_0811)
        video = load_frame_folder(FRAMES_0811)                 # ALL frames
        calib_video = load_frame_folder(FRAMES_0811, limit=args.calib_frames)
    else:
        session = load_session(RAW_0814)
        video = calib_video = TS.probe_video(session.video_path)
    print(f"session : {os.path.basename(session.path)}")
    print(f"frames  : {video.nb_frames} over {video.duration:.2f}s @ {video.avg_fps:.3f} fps")

    rep = Q.merge(Q.check_gps(session.get("gps")), Q.check_imu(session),
                  Q.check_motion(session.get("gps")))
    print(rep)

    # ------------------------------------------------------- sync + calib
    t0 = time.time()
    flow = TS.image_angular_rate(calib_video) if args.session == "0811" \
        else TS.image_angular_rate(video, t_dur=min(40.0, video.duration))
    sync = TS.synchronise(session, calib_video, cam_flow=flow)
    print(f"sync    : t_video_start={sync.t_video_start:.4f}s  [{sync.source}]  "
          f"({time.time() - t0:.0f}s)")

    vframe = estimate_vehicle_frame(session)
    traj = estimate_trajectory(session, vframe)
    print(f"traj    : {traj.meta['n']} steps, accel tier {traj.meta['accel_tier']}, "
          f"pos 1-sigma median {np.median(traj.pos_std()):.2f} m")

    vehicle = {**AUDI_A6_ETRON, "wheelbase_m": args.wheelbase,
               "steering_ratio": args.steering_ratio}
    if args.drivability:
        traj, dinfo = apply_drivability(traj, vehicle,
                                        max_wheel_rate_deg_s=args.max_wheel_rate)
        print(f"drive   : PATH rewritten, steering-rate p99 "
              f"{dinfo['raw_p99']:.0f} -> {dinfo['final_p99']:.0f} deg/s"
              + (f" (curvature low-pass {dinfo['fc']:.3f} Hz)" if dinfo["fc"] else ""))
    else:
        print("drive   : path left as estimated (rate limiting applies to the read-out only)")
    steer = estimate_steering(traj, vehicle, max_wheel_rate_deg_s=args.max_wheel_rate)
    print(steer.summary())

    cam = calibrate_camera(calib_video, session, sync, traj=traj, cam_flow=flow,
                           vframe=vframe, height_m=args.cam_height)

    # Run the *production* self-calibration so this path cannot drift away from
    # what pipeline.py does.  It needs an args-like object and a logger.
    from types import SimpleNamespace
    from pipeline import self_calibrate

    class _Log:
        def __call__(self, msg="", level="INFO"):
            print(f"calib   : {msg}" if level == "INFO" else f"calib  !: {msg}")
        def block(self, title, body, level="INFO"):
            print(f"calib   : --- {title} ---")
            for ln in str(body).splitlines():
                print(f"calib   :     {ln}")

    ca = SimpleNamespace(plane_calib=args.plane_calib, vp_calib=True, vp_frames=40,
                         lane_calib=True, scale_calib=True, lane_width=args.lane_width,
                         lateral_offset=args.lateral_offset,
                         mount_longitudinal=args.mount_longitudinal,
                         vehicle_width=args.vehicle_width)
    cam, calib = self_calibrate(video, cam, traj, sync, ca, _Log(),
                                session=session, vframe=vframe)
    args.lateral_offset = ca.lateral_offset          # lane calib may have measured it
    cam.longitudinal_m = float(args.mount_longitudinal)
    cam.lateral_m = float(args.lateral_offset)
    for k, v in calib["parameters"].items():
        print(f"calib   : {k:20s} {v['value']}  <- {v.get('source', '?')[:56]}")
    print(cam.summary())

    # ------------------------------------------------------------ select
    frame_times = sync.frame_times(video.pts)
    usable = np.nonzero((frame_times >= traj.t[0] + args.t_past) &
                        (frame_times <= traj.t[-1] - args.t_future))[0][::args.stride]
    if len(usable) == 0:
        raise SystemExit("no frame has a complete past+future window")
    fps = video.avg_fps / args.stride
    print(f"render  : {len(usable)} frames "
          f"({frame_times[usable[0]]:.1f}..{frame_times[usable[-1]]:.1f}s) at {fps:.2f} fps")

    # ------------------------------------------------------------ encode
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH")

    proc = None
    t0 = time.time()
    written = 0
    for n, fi in enumerate(usable):
        ego = ego_trajectory(traj, float(frame_times[fi]), t_past=args.t_past,
                             t_future=args.t_future, dt_out=0.1)
        if ego is not None:
            ego = to_vehicle_reference(ego, args.mount_longitudinal, args.lateral_offset)
        img = (video.read_bgr(int(fi)) if hasattr(video, "read_bgr")
               else _grab_bgr(video.path, float(video.pts[fi])))
        if ego is None or img is None:
            continue

        t_ref = float(frame_times[fi])
        over = viz.draw_trajectory_on_image(img, ego, cam, vehicle_width=args.vehicle_width,
                                           lateral_offset_m=0.0)
        over = viz.draw_hud(over, ego, {"frame": str(int(fi))})
        sw = float(np.interp(t_ref, steer.t, steer.wheel_deg))
        sr = float(np.interp(t_ref, steer.t, steer.wheel_rate_deg_s))
        sv = bool(np.interp(t_ref, steer.t, steer.valid.astype(float)) > 0.5)
        over = viz.draw_steering_wheel(over, sw, valid=sv, rate_deg_s=sr)
        bev = viz.draw_bev_cv(ego, size=(int(args.height * 0.86), args.height),
                              vehicle_width=args.vehicle_width,
                              lateral_offset_m=0.0)
        panel = viz.compose_panels(over, bev, height=args.height)

        if proc is None:
            h, w = panel.shape[:2]
            print(f"          output {w}x{h} -> {args.out}")
            proc = subprocess.Popen(
                [ffmpeg, "-y", "-v", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{fps:.4f}", "-i", "-",
                 "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
                stdin=subprocess.PIPE)
        proc.stdin.write(panel.tobytes())
        written += 1
        if n % 100 == 0:
            el = time.time() - t0
            eta = el / max(n + 1, 1) * (len(usable) - n - 1)
            print(f"          {n}/{len(usable)}  ({el:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

    if proc is not None:
        proc.stdin.close()
        proc.wait()
    size_mb = os.path.getsize(args.out) / 1e6 if os.path.exists(args.out) else 0
    print(f"\nwrote {written} frames -> {args.out}  ({size_mb:.1f} MB, "
          f"{written / fps:.1f}s of video, {time.time() - t0:.0f}s to render)")


def _grab_bgr(path, t):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


if __name__ == "__main__":
    main()
