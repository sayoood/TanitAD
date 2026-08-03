#!/usr/bin/env python3
"""OPEN-LOOP prediction inside a NuRec scene, on the Jetson Thor.

⭐ OPEN LOOP, DEFINED — and this is the whole point of the file:
the ego follows the **LOGGED trajectory**. Every frame is rendered at the pose the rig
actually had, the policy consumes that frame stack, and the plan it emits is scored
against the log's own future motion. **The plan is never executed.** No controller step,
no bicycle integration, no divergence.

WHY IT EXISTS. Every AlpaSim video and every AlpaSim number this programme has produced
is CLOSED loop, where perception error and control drift are confounded: a bad frame
moves the car, which produces a worse frame, which moves it further. MEASURED on
2026-08-03, that confound is not academic — flagship v1's driven path moved a mean 9.05 m
(max 37.78) from a *render change alone*. Open loop pins the observation distribution to
the log's own poses, so what is left is prediction.

WHAT IS SHARED WITH CLOSED LOOP, DELIBERATELY (never re-derived here):
  * `closedloop_drive.FlagshipV1Policy` / `RefCPolicy` — the same canonicalization
    (`ftheta_crop_resize(center="principal")` + `stack_frames`), the same `f_eff`
    self-check against `F_REF`, and the same hard `(1, 8, 9, 256, 256)` raster
    assertion. ⛔ Both arms are 256px SQUARE; `_BasePolicy.canon` refuses anything else.
  * `closedloop_drive.wp_to_control` — the plan is passed through the SAME pure-pursuit
    controller as the closed loop, so `v_target` and the commanded yaw rate exist and the
    LONGITUDINAL / LATERAL families are built from the same quantities as the closed-loop
    panel. ⚠️ Its output is **RECORDED, NOT APPLIED**.
  * `closedloop_drive.nav_from_route` / `gt_poses_xyv` — the programme's own v2.1
    labellers, so an open-loop manoeuvre is the same object as a training label.
  * `cl_metrics.py` scores this output **UNCHANGED**: the record schema is byte-identical
    in shape to a closed-loop rollout, so the four families and the paired
    episode-cluster bootstrap come from the same instrument that produced the
    closed-loop panel. Nothing about the estimator is re-implemented here.

⭐ BOTH ARMS SEE THE SAME PIXELS, BY CONSTRUCTION, NOT BY DETERMINISM ARGUMENT.
One render pass drives every arm in the same process, and the md5 of each rendered frame
is recorded in the payload. The pairing therefore rests on an identity, not on a claim
that the renderer reproduces — which matters, because the renderer is a step function of
pose and a 0.1 px camera rotation has been measured to move the 2 s waypoint 6.65 m.

⚠️ FIVE METRICS ARE DEGENERATE BY CONSTRUCTION HERE and must NOT be read as results:
`cross_track_abs_m`, `cross_track_signed_m`, `dist_to_gt_traj_m`, `executed_speed_err_ms`
and `route_corridor_departure_rate` are ~0 because the ego IS the logged path; and
`manoeuvre_exec_eq_plan` collapses onto `manoeuvre_plan_eq_logged` because the "executed"
manoeuvre is the logged one. `ol_report.py` marks all six on the way out. Do not quote
them from the raw JSON.

⚠️ WITHIN-SIM RELATIVE. REF-C's open-loop ADE is 1.5157 on these reconstructions vs
0.4728 on real footage — **3.21x OOD**. Orderings survive; absolute rates do not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger("openloop")

DT = 0.1
WINDOW = 8
STACK = 3
NEED_FRAMES = WINDOW + STACK - 1          # 10 native ticks -> [8, 9, 256, 256]
WP_STEPS = (5, 10, 15, 20)
LOOKAHEAD_IDX = 0
NAV_NAMES = ("follow", "left", "right", "straight")


def build_policy(arm: str, ckpt: str):
    from closedloop_drive import FlagshipV1Policy, RefCPolicy
    if arm == "flagship-v1":
        return FlagshipV1Policy(ckpt)
    return RefCPolicy(ckpt, preset=arm.split("-", 1)[1])


def segment_bounds(ticks, n_clusters):
    """Split the scoreable tick range into `n_clusters` DISJOINT contiguous segments.

    ⚠️ The resampling unit for the episode-cluster bootstrap. An open-loop sweep visits
    the clip once, so unlike the closed-loop panel there is no natural set of rollout
    starts — the clusters here are disjoint SEGMENTS of one clip. That is weaker than 40
    independent val episodes and is stated on every interval this run emits; it is not
    quietly presented as an episode bootstrap.
    """
    n_clusters = max(1, min(int(n_clusters), len(ticks)))
    return [list(c) for c in np.array_split(np.asarray(ticks), n_clusters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="arm=path, repeatable. e.g. --ckpt flagship-v1=~/models/.../ckpt.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="background,road")
    ap.add_argument("--n-clusters", type=int, default=9,
                    help="disjoint segments of the clip used as bootstrap clusters")
    ap.add_argument("--max-ticks", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--save-video-frames", action="store_true")
    ap.add_argument("--loader-dir", default=None)
    # ---- render quality: the 2026-08-03 CHOSEN configuration ----------------------
    ap.add_argument("--all-dynamic-layers", action="store_true",
                    help="render dynamic_rigids + dynamic_deformables (the real scene)")
    ap.add_argument("--cull-scale-quantile", type=float, default=None)
    ap.add_argument("--sky-gain", type=float, default=0.0)
    ap.add_argument("--sky-lo-deg", type=float, default=0.0)
    ap.add_argument("--sky-hi-deg", type=float, default=6.0)
    ap.add_argument("--rolling-shutter", action="store_true",
                    help="measured-better (grad-NCC 0.3747) at 161x the cost. OFF by "
                         "default; if you turn it on, SAY SO and quote the cost.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    from closedloop_drive import (InProcTransport, build_intr, gt_poses_xyv,
                                  nav_from_route, wp_to_control, _yaw)
    from gsplat_renderer import NuRecGsplatRenderer

    sd = Path(args.scene_dir).expanduser()
    r = NuRecGsplatRenderer(sd, layers=[x for x in args.layers.split(",") if x],
                            loader_dir=args.loader_dir)
    if args.cull_scale_quantile:
        logger.info("scale cull: %s", r.cull_by_scale(args.cull_scale_quantile))
    if args.sky_gain > 0:
        if r.attach_sky(lo_deg=args.sky_lo_deg, hi_deg=args.sky_hi_deg,
                        gain=args.sky_gain) is None:
            raise SystemExit("--sky-gain given but the scene ships no sky-env-map")
        logger.info("gated sky ON: gain=%.2f ramp %.1f-%.1f deg", args.sky_gain,
                    args.sky_lo_deg, args.sky_hi_deg)
    attach = None
    if args.all_dynamic_layers:
        from actor_map import attach_all_dynamic_layers
        attach = attach_all_dynamic_layers(r, sd)
        (out / "actor_attach.json").write_text(json.dumps(attach, indent=2))
        # cl_metrics' `--renderable-from` wants a FLAT per_track list; the all-layer
        # attach nests one per layer. Flatten it here rather than widening the scorer,
        # so the distance-keeping family can be restricted to agents the renderer
        # actually drew (crediting a lead the model never saw would measure the
        # annotation, not the policy).
        flat = []
        for L, d in attach["per_layer"].items():
            for x in d.get("per_track", []):
                flat.append({**x, "layer": L})
        (out / "renderable.json").write_text(json.dumps({"per_track": flat}, indent=2))
        logger.info("actors: %s", {k: v for k, v in attach.items()
                                   if k not in ("per_layer", "falsifier")})

    transport = InProcTransport(r)
    intr = build_intr(transport.camera())
    logger.info("camera: %dx%d cx=%.1f cy=%.1f", intr.width, intr.height, intr.cx, intr.cy)

    # 30 Hz rig -> 10 Hz control/model tick (the same sub-sampling the closed loop does;
    # feeding a 30 Hz stack to a 10 Hz-trained encoder is the same class of train/serve
    # skew as a wrong raster).
    stride = int(round(1e5 / (r.frame_timestamps_us(1)[1] - r.frame_timestamps_us(0)[1])))
    n = r.n_frames() // stride
    if args.max_ticks:
        n = min(n, args.max_ticks)
    gt_T = [r.gt_rig_to_world(f * stride) for f in range(n)]
    gt_ts = [r.frame_timestamps_us(f * stride)[1] for f in range(n)]
    gtp = gt_poses_xyv(gt_T)
    Ts_cam = r.cam.T_sensor_rig
    shutter_s = 0.0
    if args.rolling_shutter:
        t0, t1 = r.frame_timestamps_us(0)
        shutter_s = (t1 - t0) / 1e6
        logger.info("ROLLING SHUTTER ON: %s readout %.1f ms (~161x cost)",
                    r.rolling_shutter_type, shutter_s * 1e3)
    logger.info("GT: %d camera frames -> %d ticks at 10 Hz (stride %d)",
                r.n_frames(), n, stride)

    arms = {}
    for spec in args.ckpt:
        a, _, p = spec.partition("=")
        if not p:
            raise SystemExit(f"--ckpt wants arm=path, got {spec!r}")
        arms[a] = str(Path(p).expanduser())
    pols = {a: build_policy(a, p) for a, p in arms.items()}

    vdir = out / "frames"
    if args.save_video_frames:
        vdir.mkdir(exist_ok=True)

    frames = deque(maxlen=NEED_FRAMES)
    digests, per_arm_steps = [], {a: [] for a in arms}
    render_ms, plan_ms = [], {a: [] for a in arms}
    t_start = time.time()
    for k in range(n):
        T_rig = gt_T[k]
        T_start = None
        if shutter_s > 0:
            T_start = r.rig.T_rig_world(r.cam_name, k * stride, shutter=0)
        t_r = time.time()
        img = transport.render(T_rig @ Ts_cam, gt_ts[k],
                               cam_to_world_start=(T_start @ Ts_cam
                                                   if T_start is not None else None))
        render_ms.append((time.time() - t_r) * 1000.0)
        frames.append(img)
        digests.append(hashlib.md5(np.ascontiguousarray(img)).hexdigest())
        if args.save_video_frames and k >= NEED_FRAMES - 1:
            import cv2
            cv2.imwrite(str(vdir / f"{k:05d}.jpg"), img[:, :, ::-1],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if k < NEED_FRAMES - 1:
            continue

        # --- the ego state is the LOG's, not a simulation's --------------------------
        ego = [float(T_rig[0, 3]), float(T_rig[1, 3]), float(T_rig[2, 3]), _yaw(T_rig)]
        v = float(gtp[k, 3])
        nav, navd = nav_from_route(gtp, k)
        fl = list(frames)
        for a, pol in pols.items():
            t_p = time.time()
            traj, extra = pol.plan(fl, intr, v, nav)
            ms = (time.time() - t_p) * 1000.0
            plan_ms[a].append(ms)
            # SAME controller as the closed loop, RECORDED not APPLIED. It is what makes
            # `target_speed_err` and the commanded yaw rate exist, so the LONGITUDINAL
            # and LATERAL families are the same quantities as the closed-loop panel's.
            steer, accel, v_target, kappa = wp_to_control(traj[LOOKAHEAD_IDX], v)
            per_arm_steps[a].append({
                "k": k, "t_us": float(gt_ts[k]), "i_gt": int(k), "nav": int(nav),
                "nav_detail": navd, "ego": ego, "v": v, "plan": traj.tolist(),
                "steer": steer, "accel": accel, "v_target": v_target,
                "kappa_plan": kappa, "plan_ms": ms, "extra": extra,
                "frame_md5": digests[k],
            })
        if k % 20 == 0:
            logger.info("k=%3d/%d  v=%.2f  nav=%s  render %.0f ms", k, n, v,
                        NAV_NAMES[nav], render_ms[-1])
    wall = time.time() - t_start

    gt_dump = [{"f": f, "x": float(gt_T[f][0, 3]), "y": float(gt_T[f][1, 3]),
                "z": float(gt_T[f][2, 3]), "yaw": _yaw(gt_T[f]),
                "ts_us": float(gt_ts[f])} for f in range(n)]
    ticks = [s["k"] for s in per_arm_steps[next(iter(arms))]]
    segs = segment_bounds(ticks, args.n_clusters)
    rq = {"layers": args.layers, "all_dynamic_layers": bool(args.all_dynamic_layers),
          "sky_gain": args.sky_gain, "sky_ramp_deg": [args.sky_lo_deg, args.sky_hi_deg],
          "cull_scale_quantile": args.cull_scale_quantile,
          "cull": getattr(r, "cull_info", None),
          "rolling_shutter": bool(args.rolling_shutter), "shutter_s": shutter_s,
          "shutter_type": r.rolling_shutter_type}

    manifest = {}
    for a, steps in per_arm_steps.items():
        by_k = {s["k"]: s for s in steps}
        base = {"arm": a, "ckpt": arms[a], "condition": "logged", "scene": sd.name,
                "layers": args.layers, "steps": len(steps), "f_eff": pols[a].f_eff,
                "mode": "open_loop", "gt": gt_dump,
                "render_quality": rq, "transport": "inproc",
                "frames_dir": str(vdir) if args.save_video_frames else None,
                "n_ticks": n, "stride": stride,
                "wall_s": wall, "render_ms_mean": float(np.mean(render_ms)),
                "plan_ms_mean": float(np.mean(plan_ms[a])),
                "open_loop_note": (
                    "OPEN LOOP: ego pose, ego speed and the observation stack all come "
                    "from the LOGGED rig trajectory. The controller output on every step "
                    "is RECORDED, NOT APPLIED — the car does not move under the model. "
                    "cross_track, dist_to_gt_traj_m, executed_speed_err_ms and "
                    "route_corridor_departure_rate are therefore ~0 BY CONSTRUCTION, and "
                    "manoeuvre_exec_eq_plan collapses onto manoeuvre_plan_eq_logged. "
                    "See ol_report.py, which marks all of them."),
                "frame_md5": {str(i): d for i, d in enumerate(digests)}}
        # (1) SCORING file: disjoint segments -> bootstrap clusters
        score = dict(base)
        score["rollouts"] = [
            {"start_frame": int(s[0]), "n_steps": len(s), "arm": a,
             "condition": "logged", "render_ms": float(np.mean(render_ms)),
             "steps": [by_k[t] for t in s if t in by_k]}
            for s in segs]
        score["cluster_note"] = (
            f"{len(segs)} DISJOINT contiguous segments of ONE clip, used as the "
            "episode-cluster bootstrap's resampling unit. Not 40 independent episodes.")
        p_score = out / f"rollouts_{a}_openloop.json"
        p_score.write_text(json.dumps(score))
        # (2) VIDEO file: one continuous rollout, so the overlay draws an unbroken sweep
        vid = dict(base)
        vid["rollouts"] = [{"start_frame": int(ticks[0]), "n_steps": len(steps),
                            "arm": a, "condition": "logged",
                            "render_ms": float(np.mean(render_ms)), "steps": steps}]
        p_vid = out / f"video_{a}_openloop.json"
        p_vid.write_text(json.dumps(vid))
        manifest[a] = {"scoring": str(p_score), "video": str(p_vid),
                       "n_windows": len(steps), "n_clusters": len(segs)}
        logger.info("%s: %d windows over %d clusters -> %s", a, len(steps), len(segs),
                    p_score.name)

    summary = {"scene": sd.name, "n_ticks": n, "stride": stride,
               "n_frames_rendered": len(digests),
               "render_ms_mean": float(np.mean(render_ms)),
               "render_ms_p50": float(np.median(render_ms)),
               "wall_s": wall, "arms": manifest, "render_quality": rq,
               "frames_dir": str(vdir) if args.save_video_frames else None,
               "shared_observation_note": (
                   "ONE render pass drove every arm in this process, so the arms are "
                   "paired on IDENTICAL pixels by construction. `frame_md5` in each "
                   "rollout file is the per-tick digest of the rendered frame; it is the "
                   "same list in every arm's file by identity, and it lets a later "
                   "re-render be checked bit-exactly rather than assumed to reproduce.")}
    (out / "openloop_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s (%.1f s wall, render %.1f ms/frame)",
                out / "openloop_summary.json", wall, float(np.mean(render_ms)))


if __name__ == "__main__":
    main()
