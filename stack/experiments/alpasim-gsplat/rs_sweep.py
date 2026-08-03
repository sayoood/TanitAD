#!/usr/bin/env python3
"""ROLLING SHUTTER: quality vs cost. Is the +35 % affordable inside a 10 Hz loop?

WHAT THIS ANSWERS
-----------------
`RENDER_QUALITY.md` measured gsplat's NATIVE rolling shutter at grad-NCC 0.3424 -> 0.3747
(+35.1 % over the original `background+road` baseline) and **3749 ms/frame, 161x** a
global-shutter render. That is a headline, not a decision: nobody had asked whether the
gain needs 1080 poses or 4.

Three knobs are swept here, all on the SAME renderer instance so the comparison is
paired:

1. **PHASE (free).** Every render before today used the shutter-**END** pose
   (`rig_trajectories.json` stores `[shutter_start, shutter_end]` and every caller took
   index 1). Under TOP_TO_BOTTOM readout the top of the image is exposed at shutter
   START, so the end pose is the *worst* single choice for the top of the frame and the
   *best* for the bottom. Rendering at the shutter MIDPOINT costs exactly nothing.
2. **SLICES.** `render_rs_sliced(n)` renders n global-shutter frames at n phases and
   keeps each render's own horizontal band. Cost is exactly n x a normal render, versus
   161x for the native kernel (which re-derives the pose PER PIXEL inside
   `rasterize_to_pixels_from_world_3dgs_fwd`).
3. **REGION.** grad-NCC is scored per horizontal band as well as whole-frame, because
   "where does the gain live" decides whether a cheap partial fix exists.

CONTROLS THAT RUN HERE, NOT LATER
---------------------------------
* **Negative control on every arm** (correct reference vs >=5 wrong frames), reusing
  `render_quality.negative_control` — no arm reports a number without it.
* **SELF-CONSISTENCY**: every sliced arm is also compared PIXELWISE to the native-RS
  render of the same frame. If slicing is the right approximation, mean|diff| must fall
  monotonically as n grows. If it does not, the slice mapping is wrong.
* **DIRECTION FALSIFIER**: an `s*_rev` arm maps band -> phase the other way round. The
  readout direction is then decided by measurement (which arm converges to native RS),
  not by trusting the string `ROLLING_TOP_TO_BOTTOM`.
* **grad-NCC only.** PSNR/NCC/MAE are RETRACTED as decision metrics on these night
  clips; they are recorded as descriptive statistics and decide nothing.

Usage:
    python rs_sweep.py --scene-dir <scene> --out ~/rq_out/rs_sweep \
        --config chosen --frames 0,50,100,...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from frame_align import scene_ref_offset
from render_quality import (CAM, build_renderer, coverage_and_error, git_sha, grad_ncc,
                            load_refs, negative_control, parse_arm, save_sbs,
                            wrong_frames_for)

N_BANDS = 8

# The two renderer configurations a recommendation has to hold for.
#   `base`   = what the banked `panel3_rs` numbers were measured on (comparability)
#   `chosen` = what the closed loop actually runs today (RENDER_QUALITY.md HEADLINE)
CONFIGS = {
    "base": "base:background,road",
    "chosen": ("chosen:background,road,dynamic_rigids,dynamic_deformables"
               ":cull=0.95:sky=0,6,0.3"),
}


def band_grad_ncc(render_u8: np.ndarray, ref_u8: np.ndarray, n_bands: int = N_BANDS):
    """grad-NCC restricted to each horizontal band, top band first.

    Under TOP_TO_BOTTOM readout, band 0 is exposed at shutter START and the last band at
    shutter END, so a phase error shows up as a MONOTONIC gradient down this list. A
    whole-frame mean cannot see that and has been hiding it.
    """
    H = render_u8.shape[0]
    e = np.linspace(0, H, n_bands + 1).round().astype(int)
    return [round(grad_ncc(np.ascontiguousarray(render_u8[e[k]:e[k + 1]]),
                           np.ascontiguousarray(ref_u8[e[k]:e[k + 1]])), 4)
            for k in range(n_bands)]


def utgate_arm_list() -> list[dict]:
    """The panel that separates ROLLING-SHUTTER PHYSICS from the VALIDITY GATE.

    `Cameras.cuh:357` shows the rolling-shutter branch returns `valid = true`
    unconditionally while the global branch returns the real `valid_start`; with
    `require_all_sigma_points_valid = True` (gsplat's default) that means the RS path
    CULLS FAR FEWER GAUSSIANS than the global path. If that is where the "+35 % rolling
    shutter" came from, then relaxing the same gate on a plain global-shutter render
    buys the same thing for FREE — and the margin knob is a second, independent way to
    reach it.
    """
    return [
        {"name": "native", "kind": "native"},
        {"name": "native_zero_end", "kind": "native", "zero": "end"},
        {"name": "g_p1.00", "kind": "global", "phase": 1.0},
        {"name": "g_p0.50", "kind": "global", "phase": 0.5},
        # ---- DOES THE PHASE CURVE TURN OVER, OR ARE WE JUST LATE? -----------------
        # The phase sweep is monotone toward the shutter START (g_p0.00 +0.0072 >
        # g_p0.25 > g_p0.50 > g_p0.75 > g_p1.00 = 0). "Start is optimal" and "every
        # render is placed systematically late" predict the SAME ordering inside [0,1]
        # and DIFFERENT things outside it, so the question is settled by extrapolating,
        # not by argument. One readout is 30.559 ms / up to 0.63 m of ego travel.
        {"name": "g_p0.00", "kind": "global", "phase": 0.0},
        {"name": "g_pm0.50", "kind": "global", "phase": -0.5},
        {"name": "g_pm1.00", "kind": "global", "phase": -1.0},
        {"name": "g_pm2.00", "kind": "global", "phase": -2.0},
        {"name": "g_p1.00_anysigma", "kind": "global", "phase": 1.0,
         "ut": {"require_all_sigma_points_valid": False}},
        {"name": "g_p0.50_anysigma", "kind": "global", "phase": 0.5,
         "ut": {"require_all_sigma_points_valid": False}},
        {"name": "g_p0.50_margin0.5", "kind": "global", "phase": 0.5,
         "ut": {"in_image_margin_factor": 0.5}},
        {"name": "g_p0.50_margin2.0", "kind": "global", "phase": 0.5,
         "ut": {"in_image_margin_factor": 2.0}},
        {"name": "g_p0.50_any_m2.0", "kind": "global", "phase": 0.5,
         "ut": {"require_all_sigma_points_valid": False,
                "in_image_margin_factor": 2.0}},
        {"name": "s4_anysigma", "kind": "sliced", "n": 4, "sweep_actor_time": True,
         "ut": {"require_all_sigma_points_valid": False}},
        {"name": "s8_any_m2.0", "kind": "sliced", "n": 8, "sweep_actor_time": True,
         "ut": {"require_all_sigma_points_valid": False,
                "in_image_margin_factor": 2.0}},
        # restore-check: identical spec to `g_p1.00`, run LAST. If the gate was restored
        # correctly it must be BIT-IDENTICAL to it; if it is not, every number above is
        # contaminated by leaked process-global state.
        {"name": "g_p1.00_restorecheck", "kind": "global", "phase": 1.0},
    ]


def batch_arm_list() -> list[dict]:
    """SLICE BATCHING: is the sliced shutter cheaper as ONE call over N cameras?

    The sequential path pays N kernel launches AND re-poses 37 actor tracks in Python N
    times (~41.6 ms each, measured). Batching removes both. It cannot remove the
    per-pixel blending, so the floor is O(N) in the rasteriser — this panel measures how
    far above that floor the sequential path was sitting.

    Each `bN` is paired with its sequential twin `sN` on the same frames, so "batching
    helps" is a delta and not two numbers from two runs.
    """
    arms = [{"name": "g_p1.00", "kind": "global", "phase": 1.0}]
    for n in (2, 4, 8, 16):
        arms.append({"name": f"b{n}", "kind": "batched", "n": n})
        arms.append({"name": f"s{n}", "kind": "sliced", "n": n,
                     "sweep_actor_time": False})
    return arms


def arm_list(config: str, full: bool) -> list[dict]:
    """(kind, params) for every arm. `kind` in {global, sliced, native}.

    ⚠️ `native` runs FIRST on purpose: every other arm is compared PIXELWISE against it
    (the self-consistency control), so its frames have to exist before the rest run.
    """
    arms = [{"name": "native", "kind": "native"}]
    # ---- MECHANISM CONTROLS on the native kernel ---------------------------------
    # `native` raises mean_alpha by ~23 % and render_mean by ~24 % over the reference.
    # A pose SWEEP cannot create alpha — it only moves geometry — so the gain may be
    # footprint DILATION inside the RS projection rather than rolling-shutter physics.
    # These three separate the two, and they are the reason this file exists:
    #   *_zero_end   : RS kernel, start pose == end pose == the GLOBAL arm's pose.
    #                  Geometrically IDENTICAL to `g_p1.00`. Any difference is the
    #                  CODE PATH, not the shutter.
    #   *_zero_start : same, at the shutter-START pose (twin of `g_p0.00`).
    #   *_swapped    : the sweep run BACKWARDS. Physics says this must be worse;
    #                  dilation says it makes no difference.
    arms += [{"name": "native_zero_end", "kind": "native", "zero": "end"},
             {"name": "native_zero_start", "kind": "native", "zero": "start"},
             {"name": "native_swapped", "kind": "native", "swap": True}]
    arms += [{"name": f"g_p{p:.2f}", "kind": "global", "phase": p}
             for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    # s1 is a SELF-CONSISTENCY check, not an arm: render_rs_sliced(1) renders one band
    # at phase 0.5, so it must equal `g_p0.50` to the bit. If it does not, the slicing
    # or the pose interpolation is wrong and nothing below it is admissible.
    ns = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64) if full else (1, 2, 4, 8, 16)
    arms += [{"name": f"s{n}", "kind": "sliced", "n": n, "sweep_actor_time": True}
             for n in ns]
    arms += [{"name": "s16_rev", "kind": "sliced", "n": 16, "reverse": True,
              "sweep_actor_time": True}]
    if config != "base":
        arms += [{"name": "s16_fixedactor", "kind": "sliced", "n": 16,
                  "sweep_actor_time": False}]
    return arms


def render_arm_frame(r, arm: dict, f: int):
    """Render frame `f` under one arm. Returns (img, alpha, raster_ms, meta)."""
    c2n_s, c2n_e = r.gt_cam_to_nre_pair(f)
    ts0, ts1 = (float(x) for x in r.frame_timestamps_us(f)[:2])
    k = arm["kind"]
    if k == "global":
        from gsplat_renderer import se3_interp
        p = float(arm["phase"])
        # actor time follows the SAME phase as the pose. At p=1.0 this reduces exactly
        # to the pre-existing default (pose = shutter end, actors at shutter end), so
        # `g_p1.00` is the current production behaviour and not a new arm.
        # Phases outside [0,1] EXTRAPOLATE; inside, the clamp is kept so p=1.0 is the
        # production pose BIT-EXACTLY rather than a slerp round-trip of it.
        img, a, ms = r.render(se3_interp(c2n_s, c2n_e, p, clamp=(0.0 <= p <= 1.0)),
                              actor_time_us=ts0 + p * (ts1 - ts0))
        return img, a, ms, {"phases": [p], "n_render_calls": 1}
    if k == "sliced":
        img, a, ms = r.render_rs_sliced(
            c2n_s, c2n_e, int(arm["n"]), actor_time_us_start=ts0,
            actor_time_us_end=ts1, sweep_actor_time=bool(arm.get("sweep_actor_time", True)),
            reverse=bool(arm.get("reverse", False)))
        return img, a, ms, {"phases": list(r.last_rs_phases),
                            "n_render_calls": int(arm["n"])}
    if k == "batched":
        # SLICE BATCHING: the same n phases as n CAMERAS in ONE rasterization call.
        # Actors are posed once (shared gaussian set), at the shutter-END time so the
        # comparison against `g_p1.00` differs only in the camera batch.
        img, a, ms = r.render_rs_batched(c2n_s, c2n_e, int(arm["n"]),
                                         actor_time_us=ts1)
        return img, a, ms, {"phases": list(r.last_rs_phases), "n_render_calls": 1}
    if k == "native":
        # actors at the shutter-END time, which is what the banked panel3_rs arm used —
        # kept identical so this arm reproduces a banked number rather than inventing one.
        p0, p1 = c2n_s, c2n_e
        if arm.get("zero") == "end":
            p0 = p1 = c2n_e
        elif arm.get("zero") == "start":
            p0 = p1 = c2n_s
        elif arm.get("swap"):
            p0, p1 = c2n_e, c2n_s
        img, a, ms = r.render(p0, actor_time_us=ts1, cam_to_nre_end=p1)
        return img, a, ms, {"phases": "per-pixel (gsplat kernel)", "n_render_calls": 1}
    raise ValueError(f"unknown arm kind {k!r}")


def warm_up(r, arm: dict, f: int, n: int = 2):
    """Warm up with THIS arm's own settings.

    ⚠️ A single global warm-up is NOT enough: the RS kernel is a different template
    instantiation, so its first call pays its own one-time cost. Warming up only on a
    global-shutter render inflated the smoke run's native median from ~3.7 s to 5.3 s
    — a 44 % timing artifact that would have gone into the headline.
    """
    for _ in range(n):
        render_arm_frame(r, arm, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True, help="RUN DIRECTORY — quote this, not a number")
    ap.add_argument("--config", default="chosen", choices=sorted(CONFIGS))
    ap.add_argument("--frames", default=None,
                    help="default: 12 frames spread over the clip")
    ap.add_argument("--n-frames-auto", type=int, default=12)
    ap.add_argument("--loader-dir", default=None)
    ap.add_argument("--png-frame", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="fewer slice counts")
    ap.add_argument("--panel", default="rs", choices=("rs", "utgate", "batch"),
                    help="rs = phase/slice cost curve; utgate = separate the shutter "
                         "sweep from gsplat's sigma-point validity gate")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()

    scene = Path(a.scene_dir).expanduser()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if a.loader_dir:
        import sys
        sys.path.insert(0, a.loader_dir)
    from nurec_loader import RigTrajectories
    rig = RigTrajectories(scene / "rig_trajectories.json")
    cam = rig.camera(CAM)
    n_clip, size = rig.n_frames(CAM), (int(cam.width), int(cam.height))

    if a.frames:
        frames = [int(x) for x in a.frames.split(",")]
    else:
        # spread over the clip but never past the last decodable frame
        frames = sorted(set(int(round(x)) for x in
                            np.linspace(0, n_clip - 1, a.n_frames_auto)))
    wrong_map = {f: wrong_frames_for(f, n_clip) for f in frames}
    need = set(frames) | {w for ws in wrong_map.values() for w in ws}
    t0 = time.time()
    # R-2026-08-03-k: the reference is offset from the rig PER SCENE. Derived,
    # never hard-coded (+6 on 00040136, +5 on 7c72937c).
    ref_offset = scene_ref_offset(scene, n_clip)
    refs = load_refs(scene / f"{CAM}.mp4", need, size, ref_offset=ref_offset)
    print(f"[rs] decoded {len(refs)}/{len(need)} refs in {time.time() - t0:.1f}s "
          f"(clip {n_clip} frames)", flush=True)
    missing = sorted(need - set(refs))
    if missing:
        raise SystemExit(f"reference frames not decodable: {missing}")

    # ---- the SHUTTER GEOMETRY, asserted before anything is scored ------------------
    geom = {"shutter_type_declared": str(cam.shutter_type), "per_frame": {}}
    for f in frames:
        Ts, Te = rig.T_rig_world(CAM, f, shutter=0), rig.T_rig_world(CAM, f, shutter=1)
        ts = rig.frame_timestamps_us(CAM, f)
        geom["per_frame"][int(f)] = {
            "ego_translation_over_shutter_m": round(
                float(np.linalg.norm(Te[:3, 3] - Ts[:3, 3])), 4),
            "readout_ms": round((float(ts[1]) - float(ts[0])) / 1000.0, 4)}
    rmm = [v["ego_translation_over_shutter_m"] for v in geom["per_frame"].values()]
    geom["ego_translation_m_min_max_mean"] = [round(min(rmm), 4), round(max(rmm), 4),
                                              round(float(np.mean(rmm)), 4)]
    print(f"[rs] shutter {geom['shutter_type_declared']}  ego moves "
          f"{geom['ego_translation_m_min_max_mean']} m over the readout", flush=True)

    arm_spec = parse_arm(CONFIGS[a.config])
    print(f"[rs] building renderer once: {CONFIGS[a.config]}", flush=True)
    tb = time.time()
    r, attach = build_renderer(scene, arm_spec, a.loader_dir)
    build_s = time.time() - tb

    report = {"run_dir": str(out), "scene_dir": str(scene),
              "git_sha": git_sha(Path(a.repo)),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "config": a.config, "config_spec": CONFIGS[a.config],
              "frames": frames, "clip_n_frames": n_clip, "n_bands": N_BANDS,
              "build_s": round(build_s, 1), "n_static_gaussians": int(r.n_gauss),
              "layer_counts": dict(r.layer_counts),
              "cull": getattr(r, "cull_info", None),
              "shutter_geometry": geom,
              "wrong_frames_per_correct": {int(k): v for k, v in wrong_map.items()},
              "metric": ("gradient-NCC ONLY may decide; PSNR/NCC/MAE are RETRACTED on "
                         "this clip and are descriptive here"),
              "arms": []}

    arms = ({"utgate": utgate_arm_list, "batch": batch_arm_list}[a.panel]()
            if a.panel in ("utgate", "batch")
            else arm_list(a.config, full=not a.quick))
    report["panel"] = a.panel
    # warm-up: first rasterization pays CUDA autotune and is not the steady state.
    c0 = r.gt_cam_to_nre(frames[0])
    for _ in range(3):
        r.render(c0, actor_time_us=float(r.frame_timestamps_us(frames[0])[1]))

    native_imgs: dict[int, np.ndarray] = {}
    ref_imgs: dict[str, np.ndarray] = {}     # for the bit-exactness self-consistency
    for arm in arms:
        prev_ut = r.set_ut_gate(**arm["ut"]) if arm.get("ut") else None
        ut_now = r.ut_defaults()
        ut_state = {"require_all_sigma_points_valid":
                    bool(ut_now.require_all_sigma_points_valid),
                    "in_image_margin_factor": float(ut_now.in_image_margin_factor)}
        warm_up(r, arm, frames[0])
        rows = []
        for f in frames:
            t1 = time.time()
            img, alpha, ms = render_arm_frame(r, arm, f)[:3]
            wall = (time.time() - t1) * 1000.0
            ref = refs[f]
            row = {"frame": int(f), "raster_ms": round(ms, 2),
                   "wall_ms": round(wall, 2),
                   "n_actor_gaussians": int(getattr(r, "last_actors_rendered", 0))}
            row.update(coverage_and_error(img, ref, alpha))
            row["neg_control"] = negative_control(img, refs, f, wrong_map[f])
            row["grad_ncc"] = row["neg_control"]["grad_ncc_correct"]
            row["band_grad_ncc"] = band_grad_ncc(img, ref)
            if arm["name"] in ("g_p0.50", "s1", "g_p1.00",
                               "g_p1.00_restorecheck") and f == frames[0]:
                ref_imgs[arm["name"]] = img.copy()
            if arm["name"] == "native":
                native_imgs[f] = img.copy()
            elif f in native_imgs:
                d = np.abs(img.astype(np.int16) - native_imgs[f].astype(np.int16))
                row["mean_abs_diff_vs_native_u8"] = round(float(d.mean()), 4)
                row["frac_pixels_differ_gt2_vs_native"] = round(
                    float((d.max(-1) > 2).mean()), 5)
            rows.append(row)
        bands = np.array([x["band_grad_ncc"] for x in rows], float)
        s = {"arm": arm["name"], "kind": arm["kind"],
             "n_slices": int(arm.get("n", 1)),
             "phase": arm.get("phase"),
             "reverse": bool(arm.get("reverse", False)),
             "sweep_actor_time": arm.get("sweep_actor_time"),
             "n_render_calls_per_frame": (int(arm.get("n", 1))
                                          if arm["kind"] == "sliced" else 1),
             "n_frames": len(rows),
             "neg_control_pass_frames": sum(1 for x in rows if x["neg_control"]["pass"]),
             "neg_control_all_pass": all(x["neg_control"]["pass"] for x in rows),
             "gradncc_argmax_correct_frames": sum(
                 1 for x in rows if x["neg_control"]["argmax_is_correct"]),
             "mae_argmin_correct_frames": sum(
                 1 for x in rows if x["neg_control"]["mae_argmin_is_correct"]),
             "psnr_argmax_correct_frames": sum(
                 1 for x in rows if x["neg_control"]["psnr_argmax_is_correct"]),
             "grad_ncc_mean": round(float(np.mean([x["grad_ncc"] for x in rows])), 4),
             "grad_ncc_per_frame": [x["grad_ncc"] for x in rows],
             "neg_margin_mean": round(float(np.mean(
                 [x["neg_control"]["margin"] for x in rows])), 4),
             "band_grad_ncc_mean": [round(float(v), 4) for v in bands.mean(0)],
             "mean_alpha": round(float(np.mean([x["mean_alpha"] for x in rows])), 4),
             "mae_full": round(float(np.mean([x["mae_full"] for x in rows])), 4),
             "render_mean": round(float(np.mean([x["render_mean"] for x in rows])), 4),
             "ref_mean": round(float(np.mean([x["ref_mean"] for x in rows])), 4),
             "raster_ms_median": round(float(np.median([x["raster_ms"] for x in rows])), 2),
             "wall_ms_median": round(float(np.median([x["wall_ms"] for x in rows])), 2),
             "ut_gate_in_effect": ut_state, "ut_override": arm.get("ut"),
             "per_frame": rows}
        if prev_ut is not None:
            r.set_ut_gate(**prev_ut)
        dv = [x["mean_abs_diff_vs_native_u8"] for x in rows
              if "mean_abs_diff_vs_native_u8" in x]
        if dv:
            s["mean_abs_diff_vs_native_u8"] = round(float(np.mean(dv)), 4)
        report["arms"].append(s)
        print(f"[rs] {arm['name']:<16} gradNCC {s['grad_ncc_mean']:.4f}  "
              f"margin {s['neg_margin_mean']:+.4f}  "
              f"negctl {'PASS' if s['neg_control_all_pass'] else 'FAIL'}  "
              f"{s['raster_ms_median']:8.1f} ms  wall {s['wall_ms_median']:8.1f} ms"
              + (f"  |diff vs native| {s['mean_abs_diff_vs_native_u8']:.3f}"
                 if "mean_abs_diff_vs_native_u8" in s else ""), flush=True)

    # ---- SELF-CONSISTENCY checks ---------------------------------------------------
    def bitexact(k1, k2, key):
        if not ({k1, k2} <= set(ref_imgs)):
            return
        eq = bool(np.array_equal(ref_imgs[k1], ref_imgs[k2]))
        report[key] = eq
        print(f"[rs] SELF-CONSISTENCY  {k1} == {k2} bit-exact: {eq}", flush=True)
        if not eq:
            report[key + "_max_abs_diff"] = int(np.abs(
                ref_imgs[k1].astype(np.int16) - ref_imgs[k2].astype(np.int16)).max())

    # 1 slice at phase 0.5 MUST equal the global phase-0.5 arm, or the slicing is wrong
    bitexact("s1", "g_p0.50", "selfcheck_s1_equals_g_p0.50_bitexact")
    # the UT gate must be RESTORED, or every arm after the first override is polluted
    bitexact("g_p1.00_restorecheck", "g_p1.00", "selfcheck_ut_gate_restored_bitexact")

    (out / "report.json").write_text(json.dumps(report, indent=1))

    # ---- side-by-side PNGs for the decisive arms only (re-rendered, cheap) ----------
    want_png = [x for x in ("g_p1.00", "g_p0.50", "s4", "s8", "native")
                if any(s["arm"] == x for s in report["arms"])]
    for name in want_png:
        arm = next(x for x in arms if x["name"] == name)
        img, alpha, _ = render_arm_frame(r, arm, a.png_frame)[:3]
        s = next(x for x in report["arms"] if x["arm"] == name)
        save_sbs(out / f"sbs_{name}_f{a.png_frame}.png", img, refs[a.png_frame], alpha,
                 f"{a.config}/{name}  gradNCC={s['grad_ncc_mean']:.4f}  "
                 f"{s['raster_ms_median']:.0f} ms/frame")
        np.savez_compressed(out / f"render_{name}_f{a.png_frame}.npz",
                            render=img, alpha=alpha, ref=refs[a.png_frame])

    # ---- report -------------------------------------------------------------------
    base = next(s for s in report["arms"] if s["arm"] == "g_p1.00")

    def disp_key(s):
        return ({"global": 0, "sliced": 1, "batched": 2, "native": 3}[s["kind"]]
                + (1 if s["arm"].startswith("s16_") and s["arm"] != "s16" else 0),
                s["phase"] if s["kind"] == "global" else s["n_slices"], s["arm"])

    shown = sorted(report["arms"], key=disp_key)
    print(f"\n== RS SWEEP  config={a.config}  run_dir={out}  n_frames={len(frames)} ==")
    print("BASELINE = g_p1.00 (shutter-END pose): what every render before today used.")
    print(f"{'arm':<17}{'calls':>6}{'gradNCC':>9}{'d vs p1.00':>11}{'%':>7}"
          f"{'margin':>9}{'negctl':>7}{'ms/frame':>10}{'wall ms':>9}{'|d| native':>11}"
          f"{'meanA':>8}{'rmean':>8}")
    for s in shown:
        d = s["grad_ncc_mean"] - base["grad_ncc_mean"]
        print(f"{s['arm']:<17}{s['n_render_calls_per_frame']:>6}"
              f"{s['grad_ncc_mean']:>9.4f}{d:>+11.4f}"
              f"{100.0 * d / base['grad_ncc_mean']:>+7.1f}"
              f"{s['neg_margin_mean']:>+9.4f}"
              f"{('PASS' if s['neg_control_all_pass'] else 'FAIL'):>7}"
              f"{s['raster_ms_median']:>10.1f}{s['wall_ms_median']:>9.1f}"
              f"{s.get('mean_abs_diff_vs_native_u8', float('nan')):>11.3f}"
              f"{s['mean_alpha']:>8.4f}{s['render_mean']:>8.4f}")
    print(f"(reference frame mean = {base['ref_mean']:.4f}; a render_mean far ABOVE it "
          f"is over-brightening, not fidelity)")
    print(f"\nPER-BAND grad-NCC (band 0 = TOP of the frame = shutter START under "
          f"TOP_TO_BOTTOM):")
    print(f"{'arm':<17}" + "".join(f"{'b%d' % k:>8}" for k in range(N_BANDS)))
    for s in shown:
        print(f"{s['arm']:<17}" + "".join(f"{v:>8.4f}"
                                          for v in s["band_grad_ncc_mean"]))
    print(f"\nPER-BAND DELTA vs g_p1.00 (where does the gain actually live?):")
    print(f"{'arm':<17}" + "".join(f"{'b%d' % k:>8}" for k in range(N_BANDS)))
    for s in shown:
        print(f"{s['arm']:<17}" + "".join(
            f"{v - b:>+8.4f}" for v, b in zip(s["band_grad_ncc_mean"],
                                              base["band_grad_ncc_mean"])))
    print(f"\nreport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
