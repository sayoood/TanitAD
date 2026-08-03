#!/usr/bin/env python3
"""Measure NuRec render quality — negative control FIRST, coverage always beside it.

WHY THIS FILE EXISTS
--------------------
The render-quality work before 2026-08-03 went wrong in three specific ways, and this
harness is built so each of them is structurally hard to repeat:

1. ⛔ **PSNR and plain NCC are RETRACTED on this clip.** Every frame is a dark night
   street, so both rank a WRONG reference frame above the correct one (PSNR: frame 150
   scores 17.457 vs the correct frame's 16.758; NCC: frame 450 scores 0.782 vs 0.704).
   ⇒ the headline metric is **gradient-NCC**, and the NEGATIVE CONTROL (correct frame vs
   >=4 wrong frames) runs on every single arm, before the arm is allowed to report a
   number. `neg_control_pass` is part of every row.

2. ⚠️ **A headline table was once copied from a superseded run directory** (the rejected
   `xyzw` quaternion layout). ⇒ every row carries `run_dir`, `git_sha`, `render_mean`
   and the full arm config, and the report writes them next to the numbers.

3. ⚠️ **Full-frame averages hid the real residual.** 79-81 % of the total absolute error
   lives in pixels NO GAUSSIAN COVERS (`mean_alpha` = 0.5145). ⇒ every error is reported
   THREE ways: full-frame, restricted to covered pixels, restricted to uncovered — plus
   the share of the total error each holds.

WHAT AN ARM IS
--------------
A layer set + a sky setting. Arms are named, rendered on the same frames, and scored
against the scene's own shipped `camera_front_wide_120fov.mp4`.

    python render_quality.py --scene-dir <scene> --out /tmp/rq/base \
        --arm base:background,road \
        --arm all4:background,road,dynamic_rigids,dynamic_deformables \
        --arm all4_sky:background,road,dynamic_rigids,dynamic_deformables:sky=0,6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

CAM = "camera_front_wide_120fov"
DEFAULT_FRAMES = (0, 60, 150, 300, 450)
N_WRONG = 5          # >= 4 is the bar; a metric judged on fewer certifies nothing
MIN_WRONG_GAP = 40   # a "wrong" frame 5 frames away is nearly the correct one


def wrong_frames_for(correct: int, n_frames: int, n: int = N_WRONG,
                     gap: int = MIN_WRONG_GAP) -> list[int]:
    """Wrong reference frames spread across the WHOLE clip, never adjacent to the
    correct one. Clip-aware so the control does not silently shrink below 4 when an
    offset runs off the end (which is how a negative control quietly stops being one)."""
    cand = [int(round(x)) for x in np.linspace(0, n_frames - 1, n + 4)]
    out = [f for f in dict.fromkeys(cand) if abs(f - correct) >= gap]
    return out[:n]


# --------------------------------------------------------------------------------- #
# metrics                                                                            #
# --------------------------------------------------------------------------------- #
def grad_ncc(a: np.ndarray, b: np.ndarray) -> float:
    import cv2
    la = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ga = np.hypot(cv2.Sobel(la, cv2.CV_32F, 1, 0, 3), cv2.Sobel(la, cv2.CV_32F, 0, 1, 3))
    gb = np.hypot(cv2.Sobel(lb, cv2.CV_32F, 1, 0, 3), cv2.Sobel(lb, cv2.CV_32F, 0, 1, 3))
    ga = ga.ravel() - ga.mean(); gb = gb.ravel() - gb.mean()
    return float(ga @ gb / max(np.linalg.norm(ga) * np.linalg.norm(gb), 1e-12))


ALPHA_EDGES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995, 1.0001)


def coverage_and_error(render_u8, ref_u8, alpha, cov_thresh=0.05):
    """Coverage + the error split that the 'near-equal per-channel gain' claim missed.

    ⚠️ TERMINOLOGY, because the previous framing was wrong and it changed the diagnosis.
    The banked claim *"79-81 % of the absolute error lives in pixels NO GAUSSIAN COVERS,
    mean_alpha 0.5145, roughly half the frame is uncovered"* conflates two different
    things. `isp_experiment.py` defines covered as `alpha >= amin` with **amin = 0.995**,
    i.e. **fully opaque** — so "uncovered" there means "not fully opaque", which is a
    very different statement from "no gaussian covers it". The full alpha histogram is
    reported here so the distinction can never collapse again.
    """
    r = render_u8.astype(np.float32) / 255.0
    g = ref_u8.astype(np.float32) / 255.0
    ae = np.abs(r - g).mean(-1)                       # [H,W] per-pixel abs error
    cov = alpha > cov_thresh
    opaque = alpha >= 0.995
    n, ncov = ae.size, int(cov.sum())
    tot = float(ae.sum())
    hist, share = [], []
    for lo, hi in zip(ALPHA_EDGES[:-1], ALPHA_EDGES[1:]):
        m = (alpha >= lo) & (alpha < hi)
        hist.append(round(float(m.mean()), 5))
        share.append(round(float(ae[m].sum() / tot), 5) if tot > 0 and m.any() else 0.0)
    return {
        "mean_alpha": float(alpha.mean()),
        "frac_covered": ncov / n,
        "frac_alpha_ge_0.995_OPAQUE": float(opaque.mean()),
        "frac_alpha_lt_0.01_TRULY_EMPTY": float((alpha < 0.01).mean()),
        "render_mean": float(r.mean()), "ref_mean": float(g.mean()),
        "mae_full": float(ae.mean()),
        "mae_covered": float(ae[cov].mean()) if ncov else None,
        "mae_uncovered": float(ae[~cov].mean()) if ncov < n else None,
        "share_of_abs_error_uncovered": float(ae[~cov].sum() / tot) if tot > 0 else None,
        "share_of_abs_error_not_opaque": float(ae[~opaque].sum() / tot) if tot > 0 else None,
        "mae_opaque": float(ae[opaque].mean()) if opaque.any() else None,
        "alpha_hist_edges": list(ALPHA_EDGES),
        "alpha_hist_pixel_frac": hist,
        "alpha_hist_abs_error_share": share,
        "cov_thresh": cov_thresh,
        # THE decisive number for whether the sky env map belongs in the frame at all:
        # what the REFERENCE actually shows where our gaussians left the pixel thin.
        # If the reference is near-black there, then "the render is missing the sky" is
        # false and any env map we add is pure over-brightening.
        "ref_mean_where_alpha_lt_0.1": (float(g[alpha < 0.1].mean())
                                        if (alpha < 0.1).any() else None),
        "render_mean_where_alpha_lt_0.1": (float(r[alpha < 0.1].mean())
                                           if (alpha < 0.1).any() else None),
        "frac_alpha_lt_0.1": float((alpha < 0.1).mean()),
    }


def negative_control(render_u8, refs, correct_frame, wrong_frames):
    """grad-NCC of ONE render against the correct reference frame and >=4 wrong ones.

    A metric that cannot separate right from wrong certifies nothing, so this runs
    before any quality claim is admitted."""
    rows, mae_rows, psnr_rows = {}, {}, {}
    for f in (correct_frame, *wrong_frames):
        if f in rows or f not in refs:
            continue
        rows[f] = grad_ncc(render_u8, refs[f])
        # MAE and PSNR are scored against the SAME frame set so that "may this number
        # decide anything?" is answered by measurement rather than by precedent.
        e = (render_u8.astype(np.float32) - refs[f].astype(np.float32)) / 255.0
        mae_rows[f] = float(np.abs(e).mean())
        psnr_rows[f] = float(10.0 * np.log10(1.0 / max(float((e ** 2).mean()), 1e-12)))
    best_wrong_f = max((f for f in rows if f != correct_frame),
                       key=lambda f: rows[f], default=None)
    correct = rows[correct_frame]
    bw = rows[best_wrong_f] if best_wrong_f is not None else float("-inf")
    return {"grad_ncc_by_ref_frame": {int(k): round(v, 4) for k, v in rows.items()},
            "correct_frame": int(correct_frame),
            "grad_ncc_correct": round(correct, 4),
            "best_wrong_frame": (int(best_wrong_f) if best_wrong_f is not None else None),
            "grad_ncc_best_wrong": round(bw, 4),
            "margin": round(correct - bw, 4),
            "argmax_is_correct": bool(max(rows, key=lambda f: rows[f]) == correct_frame),
            "n_wrong_frames": len(rows) - 1,
            "pass": bool(max(rows, key=lambda f: rows[f]) == correct_frame
                         and len(rows) - 1 >= 4),
            # ---- is a PHOTOMETRIC metric even admissible on this clip? ----
            "mae_by_ref_frame": {int(k): round(v, 4) for k, v in mae_rows.items()},
            "mae_argmin_is_correct": bool(min(mae_rows, key=lambda f: mae_rows[f])
                                          == correct_frame),
            "psnr_argmax_is_correct": bool(max(psnr_rows, key=lambda f: psnr_rows[f])
                                           == correct_frame),
            "psnr_by_ref_frame": {int(k): round(v, 3) for k, v in psnr_rows.items()}}


def load_refs(mp4, frames, size_wh, ref_offset: int = 0):
    """Decode every needed reference frame in ONE sequential pass.

    Seeking a 3840x2160 mp4 once per (arm, frame, wrong-frame) pair dominated the
    harness's wall clock; the references do not change between arms, so they are
    decoded once and reused.

    ``frames`` are **RIG** indices and the returned dict is keyed by RIG index; the
    video frame actually decoded for rig frame ``f`` is ``f + ref_offset``.  Keeping the
    keys in rig space means every caller downstream (``negative_control``,
    ``wrong_frames_for``, ``score_arm``) is correct without knowing the offset exists —
    which is the only way an alignment fix does not have to be re-applied at N call
    sites and forgotten at one of them.  See ``R-2026-08-03-k``.
    """
    import cv2
    want = sorted(set(int(f) for f in frames))
    vid = {int(f) + int(ref_offset): int(f) for f in want}
    need = sorted(vid)
    out, cap, i, k = {}, cv2.VideoCapture(str(mp4)), 0, 0
    while k < len(need):
        ok, img = cap.read()
        if not ok:
            break
        if i == need[k]:
            img = img[:, :, ::-1]
            if (img.shape[1], img.shape[0]) != tuple(size_wh):
                img = cv2.resize(img, tuple(size_wh), interpolation=cv2.INTER_AREA)
            out[vid[need[k]]] = np.ascontiguousarray(img)
            k += 1
        i += 1
    cap.release()
    return out


# --------------------------------------------------------------------------------- #
# arms                                                                               #
# --------------------------------------------------------------------------------- #
def parse_arm(spec: str) -> dict:
    """`name:layer,layer[,...][:sky=lo,hi[,gain]]`"""
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"bad arm spec {spec!r}; want name:layers[:sky=lo,hi][:rs]")
    arm = {"name": parts[0], "layers": [x for x in parts[1].split(",") if x],
           "sky": None, "rs": False, "cull": None, "haze": None}
    for extra in parts[2:]:
        if extra.startswith("sky="):
            v = [float(x) for x in extra[4:].split(",")]
            arm["sky"] = {"lo_deg": v[0], "hi_deg": v[1],
                          "gain": v[2] if len(v) > 2 else 1.0}
        elif extra == "naive-sky":
            arm["sky"] = {"lo_deg": -90.0, "hi_deg": -90.0, "gain": 1.0}  # gate always 1
        elif extra == "rs":
            arm["rs"] = True      # rolling shutter: render start->end down the frame
        elif extra.startswith("cull="):
            arm["cull"] = float(extra[5:])   # drop static splats above this scale-quantile
        elif extra.startswith("haze="):
            v = [float(x) for x in extra[5:].split(",")]
            arm["haze"] = {"scale_quantile": v[0], "opacity_max": v[1]}
        else:
            raise ValueError(f"unknown arm option {extra!r}")
    return arm


def build_renderer(scene_dir: Path, arm: dict, loader_dir: str | None, verbose=False):
    from actor_map import attach_all_dynamic_layers
    from gsplat_renderer import NuRecGsplatRenderer
    static = [L for L in arm["layers"] if L in ("background", "road")]
    dynamic = [L for L in arm["layers"] if L.startswith("dynamic_")]
    r = NuRecGsplatRenderer(scene_dir, layers=static, loader_dir=loader_dir,
                            verbose=verbose)
    info = None
    if arm.get("haze"):
        r.cull_haze(**arm["haze"])
    if arm.get("cull"):
        r.cull_by_scale(float(arm["cull"]))
    if dynamic:
        info = attach_all_dynamic_layers(r, scene_dir, layers=tuple(dynamic), frames=(0,))
    if arm["sky"]:
        r.attach_sky(**arm["sky"])
    return r, info


def score_arm(r, scene_dir: Path, arm: dict, frames, refs, wrong_map, warmup=2):
    # warm-up: the first rasterization pays CUDA autotune/JIT and is not the steady state
    c2n0 = r.gt_cam_to_nre(frames[0])
    e0 = r.gt_cam_to_nre_pair(frames[0])[1] if arm.get("rs") else None
    ts0 = r.frame_timestamps_us(frames[0])[1]
    for _ in range(warmup):
        r.render(c2n0, actor_time_us=float(ts0), cam_to_nre_end=e0)

    rows, imgs = [], {}
    for f in frames:
        if arm.get("rs"):
            c2n, c2n_end = r.gt_cam_to_nre_pair(f)
        else:
            c2n, c2n_end = r.gt_cam_to_nre(f), None
        ts = float(r.frame_timestamps_us(f)[1])
        t0 = time.time()
        img, alpha, ms = r.render(c2n, actor_time_us=ts, cam_to_nre_end=c2n_end)
        wall = (time.time() - t0) * 1000.0
        ref = refs[f]
        row = {"frame": int(f), "raster_ms": round(ms, 2), "wall_ms": round(wall, 2),
               "n_actor_gaussians": int(getattr(r, "last_actors_rendered", 0)),
               "actors_per_layer": dict(getattr(r, "last_actors_per_layer", {}) or {})}
        row.update(coverage_and_error(img, ref, alpha))
        row["neg_control"] = negative_control(img, refs, f, wrong_map[f])
        row["grad_ncc"] = row["neg_control"]["grad_ncc_correct"]
        if r._sky is not None:
            row["sky_weight_mean"] = round(getattr(r, "last_sky_weight_mean", 0.0), 4)
        rows.append(row)
        imgs[f] = (img, alpha, ref)
    return rows, imgs


def summarize(name, rows, arm, extra=None):
    ok = [x for x in rows if x["neg_control"]["pass"]]
    s = {"arm": name, "layers": arm["layers"], "sky": arm["sky"],
         "rolling_shutter": bool(arm.get("rs")), "n_frames": len(rows),
         "neg_control_pass_frames": len(ok),
         "neg_control_all_pass": len(ok) == len(rows),
         # how often each metric picks the CORRECT reference frame out of >=5 candidates
         "gradncc_argmax_correct_frames": sum(
             1 for x in rows if x["neg_control"]["argmax_is_correct"]),
         "mae_argmin_correct_frames": sum(
             1 for x in rows if x["neg_control"]["mae_argmin_is_correct"]),
         "psnr_argmax_correct_frames": sum(
             1 for x in rows if x["neg_control"]["psnr_argmax_is_correct"]),
         "grad_ncc_mean": round(float(np.mean([x["grad_ncc"] for x in rows])), 4),
         "neg_margin_mean": round(float(np.mean(
             [x["neg_control"]["margin"] for x in rows])), 4),
         "mean_alpha": round(float(np.mean([x["mean_alpha"] for x in rows])), 4),
         "frac_covered": round(float(np.mean([x["frac_covered"] for x in rows])), 4),
         "mae_full": round(float(np.mean([x["mae_full"] for x in rows])), 4),
         "mae_covered": round(float(np.mean(
             [x["mae_covered"] for x in rows if x["mae_covered"] is not None])), 4),
         "mae_opaque": round(float(np.mean(
             [x["mae_opaque"] for x in rows if x["mae_opaque"] is not None])), 4),
         "share_err_uncovered": round(float(np.mean(
             [x["share_of_abs_error_uncovered"] for x in rows
              if x["share_of_abs_error_uncovered"] is not None])), 4),
         "share_err_not_opaque": round(float(np.mean(
             [x["share_of_abs_error_not_opaque"] for x in rows
              if x["share_of_abs_error_not_opaque"] is not None])), 4),
         "frac_opaque": round(float(np.mean(
             [x["frac_alpha_ge_0.995_OPAQUE"] for x in rows])), 4),
         "frac_truly_empty": round(float(np.mean(
             [x["frac_alpha_lt_0.01_TRULY_EMPTY"] for x in rows])), 5),
         "ref_mean_low_alpha": round(float(np.mean(
             [x["ref_mean_where_alpha_lt_0.1"] for x in rows
              if x["ref_mean_where_alpha_lt_0.1"] is not None])), 4),
         "render_mean_low_alpha": round(float(np.mean(
             [x["render_mean_where_alpha_lt_0.1"] for x in rows
              if x["render_mean_where_alpha_lt_0.1"] is not None])), 4),
         "frac_low_alpha": round(float(np.mean(
             [x["frac_alpha_lt_0.1"] for x in rows])), 4),
         "render_mean": round(float(np.mean([x["render_mean"] for x in rows])), 4),
         "ref_mean": round(float(np.mean([x["ref_mean"] for x in rows])), 4),
         "raster_ms_median": round(float(np.median([x["raster_ms"] for x in rows])), 2),
         "actor_gaussians_median": int(np.median([x["n_actor_gaussians"] for x in rows])),
         }
    if extra:
        s["attach"] = extra
    return s


# --------------------------------------------------------------------------------- #
# ⛔ THE ALIGNMENT GATE — runs BEFORE any fidelity number is produced                 #
# --------------------------------------------------------------------------------- #
def assert_reference_aligned(r, refs, frames, ref_offset: int, k: int = 3,
                             n_probe: int = 3, out_dir: Path | None = None,
                             min_prominence: float = 0.02,
                             min_modal_mass: float = 0.5) -> dict:
    """Render probe frames and score them against reference frames ``f-k .. f+k``.

    ⛔ THE FAILURE THIS PREVENTS ALREADY HAPPENED (`R-2026-08-03-k`).  Every absolute
    grad-NCC / MAE / PSNR on scene `00040136` was scored against a reference **6 frames
    too early**, for weeks, while the numbers still looked plausible and the harness's
    own negative control PASSED on every frame.  It passed because
    ``wrong_frames_for()`` enforces ``MIN_WRONG_GAP = 40``: a 6-frame error is invisible
    to it **by construction**.  The hard negatives for an alignment error are the
    IMMEDIATE NEIGHBOURS — exactly the frames that control deliberately excludes.

    So this gate is a different control, not a stricter one, and it is not optional:
    it runs before the arms, on the same renderer, and a non-zero argmax is a hard stop.

    Raises ``SystemExit`` on failure, after writing the evidence to ``out_dir``.
    """
    from frame_align import adjudicate, bootstrap_offset

    # ⚠️ A probe frame needs its WHOLE neighbourhood decoded, or its argmax sits at a
    # truncated edge and the adjudicator refuses on `boundary` — which is the correct
    # behaviour of the estimator and a FALSE alarm from the gate. Rig frame 0 has no
    # f-k neighbours by construction, so it can never be a probe frame. MEASURED: the
    # first run of this gate failed on `{0: None}` at a correct offset.
    eligible = [f for f in frames if all((f + d) in refs for d in range(-k, k + 1))]
    if not eligible:
        raise SystemExit(
            f"⛔ ALIGNMENT GATE CANNOT RUN: no frame in {list(frames)} has a complete "
            f"+-{k} reference neighbourhood decoded. Reduce --align-k or pick interior "
            "frames. A gate that silently skips itself is not a gate.")
    probe = [eligible[i] for i in
             sorted(set(int(round(x)) for x in
                        np.linspace(0, len(eligible) - 1, min(n_probe, len(eligible)))))]
    # ⚠️ A frame carries alignment information only if its curve has a PROMINENT peak.
    # On a stationary segment every neighbouring reference frame is nearly identical, so
    # the curve is flat and its argmax is noise. MEASURED on `7c72937c` frame 60: the
    # whole +-10 curve spans 0.3994-0.4041 and the argmax landed at -6 at a CORRECT
    # offset. Such a frame is UNINFORMATIVE, not misaligned, and must not vote —
    # otherwise the gate blocks a correct run, which is how gates get disabled.
    curves, informative, offwindow, per_frame = [], [], [], {}
    for f in probe:
        img = r.render(r.gt_cam_to_nre(f),
                       actor_time_us=float(r.frame_timestamps_us(f)[1]))[0]
        cur = {d: round(grad_ncc(img, refs[f + d]), 5)
               for d in range(-k, k + 1) if (f + d) in refs}
        est = adjudicate(cur, "render_neighbour_scan", min_peak=0.05,
                         min_prominence=min_prominence, require_turnover=False)
        if not est.refused:
            curves.append(cur)
        per_frame[int(f)] = {"grad_ncc_by_offset": cur,
                             "argmax_offset": est.offset, "refused": est.refused,
                             "reason": est.reason,
                             "informative": (not est.refused),
                             "prominence": est.prominence,
                             "subframe_offset": est.subframe_offset,
                             "gain_vs_offset0": (round(cur[est.offset] - cur[0], 5)
                                                 if est.offset is not None and 0 in cur
                                                 else None)}
        if not est.refused:
            informative.append(int(f))
        elif est.reason == "boundary":
            # NOT "uninformative": the curve is still climbing at the scan edge, so the
            # residual exists and is >= k. That is a misalignment of unknown magnitude.
            offwindow.append(int(f))
    boot = bootstrap_offset(curves, b=1000, seed=0, method="render_neighbour_scan",
                            min_peak=0.05, min_prominence=min_prominence,
                            require_turnover=False)
    argmaxes = {f: per_frame[f]["argmax_offset"] for f in informative}
    modal0 = float(boot.get("mass", {}).get(0, 0.0))
    subs = [per_frame[f]["subframe_offset"] for f in informative
            if per_frame[f]["subframe_offset"] is not None]
    # PASS = the INTEGER residual is 0 on the aggregate. A sub-frame preference (the
    # optimum sitting between two samples) may split single-frame argmaxes across 0/+-1
    # without any index being wrong; a genuine off-by-one moves the bootstrap mass
    # wholesale, as the offset-0 demonstration on `00040136` shows (mass {6: 1.0}).
    ok = (len(informative) > 0 and not offwindow
          and boot.get("point") == 0 and modal0 >= min_modal_mass)
    res = {"gate": "reference_alignment", "ref_offset_applied": int(ref_offset),
           "k": k, "probe_frames": [int(f) for f in probe],
           "frames_offered": [int(f) for f in frames],
           "frames_skipped_incomplete_window": [int(f) for f in frames
                                                if f not in eligible],
           "informative_frames": informative,
           "offwindow_frames": offwindow,
           "uninformative_frames": {str(f): per_frame[f]["reason"]
                                    for f in per_frame
                                    if f not in informative and f not in offwindow},
           "min_prominence": min_prominence, "min_modal_mass": min_modal_mass,
           "residual_argmax_by_frame": {str(f): v["argmax_offset"]
                                        for f, v in per_frame.items()},
           "residual_offset_bootstrap": boot,
           "residual_mass_at_zero": round(modal0, 4),
           "residual_subframe_mean": (round(float(np.mean(subs)), 3) if subs else None),
           "per_frame": per_frame,
           "pass": bool(ok),
           "why": ("the harness's other negative control uses MIN_WRONG_GAP=40 and is "
                   "structurally blind to a small index error; this one is not")}
    if out_dir is not None:
        (Path(out_dir) / "alignment_gate.json").write_text(json.dumps(res, indent=1))
    if not ok:
        bad = {f: a for f, a in argmaxes.items() if a != 0}
        if not informative and not offwindow:
            raise SystemExit(
                "\n⛔ ALIGNMENT GATE CANNOT CERTIFY — and 'cannot certify' is NOT 'aligned'.\n"
                f"   none of the probe frames {probe} has a peak prominent enough to "
                f"identify an offset (min_prominence={min_prominence}).\n"
                "   Pick probe frames where the ego is MOVING; a stationary segment "
                "carries no alignment information at all.")
        if offwindow or boot.get("point") is None:
            # the residual is at or beyond the scan edge — the ">= +3" failure again:
            # reporting a boundary as an answer is exactly what this module forbids.
            advice = (f"   the residual is AT OR BEYOND the +-{k} scan edge on frames "
                      f"{offwindow or 'the mean curve'}, so this gate cannot name the "
                      "correction.\n   Widen --align-k, or measure the offset with "
                      "frame_align.py / rs_frame_offset.py.")
        else:
            advice = (f"   => the correct offset is {ref_offset + boot['point']:+d} "
                      f"(residual {boot['point']:+d}, bootstrap mass {boot.get('mass')}). "
                      "Re-run with --ref-offset that.")
        raise SystemExit(
            "\n⛔ ALIGNMENT GATE FAILED — NO FIDELITY NUMBER FROM THIS RUN IS ADMISSIBLE.\n"
            f"   applied --ref-offset {ref_offset:+d}; residual is NOT 0 on "
            f"{len(bad)}/{len(argmaxes)} informative probe frames: {bad}\n"
            f"   bootstrap residual {boot.get('point')} mass {boot.get('mass')} "
            f"(mass at 0 = {modal0:.3f} < {min_modal_mass})\n"
            + advice + "\n"
            "   This is R-2026-08-03-k. Do not disable the gate to get a number.")
    return res


def git_sha(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def save_sbs(path, render, ref, alpha, label):
    """render | reference | |diff|x3 | alpha — the coverage panel is not optional."""
    import cv2
    pad = np.full((render.shape[0], 8, 3), 255, np.uint8)
    diff = np.clip(np.abs(render.astype(np.int16) - ref.astype(np.int16)) * 3,
                   0, 255).astype(np.uint8)
    acol = cv2.applyColorMap((np.clip(alpha, 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
    canvas = np.concatenate([render, pad, ref, pad, diff, pad, acol], 1)
    bar = np.zeros((40, canvas.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    out = np.concatenate([bar, canvas], 0)
    cv2.imwrite(str(path), out[:, :, ::-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True, help="RUN DIRECTORY — quote this, not a number")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--frames", default=",".join(str(f) for f in DEFAULT_FRAMES))
    ap.add_argument("--loader-dir", default=None)
    ap.add_argument("--png-frame", type=int, default=0)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--ref-offset", type=int, default=None,
                    help="video_index = rig_index + OFFSET. Default: read PER SCENE from "
                         "(n_mp4_decodable - n_rig). ⛔ Do NOT hard-code +6 — it is +5 on "
                         "7c72937c (R-2026-08-03-k).")
    ap.add_argument("--align-k", type=int, default=3,
                    help="neighbour half-window for the alignment gate")
    ap.add_argument("--align-probe-frames", type=int, default=3)
    ap.add_argument("--no-align-check", action="store_true",
                    help="⛔ disables the gate. Recorded in the report; any number "
                         "produced with this flag is NOT admissible as an absolute.")
    a = ap.parse_args()

    scene = Path(a.scene_dir).expanduser()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    frames = [int(x) for x in a.frames.split(",")]

    # one sequential decode of every reference frame this run will ever need
    arms = [parse_arm(s) for s in a.arm]
    if a.loader_dir:
        import sys
        sys.path.insert(0, a.loader_dir)
    from nurec_loader import RigTrajectories
    _rig = RigTrajectories(scene / "rig_trajectories.json")
    _cam = _rig.camera(CAM)
    n_frames, size = _rig.n_frames(CAM), (int(_cam.width), int(_cam.height))

    # ---- the reference offset, PER SCENE, never hard-coded --------------------------
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frame_align import count_delta, mp4_motion_series
    mp4 = scene / f"{CAM}.mp4"
    if a.ref_offset is None:
        n_dec, _meta, _M, _mn, _hd = mp4_motion_series(mp4)
        e = count_delta(n_dec, n_frames)
        if e.refused:
            raise SystemExit(f"cannot derive --ref-offset for {scene.name}: {e}")
        ref_offset, offset_src = e.offset, f"count_delta(mp4={n_dec}, rig={n_frames})"
    else:
        ref_offset, offset_src = int(a.ref_offset), "--ref-offset (explicit)"
    print(f"[rq] reference offset {ref_offset:+d} frames  [{offset_src}]", flush=True)

    wrong_map = {f: wrong_frames_for(f, n_frames) for f in frames}
    need = set(frames) | {w for ws in wrong_map.values() for w in ws}
    # the alignment gate scores against the IMMEDIATE NEIGHBOURS, so decode them too
    need |= {f + d for f in frames for d in range(-a.align_k, a.align_k + 1)
             if 0 <= f + d < n_frames}
    t_ref = time.time()
    refs = load_refs(mp4, need, size, ref_offset=ref_offset)
    print(f"[rq] decoded {len(refs)}/{len(need)} reference frames in "
          f"{time.time() - t_ref:.1f}s (clip has {n_frames} rig frames, "
          f"video index = rig index {ref_offset:+d})", flush=True)
    missing = sorted(need - set(refs))
    if missing:
        raise SystemExit(f"reference frames not decodable: {missing}")

    report = {"run_dir": str(out), "scene_dir": str(scene),
              "git_sha": git_sha(Path(a.repo)),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "frames": frames, "clip_n_frames": n_frames,
              "ref_offset": int(ref_offset), "ref_offset_source": offset_src,
              "ref_offset_rule": "video_index = rig_index + ref_offset  (R-2026-08-03-k)",
              "alignment_gate": None,
              "alignment_gate_disabled": bool(a.no_align_check),
              "wrong_frames_per_correct": {int(k): v for k, v in wrong_map.items()},
              "metric": "gradient-NCC (PSNR and plain NCC are RETRACTED on this clip)",
              "arms": []}

    # ---- ⛔ the gate, BEFORE any arm is allowed to report a number -------------------
    if a.no_align_check:
        print("[rq] ⛔ ALIGNMENT GATE DISABLED — absolutes from this run are NOT "
              "admissible", flush=True)
    else:
        _r0, _ = build_renderer(scene, arms[0], a.loader_dir)
        for _ in range(2):
            _r0.render(_r0.gt_cam_to_nre(frames[0]))
        report["alignment_gate"] = assert_reference_aligned(
            _r0, refs, frames, ref_offset, k=a.align_k,
            n_probe=a.align_probe_frames, out_dir=out)
        print(f"[rq] ✅ alignment gate PASS at offset {ref_offset:+d} "
              f"(residual argmax 0 on every probe frame)", flush=True)
        del _r0
        import torch
        torch.cuda.empty_cache()

    for arm in arms:
        print(f"[rq] === arm {arm['name']}  layers={arm['layers']} sky={arm['sky']}",
              flush=True)
        t0 = time.time()
        r, info = build_renderer(scene, arm, a.loader_dir)
        rows, imgs = score_arm(r, scene, arm, frames, refs, wrong_map)
        s = summarize(arm["name"], rows, arm, extra=info)
        s["build_s"] = round(time.time() - t0, 1)
        s["n_static_gaussians"] = int(r.n_gauss)
        s["layer_counts"] = dict(r.layer_counts)
        s["cull"] = getattr(r, "cull_info", None)
        s["shutter_type_declared"] = r.rolling_shutter_type
        s["per_frame"] = rows
        report["arms"].append(s)
        if a.png_frame in imgs:
            img, alpha, ref = imgs[a.png_frame]
            save_sbs(out / f"sbs_{arm['name']}_f{a.png_frame}.png", img, ref, alpha,
                     f"{arm['name']}  layers={'+'.join(arm['layers'])}  "
                     f"sky={arm['sky'] is not None}  gradNCC={s['grad_ncc_mean']:.4f}  "
                     f"mean_alpha={s['mean_alpha']:.4f}")
            np.savez_compressed(out / f"render_{arm['name']}_f{a.png_frame}.npz",
                                render=img, alpha=alpha, ref=ref)
        print(json.dumps({k: v for k, v in s.items()
                          if k not in ("per_frame", "attach")}, indent=1), flush=True)
        del r
        import torch
        torch.cuda.empty_cache()

    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("\n== SUMMARY (run_dir %s) ==" % out)
    hdr = (f"{'arm':<24}{'gradNCC':>9}{'margin':>9}{'negctl':>7}{'meanA':>8}"
           f"{'MAEfull':>9}{'MAEopq':>9}{'rmean':>8}{'refmean':>9}"
           f"{'r@loA':>8}{'ref@loA':>9}{'ms':>8}")
    print(hdr)
    for s in report["arms"]:
        print(f"{s['arm']:<24}{s['grad_ncc_mean']:>9.4f}{s['neg_margin_mean']:>9.4f}"
              f"{('PASS' if s['neg_control_all_pass'] else 'FAIL'):>7}"
              f"{s['mean_alpha']:>8.4f}{s['mae_full']:>9.4f}"
              f"{s['mae_opaque']:>9.4f}{s['render_mean']:>8.4f}{s['ref_mean']:>9.4f}"
              f"{s['render_mean_low_alpha']:>8.4f}{s['ref_mean_low_alpha']:>9.4f}"
              f"{s['raster_ms_median']:>8.1f}")
    n = report["arms"][0]["n_frames"] if report["arms"] else 0
    print(f"\nWHICH METRIC MAY DECIDE? frames where the metric picks the CORRECT "
          f"reference out of >=5 candidates (n={n} frames per arm):")
    print(f"{'arm':<24}{'gradNCC':>9}{'MAE':>7}{'PSNR':>7}")
    for s in report["arms"]:
        print(f"{s['arm']:<24}{s['gradncc_argmax_correct_frames']:>9d}"
              f"{s['mae_argmin_correct_frames']:>7d}{s['psnr_argmax_correct_frames']:>7d}")
    print(f"\nreport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
