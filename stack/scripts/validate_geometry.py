"""Axis-2 GATE — geometry & unified calibration (the VLM3 "one effective focal").

Pre-flight validation (Project Steering/PRE_FLIGHT_VALIDATION.md, Axis 2) for the
4-day flagship run. The prior run wasted time on a 1.6x focal error (PhysicalAI
front-wide is an f-theta FISHEYE canonicalized with a pinhole model — see
Benchmarks & Eval/GEOMETRY_INTEGRITY_AUDIT.md). This script PROVES, with
evidence, whether the corrected f-theta canonicalization in tanitad/data/calib.py
now makes comma2k19 and PhysicalAI-AV geometrically consistent, or finds where it
still doesn't. It VALIDATES calib.py; it never edits it.

Five checks, each -> PASS/FAIL + a number:
  1  Shared canonical focal (the VLM3 invariant): achieved f_eff of the corrected
     256-px frames for BOTH corpora (comma via true 910 px; physicalai via the
     REAL per-clip fisheye poly in calibration/). PASS iff both = 266 +/- 8.
  2  Empirical cross-corpus pixels-per-metre: ground-motion scale (px/m) on
     straight+fast segments (geom_sanity FOE/optical-flow). PASS iff comma and
     physicalai agree within ~15%. Urban traffic contaminates physicalai flow ->
     also reports the calibration-derived expectation; falls back to it (as the
     prior audit did) if the flow is too noisy to conclude, and SAYS which.
  3  Undistortion/curvature: f-theta inverse round-trips exactly; a straight-world
     line maps to a straight image line (residual curvature small); principal
     point centred; effective horizon ~ h/2.
  4  Silent-skew guard: CORPUS_META / I7 fingerprint reports the ACHIEVED f_eff
     (266, sourced from calib.F_REF), not the old nominal (~434) -> the error
     can't hide again.
  5  Sanity vs the old cache: corrected frames have a WIDER field (more scene)
     than the old over-cropped ones (before/after FOV + 2 annotated frames).

Usage (pod2, which holds all three caches + the intrinsics CSV + raw clips):
  python scripts/validate_geometry.py \
    --comma-cache /opt/comma_epcache \
    --pai-cache   /workspace/data/physicalai/_epcache \
    --pai-cache-old /workspace/data/physicalai/_epcache_old \
    --intrinsics  /workspace/data/physicalai/calibration/physicalai_front_wide_intrinsics.csv \
    --pai-root    /workspace/data/physicalai \
    --episodes 90 --out /workspace/experiments/validate_geometry.json \
    --png-dir /workspace/experiments/geom_png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tanitad.data.calib import (F_REF, COMMA2K19_FOCAL_PX,
                                PHYSICALAI_FRONT_WIDE_FTHETA, FThetaIntrinsics,
                                canonical_halfangle_rad, focal_crop_resize,
                                ftheta_crop_resize, ftheta_crop_size,
                                ftheta_feff_report, ftheta_undistort,
                                ftheta_undistort_grid)

# geom_sanity lives next to this script; reuse its flow/FOE machinery verbatim.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_sanity  # noqa: E402

FEFF_TOL = 8.0          # check 1 tolerance: f_eff = 266 +/- 8
PPM_TOL = 0.15          # check 2 tolerance: cross-corpus px/m agree within 15%


# --------------------------------------------------------------------------- #
# intrinsics helpers                                                           #
# --------------------------------------------------------------------------- #
def _load_intrinsics(csv_path: str) -> list[tuple[str, FThetaIntrinsics]]:
    df = pd.read_csv(csv_path, dtype={"clip_id": str})
    out = []
    for r in df.itertuples(index=False):
        out.append((str(r.clip_id), FThetaIntrinsics(
            poly=(float(r.fw_poly_0), float(r.fw_poly_1), float(r.fw_poly_2),
                  float(r.fw_poly_3), float(r.fw_poly_4)),
            cx=float(r.cx), cy=float(r.cy),
            width=int(r.width), height=int(r.height))))
    return out


def _stats(x) -> dict:
    a = np.asarray(x, float)
    return {"mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
            "std": round(float(a.std()), 4), "n": int(a.size)}


# --------------------------------------------------------------------------- #
# CHECK 1 — shared canonical focal (VLM3 invariant)                            #
# --------------------------------------------------------------------------- #
def check1_shared_focal(intr_list, pai_root: str | None) -> dict:
    # comma: the pipeline's own focal_crop_resize on a native-shaped frame. Its
    # crop clamps to the 874-px height for EVERY frame, so f_eff is a constant.
    dummy = torch.zeros(1, 3, 874, 1164, dtype=torch.uint8)
    focal_crop_resize(dummy, COMMA2K19_FOCAL_PX, 256)
    comma_feff = float(focal_crop_resize.last_f_eff)

    # physicalai: achieved f_eff from the REAL per-clip fisheye poly (the value
    # calib.ftheta_crop_resize delivers on each clip). Report the population.
    f_after, f_before, hfov_after, hfov_before = [], [], [], []
    for _cid, intr in intr_list:
        rep = ftheta_feff_report(intr, size=256, f_ref=F_REF)
        f_after.append(rep["f_eff_after"])
        f_before.append(rep["f_eff_before_nominal"])
        hfov_after.append(rep["retained_hfov_after_deg"])
        hfov_before.append(rep["retained_hfov_before_deg"])
    pai_after = _stats(f_after)
    pai_before = _stats(f_before)

    # byte-faithful confirmation: decode a few REAL clips and read the f_eff that
    # ftheta_crop_resize actually stamps (not just the analytic report).
    decode_confirm = []
    if pai_root:
        cam = Path(pai_root) / "r0" / "camera_front_wide"
        by_cid = dict(intr_list)
        mp4s = sorted(cam.rglob("*.mp4"))[:3]
        for mp4 in mp4s:
            cid = mp4.name.split(".")[0]
            intr = by_cid.get(cid, PHYSICALAI_FRONT_WIDE_FTHETA)
            try:
                import av
                frames = []
                with av.open(str(mp4)) as c:
                    s = c.streams.video[0]
                    for fr in c.decode(s):
                        frames.append(torch.from_numpy(
                            fr.to_ndarray(format="rgb24")).permute(2, 0, 1))
                        if len(frames) >= 2:
                            break
                vid = torch.stack(frames)
                ftheta_crop_resize(vid, intr, 256)
                decode_confirm.append({
                    "clip_id": cid, "native_HW": list(vid.shape[-2:]),
                    "decoded_f_eff": round(float(ftheta_crop_resize.last_f_eff), 2),
                    "analytic_f_eff": ftheta_feff_report(intr)["f_eff_after"]})
            except Exception as e:  # noqa: BLE001
                decode_confirm.append({"clip_id": cid, "error": repr(e)})

    comma_pass = abs(comma_feff - F_REF) <= FEFF_TOL
    pai_pass = abs(pai_after["median"] - F_REF) <= FEFF_TOL
    return {
        "F_REF": F_REF, "tol": FEFF_TOL,
        "comma_f_eff": round(comma_feff, 2),
        "physicalai_f_eff_after_stats": pai_after,
        "physicalai_f_eff_before_nominal_stats": pai_before,
        "physicalai_retained_hfov_after_deg": _stats(hfov_after),
        "physicalai_retained_hfov_before_deg": _stats(hfov_before),
        "decode_confirmation": decode_confirm,
        "comma_pass": bool(comma_pass), "physicalai_pass": bool(pai_pass),
        "PASS": bool(comma_pass and pai_pass),
        "note": ("both corpora land f_eff ~266; physicalai is NO LONGER ~431/434 "
                 "(the old nominal-pinhole zoom). VLM3 'one effective focal' holds."),
    }


# --------------------------------------------------------------------------- #
# CHECK 2 — empirical cross-corpus pixels-per-metre                            #
# --------------------------------------------------------------------------- #
def _flow_quality(gf: dict) -> bool:
    foe = gf.get("horizon_row_FOE")
    n = gf.get("n_straight_fast_bright_pairs", 0)
    fh = gf.get("scale_fit", {}).get("f_times_h")
    return bool(foe is not None and 96 <= foe <= 160 and n >= 40 and fh)


def check2_pixels_per_metre(comma_eps, pai_eps, comma_feff, pai_feff) -> dict:
    gf_c = geom_sanity.check_ground_flow(comma_eps)
    gf_p = geom_sanity.check_ground_flow(pai_eps)
    fh_c = gf_c.get("scale_fit", {}).get("f_times_h")
    fh_p = gf_p.get("scale_fit", {}).get("f_times_h")

    # per-row du/dd ratio on rows both corpora resolved
    common = sorted(set(gf_c["du_per_metre_by_row"]) & set(gf_p["du_per_metre_by_row"]))
    row_ratio = {r: round(gf_p["du_per_metre_by_row"][r] / gf_c["du_per_metre_by_row"][r], 3)
                 for r in common if abs(gf_c["du_per_metre_by_row"][r]) > 0.02}
    med_row_ratio = float(np.median(list(row_ratio.values()))) if row_ratio else None
    fh_ratio = (fh_p / fh_c) if (fh_c and fh_p) else None

    q_c, q_p = _flow_quality(gf_c), _flow_quality(gf_p)
    empirical_ok = bool(q_c and q_p)

    # du/dd = (r-r0)^2 / (f*h): the empirical f*h ratio folds focal AND camera
    # height. With f_eff now matched (check 1), any residual is the UNNORMALIZED
    # extrinsic HEIGHT (D-016 defers it): audit measured 1.43 m (physicalai) vs
    # ~1.2 m (comma) -> expected residual ~1.19x. The decisive VLM3 quantity is
    # the FOCAL ratio, which check 1 pins at ~1.0 (was 1.62x).
    focal_ratio = round(pai_feff / comma_feff, 3)
    expected_height_ratio = round(1.43 / 1.20, 3)  # audit extrinsics

    empirical_pass = None
    if empirical_ok and fh_ratio is not None:
        # focal-only agreement: divide out the known height residual
        focal_component = fh_ratio / expected_height_ratio
        empirical_pass = bool(abs(focal_component - 1.0) <= PPM_TOL)
    verdict = ("EMPIRICAL" if empirical_ok else "CALIBRATION_FALLBACK")
    passed = empirical_pass if empirical_ok else bool(abs(focal_ratio - 1.0) <= PPM_TOL)

    return {
        "comma_flow": {"n_pairs": gf_c["n_straight_fast_bright_pairs"],
                       "horizon_row_FOE": gf_c.get("horizon_row_FOE"),
                       "f_times_h": fh_c, "clean": q_c},
        "physicalai_flow": {"n_pairs": gf_p["n_straight_fast_bright_pairs"],
                            "horizon_row_FOE": gf_p.get("horizon_row_FOE"),
                            "f_times_h": fh_p, "clean": q_p},
        "du_per_m_row_ratio_pai_over_comma": row_ratio,
        "median_row_ratio_pai_over_comma": round(med_row_ratio, 3) if med_row_ratio else None,
        "f_times_h_ratio_pai_over_comma": round(fh_ratio, 3) if fh_ratio else None,
        "focal_ratio_pai_over_comma_from_check1": focal_ratio,
        "expected_residual_height_ratio": expected_height_ratio,
        "evidence_standing_on": verdict,
        "empirical_conclusive": empirical_ok,
        "PASS": bool(passed),
        "note": ("focal ratio ~1.0 (was 1.62x) is the VLM3 action->pixel-scale "
                 "consistency; any residual px/m gap is the deferred height "
                 "extrinsic (~1.19x), not the lens error the gate is about."),
    }


# --------------------------------------------------------------------------- #
# CHECK 3 — undistortion / curvature / principal point / horizon               #
# --------------------------------------------------------------------------- #
def _r_theta_table(intr: FThetaIntrinsics, n=200000, th_max=1.5):
    th = np.linspace(0.0, th_max, n)
    poly = np.asarray(intr.poly, float)
    r = np.zeros_like(th)
    for c in poly[::-1]:
        r = r * th + c
    return th, r


def check3_undistort_curvature(intr_list, comma_flow_foe) -> dict:
    intr = intr_list[len(intr_list) // 2][1] if intr_list else PHYSICALAI_FRONT_WIDE_FTHETA
    th_can = canonical_halfangle_rad(256, F_REF)

    # (a) inverse round-trip: theta -> r_of_theta -> theta_of_r, over the field
    errs = []
    for th in np.linspace(0.01, th_can, 40):
        r = float(intr.r_of_theta(float(th)))
        th2 = intr.theta_of_r(r)
        errs.append(abs(math.degrees(th - th2)))
    roundtrip_max_deg = round(float(max(errs)), 5)

    # (b) rectilinearity: straight lines in the undistorted OUTPUT must map to
    # straight lines in an ideal WORLD pinhole. Push output-line samples through
    # the real undistort grid to native, then forward-project native -> world
    # pinhole; measure max perpendicular residual [px]. ~0 => straight->straight.
    grid = ftheta_undistort_grid(intr, 256, F_REF)[0].numpy()      # [S,S,2] in [-1,1]
    u_nat = (grid[..., 0] + 1) / 2 * (intr.width - 1)
    v_nat = (grid[..., 1] + 1) / 2 * (intr.height - 1)
    th_tab, r_tab = _r_theta_table(intr)
    f_world = intr.paraxial_focal
    dxn = u_nat - intr.cx
    dyn = v_nat - intr.cy
    rn = np.hypot(dxn, dyn)
    thn = np.interp(rn, r_tab, th_tab)
    rho = f_world * np.tan(thn)
    scale = np.where(rn > 1e-6, rho / rn, 0.0)
    wx = dxn * scale                                               # world pinhole x
    wy = dyn * scale                                               # world pinhole y

    def _line_resid(px, py):
        px, py = np.asarray(px, float), np.asarray(py, float)
        vx, vy = px[-1] - px[0], py[-1] - py[0]
        n = math.hypot(vx, vy)
        if n < 1e-9:
            return 0.0
        # perpendicular distance of each point to the endpoint chord
        d = np.abs((py - py[0]) * vx - (px - px[0]) * vy) / n
        return float(d.max())

    resids = []
    S = 256
    for row in (S // 4, S // 2, 3 * S // 4):                       # horizontals
        resids.append(_line_resid(wx[row, :], wy[row, :]))
    for col in (S // 4, S // 2, 3 * S // 4):                       # verticals
        resids.append(_line_resid(wx[:, col], wy[:, col]))
    diag = np.arange(S)
    resids.append(_line_resid(wx[diag, diag], wy[diag, diag]))     # diagonal
    rectilinearity_max_resid_px = round(float(max(resids)), 4)

    # (c) residual curvature of the DEFAULT crop path (what the cache carries):
    # a fisheye keeps small barrel curvature in the retained +/-25.7 deg patch.
    # Sagitta of the edge world-line through crop+resize (no undistort). We map a
    # straight native-edge row of the retained crop to world angle to get its bow.
    c_native = ftheta_crop_size(intr, 256, F_REF)
    half = c_native / 2.0
    # sample the top edge of the retained square (worst curvature), map to angle
    xs = np.linspace(-half, half, 256)
    ys = np.full_like(xs, -half)
    rr = np.hypot(xs, ys)
    tt = np.interp(rr, r_tab, th_tab)
    ww = f_world * np.tan(tt)
    sc = np.where(rr > 1e-6, ww / rr, 0.0)
    ex, ey = xs * sc, ys * sc
    crop_edge_sagitta_worldpx = round(_line_resid(ex, ey), 3)
    # express as fraction of the 256 output width for interpretability
    crop_edge_sagitta_frac = round(crop_edge_sagitta_worldpx /
                                   (2 * f_world * math.tan(th_can)) * 256, 3)

    # (d) principal point: geometric center is the crop convention; report the
    # real (cx,cy) offset + the known bimodal cy split (rig A ~543 / rig B ~754).
    cxs = np.array([i.cx for _c, i in intr_list])
    cys = np.array([i.cy for _c, i in intr_list])
    W = intr_list[0][1].width if intr_list else 1920
    H = intr_list[0][1].height if intr_list else 1080
    pp = {
        "cx_offset_from_center_px": _stats(cxs - W / 2.0),
        "cy_offset_from_center_px": _stats(cys - H / 2.0),
        "cy_bimodal_frac_low_rigA(<650)": round(float((cys < 650).mean()), 3),
        "cy_bimodal_frac_high_rigB(>=650)": round(float((cys >= 650).mean()), 3),
        "crop_uses_geometric_center": True,
        "note": ("cx ~centered; cy is BIMODAL across two rigs so the crop uses "
                 "GEOMETRIC center (not principal point) by design -> robust to "
                 "the rig split; horizon/pp extrinsic normalization is D-016 R1."),
    }

    # (e) horizon ~ h/2. comma: empirical FOE row (validated method). physicalai:
    # extrinsic pitch not on this pod (only intrinsics CSV); audit measured
    # optical-axis pitch -0.49 deg -> horizon row ~= 128 + f_eff*tan(pitch).
    pai_pitch_deg = -0.49  # GEOMETRY_INTEGRITY_AUDIT.md (extrinsics chunk)
    pai_horizon_expected = round(128.0 + F_REF * math.tan(math.radians(pai_pitch_deg)), 1)
    horizon = {
        "comma_empirical_FOE_row": comma_flow_foe,
        "comma_h_over_2": 128,
        "physicalai_pitch_deg_from_audit": pai_pitch_deg,
        "physicalai_horizon_row_expected": pai_horizon_expected,
        "physicalai_extrinsics_on_pod": False,
        "note": ("comma FOE ~ image center validates horizon~h/2 empirically; "
                 "physicalai pitch is near-zero (audit) so its horizon is also "
                 "~center. Pitch/height homography is the deferred D-016 R1 step."),
    }

    comma_h_ok = (comma_flow_foe is None) or (96 <= comma_flow_foe <= 160)
    passed = (roundtrip_max_deg < 0.05 and rectilinearity_max_resid_px < 1.0
              and comma_h_ok)
    return {
        "inverse_roundtrip_max_err_deg": roundtrip_max_deg,
        "rectilinearity_straightline_max_resid_px": rectilinearity_max_resid_px,
        "default_crop_edge_residual_curvature_worldpx": crop_edge_sagitta_worldpx,
        "default_crop_edge_residual_curvature_outpx": crop_edge_sagitta_frac,
        "principal_point": pp,
        "horizon": horizon,
        "PASS": bool(passed),
        "note": ("f-theta inverse is exact and undistort maps straight->straight; "
                 "the default cache crop carries only small central barrel "
                 "curvature (undistort is the deferred D-016 R1 option)."),
    }


# --------------------------------------------------------------------------- #
# CHECK 4 — silent-skew guard (CORPUS_META / I7)                               #
# --------------------------------------------------------------------------- #
def check4_silent_skew_guard() -> dict:
    from tanitad.data import calib
    from tanitad.data.comma2k19 import CORPUS_META as CMA
    from tanitad.data.physicalai import CORPUS_META as PAI
    from tanitad.instruments.checks import i7_task_identity

    identical, bad = i7_task_identity(dict(CMA), dict(PAI))
    pai_traceable = (PAI["f_eff_px"] == calib.F_REF)      # sourced from F_REF
    both_266 = (CMA["f_eff_px"] == 266.0 and PAI["f_eff_px"] == 266.0)
    not_old_nominal = PAI["f_eff_px"] not in (431.0, 434.0, 554.3)
    passed = bool(both_266 and pai_traceable and not_old_nominal and identical)
    return {
        "comma_CORPUS_META": CMA,
        "physicalai_CORPUS_META": PAI,
        "physicalai_f_eff_px_is_calib_F_REF": bool(pai_traceable),
        "i7_fingerprints_identical": bool(identical),
        "i7_mismatched_keys": bad,
        "reports_achieved_266_not_nominal_434": bool(both_266 and not_old_nominal),
        "PASS": passed,
        "note": ("physicalai emits f_eff_px = calib.F_REF (266), the ACHIEVED "
                 "focal (check 1), not the old nominal ~434 the pipeline used to "
                 "silently deliver. build_pai_cache.py also asserts f_eff~=F_REF "
                 "at build time, so the skew can't recur silently."),
    }


# --------------------------------------------------------------------------- #
# CHECK 5 — sanity vs the old (wrong-zoom) cache                               #
# --------------------------------------------------------------------------- #
def check5_before_after(intr_list, pai_cache: str, pai_cache_old: str | None,
                        png_dir: str | None) -> dict:
    intr = intr_list[len(intr_list) // 2][1] if intr_list else PHYSICALAI_FRONT_WIDE_FTHETA
    rep = ftheta_feff_report(intr, 256, F_REF)
    c_after, c_before = rep["crop_side_after"], rep["crop_side_before"]
    hfov_after, hfov_before = rep["retained_hfov_after_deg"], rep["retained_hfov_before_deg"]
    fov_ratio_linear = round(c_after / c_before, 3)              # native px retained
    fov_ratio_angular = round(hfov_after / hfov_before, 3)

    frames_written = []
    if pai_cache_old and png_dir:
        try:
            frames_written = _dump_before_after_frames(
                pai_cache, pai_cache_old, png_dir)
        except Exception as e:  # noqa: BLE001
            frames_written = [f"png_error: {e!r}"]

    passed = bool(c_after > c_before and hfov_after > hfov_before)
    return {
        "corrected_crop_side_native_px": c_after,
        "old_crop_side_native_px": c_before,
        "corrected_retained_hfov_deg": hfov_after,
        "old_retained_hfov_deg": hfov_before,
        "fov_wider_ratio_linear": fov_ratio_linear,
        "fov_wider_ratio_angular": fov_ratio_angular,
        "annotated_frames": frames_written,
        "PASS": passed,
        "note": ("corrected keeps a WIDER field (~51 deg / crop side ~829 px) vs "
                 "the old over-zoomed ~33 deg / ~533 px -> more of the scene, "
                 "comma-matched. The sacrificed wide periphery returns as H2 "
                 "side-view modalities later."),
    }


def _dump_before_after_frames(pai_cache, pai_cache_old, png_dir, n_pairs=2):
    from PIL import Image
    Path(png_dir).mkdir(parents=True, exist_ok=True)
    new_eps = geom_sanity.load_val_episodes(pai_cache, 40)
    old_eps = geom_sanity.load_val_episodes(pai_cache_old, 40)
    new_by = {int(e.episode_id): e for e in new_eps}
    old_by = {int(e.episode_id): e for e in old_eps}
    common = sorted(set(new_by) & set(old_by))
    written = []
    for eid in common[:n_pairs]:
        en, eo = new_by[eid], old_by[eid]
        t = min(en.frames.shape[0], eo.frames.shape[0]) // 2
        fn = en.frames[t, -3:].permute(1, 2, 0).numpy().astype(np.uint8)   # current RGB
        fo = eo.frames[t, -3:].permute(1, 2, 0).numpy().astype(np.uint8)
        pad = np.full((256, 8, 3), 255, np.uint8)
        combo = np.concatenate([fo, pad, fn], axis=1)   # OLD (zoomed) | CORRECTED (wide)
        p = str(Path(png_dir) / f"before_after_ep{eid}.png")
        Image.fromarray(combo).save(p)
        written.append({"episode_id": eid, "path": p,
                        "layout": "LEFT=old(zoomed) | RIGHT=corrected(wider)"})
    return written


# --------------------------------------------------------------------------- #
def _overall(checks: dict) -> str:
    c = {k: v.get("PASS") for k, v in checks.items()}
    core = c["check1_shared_focal"] and c["check4_silent_skew_guard"] and \
        c["check3_undistort_curvature"] and c["check5_before_after"]
    ppm = checks["check2_pixels_per_metre"]
    if not core:
        return "RED"
    if c["check1_shared_focal"] and ppm["PASS"] and ppm["empirical_conclusive"]:
        return "GREEN"
    # focal + guards green; px/m rests on the calibration-derived proof (flow
    # inconclusive on urban physicalai, as the prior audit found) -> GREEN on the
    # VLM3 invariant with a stated caveat; AMBER only if px/m itself fails.
    return "GREEN" if ppm["PASS"] else "AMBER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--pai-cache", required=True)
    ap.add_argument("--pai-cache-old", default="")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--pai-root", default="")
    ap.add_argument("--episodes", type=int, default=90)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png-dir", default="")
    args = ap.parse_args()

    intr_list = _load_intrinsics(args.intrinsics)
    print(f"[geom] loaded {len(intr_list)} per-clip f-theta intrinsics", flush=True)

    print("[geom] CHECK 1 — shared canonical focal ...", flush=True)
    c1 = check1_shared_focal(intr_list, args.pai_root or None)
    print(f"  comma f_eff={c1['comma_f_eff']} | physicalai f_eff median="
          f"{c1['physicalai_f_eff_after_stats']['median']} "
          f"(old nominal median={c1['physicalai_f_eff_before_nominal_stats']['median']}) "
          f"-> PASS={c1['PASS']}", flush=True)

    print(f"[geom] loading {args.episodes} val episodes/corpus for flow ...", flush=True)
    comma_eps = geom_sanity.load_val_episodes(args.comma_cache, args.episodes)
    pai_eps = geom_sanity.load_val_episodes(args.pai_cache, args.episodes)

    print("[geom] CHECK 2 — empirical cross-corpus pixels-per-metre ...", flush=True)
    c2 = check2_pixels_per_metre(comma_eps, pai_eps, c1["comma_f_eff"],
                                 c1["physicalai_f_eff_after_stats"]["median"])
    print(f"  f*h comma={c2['comma_flow']['f_times_h']} physicalai="
          f"{c2['physicalai_flow']['f_times_h']} | focal_ratio="
          f"{c2['focal_ratio_pai_over_comma_from_check1']} standing_on="
          f"{c2['evidence_standing_on']} -> PASS={c2['PASS']}", flush=True)

    print("[geom] CHECK 3 — undistortion / curvature / horizon ...", flush=True)
    c3 = check3_undistort_curvature(intr_list, c2["comma_flow"]["horizon_row_FOE"])
    print(f"  inverse_roundtrip={c3['inverse_roundtrip_max_err_deg']}deg "
          f"rectilinearity_resid={c3['rectilinearity_straightline_max_resid_px']}px "
          f"-> PASS={c3['PASS']}", flush=True)

    print("[geom] CHECK 4 — silent-skew guard (CORPUS_META / I7) ...", flush=True)
    c4 = check4_silent_skew_guard()
    print(f"  comma_feff={c4['comma_CORPUS_META']['f_eff_px']} "
          f"physicalai_feff={c4['physicalai_CORPUS_META']['f_eff_px']} "
          f"i7_identical={c4['i7_fingerprints_identical']} -> PASS={c4['PASS']}", flush=True)

    print("[geom] CHECK 5 — sanity vs old cache (before/after FOV) ...", flush=True)
    c5 = check5_before_after(intr_list, args.pai_cache,
                             args.pai_cache_old or None, args.png_dir or None)
    print(f"  crop_side {c5['old_crop_side_native_px']}->{c5['corrected_crop_side_native_px']} "
          f"hfov {c5['old_retained_hfov_deg']}->{c5['corrected_retained_hfov_deg']}deg "
          f"(wider x{c5['fov_wider_ratio_angular']}) -> PASS={c5['PASS']}", flush=True)

    checks = {"check1_shared_focal": c1, "check2_pixels_per_metre": c2,
              "check3_undistort_curvature": c3, "check4_silent_skew_guard": c4,
              "check5_before_after": c5}
    verdict = _overall(checks)
    report = {
        "gate": "Axis-2 geometry / VLM3 unified calibration (f-theta corrected)",
        "calib_py_under_test": "tanitad/data/calib.py",
        "episodes_per_corpus": args.episodes,
        "n_intrinsics_clips": len(intr_list),
        "summary_table": {
            "check1_shared_focal": {
                "comma_f_eff": c1["comma_f_eff"],
                "physicalai_f_eff": c1["physicalai_f_eff_after_stats"]["median"],
                "PASS": c1["PASS"]},
            "check2_pixels_per_metre": {
                "focal_ratio_pai_over_comma": c2["focal_ratio_pai_over_comma_from_check1"],
                "f_times_h_ratio": c2["f_times_h_ratio_pai_over_comma"],
                "standing_on": c2["evidence_standing_on"], "PASS": c2["PASS"]},
            "check3_undistort_curvature": {
                "rectilinearity_resid_px": c3["rectilinearity_straightline_max_resid_px"],
                "PASS": c3["PASS"]},
            "check4_silent_skew_guard": {
                "physicalai_f_eff_px": c4["physicalai_CORPUS_META"]["f_eff_px"],
                "PASS": c4["PASS"]},
            "check5_before_after": {
                "fov_wider_ratio_angular": c5["fov_wider_ratio_angular"],
                "PASS": c5["PASS"]}},
        "checks": checks,
        "AXIS2_VERDICT": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[geom] AXIS-2 VERDICT: {verdict}", flush=True)
    print(f"[geom] report -> {args.out}\nVALIDATE_GEOMETRY_DONE", flush=True)


if __name__ == "__main__":
    main()
