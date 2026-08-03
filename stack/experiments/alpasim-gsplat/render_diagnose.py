#!/usr/bin/env python3
"""Localise and EXPLAIN the two named render artifacts, from source rather than by masking.

The FINDINGS write-up names two visible defects on scene 00040136 frame 0:
  * a **black band top-left**   — "FOV/coverage"
  * a **magenta smear bottom-left** — "a bad gaussian cluster or a missing layer"

Both were hypotheses. This script decides them with measurements a mask cannot fake:

BLACK BAND
  If it is the f-theta field of view, the dark pixels must be exactly the pixels whose
  ray angle exceeds the camera's own `max_angle` — a property of the CALIBRATION, known
  without rendering anything. So the test is: does the set {alpha < eps} coincide with
  {theta(r) >= max_angle}? Coincidence is reported as IoU, not eyeballed.

MAGENTA SMEAR
  Three candidate causes, separated by ablation rather than by argument:
    1. a layer — render `background` only and `road` only and see which one carries it;
    2. a few over-sized splats — sweep a max-scale cull and watch the region's colour;
    3. genuinely magenta content in the reference (i.e. not our bug at all) — compare
       the same region of the reference frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAM = "camera_front_wide_120fov"


# --------------------------------------------------------------------------------- #
def magenta_mask(img_u8, delta=18, min_rb=40):
    """Magenta = R and B both clearly above G. `delta` in 8-bit levels."""
    r, g, b = (img_u8[..., i].astype(np.int16) for i in range(3))
    return ((r - g) > delta) & ((b - g) > delta) & (r > min_rb) & (b > min_rb)


def boxes_of(mask, min_px=200):
    """Connected components of a boolean mask -> [{bbox, n_px, centroid}]."""
    import cv2
    n, lab, stats, cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        out.append({"bbox_xywh": [int(stats[i, cv2.CC_STAT_LEFT]),
                                  int(stats[i, cv2.CC_STAT_TOP]),
                                  int(stats[i, cv2.CC_STAT_WIDTH]),
                                  int(stats[i, cv2.CC_STAT_HEIGHT])],
                    "n_px": int(stats[i, cv2.CC_STAT_AREA]),
                    "centroid_xy": [round(float(cent[i][0]), 1),
                                    round(float(cent[i][1]), 1)]})
    out.sort(key=lambda d: -d["n_px"])
    return out[:8]


def theta_map(cam):
    """Per-pixel ray angle from the camera's OWN backward polynomial, and the mask of
    pixels beyond `max_angle` — the FOV boundary, known from calibration alone."""
    H, W = int(cam.height), int(cam.width)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.hypot((xs + 0.5) - (cam.cx + 0.5), (ys + 0.5) - (cam.cy + 0.5))
    bw = np.array(cam.pixeldist_to_angle_poly, np.float64)
    th = np.polyval(bw[::-1], r)
    return th, th >= cam.max_angle


def iou(a, b):
    u = float((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


def region_stats(name, mask, img, ref, alpha, depth=None):
    if not mask.any():
        return {"region": name, "n_px": 0}
    d = {"region": name, "n_px": int(mask.sum()),
         "frac_of_frame": round(float(mask.mean()), 5),
         "render_rgb_mean": [round(float(img[..., i][mask].mean()), 1) for i in range(3)],
         "ref_rgb_mean": [round(float(ref[..., i][mask].mean()), 1) for i in range(3)],
         "alpha_mean": round(float(alpha[mask].mean()), 4),
         "alpha_p95": round(float(np.percentile(alpha[mask], 95)), 4),
         "mae": round(float(np.abs(img[mask].astype(np.float32)
                                   - ref[mask].astype(np.float32)).mean() / 255.0), 4)}
    if depth is not None:
        dm = depth[mask]
        dm = dm[np.isfinite(dm)]
        if dm.size:
            d["depth_median_m"] = round(float(np.median(dm)), 2)
            d["depth_p05_m"] = round(float(np.percentile(dm, 5)), 2)
    return d


# --------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--loader-dir", default=None)
    a = ap.parse_args()

    from gsplat_renderer import NuRecGsplatRenderer, grad_ncc, read_ref_frame

    scene = Path(a.scene_dir).expanduser()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"run_dir": str(out), "scene_dir": str(scene), "frame": a.frame}

    r = NuRecGsplatRenderer(scene, layers=("background", "road"),
                            loader_dir=a.loader_dir, verbose=True)
    c2n = r.gt_cam_to_nre(a.frame)
    img, alpha, ms = r.render(c2n)
    # depth comes from a SEPARATE pass: render_mode="RGB+ED" aborts the process on the
    # f-theta + with_eval3d kernel (`channels == 3` assert) — see render_depth().
    depth = r.render_depth(c2n)
    ref = read_ref_frame(scene / f"{CAM}.mp4", a.frame, (r.width, r.height))
    rep["render_ms"] = round(ms, 1)

    # ---- 1. BLACK BAND: is it the f-theta FOV? ---------------------------------
    th, beyond = theta_map(r.cam)
    empty = alpha < 0.01
    dark = img.max(-1) < 12
    rep["black_band"] = {
        "max_angle_rad": float(r.cam.max_angle),
        "max_angle_deg": round(float(np.degrees(r.cam.max_angle)), 2),
        "frac_pixels_beyond_max_angle": round(float(beyond.mean()), 5),
        "frac_alpha_lt_0.01": round(float(empty.mean()), 5),
        "frac_render_near_black": round(float(dark.mean()), 5),
        "IoU_empty_vs_beyond_fov": round(iou(empty, beyond), 4),
        "IoU_dark_vs_beyond_fov": round(iou(dark, beyond), 4),
        "frac_of_empty_that_is_beyond_fov": (
            round(float((empty & beyond).sum() / max(empty.sum(), 1)), 4)),
        "frac_of_beyond_fov_that_is_empty": (
            round(float((empty & beyond).sum() / max(beyond.sum(), 1)), 4)),
        "empty_boxes": boxes_of(empty),
        "beyond_fov_boxes": boxes_of(beyond),
        "dark_boxes": boxes_of(dark),
    }
    rep["black_band"]["region_empty"] = region_stats("alpha<0.01", empty, img, ref,
                                                     alpha, depth)

    # ---- 2. MAGENTA SMEAR ------------------------------------------------------
    mag = magenta_mask(img)
    mag_ref = magenta_mask(ref)
    rep["magenta"] = {
        "frac_render": round(float(mag.mean()), 5),
        "frac_reference": round(float(mag_ref.mean()), 5),
        "boxes_render": boxes_of(mag),
        "boxes_reference": boxes_of(mag_ref),
        "region": region_stats("magenta", mag, img, ref, alpha, depth),
    }

    # 2a. which LAYER carries it
    per_layer = {}
    for L in ("background", "road"):
        rl = NuRecGsplatRenderer(scene, layers=(L,), loader_dir=a.loader_dir,
                                 verbose=False)
        il, al, _ = rl.render(c2n)
        ml = magenta_mask(il)
        per_layer[L] = {"frac_magenta": round(float(ml.mean()), 5),
                        "magenta_overlap_with_full_render":
                            round(iou(ml, mag), 4),
                        "boxes": boxes_of(ml),
                        "rgb_mean_in_full_magenta_region":
                            ([round(float(il[..., i][mag].mean()), 1) for i in range(3)]
                             if mag.any() else None),
                        "alpha_mean_in_full_magenta_region":
                            (round(float(al[mag].mean()), 4) if mag.any() else None)}
        del rl
        import torch
        torch.cuda.empty_cache()
    rep["magenta"]["per_layer"] = per_layer

    # 2b. is it a few OVER-SIZED splats? cull by max scale and watch the region
    sweep = []
    import torch
    smax = r.scales.max(dim=1).values
    base_ncc = grad_ncc(img, ref)
    for q in (1.0, 0.9999, 0.999, 0.99, 0.95):
        if q >= 1.0:
            thr = float("inf"); keep = torch.ones_like(smax, dtype=torch.bool)
        else:
            thr = float(torch.quantile(smax[::37].float(), q))
            keep = smax <= thr
        sub = _render_subset(r, c2n, keep)
        mg = magenta_mask(sub)
        sweep.append({"scale_quantile_kept": q, "scale_thresh_m": (None if thr == float("inf")
                                                                  else round(thr, 3)),
                      "n_culled": int((~keep).sum()),
                      "frac_magenta": round(float(mg.mean()), 5),
                      "rgb_mean_in_magenta_region":
                          ([round(float(sub[..., i][mag].mean()), 1) for i in range(3)]
                           if mag.any() else None),
                      "grad_ncc": round(grad_ncc(sub, ref), 4)})
    rep["magenta"]["scale_cull_sweep"] = sweep
    rep["magenta"]["grad_ncc_baseline"] = round(base_ncc, 4)

    # ---- 3. panel PNG ----------------------------------------------------------
    _panel(out / f"diagnose_f{a.frame}.png", img, ref, alpha, beyond, mag, depth)
    (out / "diagnose.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def _render_subset(r, c2n, keep):
    """Re-rasterise the static scene with a boolean subset of its gaussians."""
    import torch
    from gsplat import rasterization
    K = torch.from_numpy(r._K_for(r.cam).copy()[None]).to(r.device)
    vm = torch.from_numpy(np.linalg.inv(c2n)[None].astype(np.float32)).to(r.device)
    colors, _alphas, _ = rasterization(
        means=r.means[keep], quats=r.quats[keep], scales=r.scales[keep],
        opacities=r.opac[keep], colors=r.sh[keep], viewmats=vm, Ks=K,
        width=r.width, height=r.height, sh_degree=3, packed=False,
        with_ut=True, with_eval3d=True, camera_model="ftheta",
        ftheta_coeffs=r._ftheta_coeffs_for(r.cam), near_plane=0.05, far_plane=2000.0)
    return (colors[0][..., :3].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def _panel(path, img, ref, alpha, beyond, mag, depth):
    import cv2
    h, w = img.shape[:2]
    acol = cv2.applyColorMap((np.clip(alpha, 0, 1) * 255).astype(np.uint8),
                             cv2.COLORMAP_VIRIDIS)[:, :, ::-1]
    over = img.copy()
    over[beyond] = (0, 255, 0)
    over[mag] = (255, 0, 0)
    if depth is not None:
        dd = np.nan_to_num(depth, nan=0.0, posinf=0.0)
        dn = np.clip(dd / max(np.percentile(dd[dd > 0], 95) if (dd > 0).any() else 1, 1e-6),
                     0, 1)
        dcol = cv2.applyColorMap((dn * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)[:, :, ::-1]
    else:
        dcol = np.zeros_like(img)
    pad = np.full((h, 8, 3), 255, np.uint8)
    row = np.concatenate([img, pad, ref, pad, acol, pad, dcol, pad, over], 1)
    bar = np.zeros((42, row.shape[1], 3), np.uint8)
    cv2.putText(bar, "RENDER | REFERENCE | ALPHA | DEPTH | overlay: GREEN=beyond ftheta "
                     "max_angle  RED=magenta", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), np.concatenate([bar, row], 0)[:, :, ::-1])


if __name__ == "__main__":
    main()
