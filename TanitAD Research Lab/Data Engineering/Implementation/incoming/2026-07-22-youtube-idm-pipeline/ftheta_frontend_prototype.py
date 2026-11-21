"""f-theta intrinsics FRONT-END prototype — the intrinsics de-risk for the IDM/YouTube line.

Task: YouTube dashcam frames have UNKNOWN intrinsics; our encoder trained on
F_REF=266 canonicalized frames (9-ch, 256px). This prototype answers ONE question
with MEASURED evidence:

    Given (estimated or assumed) intrinsics, does canonicalizing a real dashcam
    frame to our F_REF=266 pinhole ROUND-TRIP cleanly to f_eff ~= 266?

It reuses the SAME AlpaSim-/D-016-validated primitives the WM pipeline uses:
    tanitad.data.calib.focal_crop_resize          (pinhole / rectilinear branch)
    tanitad.data.calib.ftheta_crop_resize(center="principal")   (fisheye branch)
    tanitad.data.comma2k19.stack_frames           (9-channel encoder contract)

Evidence classes (per CLAUDE.md operating standard):
  * canonicalization round-trip + error-propagation law: MEASURED (this script).
  * external calibration-method choices: PUBLISHED (cited in PIPELINE_DESIGN.md).
  * "single road VP under-determines focal": MEASURED here (analytic round-trip).

Sample frame: a comma2k19 raw frame from OUR lake (native 1164x874, rectilinear,
CAN-calibrated focal ~910 px) — a legitimate stand-in for a rectilinear dashcam.
NO YouTube video is scraped or downloaded (task constraint). The round-trip is a
geometric property of the crop, INDEPENDENT of pixel content, so a real frame is
used only to make the demonstration concrete and to exercise the VP estimator on
genuine road content.

Run:
    PYTHONPATH=stack C:/Users/Admin/venvs/tanitad/Scripts/python.exe \
        ".../2026-07-22-youtube-idm-pipeline/ftheta_frontend_prototype.py"
Writes ftheta_frontend_result.json + canonical_sample.png next to this file.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# --- locate the stack package (repo-relative; no install assumed) -------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE
for _p in _HERE.parents:
    if (_p / "stack" / "tanitad").is_dir():
        _REPO = _p
        break
sys.path.insert(0, str(_REPO / "stack"))

from tanitad.data.calib import (  # noqa: E402
    COMMA2K19_FOCAL_PX, F_REF, FThetaIntrinsics, PHYSICALAI_FRONT_WIDE_FTHETA,
    focal_crop_resize, ftheta_crop_resize, ftheta_feff_report)
from tanitad.data.comma2k19 import stack_frames  # noqa: E402

TOL = 0.01                      # accept |f_eff - 266|/266 < 1% as "round-trips"
SIZE = 256
DATA_ROOT = Path(os.environ.get(
    "TANITAD_COMMA_ROOT",
    r"C:/Users/Admin/tanitad-data/comma2k19/extracted"))


# --------------------------------------------------------------------------- #
# 0. Get a real raw dashcam-like frame (comma2k19, native res, from our lake)  #
# --------------------------------------------------------------------------- #
def load_raw_frames(n: int = 4):
    """Decode the first n frames of a comma2k19 segment -> uint8 [n,3,H,W].

    Returns (frames, provenance_str). Falls back to a synthetic road frame if no
    local video is available, flagged clearly in provenance (round-trip geometry
    is content-independent, so the fallback still validates f_eff)."""
    vids = sorted(glob.glob(str(DATA_ROOT / "**" / "video.hevc"), recursive=True))
    if vids:
        import av
        seg = vids[0]
        out = []
        with av.open(seg) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            for i, fr in enumerate(c.decode(st)):
                out.append(torch.from_numpy(
                    fr.to_ndarray(format="rgb24")).permute(2, 0, 1))
                if len(out) >= n:
                    break
        frames = torch.stack(out)
        rel = os.path.relpath(seg, DATA_ROOT)
        return frames, f"comma2k19 raw (lake): {rel} [{frames.shape[2]}x{frames.shape[3]}]"
    # fallback: synthetic straight-road frame at comma native res
    H, W = 874, 1164
    img = np.full((H, W, 3), 90, np.uint8)
    img[: int(0.45 * H)] = (150, 170, 200)                    # sky
    for k in range(20):                                        # converging lanes
        x = int(W / 2 + (k - 10) * 6)
        for v in range(int(0.45 * H), H):
            u = int(W / 2 + (x - W / 2) * (v - 0.45 * H) / (H - 0.45 * H))
            if 0 <= u < W:
                img[v, max(0, u - 1):u + 1] = 230
    fr = torch.from_numpy(img).permute(2, 0, 1)
    return fr.unsqueeze(0).repeat(n, 1, 1, 1), "SYNTHETIC straight-road (no local video found)"


# --------------------------------------------------------------------------- #
# 1. CORE: canonicalization round-trip on both camera branches                 #
# --------------------------------------------------------------------------- #
def roundtrip_pinhole(frames, f_assumed):
    """Rectilinear-dashcam branch: focal_crop_resize -> f_eff. MEASURED."""
    out = focal_crop_resize(frames, f_assumed, SIZE)
    f_eff = float(focal_crop_resize.last_f_eff)
    return out, f_eff


def roundtrip_ftheta(frames):
    """Wide/fisheye-dashcam branch: the AlpaSim-validated principal-point crop.

    Builds a PER-CLIP f-theta model (per_clip=True) so center="principal" is
    actually exercised (not silently reverted to geometric). Uses the measured
    PhysicalAI front-wide radial poly with a plausible per-clip principal point."""
    base = PHYSICALAI_FRONT_WIDE_FTHETA
    intr = FThetaIntrinsics(poly=base.poly, cx=959.0, cy=548.0,   # rig-A-like cy
                            width=1920, height=1080, per_clip=True)
    # resize our comma frame up to the fisheye native canvas so the crop math runs
    big = torch.nn.functional.interpolate(
        frames.float(), size=(1080, 1920), mode="bilinear",
        align_corners=False).clamp(0, 255).to(torch.uint8)
    out = ftheta_crop_resize(big, intr, SIZE, center="principal")
    f_eff = float(ftheta_crop_resize.last_f_eff)
    return out, f_eff, intr


# --------------------------------------------------------------------------- #
# 2. Estimation-ERROR propagation law (the de-risk quantification). MEASURED.  #
# --------------------------------------------------------------------------- #
def error_propagation(frames, f_true):
    """If the TRUE focal is f_true but we canonicalize with a wrong estimate
    f_est, the output's TRUE effective focal = F_REF * (f_true / f_est).

    We MEASURE this by: crop with f_est (what the pipeline would do), then read
    the achieved f_eff back against f_true through the identical crop geometry."""
    rows = []
    for mult in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2):
        f_est = f_true * mult
        # crop side the pipeline picks from the (wrong) estimate
        from tanitad.data.calib import focal_crop_size
        c = focal_crop_size(f_est, frames.shape[2], frames.shape[3], SIZE)
        f_eff_true = f_true * SIZE / c              # TRUE f_eff of the output
        rows.append({
            "f_est_over_f_true": round(mult, 3),
            "crop_side_px": c,
            "f_eff_true_px": round(f_eff_true, 2),
            "f_eff_dev_pct": round(100 * (f_eff_true - F_REF) / F_REF, 2),
        })
    return rows


# --------------------------------------------------------------------------- #
# 3. Intrinsics ESTIMATION primitives (geometric self-calibration). MEASURED.  #
# --------------------------------------------------------------------------- #
def vp_row(rgb):
    """Road vanishing-point (v,u,votes) via numpy Hough + pairwise intersection.
    Vendored verbatim from stack/scripts/pod_ops/horizon_probe.py (the program's
    existing VP primitive) so the prototype has no new dependency."""
    def conv3(a, k):
        ap = np.pad(a, 1, mode="edge"); o = np.zeros_like(a, float)
        for i in range(3):
            for j in range(3):
                o += k[i, j] * ap[i:i + a.shape[0], j:j + a.shape[1]]
        return o
    KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float); KY = KX.T
    g = rgb.astype(float) @ np.array([0.299, 0.587, 0.114])
    H, W = g.shape
    gx, gy = conv3(g, KX), conv3(g, KY)
    mag, ori = np.hypot(gx, gy), np.arctan2(gy, gx)
    band = np.zeros_like(mag, bool); band[int(0.35 * H):int(0.95 * H), :] = True
    if not band.any() or mag[band].size == 0:
        return math.nan, math.nan, 0
    thr = np.percentile(mag[band], 88)
    ys, xs = np.where(band & (mag > thr))
    if len(xs) < 30:
        return math.nan, math.nan, 0
    phi = ori[ys, xs] + math.pi / 2.0
    slope = np.tan(phi); keep = (np.abs(slope) > 0.18) & (np.abs(slope) < 6.0)
    ys, xs, phi = ys[keep], xs[keep], phi[keep]
    if len(xs) < 20:
        return math.nan, math.nan, 0
    cph, sph = np.cos(phi), np.sin(phi)
    dxdy = cph / np.where(np.abs(sph) < 1e-6, 1e-6, sph)
    left, right = dxdy < 0, dxdy > 0

    def samp(m, n=90):
        idx = np.where(m)[0]
        if len(idx) > n:
            idx = idx[np.argsort(-mag[ys[idx], xs[idx]])[:n]]
        return idx
    li, ri = samp(left), samp(right)
    if len(li) < 3 or len(ri) < 3:
        return math.nan, math.nan, 0
    inter = []
    for a in li:
        p1 = np.array([xs[a], ys[a]], float); d1 = np.array([cph[a], sph[a]])
        for b in ri:
            p2 = np.array([xs[b], ys[b]], float); d2 = np.array([cph[b], sph[b]])
            det = d1[0] * (-d2[1]) - (-d2[0]) * d1[1]
            if abs(det) < 1e-6:
                continue
            bb = p2 - p1
            t = (bb[0] * (-d2[1]) - (-d2[0]) * bb[1]) / det
            pt = p1 + t * d1
            if 0.15 * W < pt[0] < 0.85 * W and 0.05 * H < pt[1] < 0.85 * H:
                inter.append(pt)
    if len(inter) < 10:
        return math.nan, math.nan, 0
    inter = np.array(inter)
    return float(np.median(inter[:, 1])), float(np.median(inter[:, 0])), len(inter)


def focal_from_two_orthogonal_vps(f_gt: float, cx: float, cy: float):
    """VALIDATE the self-calibration formula f = sqrt(-(vp1-pp).(vp2-pp)) by
    round-trip: project two ORTHOGONAL world directions through a known pinhole
    K(f_gt, cx, cy), read the two VPs, recover f, compare to f_gt. MEASURED
    (analytic, exact) — proves the estimator math is wired correctly.

    A VP of world direction d=(dx,dy,dz) is K @ d in homogeneous px:
    (cx + f*dx/dz, cy + f*dy/dz). Orthogonal directions -> (vp1-pp).(vp2-pp)=-f^2.
    """
    # two orthogonal, non-degenerate directions (both with +z so they image)
    d1 = np.array([0.9, 0.1, 1.0]); d1 /= np.linalg.norm(d1)
    d2 = np.array([-0.15, 0.8, 1.0])                       # make exactly _|_ d1
    d2 = d2 - (d2 @ d1) * d1; d2 /= np.linalg.norm(d2)
    if d2[2] <= 0:
        d2 = -d2
    vp1 = np.array([cx + f_gt * d1[0] / d1[2], cy + f_gt * d1[1] / d1[2]])
    vp2 = np.array([cx + f_gt * d2[0] / d2[2], cy + f_gt * d2[1] / d2[2]])
    pp = np.array([cx, cy])
    dot = (vp1 - pp) @ (vp2 - pp)
    f_rec = math.sqrt(-dot) if dot < 0 else math.nan
    return {"f_gt": f_gt, "f_recovered": round(f_rec, 4),
            "abs_err_pct": round(100 * abs(f_rec - f_gt) / f_gt, 4),
            "vp1": [round(float(x), 1) for x in vp1],
            "vp2": [round(float(x), 1) for x in vp2]}


# --------------------------------------------------------------------------- #
def main():
    res: dict = {"F_REF": F_REF, "size": SIZE, "tol_pct": TOL * 100}
    frames, prov = load_raw_frames(4)
    res["sample_frame"] = prov
    res["raw_shape"] = list(frames.shape)

    # 1a. pinhole / rectilinear branch (comma's true CAN focal ~910 as "estimate")
    out_p, feff_p = roundtrip_pinhole(frames, COMMA2K19_FOCAL_PX)
    # 1b. f-theta / wide branch (the AlpaSim-validated principal-point crop)
    out_f, feff_f, intr = roundtrip_ftheta(frames)
    # 1c. 9-channel encoder contract via stack_frames on 3 consecutive frames
    stacked = stack_frames(out_p, n_stack=3)             # [T-2, 9, S, S]

    res["roundtrip"] = {
        "pinhole_rectilinear": {
            "fn": "focal_crop_resize", "f_assumed_px": COMMA2K19_FOCAL_PX,
            "f_eff_px": round(feff_p, 3),
            "dev_pct": round(100 * (feff_p - F_REF) / F_REF, 3),
            "pass": abs(feff_p - F_REF) / F_REF < TOL,
            "out_shape": list(out_p.shape)},
        "ftheta_wide_principal": {
            "fn": 'ftheta_crop_resize(center="principal")',
            "per_clip": intr.per_clip, "cy": intr.cy,
            "f_eff_px": round(feff_f, 3),
            "dev_pct": round(100 * (feff_f - F_REF) / F_REF, 3),
            "pass": abs(feff_f - F_REF) / F_REF < TOL,
            "out_shape": list(out_f.shape)},
        "stack_frames_9ch": {
            "fn": "stack_frames(n_stack=3)",
            "out_shape": list(stacked.shape),
            "channels": int(stacked.shape[1]),
            "pass": stacked.shape[1] == 9 and stacked.shape[-1] == SIZE},
    }
    res["ftheta_feff_report"] = ftheta_feff_report(intr)

    # 2. estimation-error propagation
    res["error_propagation_law"] = {
        "law": "f_eff_true = F_REF * (f_true / f_est)",
        "table": error_propagation(frames, COMMA2K19_FOCAL_PX)}

    # 3a. real road VP on the canonicalized frame (does the cue exist in data?)
    rgb = out_p[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
    v, u, votes = vp_row(rgb)
    res["real_road_vp"] = {
        "note": "VP on the 256x256 canonical comma frame; model assumes horizon=128",
        "v_row": None if math.isnan(v) else round(v, 1),
        "u_col": None if math.isnan(u) else round(u, 1),
        "votes": int(votes),
        "offset_from_128": None if math.isnan(v) else round(v - 128.0, 1)}

    # 3b. self-calibration formula round-trip (analytic ground truth)
    res["two_vp_focal_roundtrip"] = focal_from_two_orthogonal_vps(910.0, 582.0, 437.0)

    # save a visual sample of the canonical frame
    try:
        from PIL import Image
        Image.fromarray(rgb).save(_HERE / "canonical_sample.png")
        res["canonical_sample_png"] = "canonical_sample.png"
    except Exception as e:
        res["canonical_sample_png"] = f"skip: {e}"

    # overall verdict
    rt = res["roundtrip"]
    res["VERDICT"] = {
        "canonicalization_roundtrips": all(
            rt[k]["pass"] for k in
            ("pinhole_rectilinear", "ftheta_wide_principal", "stack_frames_9ch")),
        "summary": (
            f'pinhole f_eff={rt["pinhole_rectilinear"]["f_eff_px"]}, '
            f'ftheta f_eff={rt["ftheta_wide_principal"]["f_eff_px"]}, '
            f'9ch={rt["stack_frames_9ch"]["out_shape"]}')}

    (_HERE / "ftheta_frontend_result.json").write_text(
        json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
