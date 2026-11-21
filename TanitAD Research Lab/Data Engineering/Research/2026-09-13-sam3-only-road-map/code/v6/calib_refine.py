"""Per-camera ground-warp calibration refined from the paint itself (small rotation + height), fit / held-out split.

Why (MEASURED 2026-09-13, cross_cam_registration.py, day clip frames 0-38): CAM_FW and CAM_CR agree within 0.1 m, but
CAM_CL peaks at a (+0.3, -0.4) m rig shift against the other cameras (NCC 0.41 at zero -> 0.58 at the peak, mirrored
control -0.01) and the rear cameras give flat or range-edge peaks. A constant shift is the wrong model for a rotation
error (a pitch error of 0.5 deg moves ground 0.21 m at 6 m range and 0.84 m at 12 m), so this fits rotations.
  reference  CALIB_REFS (default CAM_FW; a FW-vs-CR run showed they differ by ~0.5-0.75 deg yaw) -> world paint share at
             0.10 m (near <= 12 m)
  sample     for each other camera, pixels of its own SAM3 classes 1-7 (not ego) within 12 m, bright-paint flag from a
             127 px image top-hat inside its thin / area paint masks; selected by the camera's own labels only
  search     yaw, pitch, roll in {-1, 0, +1} deg x height {-0.10, 0, +0.10} m on every 2nd fit frame, then coordinate
             descent (0.5 deg / 0.05 m, then 0.25 deg / 0.025 m) on all fit frames; score = pooled NCC of (paint flag,
             reference share) over the FIT frames (even tokens)
  verdict    the chosen correction must raise the NCC on the HELD-OUT frames (odd tokens) over the identity, and a
             reference camera run through the same search must choose ~identity (<= 0.25 deg, 0 m) -- else no correction
Output: calib_refine_<c8>.json (per camera: correction, NCC identity / corrected on fit and held-out, verdict).
Usage: calib_refine.py <c8> <npz dir> <out json>   (SAM3MAP_ROOT = sequence root)
"""
import itertools, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p_ in ("/home/nvidia/sam3paint", "/home/nvidia/sam3map", "/home/nvidia/sam3map/eval", str(HERE)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import numpy as np
import cv2
from PIL import Image
import sam3_paint as P
import camera_model as CM
import ground_surface as GS

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
RES, NEAR_M = 0.10, 12.0
REFS = tuple(os.environ.get("CALIB_REFS", "CAM_FW").split(","))          # the anchor camera(s); every other camera is fit to them
MODE = os.environ.get("CALIB_MODE", "anchor")                          # anchor: fit to CALIB_REFS; self: range-split self-consistency
SELF_NEAR_M = 7.0                                                        # self mode: REFUTED as an absolute method (2026-09-13: FW pitch +2 deg, RR yaw -2.5 deg,
#   CR yaw of the opposite sign to the anchor run; fit gains far above held-out -- the near/far split and the reference both move with the candidate)
TOPHAT = cv2.getStructuringElement(cv2.MORPH_RECT, (127, 127))          # rect: 0.03 s vs 0.64 s (ellipse) per image on Thor


def rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def perturbed(C, yaw, pitch, roll, dz):
    """Camera frame: x right, y down, z forward -> pitch about x, yaw about y, roll about z; R maps camera -> rig."""
    dR = rot("y", yaw) @ rot("x", pitch) @ rot("z", roll)
    return CM.Camera(C.R @ dR, C.t + np.array([0.0, 0.0, dz]), K=C.K, ftheta=C.ft, width=C.W, height=C.H)


def image_paint(img, cls_small, ego_small):
    """(valid pixels, bright-paint flag) on the 960x540 grid: valid = SAM3 classes 1-7 not ego; paint = 127 px top-hat
    brighter than max(10, 2.5 x road MAD) inside the camera's thin / area paint classes."""
    Y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    th = cv2.resize(cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, TOPHAT), (960, 540), interpolation=cv2.INTER_AREA)
    road = cls_small == 1
    mad = float(np.median(np.abs(th[road] - np.median(th[road])))) if road.sum() > 100 else 3.0
    valid = (cls_small >= 1) & (cls_small <= 7)
    if ego_small is not None:
        valid &= ~ego_small
    paint = valid & np.isin(cls_small, (2, 3, 4, 6)) & (th > max(10.0, 2.5 * mad))
    return valid, paint


def lift(C, sg, u, v):
    d = C.rays_rig(u, v)
    good = np.isfinite(d).all(axis=1) & (d[:, 2] < -1e-3)
    d = np.where(good[:, None], d, np.array([0.0, 0.0, -1.0]))
    z = np.full(len(u), float(np.median(sg)))
    for _ in range(3):
        s = (z - C.t[2]) / d[:, 2]
        xy = C.t[:2] + d[:, :2] * s[:, None]
        z = GS.height(xy, sg)
    return xy, good & (s > 0) & (np.hypot(xy[:, 0] - C.t[0], xy[:, 1] - C.t[1]) <= NEAR_M)


def main():
    c8, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    t0 = time.time()
    files = sorted(src.glob("[0-9][0-9][0-9].npz"))
    sd = ROOT / f"seq_{c8}"
    data = []                                              # per frame: T, sg, {cam: (C, u, v, paint)}
    cams = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        cams = cams or [c for c in ("CAM_FW", "CAM_CL", "CAM_CR", "CAM_RL", "CAM_RR", "CAM_RT", "CAM_FT", "CAM_F0") if f"cls_{c}" in z.files]
        fd = sd / str(z["tok"])
        c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
        grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
        per = {}
        for cam in cams:
            C = CM.Camera.from_calib(c, fr["cam_order"].index(cam))
            img = np.asarray(Image.open(fd / "images" / f"{cam}.jpg").convert("RGB"))
            ego = z[f"ego_{cam}"] if f"ego_{cam}" in z.files else None
            valid, paint = image_paint(img, z[f"cls_{cam}"], ego)
            vv, uu = np.nonzero(valid[::3, ::3]); vv, uu = vv * 3, uu * 3            # every third 960x540 pixel
            u, v = uu * 2 + 1.0, vv * 2 + 1.0
            xy, ok = lift(C, sg, u, v)
            dd = np.hypot(xy[:, 0] - C.t[0], xy[:, 1] - C.t[1])
            keep = np.isfinite(dd) & (dd <= NEAR_M + 3.0)                             # candidates that can land within 12 m
            per[cam] = (C, u[keep], v[keep], paint[vv, uu][keep])
        data.append((np.array(z["T_world_rig"]), sg, per))
    print(f"loaded {len(data)} frames x {len(cams)} cameras  {time.time() - t0:.0f}s", flush=True)
    P_ = np.array([d[0][:2, 3] for d in data]); x0, y0 = P_.min(axis=0) - 20; x1, y1 = P_.max(axis=0) + 20
    WW, WH = int((x1 - x0) / RES) + 1, int((y1 - y0) / RES) + 1

    def cells(T, xy):
        w = xy @ T[:2, :2].T + T[:2, 3]
        i = np.floor((w[:, 0] - x0) / RES).astype(np.int64); j = np.floor((w[:, 1] - y0) / RES).astype(np.int64)
        return i, j, (i >= 0) & (i < WW) & (j >= 0) & (j < WH)

    accepted = {}                                          # camera -> correction, joins the reference once accepted

    def reference(frames_idx, exclude=None):
        obs = np.zeros((WW, WH), np.float32); pnt = np.zeros((WW, WH), np.float32)
        for n in frames_idx:
            T, sg, per = data[n]
            for cam in tuple(REFS) + tuple(accepted):
                if cam not in per or cam == exclude:
                    continue
                C, u, v, pf = per[cam]
                xy, ok = lift(perturbed(C, *accepted[cam]) if cam in accepted else C, sg, u, v)
                i, j, inb = cells(T, xy); m = ok & inb
                np.add.at(obs, (i[m], j[m]), 1.0); np.add.at(pnt, (i[m], j[m]), pf[m].astype(np.float32))
        return np.where(obs >= 2, pnt / np.maximum(obs, 1), np.nan).astype(np.float32)

    def score(cam, frames_idx, ref, pert):
        acc = np.zeros(6)
        for n in frames_idx:
            T, sg, per = data[n]
            C, u, v, pf = per[cam]
            xy, ok = lift(perturbed(C, *pert), sg, u, v)
            i, j, inb = cells(T, xy); m = ok & inb
            b = ref[i[m], j[m]]; a = pf[m].astype(np.float64); fin = np.isfinite(b); a, b = a[fin], b[fin].astype(np.float64)
            acc += [(a * b).sum(), a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), len(a)]
        nn = max(acc[5], 1)
        cov = acc[0] / nn - (acc[1] / nn) * (acc[2] / nn); va = acc[3] / nn - (acc[1] / nn) ** 2; vb = acc[4] / nn - (acc[2] / nn) ** 2
        return float(cov / np.sqrt(max(va * vb, 1e-12))), int(acc[5])

    def score_self(cam, frames_idx, _ref, pert):
        """Range-split self-consistency (no other camera, no LiDAR): the camera's own paint share from NEAR observations
        (<= SELF_NEAR_M, where angle errors move the ground little) is the reference for its FAR observations (SELF_NEAR_M..
        NEAR_M, where a pitch error moves ground ~r^2 * dtheta / h). The ego poses carry the cells from far to near."""
        C0 = perturbed(data[frames_idx[0]][2][cam][0], *pert)
        obs = np.zeros((WW, WH), np.float32); pnt = np.zeros((WW, WH), np.float32); far = []
        for n in frames_idx:
            T, sg, per = data[n]
            C, u, v, pf = per[cam]
            Cp = perturbed(C, *pert)
            xy, ok = lift(Cp, sg, u, v)
            rr = np.hypot(xy[:, 0] - Cp.t[0], xy[:, 1] - Cp.t[1])
            i, j, inb = cells(T, xy)
            m = ok & inb & (rr <= SELF_NEAR_M)
            np.add.at(obs, (i[m], j[m]), 1.0); np.add.at(pnt, (i[m], j[m]), pf[m].astype(np.float32))
            f_ = ok & inb & (rr > SELF_NEAR_M)
            far.append((i[f_], j[f_], pf[f_]))
        ref = np.where(obs >= 2, pnt / np.maximum(obs, 1), np.nan)
        acc = np.zeros(6)
        for i, j, a in far:
            b = ref[i, j]; fin = np.isfinite(b); a, b = a[fin].astype(np.float64), b[fin]
            acc += [(a * b).sum(), a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), len(a)]
        nn = max(acc[5], 1)
        cov = acc[0] / nn - (acc[1] / nn) * (acc[2] / nn); va = acc[3] / nn - (acc[1] / nn) ** 2; vb = acc[4] / nn - (acc[2] / nn) ** 2
        return float(cov / np.sqrt(max(va * vb, 1e-12))), int(acc[5])

    if MODE == "self":
        score = score_self                                   # noqa: F811 -- the same search, a different reference

    fit = list(range(0, len(data), 2)); held = list(range(1, len(data), 2))
    grid1 = list(itertools.product((-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (-0.1, 0.0, 0.1)))
    res = {}
    order = [c for c in os.environ.get("CALIB_CHAIN", "").split(",") if c in cams] or cams
    for cam in order:
        if cam == "CAM_FT":
            continue                                        # the front tele sees no ground within 12 m
        ref_fit = reference(fit, exclude=cam) if MODE == "anchor" else None
        ref_held = reference(held, exclude=cam) if MODE == "anchor" else None
        s0, n0 = score(cam, fit, ref_fit, (0, 0, 0, 0))
        if n0 < 2000:
            res[cam] = {"verdict": "not enough overlap with the reference", "cells": n0}; print(cam, json.dumps(res[cam]), flush=True); continue
        coarse = fit[::2]
        best = max(grid1, key=lambda p: score(cam, coarse, ref_fit, p)[0])
        sf, _ = score(cam, fit, ref_fit, best)
        for step, dstep in ((0.5, 0.05), (0.25, 0.025)):                           # coordinate descent on all fit frames
            improved = True
            while improved:
                improved = False
                for k_ in range(4):
                    for sgn in (-1, 1):
                        cand = list(best); cand[k_] = round(cand[k_] + sgn * (dstep if k_ == 3 else step), 3); cand = tuple(cand)
                        sc, _ = score(cam, fit, ref_fit, cand)
                        if sc > sf + 1e-4:
                            best, sf, improved = cand, sc, True
        h0, nh = score(cam, held, ref_held, (0, 0, 0, 0)); h1, _ = score(cam, held, ref_held, best)
        is_ref = cam in REFS and MODE == "anchor"
        if is_ref:
            verdict = "reference self-check PASS" if max(abs(best[0]), abs(best[1]), abs(best[2])) <= 0.25 and abs(best[3]) <= 0.025 else "reference self-check FAIL"
        else:
            verdict = ("apply" if (h1 > h0 + 0.02 and sf > s0 + 0.02 and h1 >= 0.3 and any(abs(x) > 1e-9 for x in best))
                       else "keep identity (needs held-out AND fit gain > 0.02 and held-out NCC >= 0.3)")
        if MODE == "anchor" and not is_ref:
            if verdict == "apply":
                accepted[cam] = best
            elif h0 >= 0.3:
                accepted[cam] = (0.0, 0.0, 0.0, 0.0)          # consistent at identity: joins the reference unchanged
        res[cam] = {"correction": {"yaw_deg": best[0], "pitch_deg": best[1], "roll_deg": best[2], "dz_m": best[3]},
                    "joined_reference": cam in accepted,
                    "ncc_fit": {"identity": round(s0, 3), "corrected": round(sf, 3)}, "ncc_held_out": {"identity": round(h0, 3), "corrected": round(h1, 3)},
                    "cells_fit": n0, "cells_held_out": nh, "verdict": verdict}
        print(cam, json.dumps(res[cam]), f"{time.time() - t0:.0f}s", flush=True)
    override = {c: {"yaw_deg": v[0], "pitch_deg": v[1], "roll_deg": v[2], "dz_m": v[3]} for c, v in accepted.items() if any(abs(x) > 1e-9 for x in v)}
    out.write_text(json.dumps({"mode": MODE, "chain": order, "override": override, "self_near_m": SELF_NEAR_M, "reference_cameras": REFS, "near_m": NEAR_M, "frames": len(data), "cameras": res, "seconds": round(time.time() - t0, 1)}, indent=1),
                   encoding="utf-8")
    print("ZZCALIB-DONEZZ")


if __name__ == "__main__":
    main()
