"""Is the projection-aware re-render CORRECT on the frame v5 actually uses?

Verifies ``taniteval.clhorizon``'s 2026-07-27 projection-aware re-render on
**real decoded frames from pod2's v5 cache**, at the mid-run held-out gate's own
``+/-8 deg`` probe, against the same standard the defect was measured with
(``…/2026-07-28-v5-gateable/code/warp_geometry_audit.py``).

FIVE LEGS, and the control is not optional
------------------------------------------
A. **COORDINATE FIELD** (exact-vs-exact, no data): shipped
   ``sampling_homography(f=266, c=128)`` vs the new
   ``sampling_source_grid(frame)`` on v5's 176x624 cylindrical frame. This
   reproduces the audit's 46.30 / 189.20 / 99.08 % numbers from the SHIPPED
   code rather than a reimplementation.
B. ⭐ **REAL PIXELS, against an INDEPENDENT ORACLE.** On a cylinder a yaw is
   ``u -> u + f_ref*psi``. Choose ``psi`` so ``f_ref*psi`` is an EXACT INTEGER
   number of pixels: the correct re-render is then a pure column ROLL of the
   real frame, computable by array slicing with **no** ``grid_sample``, no
   homography and no shared code. That is a bit-level oracle on real camera
   pixels. The sub-pixel case uses an independent numpy bilinear column
   interpolation. Both the NEW path and the SHIPPED path are scored against it.
C. ⭐ **THE CONTROL — the deployed path must not move.** On a 256x256 raster of
   real camera content, ``warp_frames(..., frame=None)`` must be
   ``torch.equal`` to the pre-2026-07-27 ``warp_batch(sampling_homography(...))``
   call, and the coordinate fields must agree to ``max 0.118 px``. A guard that
   fires everywhere proves nothing.
D. **REPRESENTATION**: the best-fit 3x3 (DLT over the whole field). Yaw on a
   cylinder IS a homography (a translation) — lateral is NOT, so the 3x3
   representation genuinely cannot carry the lateral axis while the resampler
   can carry both.
E. **GEOMETRY PROVENANCE**: every frame used here is READ from the cache's own
   ``_geometry.json`` / the ``*.v2ep.pt`` payload — never re-derived, never
   re-hard-coded.

Usage (pod2)::

    PYTHONPATH=/workspace/v5gate/stack:/workspace/v5gate/stack/scripts:\\
    /workspace/v5gate/taniteval \\
    python3 warp_realframes_verify.py \\
        --cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \\
        --subframe 176x624 --clips 8 --out realframes.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

# ⚠️ `tanitad` FIRST and deliberately: several `taniteval` submodules still do
# `sys.path.insert(0, "/root/TanitAD/stack")`, which silently shadows a
# working-tree stack. Importing here pins it in `sys.modules` before that can
# happen, and the resolved paths are recorded in the output (see `main`). The
# supported alternative is `TANITEVAL_STACK_OVERRIDE`; the runner sets both.
from tanitad.data.calib import CanonicalFrame, centred_subframe, subframe_slice
from tanitad.data.v2_dataset import build_v2_providers, stored_frame_of
from taniteval import clhorizon as CH

PROBE_DYAW = (2.0, 8.0, 12.0)          # 8 = heldout_gate.probe_grid
PROBE_DLAT = (0.5, 1.0, 2.0)
CONTROL_MAX_PX = 0.118                 # the audit's control, to be reproduced


# --------------------------------------------------------------------------- #
# A. the coordinate field                                                      #
# --------------------------------------------------------------------------- #
def _dest_grid(h, w):
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64),
                            torch.arange(w, dtype=torch.float64), indexing="ij")
    return xs, ys


def _apply_H(H, u, v):
    P = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(-1, 3).T
    s = H.to(torch.float64) @ P
    return ((s[0] / s[2]).reshape(u.shape), (s[1] / s[2]).reshape(u.shape))


def field_compare(frame, dlat, dyaw):
    """SHIPPED warp vs the CORRECT one for ``frame``, in source pixels."""
    fr = CH.as_warp_frame(frame)
    xs, ys = _dest_grid(fr.height, fr.width)
    hu, hv = _apply_H(CH.sampling_homography(dlat, dyaw), xs, ys)
    tu, tv, valid = CH.sampling_source_grid(dlat, dyaw, frame)
    e = torch.hypot(hu - tu, hv - tv)
    row = {
        "dlat_m": float(dlat), "dyaw_deg": float(dyaw),
        "err_px_mean": round(float(e.mean()), 4),
        "err_px_median": round(float(e.median()), 4),
        "err_px_p95": round(float(e.flatten().quantile(0.95)), 4),
        "err_px_max": round(float(e.max()), 4),
        "err_u_px_max": round(float((hu - tu).abs().max()), 4),
        "err_v_px_max": round(float((hv - tv).abs().max()), 4),
        "frac_px_err_gt_1": round(float((e > 1.0).double().mean()), 6),
        "frac_px_err_gt_8": round(float((e > 8.0).double().mean()), 6),
        "no_ground_preimage_frac": round(float((~valid).double().mean()), 6),
    }
    if fr.projection == "cylindrical" and dlat == 0.0:
        row["true_shift_px"] = round(fr.f_ref * math.radians(dyaw), 4)
        row["err_exceeds_the_whole_correct_displacement"] = bool(
            row["err_px_mean"] > abs(row["true_shift_px"]))
    return row


# --------------------------------------------------------------------------- #
# B. the independent oracle on real pixels                                     #
# --------------------------------------------------------------------------- #
def roll_oracle(img: np.ndarray, shift_px: int) -> np.ndarray:
    """Pure column ROLL with border replication. numpy slicing only.

    ``img`` is ``[..., H, W]``. The re-rendered pixel ``u`` samples the source at
    ``u + shift``, so the output is the source shifted LEFT by ``shift``.
    """
    w = img.shape[-1]
    idx = np.clip(np.arange(w) + int(shift_px), 0, w - 1)
    return img[..., idx]


def bilinear_column_oracle(img: np.ndarray, shift_px: float) -> np.ndarray:
    """Sub-pixel horizontal shift, bilinear, border-clamped. numpy only —
    shares no line of code with ``grid_sample`` or with the module under test."""
    w = img.shape[-1]
    src = np.arange(w, dtype=np.float64) + float(shift_px)
    src = np.clip(src, 0.0, w - 1.0)
    i0 = np.floor(src).astype(int)
    i1 = np.clip(i0 + 1, 0, w - 1)
    a = (src - i0)[None, :]
    return img[..., i0] * (1.0 - a) + img[..., i1] * a


def _interior(x, pad):
    return x[..., :, pad:-pad] if pad else x


def real_pixel_leg(frames_u8, frame, pad=64):
    """``frames_u8`` ``[n, C, H, W]`` uint8 REAL decoded pixels."""
    fr = CH.as_warp_frame(frame)
    img = frames_u8.float().div(255.0)                       # [n, C, H, W]
    fw = img[:, None]                                        # [n, 1, C, H, W]
    np_img = img.numpy().astype(np.float64)
    out = {"n_frames": int(img.shape[0]), "interior_pad_px": pad, "cases": []}

    # -- B1. EXACT-INTEGER shift: the bit-level oracle --------------------- #
    for shift in (10, 40):
        psi = shift / fr.f_ref                                # radians
        dyaw = math.degrees(psi)
        oracle = roll_oracle(np_img, shift)
        new = CH.warp_frames(fw, 0.0, dyaw, frame)[:, 0].numpy().astype(np.float64)
        old = CH.warp_frames(fw, 0.0, dyaw, CH.LEGACY_WARP)[:, 0].numpy().astype(np.float64)
        o_i, n_i, l_i = (_interior(oracle, pad), _interior(new, pad),
                         _interior(old, pad))
        out["cases"].append({
            "case": f"EXACT_INTEGER_SHIFT_{shift}px",
            "dyaw_deg": round(dyaw, 6),
            "oracle": "numpy column roll (array slicing; no grid_sample)",
            "NEW_max_abs_intensity_err": float(np.abs(n_i - o_i).max()),
            "NEW_mean_abs_intensity_err": float(np.abs(n_i - o_i).mean()),
            "SHIPPED_max_abs_intensity_err": float(np.abs(l_i - o_i).max()),
            "SHIPPED_mean_abs_intensity_err": float(np.abs(l_i - o_i).mean()),
            "SHIPPED_over_NEW_mean_ratio": float(
                np.abs(l_i - o_i).mean() / max(np.abs(n_i - o_i).mean(), 1e-12)),
        })

    # -- B2. the gate's own probe angles, sub-pixel oracle ----------------- #
    for dyaw in PROBE_DYAW:
        shift = fr.f_ref * math.radians(dyaw)
        oracle = bilinear_column_oracle(np_img, shift)
        new = CH.warp_frames(fw, 0.0, dyaw, frame)[:, 0].numpy().astype(np.float64)
        old = CH.warp_frames(fw, 0.0, dyaw, CH.LEGACY_WARP)[:, 0].numpy().astype(np.float64)
        o_i, n_i, l_i = (_interior(oracle, pad), _interior(new, pad),
                         _interior(old, pad))
        out["cases"].append({
            "case": f"PROBE_{dyaw:g}deg",
            "true_shift_px": round(float(shift), 4),
            "oracle": "numpy bilinear column interpolation",
            "NEW_max_abs_intensity_err": float(np.abs(n_i - o_i).max()),
            "NEW_mean_abs_intensity_err": float(np.abs(n_i - o_i).mean()),
            "SHIPPED_max_abs_intensity_err": float(np.abs(l_i - o_i).max()),
            "SHIPPED_mean_abs_intensity_err": float(np.abs(l_i - o_i).mean()),
            "SHIPPED_over_NEW_mean_ratio": float(
                np.abs(l_i - o_i).mean() / max(np.abs(n_i - o_i).mean(), 1e-12)),
            "frame_dynamic_range": float(np_img.max() - np_img.min()),
        })
    return out


# --------------------------------------------------------------------------- #
# C. the control                                                               #
# --------------------------------------------------------------------------- #
def control_leg(frames_u8):
    """The DEPLOYED 256x256 path, on a 256x256 raster of REAL camera content."""
    n, c, h, w = frames_u8.shape
    t = (h - 256) // 2
    left = (w - 256) // 2
    crop = frames_u8[..., max(t, 0):max(t, 0) + 256, left:left + 256]
    if crop.shape[-2] != 256:                    # pad rows if the frame is short
        crop = torch.nn.functional.pad(
            crop.float(), (0, 0, 0, 256 - crop.shape[-2]), mode="replicate"
        ).to(torch.uint8)
    fw = crop.float().div(255.0)[:, None]
    dep = {"height": 256, "width": 256, "f_ref": 266.0, "projection": "pinhole"}
    rows = []
    for dlat, dyaw in ((0.0, 8.0), (0.0, -8.0), (1.5, 3.0), (3.0, 12.0)):
        legacy = CH.warp_batch(fw, torch.stack(
            [CH.sampling_homography(dlat, dyaw)] * fw.shape[0]))
        new = CH.warp_frames(fw, dlat, dyaw, None)
        declared = CH.warp_frames(fw, dlat, dyaw, dep)
        f = field_compare(dep, dlat, dyaw)
        rows.append({
            "dlat_m": dlat, "dyaw_deg": dyaw,
            "bit_identical_frame_None": bool(torch.equal(legacy, new)),
            "bit_identical_declared_canonical": bool(
                torch.equal(legacy, declared)),
            "max_abs_pixel_diff": float((legacy - new).abs().max()),
            "coord_field_err_px_max": f["err_px_max"],
            "coord_field_err_px_mean": f["err_px_mean"],
            "within_audit_control_0_118px": bool(
                f["err_px_max"] <= CONTROL_MAX_PX),
        })
    return {"raster": "256x256 centre crop of REAL v5 camera pixels",
            "n_frames": int(fw.shape[0]), "by_condition": rows}


# --------------------------------------------------------------------------- #
# D. can a 3x3 express it?                                                     #
# --------------------------------------------------------------------------- #
def dlt_residual(frame, dlat, dyaw, n_max=6000):
    fr = CH.as_warp_frame(frame)
    su, sv, valid = CH.sampling_source_grid(dlat, dyaw, frame)
    xs, ys = _dest_grid(fr.height, fr.width)
    m = valid & torch.isfinite(su) & torch.isfinite(sv)
    u = xs[m].double().numpy(); v = ys[m].double().numpy()
    a_ = su[m].double().numpy(); b_ = sv[m].double().numpy()
    ix = np.linspace(0, len(u) - 1, min(len(u), n_max)).astype(int)
    u, v, a_, b_ = u[ix], v[ix], a_[ix], b_[ix]
    n = len(u); o = np.ones(n); z = np.zeros(n)
    A = np.empty((2 * n, 9))
    A[0::2] = np.stack([u, v, o, z, z, z, -a_ * u, -a_ * v, -a_], axis=1)
    A[1::2] = np.stack([z, z, z, u, v, o, -b_ * u, -b_ * v, -b_], axis=1)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    H = Vt[-1].reshape(3, 3)
    d = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    pu = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / d
    pv = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / d
    e = np.hypot(pu - a_, pv - b_)
    return {"dlat_m": float(dlat), "dyaw_deg": float(dyaw), "n_points": int(n),
            "best_homography_residual_px_max": round(float(e.max()), 6),
            "best_homography_residual_px_mean": round(float(e.mean()), 6),
            "best_homography_residual_px_p95": round(
                float(np.percentile(e, 95)), 6),
            "expressible_as_a_3x3": bool(e.max() < 1e-6)}


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--subframe", default="176x624")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    cache = Path(a.cache)
    geo = json.loads((cache / "_geometry.json").read_text())
    cache_frame = CanonicalFrame.from_dict(geo["frame"])       # READ, not derived
    if a.subframe.lower() in ("none", ""):
        train_frame = cache_frame
        rs = cs = None
    else:
        h, w = (int(x) for x in a.subframe.lower().split("x"))
        train_frame = centred_subframe(cache_frame, h, w)
        rs, cs = subframe_slice(cache_frame, train_frame)

    eps = build_v2_providers([str(cache)], lru_size=4, frame=train_frame,
                             verbose=False)[:a.clips]
    # the payload's OWN declared frame, independently of _geometry.json
    payload_frame = stored_frame_of(torch.load(
        sorted(cache.glob("*.v2ep.pt"))[0], weights_only=False, map_location="cpu"))

    fr_list = []
    for ep in eps:
        f = torch.as_tensor(ep.frames[0:a.window])
        fr_list.append(f if f.dtype == torch.uint8 else
                       (f * 255).clamp(0, 255).to(torch.uint8))
    frames_u8 = torch.cat(fr_list)                    # [n*window, C, H, W]

    out = {
        "what": ("does taniteval.clhorizon's projection-aware re-render land the "
                 "CORRECT pixels on the frame v5 actually uses, and does the "
                 "DEPLOYED path still land the pixels it used to?"),
        "host": "pod2", "cache": str(cache),
        "code_provenance": {
            "taniteval.clhorizon": CH.__file__,
            "tanitad.data.calib": CanonicalFrame.__module__ and __import__(
                "tanitad.data.calib", fromlist=["x"]).__file__,
        },
        "geometry_provenance": {
            "cache_frame_from": str(cache / "_geometry.json"),
            "cache_frame": cache_frame.to_dict(),
            "payload_declared_frame": payload_frame.to_dict(),
            "cache_and_payload_agree": bool(payload_frame == cache_frame),
            "train_frame": train_frame.to_dict(),
            "train_frame_is_centred_slice_rows": (
                None if rs is None else [rs.start, rs.stop]),
            "train_frame_is_centred_slice_cols": (
                None if cs is None else [cs.start, cs.stop]),
            "principal_point_px_in_train_frame": [
                CH.as_warp_frame(train_frame).cx,
                CH.as_warp_frame(train_frame).cy],
            "_note": ("centred_subframe/subframe_slice keep the boresight at the "
                      "CHILD's own geometric centre exactly (calib.py: the "
                      "(W-1)/2 and (w-1)/2 halves cancel against the integer "
                      "margin), and the clip's NATIVE per-clip (cx, cy) was "
                      "already absorbed by cylindrical_rectify at build time."),
        },
        "shipped_warp": ("clhorizon.sampling_homography(f=F_EFF=266.0, "
                         "c=CXY=128.0) — the DEPLOYED 256x256 pinhole crop"),
        "n_clips": len(eps), "window": a.window,
        "frames_decoded": int(frames_u8.shape[0]),
        "frame_shape": list(frames_u8.shape[-2:]),
    }

    # A ------------------------------------------------------------------- #
    out["A_coordinate_field_v5_frame"] = (
        [field_compare(train_frame, 0.0, d) for d in PROBE_DYAW]
        + [field_compare(train_frame, d, 0.0) for d in PROBE_DLAT])
    out["A_coordinate_field_CONTROL_deployed"] = [
        field_compare({"height": 256, "width": 256, "f_ref": 266.0,
                       "projection": "pinhole"}, 0.0, d) for d in PROBE_DYAW]

    # B ------------------------------------------------------------------- #
    out["B_real_pixels_vs_independent_oracle"] = real_pixel_leg(
        frames_u8, train_frame)

    # C ------------------------------------------------------------------- #
    out["C_control_deployed_path_unregressed"] = control_leg(frames_u8)

    # D ------------------------------------------------------------------- #
    out["D_representation"] = {
        "yaw_on_a_cylinder": dlt_residual(train_frame, 0.0, 8.0),
        "lateral_on_a_cylinder": dlt_residual(train_frame, 2.0, 0.0),
        "lateral_on_a_pinhole": dlt_residual(
            {"height": 256, "width": 256, "f_ref": 266.0,
             "projection": "pinhole"}, 2.0, 0.0),
        "_reading": ("a yaw on a cylinder IS a 3x3 (a translation) — the "
                     "shipped code simply computes a DIFFERENT 3x3. A LATERAL "
                     "displacement on a cylinder is NOT a 3x3 at all, so the "
                     "matrix representation cannot carry it while the "
                     "grid_sample resampler carries both."),
    }

    # E ------------------------------------------------------------------- #
    fw = frames_u8.float().div(255.0)[:, None]
    refused = {}
    try:
        CH.warp_frames(fw, 0.0, 8.0, None)
        refused["no_frame_on_a_176x624_raster"] = "NOT REFUSED — GUARD BROKEN"
    except CH.WarpFrameRefused as ex:
        refused["no_frame_on_a_176x624_raster"] = f"REFUSED: {str(ex)[:120]}…"
    try:
        CH.warp_frames(fw, 0.0, 8.0, cache_frame)     # 256x640 != 176x624
        refused["declared_frame_is_not_the_raster"] = "NOT REFUSED — GUARD BROKEN"
    except CH.WarpFrameRefused as ex:
        refused["declared_frame_is_not_the_raster"] = f"REFUSED: {str(ex)[:120]}…"
    out["E_guard_demonstrated_failing_on_real_frames"] = refused

    txt = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"[warp-realframes] wrote {a.out}")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
