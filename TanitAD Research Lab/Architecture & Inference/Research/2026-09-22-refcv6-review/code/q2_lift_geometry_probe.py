"""Q2 / Q3 — does the BEV lift consume REAL geometry, and is `f_ref` expressed
at the resolution the network actually sees?

Advisory class B2: *"does anything named 3D actually consume geometry (K,
extrinsics), or only an index?"*  B3: *"correct intrinsics applied at the wrong
resolution"* — REFe's 1920-wide intrinsics on a 960-wide input halved the FOV.

Three measurements, each with a control:

**B2 — does the lift read the extrinsics?**  Build the lift geometry twice with
DIFFERENT mount poses and measure how far the sampling grid moves. A lift that
only consumed an index would move ZERO. The control is the same pose twice,
which must move EXACTLY zero.

**B3 — f_ref at the frame the network sees.** `refcv6_perception_branch.
frame_for_model` derives the lift's frame with `trunk_shapes.frame_for_width`,
which scales `f_ref` by `width/640`. Cross-check EVERY geometry declared in
`refc_v3_train._agent_cam_frames()` against what `frame_for_model` would
produce for the same `(h, w)`, and convert any disagreement into METRES of
lateral error on a BEV cell.

**The feature-centre offset for the TIMM trunk.** `bev_lift.
REFC_FEATURE_CENTRE_OFFSET_PX = 0.0` is derived and pinned for REF-C's OWN
`ResNetEncoder`. refcv6's trunk is a `timm` ResNet. Measured here by an IMPULSE
forward through the real backbone, not by reading the stem's declaration.
"""
from __future__ import annotations

import json
import math
import sys

import torch

OUT = {}


def main() -> int:
    from tanitad.data import calib as C
    from tanitad.models import trunk_shapes as TS
    from tanitad.models.bev_lift import (REFC_FEATURE_CENTRE_OFFSET_PX,
                                         build_lift_geometry)

    sys.path.insert(0, "D:/Projects/TanitAD/stack/scripts")
    import importlib
    rv3 = importlib.import_module("refc_v3_train")
    tbl = rv3._agent_cam_frames()

    # ---------------------------------------------------------------- B3 --- #
    rows = []
    for (h, w), fr in sorted(tbl.items()):
        derived = TS.frame_for_width(int(w), int(h))
        d_fref = float(derived.f_ref) - float(fr.f_ref)
        # lateral error at 20 m ahead for a 1-column-equivalent azimuth error:
        # az = (col - (W-1)/2)/f_ref, so the SAME rig point lands at columns
        # differing by |col_off| * |1 - f_declared/f_derived|; convert the
        # azimuth error back to metres at 20 m.
        col_off = (w / 2.0)                        # the frame EDGE, worst case
        az_declared = col_off / float(fr.f_ref)
        az_derived = col_off / float(derived.f_ref)
        rows.append({
            "hw": [h, w],
            "declared_f_ref": float(fr.f_ref),
            "frame_for_width_f_ref": float(derived.f_ref),
            "delta_f_ref": d_fref,
            "IDENTICAL": d_fref == 0.0,
            "declared_hfov_deg": round(math.degrees(2 * az_declared), 4),
            "frame_for_width_hfov_deg": round(math.degrees(2 * az_derived), 4),
            "edge_azimuth_error_deg":
                round(math.degrees(abs(az_declared - az_derived)), 4),
            "lateral_error_m_at_20m":
                round(20.0 * abs(math.tan(az_declared) - math.tan(az_derived)), 4),
        })
    OUT["B3_frame_for_width_vs_declared"] = rows
    OUT["B3_n_mismatched"] = sum(1 for r in rows if not r["IDENTICAL"])
    OUT["B3_CONTROL_n_total"] = len(rows)     # must be > 0 or the probe read nothing

    # ---------------------------------------------------------------- B2 --- #
    # two mount poses that differ ONLY in pitch; the lift must move.
    def extr(qx=0.0, qy=0.0, qz=0.0, qw=1.0, x=1.6, y=0.0, z=1.45):
        return {"qx": qx, "qy": qy, "qz": qz, "qw": qw, "x": x, "y": y, "z": z}

    f416 = TS.FRAME_416x1024
    a = build_lift_geometry(extr(), frame=f416, stride=16)
    b = build_lift_geometry(extr(z=1.20), frame=f416, stride=16)   # 25 cm lower
    c = build_lift_geometry(extr(qx=0.02), frame=f416, stride=16)  # small pitch
    ctl = build_lift_geometry(extr(), frame=f416, stride=16)       # SAME pose

    def moved(p, q):
        m = p.valid & q.valid
        if int(m.sum()) == 0:
            return {"n_common_valid": 0}
        return {"n_common_valid": int(m.sum()),
                "max_abs_row_shift_px": round(float((p.row - q.row)[m].abs().max()), 4),
                "max_abs_col_shift_px": round(float((p.col - q.col)[m].abs().max()), 4),
                "mean_abs_row_shift_px": round(float((p.row - q.row)[m].abs().mean()), 4)}

    OUT["B2_lift_reads_geometry"] = {
        "mount_height_-0.25m": moved(a, b),
        "pitch_qx_+0.02": moved(a, c),
        "CONTROL_same_pose_twice": moved(a, ctl),
        "grid_identical_for_same_pose": bool(torch.equal(a.grid, ctl.grid)),
        "valid_frac_at_416x1024": round(float(a.valid.any(dim=0).float().mean()), 4),
        "grid_shape": list(a.grid.shape), "feat_hw": list(a.feat_hw),
    }
    # a real intrinsics object is NOT consumed by the lift -- state that plainly
    OUT["B2_what_the_lift_consumes"] = {
        "per_clip_extrinsics_quaternion_and_translation": True,
        "frame_f_ref_and_projection": True,
        "per_clip_FTHETA_INTRINSICS": False,
        "note": ("the f-theta intrinsics were consumed ONCE, at cache BUILD "
                 "time by calib.cylindrical_rectify; the lift inverts the "
                 "CANONICAL frame, so it needs f_ref + projection only. That is "
                 "correct AND it is the reason a wrong f_ref is silent."),
    }

    # ---- the frame the lift would get, end to end, for a 416x1024 build ---- #
    import types
    fake = types.SimpleNamespace(
        cfg=types.SimpleNamespace(core=types.SimpleNamespace(
            encoder=types.SimpleNamespace(image_hw=lambda: (416, 1024)))))
    from tanitad.models.refcv6_perception_branch import frame_for_model
    fm = frame_for_model(fake)
    OUT["end_to_end_frame_for_416x1024_build"] = {
        "h": fm.height, "w": fm.width, "f_ref": float(fm.f_ref),
        "equals_FRAME_416x1024": (fm.height, fm.width, float(fm.f_ref)) ==
                                 (f416.height, f416.width, float(f416.f_ref))}
    # ... and for a RIG-CLEAN build, which is a DECLARED geometry in the table
    fake2 = types.SimpleNamespace(
        cfg=types.SimpleNamespace(core=types.SimpleNamespace(
            encoder=types.SimpleNamespace(image_hw=lambda: (176, 624)))))
    fm2 = frame_for_model(fake2)
    dec2 = tbl[(176, 624)]
    OUT["end_to_end_frame_for_176x624_build"] = {
        "frame_for_model_f_ref": float(fm2.f_ref),
        "declared_f_ref": float(dec2.f_ref),
        "AGREE": float(fm2.f_ref) == float(dec2.f_ref)}

    # -------------------------------------------- feature-centre offset --- #
    # IMPULSE through the REAL timm backbones: a single 1.0 pixel, find the
    # stride-16 column that lights up. offset == stride*k - col_of_impulse.
    import timm
    off = {}
    for name in ("resnet34.a1_in1k", "resnet101.a1_in1k"):
        net = timm.create_model(name, pretrained=False, features_only=True,
                                out_indices=(3, 4)).eval()
        H, W = 256, 256
        peaks = {}
        for stride, oi in ((16, 0), (32, 1)):
            cols = []
            for c in (64, 96, 128, 160):
                x = torch.zeros(1, 3, H, W)
                x[0, :, H // 2, c] = 1.0
                with torch.no_grad():
                    fmaps = net(x)
                m = fmaps[oi][0].abs().sum(dim=0)         # [h, w]
                k = int(m.sum(dim=0).argmax())
                cols.append({"impulse_col": c, "peak_feature_col": k,
                             "implied_centre_px": stride * k,
                             "offset_px": stride * k - c})
            peaks[f"stride{stride}"] = cols
        off[name] = peaks
    OUT["feature_centre_offset_IMPULSE"] = off
    OUT["REFC_FEATURE_CENTRE_OFFSET_PX_in_code"] = float(
        REFC_FEATURE_CENTRE_OFFSET_PX)

    json.dump(OUT, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
