"""Is the closed-loop / pseudo-simulation WARP valid on the v5 frame?

WHY THIS EXISTS
---------------
``GATE_PROTOCOL`` §0 makes ``corridor_departure_rate`` @ a pre-registered
``K`` the gate CO-PRIMARY, and §0.6 states that a ``K > 20`` corridor read
"requires a closed-loop rollout". Both closed-loop surfaces in the package --
``taniteval.clhorizon.corridor_rollout`` (sequential) and
``taniteval.pseudosim.pseudo_evaluate`` (bounded grid) -- re-render the real
camera frame through ONE function::

    taniteval.clhorizon.sampling_homography(dlat_m, dyaw_deg,
                                            f=F_EFF=266.0, c=CXY=128.0)

``F_EFF`` / ``CXY`` are the **deployed 256x256 pinhole crop's** intrinsics
(``calib.py`` ``F_REF = 266`` at 256 px, principal point at the geometric
centre). v5's frame is **256x640 cylindrical at f_ref 305.5775**, sub-framed to
**176x624**. So the warp is applied with

  1. the WRONG focal length and the WRONG principal point, and
  2. the WRONG PROJECTION MODEL -- a homography is a pinhole operation; the v5
     raster is an equidistant-azimuth cylinder (``calib.cylindrical_rays``:
     ray ``(sin phi, y_n, cos phi)``, ``phi = (u - (W-1)/2)/f_ref``).

WHAT IS MEASURED
----------------
For the HEADING axis -- the only axis pseudo-simulation uses, since its lateral
axis is REFUSED on measured geometry -- the exact answer is available in closed
form on BOTH projections, so this is an exact-vs-exact comparison and not an
approximation:

* **cylindrical**: a yaw rotation by ``dpsi`` maps the ray ``(sin phi, y_n,
  cos phi)`` to ``(sin(phi+dpsi), y_n, cos(phi+dpsi))``. Therefore
  ``u -> u + f_ref*dpsi`` EXACTLY, uniformly, at every row, and ``v`` is
  UNCHANGED. Depth-independent, like the pinhole ``K R K^-1``.
* **pinhole**: ``H = K R K^-1`` with the shipped ``K``.

The disagreement between the two source-pixel fields is the number: it is how
far the perturbed observation the model is scored on sits from the observation
the camera would actually have produced.

CONTROL. The same comparison is run on the DEPLOYED 256x256 pinhole frame,
where the shipped warp is the correct model. A guard that fires everywhere is
not a guard; this one must be ~0 there and large on v5.

No data, no GPU, no cache: the frames' geometry is fully specified by
``CanonicalFrame``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _sampling_homography_np(dlat_m, dyaw_deg, h_cam=1.5, pitch_deg=0.0,
                            f=266.0, c=128.0):
    """``taniteval.clhorizon.sampling_homography``, numpy, verbatim algebra.

    Reproduced here (rather than imported) so this audit runs on any host; the
    ``--check-import`` leg asserts it is bit-equal to the shipped function when
    ``taniteval`` is importable."""
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]], dtype=np.float64)
    Ki = np.linalg.inv(K)
    p = math.radians(pitch_deg)
    n = np.array([0.0, math.cos(p), math.sin(p)], dtype=np.float64)
    d = float(h_cam)
    psi = math.radians(dyaw_deg)
    Ry = np.array([[math.cos(-psi), 0, math.sin(-psi)],
                   [0, 1.0, 0],
                   [-math.sin(-psi), 0, math.cos(-psi)]], dtype=np.float64)
    Cc = np.array([dlat_m, 0.0, 0.0], dtype=np.float64)
    t = -(Ry @ Cc)
    H_1to2 = K @ (Ry + np.outer(t, n) / d) @ Ki
    return np.linalg.inv(H_1to2)


def _apply_H(H, u, v):
    """``warp_batch``'s source lookup: src = H @ [u, v, 1]; divide by w."""
    ones = np.ones_like(u)
    P = np.stack([u, v, ones], axis=0).reshape(3, -1)
    s = H @ P
    su = (s[0] / s[2]).reshape(u.shape)
    sv = (s[1] / s[2]).reshape(u.shape)
    return su, sv


def _grid(h, w):
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64), indexing="ij")
    return xs, ys


def cylindrical_yaw_source(u, v, dyaw_deg, f_ref):
    """EXACT source pixel for a camera yawed by ``dyaw_deg``, cylindrical raster.

    From ``calib.cylindrical_rays``: ``phi = (u - (W-1)/2)/f_ref`` and the ray is
    ``(sin phi, y_n, cos phi)`` with ``y_n = (v - (H-1)/2)/f_ref``. A rotation
    about the camera's +y (down) axis by ``psi`` sends ``phi -> phi + psi`` and
    leaves ``y_n`` untouched, so in PIXELS the map is a pure horizontal
    translation by ``f_ref * psi`` and the rows do not move at all.

    The sign is taken to MATCH the shipped homography's own sign convention,
    which is established empirically by :func:`_match_sign` rather than argued --
    a sign error would otherwise be reported as a geometry error."""
    return u + f_ref * math.radians(dyaw_deg), v.copy()


def _match_sign(f_ref, dyaw_deg, h, w, c):
    """Which sign of the cylindrical shift agrees with the shipped homography?

    Compared at the IMAGE CENTRE COLUMN only, where the two models must agree to
    first order whatever the projection -- so this fixes the convention without
    importing the disagreement it is meant to measure."""
    H = _sampling_homography_np(0.0, dyaw_deg, c=c, f=f_ref)
    u0 = np.array([[c]], dtype=np.float64)
    v0 = np.array([[c]], dtype=np.float64)
    su, _ = _apply_H(H, u0, v0)
    return 1.0 if (su[0, 0] - c) * (f_ref * math.radians(dyaw_deg)) >= 0 else -1.0


def audit_frame(name, h, w, f_ref, projection, cx, cy, dyaw_list,
                shipped_f=266.0, shipped_c=128.0):
    u, v = _grid(h, w)
    rows = []
    sgn = _match_sign(f_ref, 8.0, h, w, cx)
    for dyaw in dyaw_list:
        H = _sampling_homography_np(0.0, dyaw, f=shipped_f, c=shipped_c)
        su_h, sv_h = _apply_H(H, u, v)
        if projection == "cylindrical":
            su_t = u + sgn * f_ref * math.radians(dyaw)
            sv_t = v.copy()
        else:
            H_true = _sampling_homography_np(0.0, dyaw, f=f_ref, c=cx)
            su_t, sv_t = _apply_H(H_true, u, v)
        du, dv = su_h - su_t, sv_h - sv_t
        err = np.hypot(du, dv)
        rows.append({
            "dyaw_deg": float(dyaw),
            "true_shift_px": (float(sgn * f_ref * math.radians(dyaw))
                              if projection == "cylindrical" else None),
            "err_px_mean": round(float(err.mean()), 4),
            "err_px_median": round(float(np.median(err)), 4),
            "err_px_p95": round(float(np.percentile(err, 95)), 4),
            "err_px_max": round(float(err.max()), 4),
            "err_u_px_max": round(float(np.abs(du).max()), 4),
            "err_v_px_max": round(float(np.abs(dv).max()), 4),
            "frac_px_err_gt_1": round(float((err > 1.0).mean()), 6),
            "frac_px_err_gt_8": round(float((err > 8.0).mean()), 6),
            "err_px_mean_over_width": round(float(err.mean() / w), 6),
        })
    return {
        "frame": name, "height": h, "width": w, "f_ref": f_ref,
        "projection": projection, "principal_point": [cx, cy],
        "shipped_warp_intrinsics": {"f": shipped_f, "c": shipped_c},
        "principal_point_error_px": [round(shipped_c - cx, 3),
                                     round(shipped_c - cy, 3)],
        "f_ref_error_frac": round((shipped_f - f_ref) / f_ref, 6),
        "sign_convention_matched": sgn,
        "by_dyaw": rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--check-import", action="store_true",
                    help="assert the reproduced homography is bit-equal to "
                         "taniteval.clhorizon.sampling_homography")
    a = ap.parse_args(argv)

    dyaws = [2.0, 8.0, 12.0]      # 8 = heldout_gate.probe_grid; 12 = envelope edge
    out = {
        "what": "shipped closed-loop/pseudo-sim yaw warp vs the EXACT warp for "
                "each frame's own projection",
        "shipped_warp": "taniteval.clhorizon.sampling_homography (f=F_EFF=266.0,"
                        " c=CXY=128.0) — used by clhorizon.corridor_rollout AND "
                        "pseudosim.pseudo_evaluate",
        "dyaw_deg_probed": dyaws,
        "note_8deg": "the mid-run held-out gate's own probe grid is "
                     "(-8, 0, +8) deg (tanitad.train.heldout_gate.probe_grid)",
        "frames": [],
    }
    if a.check_import:
        try:
            sys.path.insert(0, str(Path(__file__).resolve()
                                   .parents[6] / "taniteval"))
            from taniteval import clhorizon as CH
            import torch
            ok, worst = True, 0.0
            for dl, dy in ((0.0, 8.0), (0.0, 12.0), (1.0, 3.0)):
                a_ = CH.sampling_homography(dl, dy).numpy()
                b_ = _sampling_homography_np(dl, dy)
                worst = max(worst, float(np.abs(a_ - b_).max()))
                ok &= bool(np.allclose(a_, b_, atol=0, rtol=0))
            out["reproduction_of_shipped_fn"] = {
                "bit_identical": ok, "max_abs_diff": worst,
                "F_EFF": float(CH.F_EFF), "CXY": float(CH.CXY)}
        except Exception as ex:                                # noqa: BLE001
            out["reproduction_of_shipped_fn"] = {"error": repr(ex)}

    # v5 as staged: 176x624 sub-frame of the 256x640 cylindrical cache.
    # A CENTRED slice keeps f_ref and moves the principal point with the crop:
    # parent centre (319.5, 127.5) - (8, 40) = (311.5, 87.5) = the sub-frame's
    # own geometric centre, which is what calib.cylindrical_rays assumes.
    out["frames"].append(audit_frame(
        "v5 rig-clean 176x624 cylindrical (--v2-subframe 176x624)",
        176, 624, 305.5774907364391, "cylindrical", 311.5, 87.5, dyaws))
    out["frames"].append(audit_frame(
        "v5 cache as built 256x640 cylindrical",
        256, 640, 305.5774907364391, "cylindrical", 319.5, 127.5, dyaws))
    out["frames"].append(audit_frame(
        "v5 alternative 128x576 cylindrical (--v2-subframe 128x576)",
        128, 576, 305.5774907364391, "cylindrical", 287.5, 63.5, dyaws))
    # CONTROL — the frame the warp was BUILT for.
    out["frames"].append(audit_frame(
        "CONTROL: deployed 256x256 pinhole crop (calib F_REF=266)",
        256, 256, 266.0, "pinhole", 127.5, 127.5, dyaws))

    txt = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(txt)
        print(f"[warp-audit] wrote {a.out}")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
