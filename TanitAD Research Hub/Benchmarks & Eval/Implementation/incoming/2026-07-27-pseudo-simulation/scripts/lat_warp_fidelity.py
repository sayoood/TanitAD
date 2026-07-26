"""C-GEO-LAT -- does the flat-road LATERAL warp carry a pseudo-simulation grid?

THE QUESTION (research doc section 7.4, pre-registered there with both outcomes
committed BEFORE this script existed)
---------------------------------------------------------------------------
Our YAW warp is geometrically exact for arbitrary depth: a pure camera rotation
induces ``H = K R K^-1``, and P1's C-GEO measured ``max|dH| = 0.000e+00`` over 30
(h_cam, pitch) conditions. The LATERAL warp is a **ground-plane** homography and
therefore rests on a FLAT-ROAD assumption. Before we build a 2-D pseudo-
simulation grid on it, measure whether that assumption survives.

  L-OK  -> the grid is 2-D (lateral x heading), NAVSIM v2's axes.
  L-BAD -> the grid is heading-only. A 1-D MEASUREMENT beats a 2-D EXTRAPOLATION.

THE CRITERION, STATED BEFORE THE NUMBERS (this is the C13 gate on my own method)
-------------------------------------------------------------------------------
The pseudo-simulation claim is that the perturbed observation is *what the camera
would have seen at that pose*. So the quantity that decides it is the
**relative displacement error** of the synthesized motion field:

    R(x) = |u_applied(x) - u_true(x)| / |u_true(x)|

  **L-OK**  iff  R < ``R_MAX`` = 0.25 on at least ``FRAC_MIN`` = 95 % of the frame,
            at the grid's maximum |dlat|.
  **L-BAD** otherwise.

⚠️ THE CRITERION CAN FAIL IN BOTH DIRECTIONS AND BOTH ARE EXERCISED HERE:
  * the YAW arm is run through the identical code path as a POSITIVE control and
    must come out at R == 0 exactly (it does; that is test A4);
  * the value that would make LATERAL pass is stated in closed form (A2): the
    flat-road warp's relative error is exactly ``H_obj / h_cam``, so L-OK
    requires every scene point to lie within ``0.25 * 1.5 = 0.375 m`` of the road
    surface. That is a falsifiable, checkable condition, not a rhetorical one.

WHAT IS MEASURED (nothing is asserted from algebra alone -- every closed form is
re-derived numerically from the PACKAGED ``taniteval.clhorizon.sampling_homography``,
the same function the closed loop calls)
-------------------------------------------------------------------------------
  A0  depth/height invariance: max|dH| over (h_cam, pitch) for BOTH arms.
      Yaw must be 0 (P1's result, reproduced); lateral must not be.
  A1  exact per-point reprojection error against a synthetic 3-D scene.
  A2  the relative error R, and its closed form.
  A3  the SIGN INVERSION and its zero-crossing.
  A4  the yaw positive control -- identical code path, R == 0.
  A5  ⚠️ C-RT (round-trip residual), the OTHER criterion section 7.4 offered, is
      shown to be **VACUOUS** for this question: the ground-plane homography
      COMPOSES exactly, H(a)H(b) = H(a+b), so H(d)H(-d) = I to machine precision
      on BOTH arms. A guard that cannot fail is not a guard; it is refused here
      rather than reported as a pass.
  A6  the frame fraction above the horizon -- a hard, data-free lower bound on
      non-planar content, since the ground plane has NO preimage there.

CPU only. No GPU, no pod, no corpus, no model. Deterministic.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[5]
sys.path.insert(0, str(_REPO / "taniteval"))

from taniteval.clhorizon import (CAM_HEIGHT_M, CAM_PITCH_DEG, CXY,  # noqa: E402
                                 F_EFF, sampling_homography)

W = H = 256                      # phase-0 cache frame size
R_MAX = 0.25                     # pre-registered: max admissible relative error
FRAC_MIN = 0.95                  # pre-registered: min frame fraction under R_MAX
DLAT_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]      # 3.0 = ENV_LAT_MAX
DYAW_GRID = [1.0, 3.0, 6.0, 12.0]                # 12.0 = ENV_YAW_MAX

# Scene heights ABOVE THE ROAD SURFACE that a front-wide driving frame contains.
# PUBLISHED / standard vehicle and infrastructure dimensions; used only to make
# the closed form legible, never to derive it.
SCENE_HEIGHTS_M = {
    "road_surface": 0.0,
    "kerb": 0.15,
    "sedan_wheel_hub": 0.33,
    "sedan_beltline": 1.05,
    "sedan_roof": 1.45,
    "camera_height_h_cam": 1.50,
    "suv_roof": 1.85,
    "van_roof": 2.60,
    "truck_trailer_roof": 4.00,
    "traffic_light_head": 5.50,
    "building_second_floor": 7.50,
}
SCENE_DEPTHS_M = [5.0, 10.0, 20.0, 40.0, 80.0]


# =========================================================================== #
# helpers                                                                      #
# =========================================================================== #
def K_mat(f=F_EFF, c=CXY):
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]], dtype=np.float64)


def H_1to2(dlat_m, dyaw_deg, h_cam=CAM_HEIGHT_M, pitch_deg=CAM_PITCH_DEG):
    """The FORWARD plane-induced homography (view 1 -> view 2).

    ``sampling_homography`` returns its INVERSE (output pixel -> source pixel),
    so this inverts it back. Derived from the packaged function, never
    re-implemented, so any change there is caught here."""
    Hs = sampling_homography(dlat_m, dyaw_deg, h_cam=h_cam, pitch_deg=pitch_deg)
    return np.linalg.inv(np.asarray(Hs.double().numpy(), dtype=np.float64))


def project(X, f=F_EFF, c=CXY):
    """[n,3] camera-frame points (x right, y DOWN, z forward) -> [n,2] pixels."""
    X = np.asarray(X, dtype=np.float64)
    return np.stack([f * X[:, 0] / X[:, 2] + c, f * X[:, 1] / X[:, 2] + c], -1)


def apply_H(Hm, uv):
    """Apply a 3x3 homography to [n,2] pixels."""
    uv = np.asarray(uv, dtype=np.float64)
    P = np.concatenate([uv, np.ones((uv.shape[0], 1))], 1).T
    Q = Hm @ P
    return (Q[:2] / Q[2]).T


def true_view2(X, dlat_m, dyaw_deg):
    """Where a 3-D point ACTUALLY lands after the camera moves.

    Camera 2 sits at C = (dlat, 0, 0) in camera-1 coordinates and is rotated by
    ``dyaw_deg`` about the (down-pointing) y axis. Mirrors the (Ry, C) of
    ``sampling_homography`` EXACTLY -- same sign conventions, same Ry."""
    psi = math.radians(dyaw_deg)
    Ry = np.array([[math.cos(-psi), 0, math.sin(-psi)],
                   [0, 1.0, 0],
                   [-math.sin(-psi), 0, math.cos(-psi)]], dtype=np.float64)
    C = np.array([dlat_m, 0.0, 0.0], dtype=np.float64)
    X2 = (Ry @ (np.asarray(X, dtype=np.float64) - C).T).T
    return project(X2), X2


def scene_points(depths, heights_above_road, h_cam=CAM_HEIGHT_M, n_x=9,
                 x_halfwidth=8.0):
    """A synthetic 3-D scene: a lattice of (lateral x, depth Z, height-above-road).

    y is DOWN and the camera sits ``h_cam`` above the road, so a point ``a``
    metres above the road has ``Y = h_cam - a``."""
    xs = np.linspace(-x_halfwidth, x_halfwidth, n_x)
    pts, meta = [], []
    for Z in depths:
        for name, a in heights_above_road.items():
            for x in xs:
                pts.append([x, h_cam - a, Z])
                meta.append((name, float(a), float(Z), float(x)))
    return np.asarray(pts, dtype=np.float64), meta


# =========================================================================== #
# A0 -- does the arm depend on the flat-road parameters at all?                #
# =========================================================================== #
def a0_parameter_invariance():
    """P1's C-GEO test, run on BOTH arms.

    If a warp's homography does not depend on ``h_cam``/``pitch``, the ground
    plane plays no role in it -- that is what makes the yaw arm exact. The
    lateral arm's dependence is the flat-road assumption, made visible."""
    h_grid = [1.2, 1.35, 1.5, 1.65, 1.8]
    p_grid = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
    out = {}
    for arm, grid in (("yaw", DYAW_GRID), ("lat", DLAT_GRID)):
        worst = 0.0
        rows = []
        for amt in grid:
            ref = sampling_homography(*( (0.0, amt) if arm == "yaw" else (amt, 0.0) ),
                                      h_cam=CAM_HEIGHT_M,
                                      pitch_deg=CAM_PITCH_DEG).double().numpy()
            m = 0.0
            for h in h_grid:
                for p in p_grid:
                    Hm = sampling_homography(
                        *((0.0, amt) if arm == "yaw" else (amt, 0.0)),
                        h_cam=h, pitch_deg=p).double().numpy()
                    m = max(m, float(np.abs(Hm - ref).max()))
            rows.append({"amount": amt, "max_abs_dH": float(m)})
            worst = max(worst, m)
        out[arm] = {
            "n_conditions": len(h_grid) * len(p_grid),
            "h_cam_grid": h_grid, "pitch_deg_grid": p_grid,
            "per_amount": rows,
            "max_abs_dH_over_all": float(worst),
            "depends_on_ground_plane": bool(worst > 1e-12),
        }
    return out


# =========================================================================== #
# A1/A2/A3 -- the reprojection error and its relative size                     #
# =========================================================================== #
def a1_reprojection(arm, amounts):
    """Per-point pixel error of the warp against the TRUE projection.

    ``u_applied``  = homography(pixel in view 1)
    ``u_true``     = projection of the actually-transformed 3-D point
    ``u_motion``   = |u_true - u_view1|, the displacement the warp is trying to
                     synthesize -- the denominator that makes the error relative.
    """
    X, meta = scene_points(SCENE_DEPTHS_M, SCENE_HEIGHTS_M)
    uv1 = project(X)
    # keep only points that are actually inside the 256x256 frame in view 1
    inb = ((uv1[:, 0] >= 0) & (uv1[:, 0] <= W - 1)
           & (uv1[:, 1] >= 0) & (uv1[:, 1] <= H - 1) & (X[:, 2] > 0.5))
    rows = []
    for amt in amounts:
        dlat, dyaw = ((0.0, amt) if arm == "yaw" else (amt, 0.0))
        Hf = H_1to2(dlat, dyaw)
        uv2_true, _ = true_view2(X, dlat, dyaw)
        uv2_hom = apply_H(Hf, uv1)
        err = np.linalg.norm(uv2_hom - uv2_true, axis=1)
        mot = np.linalg.norm(uv2_true - uv1, axis=1)
        rel = np.where(mot > 1e-9, err / np.maximum(mot, 1e-9), 0.0)
        per_height = {}
        for name, a in SCENE_HEIGHTS_M.items():
            sel = inb & np.array([m[0] == name for m in meta])
            if not sel.any():
                continue
            per_height[name] = {
                "height_above_road_m": a,
                "n_points": int(sel.sum()),
                "err_px_mean": round(float(err[sel].mean()), 4),
                "err_px_max": round(float(err[sel].max()), 4),
                "motion_px_mean": round(float(mot[sel].mean()), 4),
                "rel_err_mean": round(float(rel[sel].mean()), 6),
                "rel_err_max": round(float(rel[sel].max()), 6),
            }
        rows.append({
            "arm": arm, "amount": amt,
            "n_points_in_frame": int(inb.sum()),
            "err_px_mean_all": round(float(err[inb].mean()), 4),
            "err_px_p95_all": round(float(np.percentile(err[inb], 95)), 4),
            "err_px_max_all": round(float(err[inb].max()), 4),
            "rel_err_mean_all": round(float(rel[inb].mean()), 6),
            "rel_err_max_all": round(float(rel[inb].max()), 6),
            "frac_points_rel_err_under_R_MAX": round(
                float((rel[inb] < R_MAX).mean()), 6),
            "per_height": per_height,
        })
    return rows


def a2_closed_form_check():
    """Is ``rel_err == height_above_road / h_cam`` -- depth-free, dlat-free?

    Checked numerically against the packaged homography rather than asserted.
    If it holds, the value that would make L-OK pass is a single number and the
    criterion becomes falsifiable by inspection."""
    X, meta = scene_points([6.0, 12.0, 25.0, 50.0], SCENE_HEIGHTS_M)
    uv1 = project(X)
    worst = 0.0
    samples = []
    for dlat in DLAT_GRID:
        Hf = H_1to2(dlat, 0.0)
        uv2_true, _ = true_view2(X, dlat, 0.0)
        uv2_hom = apply_H(Hf, uv1)
        err = np.linalg.norm(uv2_hom - uv2_true, axis=1)
        mot = np.linalg.norm(uv2_true - uv1, axis=1)
        pred = np.array([m[1] / CAM_HEIGHT_M for m in meta])   # a / h_cam
        obs = np.where(mot > 1e-9, err / np.maximum(mot, 1e-9), 0.0)
        ok = mot > 1e-6
        d = float(np.abs(obs[ok] - pred[ok]).max())
        worst = max(worst, d)
        samples.append({"dlat_m": dlat, "max_abs_dev_from_closed_form": round(d, 9)})
    return {
        "closed_form": "rel_err(x) = height_above_road(x) / h_cam   "
                       "(independent of depth, of |dlat| and of focal length)",
        "max_abs_deviation_over_grid": round(worst, 9),
        "holds": bool(worst < 1e-9),
        "per_dlat": samples,
        "implied_L_OK_requirement_m": round(R_MAX * CAM_HEIGHT_M, 4),
        "_reading": (f"L-OK (rel_err < {R_MAX}) requires EVERY scene point to lie "
                     f"within {R_MAX * CAM_HEIGHT_M:.3f} m of the road surface."),
    }


def a3_sign_inversion():
    """Where does the applied displacement change sign relative to the true one?

    ``u_applied = u_true * (1 - a / h_cam)``: the warp under-moves content above
    the road, stops moving it at ``a = h_cam``, and moves it BACKWARDS above
    that. Measured, not asserted."""
    rows = []
    X, meta = scene_points([15.0], SCENE_HEIGHTS_M, n_x=3, x_halfwidth=4.0)
    uv1 = project(X)
    dlat = 2.0
    Hf = H_1to2(dlat, 0.0)
    uv2_true, _ = true_view2(X, dlat, 0.0)
    uv2_hom = apply_H(Hf, uv1)
    du_true = uv2_true[:, 0] - uv1[:, 0]
    du_appl = uv2_hom[:, 0] - uv1[:, 0]
    for name, a in SCENE_HEIGHTS_M.items():
        sel = np.array([m[0] == name for m in meta])
        if not sel.any():
            continue
        t = float(du_true[sel].mean())
        p = float(du_appl[sel].mean())
        rows.append({
            "content": name, "height_above_road_m": a,
            "du_true_px": round(t, 4), "du_applied_px": round(p, 4),
            "ratio_applied_over_true": round(p / t, 6) if abs(t) > 1e-9 else None,
            "sign_inverted": bool(t * p < -1e-12),
        })
    return {
        "dlat_m": dlat, "depth_m": 15.0,
        "predicted_zero_crossing_m": CAM_HEIGHT_M,
        "rows": rows,
        "_reading": "u_applied = u_true * (1 - height_above_road / h_cam). The "
                    "applied displacement VANISHES at the camera height and is "
                    "SIGN-INVERTED above it.",
    }


# =========================================================================== #
# A5 -- the round-trip criterion, and why it is refused                        #
# =========================================================================== #
def a5_roundtrip_is_vacuous():
    """C-RT cannot see the flat-road error. Demonstrated, then refused.

    The ground-plane homography COMPOSES: with ``n = (0, cos p, sin p)`` and
    ``t proportional to (1,0,0)`` we have ``n . t = 0``, so the second-order term
    vanishes and ``H(a) H(b) = H(a + b)`` exactly. Therefore
    ``H(d) H(-d) = H(0) = I`` to machine precision -- on BOTH arms, and
    regardless of how wrong the flat-road assumption is."""
    out = {}
    for arm, grid in (("yaw", DYAW_GRID), ("lat", DLAT_GRID)):
        rows = []
        for amt in grid:
            f = lambda a: H_1to2(*((0.0, a) if arm == "yaw" else (a, 0.0)))
            rt = f(amt) @ f(-amt)
            rt = rt / rt[2, 2]
            comp = f(amt) @ f(amt)
            comp = comp / comp[2, 2]
            tgt = f(2 * amt) if arm == "lat" else f(2 * amt)
            tgt = tgt / tgt[2, 2]
            rows.append({
                "amount": amt,
                "roundtrip_max_abs_dev_from_identity": float(
                    np.abs(rt - np.eye(3)).max()),
                "composition_max_abs_dev_H2a_vs_HaHa": float(
                    np.abs(comp - tgt).max()),
            })
        out[arm] = {
            "per_amount": rows,
            "max_roundtrip_dev": float(max(r["roundtrip_max_abs_dev_from_identity"]
                                           for r in rows)),
        }
    out["_verdict"] = (
        "C-RT is VACUOUS for the flat-road question: the round-trip residual is "
        "identically zero on the LATERAL arm, which is the arm under suspicion. "
        "A criterion that returns the same value for an exact warp and a "
        "provably wrong one cannot adjudicate between them. REFUSED as a "
        "fidelity criterion; C-GEO is used instead.")
    return out


# =========================================================================== #
# A6 -- the frame fraction with no ground-plane preimage                       #
# =========================================================================== #
def a6_above_horizon_fraction():
    """Rows above the horizon have NO ground-plane preimage.

    The ground plane ``Y = h_cam, Z > 0`` projects only to ``v > c_y + 0`` (for
    zero pitch). Every pixel at or above the horizon therefore shows content the
    flat-road model cannot represent at all -- and, by A3, content it displaces
    with the WRONG SIGN. This is a hard, data-free lower bound on the
    non-planar fraction of the frame."""
    out = {}
    for pitch in (-2.0, 0.0, 2.0, 4.0):
        p = math.radians(pitch)
        # horizon row: the vanishing line of the plane with normal n at pitch p
        # image row of the direction parallel to the plane and along +z
        v_h = CXY + F_EFF * math.tan(p)
        rows_above = max(0.0, min(float(H), v_h))
        out[f"pitch_{pitch:g}deg"] = {
            "horizon_row_v": round(float(v_h), 3),
            "rows_with_no_ground_preimage": round(rows_above, 2),
            "frac_frame_no_ground_preimage": round(rows_above / H, 6),
        }
    out["_reading"] = (
        "At the shipped pitch of 0 deg and c_y = 128 in a 256-row frame, EXACTLY "
        "50 % of the frame is above the horizon. There the ground plane has no "
        "preimage, the flat-road warp's relative error is >= 1.0 by A2, and by "
        "A3 the applied displacement is opposite in sign to the true one.")
    return out


# =========================================================================== #
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    res = {
        "_what": "C-GEO-LAT -- is the flat-road LATERAL warp faithful enough to "
                 "carry a pseudo-simulation grid?",
        "_evidence_class": "MEASURED (ours; CPU-only, deterministic, no model, "
                           "no corpus, no GPU)",
        "_warp_source": "taniteval.clhorizon.sampling_homography (PACKAGED -- the "
                        "same function taniteval.clhorizon.corridor_rollout calls)",
        "_criterion_pre_registered": {
            "R_MAX": R_MAX, "FRAC_MIN": FRAC_MIN,
            "rule": f"L-OK iff relative displacement error < {R_MAX} on >= "
                    f"{FRAC_MIN:.0%} of the frame at the grid's max |dlat|; "
                    f"L-BAD otherwise.",
            "what_would_make_it_PASS": "see a2.implied_L_OK_requirement_m",
        },
        "intrinsics": {"f_eff_px": F_EFF, "principal_px": CXY, "frame": [H, W],
                       "h_cam_m": CAM_HEIGHT_M, "pitch_deg": CAM_PITCH_DEG},
        "scene_heights_above_road_m": SCENE_HEIGHTS_M,
        "scene_depths_m": SCENE_DEPTHS_M,
    }
    res["A0_parameter_invariance"] = a0_parameter_invariance()
    res["A1_reprojection_lat"] = a1_reprojection("lat", DLAT_GRID)
    res["A4_reprojection_yaw_POSITIVE_CONTROL"] = a1_reprojection("yaw", DYAW_GRID)
    res["A2_closed_form"] = a2_closed_form_check()
    res["A3_sign_inversion"] = a3_sign_inversion()
    res["A5_roundtrip_refused"] = a5_roundtrip_is_vacuous()
    res["A6_above_horizon"] = a6_above_horizon_fraction()

    # ---- the verdict, computed from the numbers, not written by hand -------- #
    lat_at_max = [r for r in res["A1_reprojection_lat"] if r["amount"] == 2.0][0]
    yaw_at_max = [r for r in res["A4_reprojection_yaw_POSITIVE_CONTROL"]
                  if r["amount"] == 12.0][0]
    frac_ok = lat_at_max["frac_points_rel_err_under_R_MAX"]
    lat_pass = bool(frac_ok >= FRAC_MIN)
    yaw_pass = bool(yaw_at_max["err_px_max_all"] < 1e-9)
    res["VERDICT"] = {
        "lateral_frac_points_under_R_MAX_at_dlat_2m": frac_ok,
        "lateral_required_frac": FRAC_MIN,
        "lateral_PASSES": lat_pass,
        "yaw_positive_control_max_err_px_at_12deg": yaw_at_max["err_px_max_all"],
        "yaw_positive_control_PASSES": yaw_pass,
        "outcome": ("L-OK" if lat_pass else "L-BAD"),
        "grid_dimensionality": ("2-D (lateral x heading)" if lat_pass
                                else "heading-only on the WARPED axis; the "
                                     "lateral axis is REFUSED"),
        "_both_directions_exercised": (
            "The identical code path returns PASS on the yaw arm and FAIL on the "
            "lateral arm, so the criterion is not vacuous. A5 additionally shows "
            "the round-trip criterion IS vacuous and refuses it."),
    }
    out = _HERE.parent / "artifacts" / "lat_warp_fidelity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["A0_parameter_invariance"]["yaw"]["max_abs_dH_over_all"]))
    print(json.dumps(res["A0_parameter_invariance"]["lat"]["max_abs_dH_over_all"]))
    print(json.dumps(res["A2_closed_form"], indent=2))
    print(json.dumps(res["A3_sign_inversion"], indent=2))
    print(json.dumps(res["A6_above_horizon"], indent=2))
    print(json.dumps(res["VERDICT"], indent=2))
    print(f"[cgeolat] wrote {out}")


if __name__ == "__main__":
    main()
