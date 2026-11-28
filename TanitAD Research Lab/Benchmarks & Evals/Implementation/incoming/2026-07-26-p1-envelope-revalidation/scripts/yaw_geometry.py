"""C-GEO -- the model-free geometry of the yaw warp. No model, no GPU, no renderer.

Criterion **C-GEO** from the pre-registration: on the yaw arm the warp model
contributes ZERO geometric error (a pure camera rotation induces `H = K R K^-1`,
exact for ANY scene geometry), so the only degradation mechanisms are

  1. finite field of view -- content rotating in from outside the frame does not
     exist and is FABRICATED by `padding_mode="border"` replication;
  2. resampling -- bilinear interpolation blur.

This script measures (1) exactly, and proves the "depth-independent" claim with
numbers rather than by quoting P1's docstring:

  A0  the yaw homography is INVARIANT to camera height and pitch (=> the
      flat-road assumption plays no role on this arm, unlike the lateral arm)
  A1  the yaw homography COMPOSES exactly, H(a)H(b) == H(a+b) (=> it is a true
      rotation, not an approximation that drifts)
  A2  fabricated-pixel fraction f(psi) -- the hard information bound
  A3  the FOV half-angle, which is where f crosses 0.5

⚠️ C13 GATE: f(psi) is monotone 0 -> 1 and reaches 0.5 at a geometric constant.
It CAN fail, and the value that makes it fail is stated before it is cited.

Uses the PACKAGED `taniteval.clhorizon.sampling_homography` -- the same function
the closed loop actually calls -- never a re-implementation.
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

from taniteval.clhorizon import (CXY, F_EFF, sampling_homography)  # noqa: E402

W = H = 256           # the phase-0 cache frame size
PSI_GRID = [0, 1, 2, 3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 25, 25.7,
            28, 30, 32, 35, 40, 45, 50, 60]


def source_coords(psi_deg):
    """Where each OUTPUT pixel samples FROM, in input-pixel coordinates.

    Mirrors `clhorizon.warp_batch` exactly: build the output pixel grid, push it
    through the sampling homography, de-homogenise.
    """
    Hm = sampling_homography(0.0, float(psi_deg))          # dlat = 0 => pure yaw
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    P = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3).T
    src = Hm @ P
    su = (src[0] / src[2]).reshape(H, W)
    sv = (src[1] / src[2]).reshape(H, W)
    return su.numpy(), sv.numpy()


def fabricated_fraction(psi_deg):
    """Fraction of the output frame with NO source information.

    `padding_mode="border"` means any sample falling outside the input frame is
    REPLICATED from the edge -- it is fabricated content, not observed content.
    """
    su, sv = source_coords(psi_deg)
    outside = (su < 0) | (su > W - 1) | (sv < 0) | (sv > H - 1)
    # column-wise too: the yaw warp fabricates a vertical BAND, so the column
    # fraction is the interpretable one.
    return {
        "psi_deg": float(psi_deg),
        "frac_pixels_fabricated": round(float(outside.mean()), 6),
        "frac_columns_fully_fabricated": round(
            float(outside.all(axis=0).mean()), 6),
        "n_columns_fully_fabricated": int(outside.all(axis=0).sum()),
        "max_source_col": round(float(np.nanmax(su)), 2),
        "min_source_col": round(float(np.nanmin(su)), 2),
    }


def main():
    out = {
        "_what": "C-GEO: model-free geometry of the P1 yaw warp",
        "_evidence_class": "MEASURED (ours) -- exact, closed-form; no model, "
                           "no GPU, no renderer, no scene data",
        "_intrinsics": {"f_eff_px": F_EFF, "principal_px": CXY,
                        "frame_px": [H, W],
                        "_source": "taniteval.clhorizon (packaged)"},
        "_c13_gate": {
            "criterion": "frac_pixels_fabricated",
            "fails_when": "grows without bound; >= 0.5 is a HARD ceiling "
                          "(the FOV half-angle -- a geometric constant, not a "
                          "chosen threshold)",
            "estimator_can_reach_it": True,
            "_why": "f(psi) is monotone 0 -> 1 by construction",
        },
    }

    # ---- A0: is the YAW homography really depth/plane independent? ---------- #
    ref = sampling_homography(0.0, 12.0, h_cam=1.5, pitch_deg=0.0)
    devs = []
    for h_cam in (0.8, 1.0, 1.5, 2.5, 3.0, 10.0):
        for pitch in (-5.0, -2.0, 0.0, 2.0, 5.0):
            Hm = sampling_homography(0.0, 12.0, h_cam=h_cam, pitch_deg=pitch)
            devs.append(float((Hm - ref).abs().max()))
    out["A0_yaw_is_depth_and_plane_independent"] = {
        "max_abs_deviation_over_h_cam_and_pitch": float(max(devs)),
        "n_conditions": len(devs),
        "_claim": "with dlat = 0 the plane term outer(t, n)/d VANISHES (t = 0), "
                  "so H = K @ Ry @ K^-1 exactly -- no ground plane, no depth. "
                  "The flat-road assumption that makes the LATERAL envelope an "
                  "optimistic bound plays NO role on the yaw arm.",
        "_verdict": "CONFIRMED" if max(devs) < 1e-12 else "REFUTED",
    }

    # ---- A1: does it compose exactly? (a true rotation does) --------------- #
    comps = []
    for a, b in ((5.0, 7.0), (10.0, 10.0), (12.0, 18.0), (15.0, 15.0),
                 (20.0, 25.0), (30.0, 30.0)):
        # sampling_homography returns cam2->cam1; composing the INVERSE-direction
        # maps means H(a) @ H(b) should equal H(a+b) up to scale.
        Ha, Hb, Hab = (sampling_homography(0.0, a), sampling_homography(0.0, b),
                       sampling_homography(0.0, a + b))
        prod = Hb @ Ha
        prod = prod / prod[2, 2]
        tgt = Hab / Hab[2, 2]
        comps.append({"a": a, "b": b,
                      "max_abs_err": float((prod - tgt).abs().max())})
    out["A1_composition_is_exact"] = {
        "pairs": comps,
        "max_abs_err_overall": max(c["max_abs_err"] for c in comps),
        "_verdict": "CONFIRMED" if max(
            c["max_abs_err"] for c in comps) < 1e-9 else "REFUTED",
    }

    # ---- A2/A3: the fabrication curve and the FOV constant ------------------ #
    half_fov = math.degrees(math.atan(CXY / F_EFF))
    out["A3_fov"] = {
        "half_fov_deg": round(half_fov, 4),
        "full_horizontal_fov_deg": round(2 * half_fov, 4),
        "_meaning": "at psi = half_fov the optical axis of the offset view "
                    "points at the edge of the real frame: HALF the output "
                    "frame is fabricated. This is the hard ceiling.",
    }
    out["A2_fabrication_curve"] = [fabricated_fraction(p) for p in PSI_GRID]

    dest = _HERE.parent / "artifacts" / "yaw_geometry.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}\n")

    a0 = out["A0_yaw_is_depth_and_plane_independent"]
    a1 = out["A1_composition_is_exact"]
    print(f"A0 depth/plane independence : max|dH| = {a0['max_abs_deviation_over_h_cam_and_pitch']:.3e} "
          f"over {a0['n_conditions']} (h_cam, pitch) conditions -> {a0['_verdict']}")
    print(f"A1 exact composition        : max|err| = {a1['max_abs_err_overall']:.3e} -> {a1['_verdict']}")
    print(f"A3 FOV                      : half = {half_fov:.2f} deg, full = {2*half_fov:.2f} deg\n")
    print(f"{'psi(deg)':>9} {'frac_px_fabricated':>19} {'cols_fabricated':>16}")
    for r in out["A2_fabrication_curve"]:
        mark = ""
        if abs(r["psi_deg"] - 12.0) < 1e-9:
            mark = "   <-- SHIPPED ENV_YAW_MAX"
        elif abs(r["psi_deg"] - 25.7) < 1e-9:
            mark = "   <-- FOV half-angle (50% fabricated)"
        print(f"{r['psi_deg']:>9.4g} {r['frac_pixels_fabricated']:>19.4f} "
              f"{r['n_columns_fully_fabricated']:>16d}{mark}")


if __name__ == "__main__":
    main()
