"""Pixel extent of typical targets in refcv6's camera frame, and where published pixel-size cut-offs fall.

ANALYTIC. Frame from D:/refcv6_eval_kit/ckpt_final/config.json (`refcv6_perception.frame`): 416 x 1024
CYLINDRICAL, f_ref 488.924 px/rad. On a cylindrical projection a vertical extent H at horizontal distance rho
spans f*H/rho px and a horizontal extent W (seen broadside, near the boresight) spans ~f*W/rho px.
Object sizes are the GT-validation per-class medians (EvalFlyWheel 2026-09-26-refcv6-gt-validation RESULT §2):
person 0.66 m wide x 1.66 m tall; automobile 1.92 m wide x 1.59 m tall (rear view).
Published cut-offs: KITTI Easy >= 40 px, Moderate/Hard >= 25 px box height (cvlibs.net eval_object);
CityPersons 'Reasonable' >= 50 px (arXiv:1702.05693 §3.3), also the only pedestrians it TRAINS on (§3.4).
The perception map is stride 16 (26 x 64 tokens, config `refcv6_perception.fmap_s16_hw`).
"""
import json, sys
F, STRIDE = 488.92398517830253, 16
objs = {"person": (0.66, 1.66), "automobile_rear": (1.92, 1.59)}
out = {"doc": __doc__.strip(), "per_range": {}, "range_at_height_px": {}}
for name, (w, h) in objs.items():
    out["per_range"][name] = {f"{r} m": {"h_px": round(F * h / r, 1), "w_px": round(F * w / r, 1),
                                         "h_tokens": round(F * h / r / STRIDE, 2),
                                         "w_tokens": round(F * w / r / STRIDE, 2)}
                              for r in (5, 10, 20, 30, 40, 50, 60)}
    out["range_at_height_px"][name] = {f"{px} px": round(F * h / px, 1) for px in (25, 40, 50)}
json.dump(out, sys.stdout, indent=1)
