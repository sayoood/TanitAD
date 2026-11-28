"""WP-D -- how much IMAGE supports each range band of the 10 cm map? (ANALYTIC, zero GPU)

The refcv7 map is decoded at 0.1 m over x in [0, 100) m from the trunk's stride-8 map of a 416 x 1024 CYLINDRICAL
frame (config.json: refcv6_perception.frame = cylindrical, f_ref 488.92398517830253 px; map_hires.trunk_tap stride 8,
s8_hw 52 x 128). This script prices the geometry of that choice per 20 m band, from the projection the lift itself uses
(stack/tanitad/models/bev_lift.py docstring; rig_projection.project_cam_to_frame):

    col = (W-1)/2 + f * atan2(x_cam, z_cam)        (column LINEAR in azimuth -- cylindrical, NOT pinhole)
    row = (H-1)/2 + f * y_cam / hypot(x_cam, z_cam)

For a ground point straight ahead at horizontal distance d with the camera h metres above the road and zero pitch:
row offset below the horizon = f * h / d. The per-clip PITCH moves the horizon (1 deg = f * pi/180 = 8.5 px); it is a
STATED simplification here (the lift uses each clip's own extrinsic), so every row count carries a +-1 deg column.

Outputs raw/geom_bound.json: per band and camera height (1.20 / 1.45 / 1.69 m = the run's extrinsics census range and a
mid value, quoted in the diagnostics RESULT sec. 1.4): image rows spanned, stride-4 / -8 / -16 feature rows spanned,
ground depth per image row, lateral metres per pixel and per stride-8 column, painted-line width in pixels, and how many
10 cm label rows each stride-8 feature row must explain. Pure arithmetic; controls are asserted in-line.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

F = 488.92398517830253          # f_ref, px (config.json refcv6_perception.frame.f_ref)
H, W = 416, 1024                # image_hw (config.json)
BANDS = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 100.0)]
HEIGHTS = [1.20, 1.45, 1.69]    # camera height census range + mid (diagnostics RESULT sec. 1.4)
STRIDES = [1, 4, 8, 16]
LINE_W = [0.12, 0.15, 0.20]     # painted line widths, metres
CELL = 0.1                      # label cell, metres
HALF_ROWS = (H - 1) / 2.0       # rows below the horizon at the bottom edge (zero pitch)
RIGB_STRIP = 43                 # the trainer's --equalize-bottom-rows (config argv)
PKG = Path(__file__).resolve().parents[1]


def row_off(d: float, h: float, pitch_deg: float = 0.0) -> float:
    """rows below the FRAME CENTRE of a ground point at horizontal distance d (straight ahead). Zero pitch: exactly
    f * h / d (cylindrical row = f * tan(elevation)). A positive pitch (camera tilted DOWN by pitch_deg) moves every
    ground point UP the frame by ~f * pitch (small-angle); band spans are invariant to it except where the image
    bottom clips the band."""
    return F * h / d - F * math.radians(pitch_deg)


def d_min(h: float, bottom_rows: float, pitch_deg: float = 0.0) -> float:
    """nearest visible ground distance straight ahead for the given number of usable rows below the horizon."""
    off = bottom_rows + F * math.radians(pitch_deg)
    return F * h / off


def main():
    # controls that must read known values
    assert abs(row_off(F * 1.45 / 100.0, 1.45) - 100.0) < 1e-9          # a point placed at offset 100 reads 100
    assert abs((F * math.radians(1.0)) - 8.533) < 1e-3                  # 1 deg of pitch = 8.53 px
    assert abs(2 * math.degrees((W / 2) / F) - 120.0) < 0.01            # cylindrical: W/2/f = 60 deg half-FOV
    out = {"inputs": {"f_ref_px": F, "image_hw": [H, W], "projection": "cylindrical", "bands_m": BANDS,
                      "camera_heights_m": HEIGHTS, "line_widths_m": LINE_W, "label_cell_m": CELL,
                      "pitch_assumption": "zero pitch; +-1 deg shown as a sensitivity (8.53 px of horizon shift)",
                      "evidence": "ANALYTIC (arithmetic on MEASURED config stamps)"},
           "bands": [], "near_limit": {}}
    for h in HEIGHTS:
        out["near_limit"][f"h{h:.2f}"] = {
            "d_min_full_frame_m": d_min(h, HALF_ROWS),
            "d_min_rigB_strip_m": d_min(h, HALF_ROWS - RIGB_STRIP)}
    for (x0, x1) in BANDS:
        rec = {"band": f"{int(x0)}_{int(x1)}", "by_height": {}}
        xc = 0.5 * (x0 + x1)
        for h in HEIGHTS:
            per = {}
            for pitch in (-1.0, 0.0, 1.0):
                top = row_off(x1, h, pitch)
                lo = min(HALF_ROWS, row_off(max(x0, 1e-6), h, pitch)) if x0 > 0 else HALF_ROWS
                rows = max(0.0, lo - top)
                per[f"pitch{pitch:+.0f}"] = {"image_rows": rows,
                                             **{f"s{s}_feature_rows": rows / s for s in STRIDES if s > 1}}
            rows0 = per["pitch+0"]["image_rows"]
            label_rows = (x1 - x0) / CELL
            depth_per_px = xc * xc / (F * h)                  # d(row)/dd = -f h / d^2
            lat_per_px = xc / F                               # azimuth-linear columns, small azimuth
            rec["by_height"][f"h{h:.2f}"] = {
                **per,
                "label_rows_10cm": label_rows,
                "label_rows_per_s8_feature_row": label_rows / max(rows0 / 8.0, 1e-9),
                "ground_depth_per_image_row_m_at_centre": depth_per_px,
                "ground_depth_per_s8_row_m_at_centre": 8 * depth_per_px,
                "lateral_m_per_px_at_centre": lat_per_px,
                "lateral_m_per_s8_col_at_centre": 8 * lat_per_px,
                "lateral_m_per_s4_col_at_centre": 4 * lat_per_px,
                **{f"line_{w:.2f}m_width_px": w / lat_per_px for w in LINE_W},
                **{f"line_{w:.2f}m_width_s8_cols": w / lat_per_px / 8 for w in LINE_W},
            }
        out["bands"].append(rec)
    p = PKG / "raw" / "geom_bound.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    # compact table (h = 1.45 m, zero pitch)
    print(f"{'band':8s} {'img rows':>8s} {'s4 rows':>7s} {'s8 rows':>7s} {'(-1/+1 deg s8)':>15s} "
          f"{'lbl rows/s8 row':>15s} {'depth/px m':>10s} {'lat/px m':>8s} {'lat/s8col m':>11s} {'0.15m line px':>13s}")
    for rec in out["bands"]:
        b = rec["by_height"]["h1.45"]
        print(f"{rec['band']:8s} {b['pitch+0']['image_rows']:8.1f} {b['pitch+0']['s4_feature_rows']:7.2f} "
              f"{b['pitch+0']['s8_feature_rows']:7.2f} {b['pitch-1']['s8_feature_rows']:6.2f}/{b['pitch+1']['s8_feature_rows']:<6.2f}  "
              f"{b['label_rows_per_s8_feature_row']:15.0f} {b['ground_depth_per_image_row_m_at_centre']:10.2f} "
              f"{b['lateral_m_per_px_at_centre']:8.3f} {b['lateral_m_per_s8_col_at_centre']:11.2f} "
              f"{b['line_0.15m_width_px']:13.2f}")
    for k, v in out["near_limit"].items():
        print(k, {kk: round(vv, 2) for kk, vv in v.items()})
    print("wrote", p)


if __name__ == "__main__":
    main()
