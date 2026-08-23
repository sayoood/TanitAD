"""D-016 R1 measured-numbers report: naive square-crop vs pinhole-rectify, per camera.

Pure geometry on grounded real intrinsics -- no download, no GPU, ~1 s on CPU.
Prints the table the research note + data card consume.
"""

from __future__ import annotations

import json

from calib_r1 import (COMMA2K19_INTR, PANDASET_FRONT_INTR, PinholeIntrinsics,
                      pinhole_geometry_report)

# Udacity CH2 (nominal ~1590 px @ 640x480 after the challenge resize; a NARROW-FOV
# pinhole that also over-runs the square crop on a short frame). Illustrative only:
# real per-clip intrinsics land at ingest. Shown to demonstrate the same failure mode.
UDACITY_LIKE = PinholeIntrinsics(fx=1590.0, fy=1590.0, cx=320.0, cy=240.0,
                                 width=640, height=480)

CAMERAS = {
    "comma2k19 (F_REF reference, ~pinhole)": COMMA2K19_INTR,
    "PandaSet front (fx=1970, k1=-0.589)": PANDASET_FRONT_INTR,
    "Udacity-like (fx=1590 @ 640x480)": UDACITY_LIKE,
}

if __name__ == "__main__":
    print("=" * 78)
    print("D-016 R1 pinhole-rectify -- naive square-crop vs rectify (f_ref=266)")
    print("=" * 78)
    rows = {}
    for name, intr in CAMERAS.items():
        rep = pinhole_geometry_report(intr)
        rows[name] = rep
        n = rep["naive_square_crop"]
        print(f"\n{name}")
        print(f"  camera FOV            : H={rep['camera_hfov_deg']} deg  "
              f"V={rep['camera_vfov_deg']} deg   (canonical H={rep['canonical_hfov_deg']} deg)")
        print(f"  NAIVE square crop     : crop {n['used_crop_px']}px "
              f"(ideal {n['ideal_crop_px']}px) -> f_eff {n['achieved_feff_px']}px  "
              f"height_clamped={n['height_clamped']}  drop_in={n['drop_in']}")
        print(f"  RECTIFY (this work)   : f_eff {rep['rectify_feff_px']}px  "
              f"drop_in={rep['rectify_drop_in']}  observed_frac={rep['rectify_observed_frac']}")
        print(f"  distortion corrected  : k1={rep['k1']}  "
              f"max edge displacement {rep['max_distort_px_at_edge']}px")
    print("\n" + "-" * 78)
    print("JSON:")
    print(json.dumps(rows, indent=1))
