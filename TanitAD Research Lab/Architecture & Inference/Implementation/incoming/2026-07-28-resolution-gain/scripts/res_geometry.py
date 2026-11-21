"""RESOLUTION-GAIN — the geometry ledger, and the CORRECTION to the brief's px/deg figure.

Deterministic. No data, no GPU, no model — every number here follows from `calib.py`'s own two
azimuth conventions and the frame parameters, so it is checkable by hand.

The brief states that `256x640 @ 120 deg` has "essentially the same on-axis figure (4.686)" as
today's deployed 4.643 px/deg. **4.686 belongs to a DIFFERENT frame** — the FOV audit's
`256x640 @ 100 deg PINHOLE letterbox` (`f_eff` 268.5). The 120 deg CYLINDRICAL frame v5 is actually
being built at has `f_ref` 305.5775 and therefore **5.3333 px/deg, uniformly**. This script computes
both so the correction is arithmetic rather than assertion.

usage:  python res_geometry.py <out_json>
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.environ.get("TANITAD_STACK",
                                  r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack"))
from tanitad.data.calib import F_REF, CanonicalFrame                    # noqa: E402


def cyl(h, w, hfov):
    return CanonicalFrame(height=h, width=w, f_ref=(w / 2.0) / math.radians(hfov / 2.0),
                          projection="cylindrical")


def main():
    out = sys.argv[1]
    today_half = math.atan(128.0 / F_REF)
    today = {
        "frame": "256x256 deployed", "projection": "pinhole", "f_ref": F_REF,
        "hfov_deg": 2 * math.degrees(today_half),
        "px_per_deg_on_axis": F_REF * math.pi / 180.0,
        "px_per_deg_at_edge": F_REF * math.pi / 180.0 / math.cos(today_half) ** 2,
        "px_per_deg_average_width_over_hfov": 256.0 / (2 * math.degrees(today_half)),
        "tokens": 256}
    v5, up = cyl(256, 640, 120.0), cyl(384, 960, 120.0)
    alt = cyl(320, 800, 120.0)
    rows = {"today_deployed": today}
    for nm, fr in (("v5_256x640_120cyl", v5), ("up_384x960_120cyl", up),
                   ("alt_320x800_120cyl", alt)):
        rows[nm] = {
            "frame": fr.tag(), "projection": "cylindrical", "f_ref": fr.f_ref,
            "hfov_deg": 2 * math.degrees(fr.width / 2.0 / fr.f_ref),
            "vfov_deg": 2 * math.degrees(math.atan((fr.height / 2.0) / fr.f_ref)),
            "px_per_deg_on_axis": fr.f_ref * math.pi / 180.0,
            "px_per_deg_at_edge": fr.f_ref * math.pi / 180.0,      # uniform by construction
            "px_per_deg_average_width_over_hfov": fr.width / (2 * math.degrees(
                fr.width / 2.0 / fr.f_ref)),
            "tokens": (fr.height // 16) * (fr.width // 16)}
    # the FOV audit's 100 deg PINHOLE letterbox, i.e. where the brief's 4.686 actually comes from
    f100 = (640 / 2.0) / math.tan(math.radians(50.0))
    rows["fov_audit_256x640_100deg_PINHOLE"] = {
        "frame": "256x640 @100 deg pinhole", "projection": "pinhole", "f_ref": f100,
        "hfov_deg": 100.0, "px_per_deg_on_axis": f100 * math.pi / 180.0,
        "px_per_deg_at_edge": f100 * math.pi / 180.0 / math.cos(math.radians(50.0)) ** 2,
        "tokens": 640,
        "note": "THIS is the 4.686 the brief quotes. It is NOT the frame v5 is built at."}

    t_on = today["px_per_deg_on_axis"]
    led = {
        "class": "MEASURED (deterministic arithmetic on calib.py's own conventions)",
        "frames": rows,
        "ladder_rungs_ladder_A": {},
        "correction_to_brief": {
            "brief_says": "256x640 @120 deg has essentially the same on-axis figure (4.686) as "
                          "today's 4.643",
            "measured": {
                "today_on_axis": round(t_on, 6),
                "v5_120cyl_uniform": round(rows["v5_256x640_120cyl"]["px_per_deg_on_axis"], 6),
                "ratio_v5_over_today_on_axis": round(
                    rows["v5_256x640_120cyl"]["px_per_deg_on_axis"] / t_on, 6),
                "ratio_v5_over_today_at_edge": round(
                    rows["v5_256x640_120cyl"]["px_per_deg_on_axis"]
                    / today["px_per_deg_at_edge"], 6),
                "where_4p686_comes_from": round(
                    rows["fov_audit_256x640_100deg_PINHOLE"]["px_per_deg_on_axis"], 6)},
            "verdict": "the QUALITATIVE claim survives (v5 is 1.149x today's on-axis and 0.933x "
                       "today's edge density — genuinely comparable); the NUMBER 4.686 does not, "
                       "it belongs to the 100 deg pinhole letterbox",
            "consequence": "the ladder carries an explicit rung k = "
                           + str(round(rows["v5_256x640_120cyl"]["px_per_deg_on_axis"] / t_on, 6))
                           + " (D_today) that lands the wide frame EXACTLY on today's deployed "
                             "on-axis angular resolution — that is the calibration point"},
    }
    v5_px = rows["v5_256x640_120cyl"]["px_per_deg_on_axis"]
    for nm, k in (("V5_640", 1.0), ("D_today", v5_px / t_on), ("D_1p5", 1.5), ("D_2", 2.0),
                  ("D_3", 3.0), ("D_6", 6.0)):
        led["ladder_rungs_ladder_A"][nm] = {
            "k": round(k, 6), "px_per_deg": round(v5_px / k, 6),
            "equivalent_width_px_at_120deg": round(640.0 / k, 2),
            "ratio_to_todays_on_axis": round((v5_px / k) / t_on, 4)}
    json.dump(led, open(out, "w"), indent=2)
    print(json.dumps(led, indent=2))


if __name__ == "__main__":
    main()
