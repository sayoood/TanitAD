"""GeoCalib integration contract + fixed-HFOV fallback (scale-up 2026-07-25).

The scale-up harvest (`harvest_scaleup.py`) accepts per-video intrinsics from the
parallel GeoCalib agent (`…/incoming/2026-07-25-geocalib/geocalib_intrinsics.py`).
As of harvest launch that deliverable had NOT landed, so the harvest ran with the
fixed-HFOV fallback (100 deg), recording `geometry_source: "fixed"` and `hfov_used_deg`
per pointer. Because the geometry is recorded per pointer, a fixed-HFOV run is fully
**re-runnable with GeoCalib later** — no new videos are harvested; the same public
pointers are re-decoded with per-video intrinsics.

CONTRACT — `geocalib_intrinsics.json` (drop at /workspace/tmp/yt_scaleup/geocalib_intrinsics.json):
    { "<video_id>": {"hfov_deg": <float>}, ... }          # preferred (resolution-independent)
  or
    { "<video_id>": {"focal_px": <float>}, ... }          # converted using yt-dlp info['width']
harvest_scaleup.py auto-detects the file and uses per-video geometry when a video_id is present,
else the fixed-HFOV fallback for that video. `manifest.geometry` reports which was used.

To PRODUCE the JSON once the GeoCalib agent lands: run its estimator over one representative
frame per harvested video (video_ids are in pointers.jsonl) and emit the mapping above. Then:

    # re-decode the SAME pointers with per-video intrinsics (no re-harvest), re-pseudo-label, re-P4
    # (a re-decode helper is intentionally left to the GeoCalib intake, which owns the estimator).

This module is a documentation/adapter stub; the live integration is in harvest_scaleup.py
(`--geocalib-json` + `per_video_hfov`). Kept separate so the GeoCalib intake has one obvious
contract to satisfy.
"""
from __future__ import annotations
import json, math, os, sys


def poll(path="/workspace/tmp/yt_scaleup/geocalib_intrinsics.json") -> bool:
    """True if the GeoCalib intrinsics JSON has landed."""
    return os.path.exists(path)


def focal_to_hfov(focal_px: float, width_px: float) -> float:
    return math.degrees(2.0 * math.atan(width_px / (2.0 * focal_px)))


def validate(path: str) -> dict:
    """Sanity-check a geocalib_intrinsics.json: every record has hfov_deg or focal_px,
    hfov in a plausible dashcam band [40, 140]."""
    m = json.loads(open(path).read())
    bad = []
    for vid, rec in m.items():
        h = rec.get("hfov_deg") if isinstance(rec, dict) else None
        if h is None and isinstance(rec, dict) and rec.get("focal_px"):
            h = None  # needs width at decode; accepted
        if h is not None and not (40.0 <= float(h) <= 140.0):
            bad.append((vid, h))
    return {"n": len(m), "n_out_of_band_hfov": len(bad), "examples": bad[:5]}


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "/workspace/tmp/yt_scaleup/geocalib_intrinsics.json"
    print(f"geocalib intrinsics present: {poll(p)}")
    if poll(p):
        print(validate(p))
