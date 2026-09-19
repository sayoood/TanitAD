#!/usr/bin/env python3
"""Build the self-contained interactive calibration bench.

WHY THIS EXISTS
---------------
The ground plane cannot separate focal length from camera height
(``scale_calib.py``: lateral sees ``h`` alone, longitudinal sees only the product
``f*h``), and on the 2026-08-08 recording every automatic estimator either
declined or disagreed:

    plane_calib  height 1.691 m  DECLINED (spread +/-0.56)
    scale_calib  f*h 1608.1      DECLINED (spread 62%)
    FOE pitch -2.93 deg  vs  lane-VP horizon row 523.4 px  ->  2.29 deg apart

So the remaining move is to put a human in the loop with live visual feedback and
let them set the parameters against the actual road, then feed those values back
as declared priors. This builds that instrument as ONE self-contained HTML file:
video, trajectory, geometry and UI inlined, no server and no network (the artifact
CSP blocks every external host).

⚠️ The output is a VALIDATION instrument, not a measurement one. Whatever the
operator dials in is a prior, and the pipeline records it as
``OPERATOR OVERRIDE ... (not measured)``.

THE ONE INVARIANT THAT MATTERS
------------------------------
The page's JavaScript projection must be the pipeline's projection, or the
operator calibrates against a lie. It mirrors ``camera.py`` exactly:

    R_cv = Rz(roll) @ Rx(pitch) @ Ry(yaw) @ R_CV_NOMINAL
    t_v  = [longitudinal, lateral, height]
    pc   = (p_vehicle - t_v) @ R_cv.T
    u    = fx*pc.x/pc.z + cx      v = fy*pc.y/pc.z + cy

VERIFIED 2026-08-08 against `trajlib.camera` at yaw -7.01 / pitch -0.64 / roll 0,
h 1.21, lon 2.10, lat -0.12, f 1478.3: horizon row 523.487 in BOTH, and four
ground points agreeing to the last printed digit (e.g. (10, +1.75) -> 413.30,
758.07). Re-run that check after touching the projection in either place.

Usage
-----
    python build_calib_bench.py --run-dir <pipeline out>/<recording> \
        --video <raw Camera/*.mp4> --out calib_bench.html

The raw video is re-encoded small (854x480, CRF 36 ~ 5.5 MB) because it is
base64-inlined and base64 costs 4/3; the published artifact ceiling is 16 MB.
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import subprocess
import tempfile

import numpy as np

TIME_GRID = np.arange(-2.0, 5.01, 0.5)      # window kept, seconds


def trajectory_payload(run_dir: pathlib.Path) -> dict:
    """Compact the per-frame windows: cm as ints, 15 points instead of 81."""
    recs = [json.loads(l) for l in (run_dir / "trajectory.jsonl").open()]
    out = {"fps": 29.922, "t": [round(float(x), 2) for x in TIME_GRID],
           "n": len(recs), "frames": []}
    for r in recs:
        t = np.asarray(r["t"], dtype=float)
        out["frames"].append({
            "p": round(float(r["pts_s"]), 3),
            "s": round(float(r["speed_ms"]), 2),
            "w": round(float(r["steer_wheel_deg"]), 1),
            "x": [int(round(v * 100)) for v in np.interp(TIME_GRID, t, r["x"])],
            "y": [int(round(v * 100)) for v in np.interp(TIME_GRID, t, r["y"])],
        })
    return out


def shrink_video(src: pathlib.Path, width: int, crf: int) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        dst = pathlib.Path(td) / "small.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
             "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-an", str(dst)], check=True)
        return dst.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=pathlib.Path,
                    help="a pipeline output dir containing trajectory.jsonl")
    ap.add_argument("--video", required=True, type=pathlib.Path,
                    help="the RAW Camera/*.mp4 (not overlay.mp4)")
    ap.add_argument("--template", type=pathlib.Path,
                    default=pathlib.Path(__file__).with_name("calib_bench.html.tmpl"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("calib_bench.html"))
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--crf", type=int, default=36)
    ap.add_argument("--max-mb", type=float, default=16.0)
    a = ap.parse_args()

    html = a.template.read_text(encoding="utf-8")
    traj = json.dumps(trajectory_payload(a.run_dir), separators=(",", ":"))
    b64 = base64.b64encode(shrink_video(a.video, a.width, a.crf)).decode("ascii")

    page = html.replace("__TRAJ_JSON__", traj).replace("__VIDEO_B64__", b64)
    for token in ("__TRAJ_JSON__", "__VIDEO_B64__"):
        if token in page:                     # a silent miss ships a broken page
            raise SystemExit(f"placeholder {token} was not substituted")
    a.out.write_text(page, encoding="utf-8")

    mb = a.out.stat().st_size / 1048576
    print(f"{a.out}  {mb:.2f} MB  ({len(traj)/1048576:.2f} MB trajectory, "
          f"{len(b64)/1048576:.2f} MB video base64)")
    if mb > a.max_mb:
        raise SystemExit(f"OVER the {a.max_mb} MB artifact ceiling — raise --crf "
                         f"or lower --width")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
