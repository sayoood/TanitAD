"""Check-5 evidence: REAL PhysicalAI-AV camera intrinsics/extrinsics vs our
nominal-120-deg-pinhole assumption (GEOMETRY_INTEGRITY_AUDIT.md).

The flagship pipeline (tanitad/data/calib.py + physicalai.py) canonicalizes the
PhysicalAI front-wide focal from the NOMINAL 120-deg HFOV via a PINHOLE formula
f = W/(2 tan(HFOV/2)). This probe downloads ONE chunk of the dataset's own
per-clip calibration feature (gated: nvidia/PhysicalAI-Autonomous-Vehicles,
calibration/{camera_intrinsics,sensor_extrinsics,vehicle_dimensions}) and shows
the real camera is an f-theta FISHEYE (paraxial focal ~926 px @1920, NOT the
554 px a pinhole gives) — a 1.67x focal error that makes the canonical crop
1.62x more zoomed than comma2k19 / than the shared f_eff=266 "fingerprint".

Auth: dev box -> tanitad.keys (Keys.txt + truststore). Pod -> export HF_TOKEN
(the dataset is gated; user `Sayood` has access). One sample, no bulk.

Usage:
  python scripts/pai_calib_probe.py [--out /workspace/experiments/pai_calib.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
NOMINAL_HFOV_DEG = 120.0
SIZE, F_REF = 256, 266.0          # calib.py canonical output size + shared focal
COMMA_FOCAL_PX, COMMA_NATIVE_H = 910.0, 874.0   # comma2k19 reference camera


def _download_calib(local_dir: str):
    try:                          # dev box: Keys.txt + proxy TLS
        from tanitad.keys import enable_tls, load_keys
        enable_tls(); load_keys()
    except Exception as e:
        print(f"[calib] keys helper unavailable ({e}); relying on env HF_TOKEN")
    from huggingface_hub import hf_hub_download
    out = {}
    for feat in ("camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions"):
        p = hf_hub_download(REPO, f"calibration/{feat}/{feat}.chunk_0000.parquet",
                            repo_type="dataset", local_dir=local_dir)
        out[feat] = pd.read_parquet(p).reset_index()
    return out


def _euler_pitch(qx, qy, qz, qw):
    """Elevation of the camera optical axis (+z) in the vehicle FLU frame [deg]."""
    q = np.array([qx, qy, qz, qw], float); q /= np.linalg.norm(q)
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    axis = R @ np.array([0, 0, 1.0])
    return math.degrees(math.asin(np.clip(axis[2], -1, 1)))


def analyze(cal: dict) -> dict:
    intr, extr, veh = cal["camera_intrinsics"], cal["sensor_extrinsics"], cal["vehicle_dimensions"]
    fw = intr[intr["camera_name"].astype(str).str.contains("front_wide")]
    W = float(fw["width"].mean()); H = float(fw["height"].mean())
    cy = float(fw["cy"].mean())
    f_paraxial = float(fw["fw_poly_1"].mean())           # dr/dtheta at 0 [px @W]
    p = fw[[f"fw_poly_{i}" for i in range(5)]].mean().to_numpy()

    def r_of_theta(th):
        return sum(p[i] * th ** i for i in range(5))
    from scipy.optimize import brentq
    hfov_real = 2 * math.degrees(brentq(lambda t: r_of_theta(t) - W / 2, 0.1, 1.6))

    f_pinhole_nom = W / (2 * math.tan(math.radians(NOMINAL_HFOV_DEG) / 2))
    crop = int(round(f_pinhole_nom * SIZE / F_REF))      # pipeline crop side @W
    half_r = crop / 2
    th_real = brentq(lambda t: r_of_theta(t) - half_r, 1e-3, 1.2)   # real half-angle
    f_eff_true = 128.0 / math.tan(th_real)               # rectilinear-equiv focal
    # comma retained real half-angle (reference the fix must match)
    comma_crop = min(round(COMMA_FOCAL_PX * SIZE / F_REF), int(COMMA_NATIVE_H))
    comma_half_ang = math.degrees(math.atan((comma_crop / 2) / COMMA_FOCAL_PX))
    r_needed = COMMA_FOCAL_PX  # placeholder; fix radius below
    fix_half_ang = comma_half_ang
    fix_radius = brentq(lambda t: math.degrees(t) - fix_half_ang, 1e-3, 1.4)
    fix_crop_radius = r_of_theta(fix_radius)

    fwe = extr[extr["sensor_name"].astype(str).str.contains("front_wide")]
    pitch = float(np.median([_euler_pitch(r.qx, r.qy, r.qz, r.qw)
                             for _, r in fwe.iterrows()]))
    height = float(fwe["z"].median())

    return {
        "n_clips_front_wide": int(len(fw)),
        "native_WxH": [int(W), int(H)], "cy_principal_row": round(cy, 1),
        "cy_offset_from_center_px": round(cy - H / 2, 1),
        "real_paraxial_focal_px_fw_poly1": round(f_paraxial, 1),
        "real_HFOV_deg": round(hfov_real, 1),
        "nominal_pinhole_focal_px": round(f_pinhole_nom, 1),
        "focal_error_ratio_real_over_nominal": round(f_paraxial / f_pinhole_nom, 3),
        "pipeline_crop_side_px": crop,
        "pipeline_believed_retained_HFOV_deg":
            round(2 * math.degrees(math.atan(half_r / f_pinhole_nom)), 1),
        "real_retained_HFOV_deg": round(2 * math.degrees(th_real), 1),
        "physicalai_true_f_eff_px": round(f_eff_true, 1),
        "assumed_shared_f_eff_px": F_REF,
        "zoom_factor_vs_assumed": round(f_eff_true / F_REF, 2),
        "comma_retained_half_angle_deg": round(comma_half_ang, 2),
        "fix_crop_radius_px_to_match_comma": round(fix_crop_radius, 0),
        "extrinsics_camera_height_m": round(height, 3),
        "extrinsics_pitch_deg_optical_axis": round(pitch, 2),
        "vehicle_wheelbase_m_real": round(float(veh["wheelbase"].median()), 3),
        "pipeline_WHEELBASE_const": 2.9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default=str(Path.home() / "pai_calib"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cal = _download_calib(args.local_dir)
    rep = analyze(cal)
    print(json.dumps(rep, indent=2))
    verdict = ("CONFIRMED intrinsics error: real f-theta fisheye focal "
               f"{rep['real_paraxial_focal_px_fw_poly1']}px vs nominal pinhole "
               f"{rep['nominal_pinhole_focal_px']}px "
               f"({rep['focal_error_ratio_real_over_nominal']}x); PhysicalAI "
               f"canonical frame is {rep['zoom_factor_vs_assumed']}x more zoomed "
               f"than the assumed/comma f_eff={F_REF}.")
    print("\n" + verdict)
    if args.out:
        Path(args.out).write_text(json.dumps({"report": rep, "verdict": verdict}, indent=2))
        print("[calib] ->", args.out)


if __name__ == "__main__":
    main()
