#!/usr/bin/env python3
"""camera_smoke_r8.py -- refcv8 NAVSIM feasibility smoke, CAMERA INPUT ONLY (CPU, no model, no GPU).

EvalFlyWheel, 2026-10-19.  Run in the TanitAD venv with the refcv8 launch tree as the code source:

    set TANITAD_REPO=C:/lgs/Lfb47a58376 & set PYTHONDONTWRITEBYTECODE=1
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe camera_smoke_r8.py --out <pkg>/raw/camera_smoke

What it does (every step asserts on CONTENT, not on exit codes):
  S1  the model's frame contract: refcv8 argv `--image-hw 416 1024` -> `trunk_shapes.FRAME_416x1024`
      (CYLINDRICAL, f_ref = 305.5774907364391 * 1024/640 = 488.924 px, hfov 120 deg).  The brief's
      "256x640, f_ref 305.577" is the refcv4b-era frame; it is NOT what refcv8 consumes.
  S2  ONE NAVSIM warmup_two_stage scene is re-stitched FROM ITS RAW JPGs (CAM_L0 / F0 / R0) through the
      refcv6/7 suite's own stitch (`frames416.build_one`, E2's `build_frames.build_map/sample`), and the
      result's sha256[:16] is compared with the bank the refcv7 NAVSIM suite fed its model
      (`tanitad-caches/refcv6-navsim-20260923/warmup_two_stage`).  Equal => the bank is reproducible
      from raw pixels by the current launch tree (a reproduction / determinism control, NOT an
      independent geometry check -- that is S3).
  S3  an INDEPENDENT geometry check that shares no code with the stitch: for the rebuilt frame, project
      CAM_F0's principal point (pinhole, cx, cy from `cam_intrinsic`) through the camera's own rotation
      into the virtual cylinder and assert the column lands at the analytic u = (W-1)/2 + az*f_ref
      within 1.5 px (az = CAM_F0's boresight azimuth relative to the virtual boresight = 0 by
      construction, so the expected column is the centre column 511.5).
  S4  the t0 frame of ONE navtest token from the navtest 416 bank (sha re-checked against index.json).
  S5  the t0 frame of ONE PhysicalAI eval139 clip from the refcv6/7/8 eval cache (`*.v2ep.pt`; its
      stored `frame` dict must equal FRAME_416x1024 -- the same canonical frame on both sides).
  S6  one side-by-side PNG (PhysicalAI / NAVSIM warmup rebuilt / NAVSIM navtest banked) with azimuth
      ticks at -60, -30, 0, +30, +60 deg (analytic, u = (W-1)/2 + az*f_ref) and the unobserved band
      marked, plus `camera_smoke.json` with per-panel statistics (observed fraction, mean px,
      per-channel means, black rows top/bottom, zero-pixel fraction) and the ANALYTIC horizon row of each
      camera (row = (H-1)/2 - tan(pitch_down) * f_ref; PhysicalAI pitch from the clip extrinsics, NAVSIM from the
      stitch rotation M).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

REPO = os.environ.get("TANITAD_REPO", "C:/Lfb_missing")
if not os.path.isdir(os.path.join(REPO, "stack", "tanitad")):
    raise SystemExit(f"TANITAD_REPO={REPO!r} has no stack/tanitad")
sys.path.insert(0, os.path.join(REPO, "stack"))
F416_PATH = os.path.join(REPO, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming", "2026-09-23-refcv6-standard-tests",
                         "navsim", "code", "frames416.py")

NAVSIM_ROOT = pathlib.Path("D:/Archive/devbox-C/navsim/data/openscene")
WARMUP_BANK = pathlib.Path("C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/warmup_two_stage")
NAVTEST_BANK = pathlib.Path("D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank")
PAI_EVAL = pathlib.Path("D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139")
EXTR_JSON = "D:/refcv8_eval_kit/data/refcv8_train4572_eval139_extrinsics.json"
ARGV_JSON = os.path.join(REPO, "stack", "ops", "runs.d", "refcv8-r101-s0.argv.json")


def sha16(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def load_mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def panel_stats(img: np.ndarray, src: np.ndarray | None) -> dict:
    """img u8 [H,W,3]."""
    rowmax = img.max(axis=(1, 2))
    nz = np.nonzero(rowmax > 0)[0]
    top_black = int(nz[0]) if len(nz) else img.shape[0]
    bot_black = int(img.shape[0] - 1 - nz[-1]) if len(nz) else img.shape[0]
    out = {"mean_px": round(float(img.mean()), 3),
           "mean_rgb": [round(float(img[..., k].mean()), 3) for k in range(3)],
           "black_rows_top": top_black, "black_rows_bottom": bot_black,
           "zero_px_frac": round(float((img.max(-1) == 0).mean()), 4)}
    if src is not None:
        out["observed_frac"] = round(float((src >= 0).mean()), 4)
        out["src_counts"] = {int(k): int(v) for k, v in zip(*np.unique(src, return_counts=True))}
    return out


def main() -> int:
    global F_REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-key", default=None, help="a warmup stage-2 scene_token (default: first in the bank)")
    ap.add_argument("--navtest-token", default=None)
    ap.add_argument("--pai-clip", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"script": os.path.abspath(__file__), "repo": REPO, "t_start": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---- S1: the frame contract, from the argv and the code -------------------------------------------------------- #
    argv = json.load(open(ARGV_JSON, encoding="utf-8"))["argv"]
    i = argv.index("--image-hw")
    hw = (int(argv[i + 1]), int(argv[i + 2]))
    from tanitad.models.trunk_shapes import FRAME_416x1024
    from tanitad.data.calib import PHYSICALAI_WIDE120_256x640, cylindrical_rays
    F_REF = float(FRAME_416x1024.f_ref)
    s1 = {"argv_image_hw": hw, "frame": FRAME_416x1024.to_dict(), "hfov_deg": FRAME_416x1024.hfov_deg,
          "vfov_deg": FRAME_416x1024.vfov_deg, "f_ref_256x640": PHYSICALAI_WIDE120_256x640.f_ref,
          "f_ref_ratio": F_REF / PHYSICALAI_WIDE120_256x640.f_ref}
    assert hw == (FRAME_416x1024.height, FRAME_416x1024.width), (hw, FRAME_416x1024)
    assert FRAME_416x1024.projection == "cylindrical"
    assert abs(FRAME_416x1024.hfov_deg - 120.0) < 1e-9
    assert abs(s1["f_ref_ratio"] - 1024 / 640) < 1e-12
    # analytic column of an azimuth: cylindrical => u linear in azimuth
    xc, yc, zc = (t.double().numpy() for t in cylindrical_rays(FRAME_416x1024))
    az_row = np.degrees(np.arctan2(xc[208], zc[208]))
    u_expect = (FRAME_416x1024.width - 1) / 2 + np.radians(az_row) * F_REF
    s1["cyl_linear_max_abs_px"] = float(np.abs(u_expect - np.arange(FRAME_416x1024.width)).max())
    # cylindrical_rays is computed in float32 (MEASURED 4.1e-5 px on the first run): tolerance 1e-3 px, i.e. linear
    assert s1["cyl_linear_max_abs_px"] < 1e-3, s1
    rep["S1_frame_contract"] = s1
    print("[S1]", json.dumps(s1)[:400], flush=True)

    # ---- S2: one warmup scene re-stitched from raw jpgs, compared with the refcv7 suite's bank --------------------- #
    F4 = load_mod("frames416_smoke", F416_PATH)
    F4.configure(FRAME_416x1024)
    import pandas as pd
    prov = pd.read_parquet(WARMUP_BANK / "frames_provenance.parquet")
    key = a.warmup_key or sorted(prov.scene_token)[0]
    row = prov[prov.scene_token == key].iloc[0]
    pk = NAVSIM_ROOT / "warmup_two_stage" / "synthetic_scene_pickles" / f"{key}.pkl"
    sc = F4.E2BF.load(pk)
    cams = [fr["camera_dict"] for fr in sc["frames"]]
    bank_rep = json.load(open(WARMUP_BANK / "BUILD_REPORT.json", encoding="utf-8"))
    keep = int(bank_rep["keep"])
    t0 = time.time()
    arr, src, rk, obs, rrec = F4.build_one(cams, NAVSIM_ROOT / "warmup_two_stage" / "sensor_blobs", {}, keep)
    t_stitch = time.time() - t0
    banked = np.load(WARMUP_BANK / "frames" / f"{key}.npy")
    s2 = {"scene_token": key, "log_name": sc["scene_metadata"]["log_name"], "keep": keep,
          "rebuilt_shape": list(arr.shape), "rebuilt_sha16": sha16(arr), "bank_sha16": str(row.sha256),
          "bank_file_sha16": sha16(banked), "rig_key": rk, "bank_rig_key": str(row.rig_key),
          "observed_frac_min": round(obs, 4), "stitch_seconds_cpu": round(t_stitch, 2),
          "max_abs_px_vs_bank": int(np.abs(arr.astype(np.int16) - banked.astype(np.int16)).max()),
          "f0_sensor2lidar_translation_m": rrec["f0_sensor2lidar_translation"],
          "f0_intrinsic": rrec["f0_intrinsic"]}
    s2["REPRODUCED"] = (s2["rebuilt_sha16"] == s2["bank_sha16"] == s2["bank_file_sha16"])
    rep["S2_warmup_restitch"] = s2
    print("[S2]", {k: v for k, v in s2.items() if k not in ("f0_intrinsic",)}, flush=True)

    # ---- S3: independent geometry check (no stitch code): CAM_F0 principal point -> virtual cylinder column ---------- #
    f0 = F4.E2BF.cam_get(cams[-1], "cam_f0")
    K = np.asarray(f0["cam_intrinsic"], np.float64)
    R_f0 = np.asarray(f0["sensor2lidar_rotation"], np.float64)
    M = F4.virtual_rotation(cams[-1])           # columns right/down/fwd of the virtual camera in the lidar frame
    # ray through F0's principal point, in F0's camera frame = +z; in lidar frame = R_f0 @ z; in virtual frame:
    d_virt = M.T @ (R_f0 @ np.array([0.0, 0.0, 1.0]))
    az = math.atan2(d_virt[0], d_virt[2])
    u_pp = (FRAME_416x1024.width - 1) / 2 + az * F_REF
    # and a ray 30 deg to the RIGHT of the virtual boresight must land at u = 511.5 + 0.5236 * f_ref = 767.5
    s3 = {"f0_principal_point_px": [K[0, 2], K[1, 2]], "f0_fx_px": K[0, 0],
          "f0_hfov_deg_pinhole": round(2 * math.degrees(math.atan(1920 / 2 / K[0, 0])), 2),
          "f0_vfov_deg_pinhole": round(2 * math.degrees(math.atan(1080 / 2 / K[1, 1])), 2),
          "virtual_az_of_f0_boresight_deg": round(math.degrees(az), 6), "column_of_f0_boresight": round(u_pp, 4),
          "expected_column": (FRAME_416x1024.width - 1) / 2,
          "virtual_pitch_down_deg": round(math.degrees(math.asin(-M[2, 2])), 3),
          "u_at_plus30deg_expected": round((FRAME_416x1024.width - 1) / 2 + math.radians(30) * F_REF, 3)}
    s3["PASS"] = abs(s3["column_of_f0_boresight"] - s3["expected_column"]) < 1.5
    # ⚠️ S3 above is close to a STRUCTURAL IDENTITY (the virtual boresight is DEFINED as F0's boresight), so it can
    # only catch a wrong axis convention in M.  S3b is the discriminating check: an INDEPENDENT remap written here from
    # the docstring convention (column linear in azimuth, pinhole per column; plain pinhole for CAM_F0, NO distortion,
    # nearest-neighbour) of the raw CAM_F0 jpg into the central window |az| < 15 deg, |el| < 8 deg, compared with the
    # stitched t0 frame; CONTROL: the same remap shifted by +2 deg azimuth must be clearly worse.
    from PIL import Image as _Im
    p_f0 = NAVSIM_ROOT / "warmup_two_stage" / "sensor_blobs" / str(f0["data_path"])
    raw = np.asarray(_Im.open(p_f0).convert("RGB")).astype(np.float32)
    Wd, Hd = FRAME_416x1024.width, FRAME_416x1024.height
    uu, vv = np.meshgrid(np.arange(Wd, dtype=np.float64), np.arange(Hd, dtype=np.float64))
    stitched = arr[-1].astype(np.float32)

    def indep_mad(shift_deg: float) -> tuple:
        phi = (uu - (Wd - 1) / 2) / F_REF + math.radians(shift_deg)
        yn = (vv - (Hd - 1) / 2) / F_REF
        ray = np.stack([np.sin(phi), yn, np.cos(phi)], -1)              # virtual cam: x right, y down, z fwd
        d_l = ray @ M.T                                                  # -> lidar frame (M columns = axes)
        d_c = d_l @ R_f0                                                 # -> CAM_F0 frame (R_f0^T d)
        uc = K[0, 0] * d_c[..., 0] / d_c[..., 2] + K[0, 2]
        vc = K[1, 1] * d_c[..., 1] / d_c[..., 2] + K[1, 2]
        win = (np.abs((uu - (Wd - 1) / 2) / F_REF) < math.radians(15)) & (np.abs(yn) < math.tan(math.radians(8)))
        ok = win & (d_c[..., 2] > 0) & (uc >= 0) & (uc <= raw.shape[1] - 1) & (vc >= 0) & (vc <= raw.shape[0] - 1)
        ok &= (src == 1)                                                 # pixels the stitch took from CAM_F0
        rem = raw[np.rint(vc[ok]).astype(int), np.rint(uc[ok]).astype(int)]
        return float(np.abs(rem - stitched[ok]).mean()), int(ok.sum())

    mad0, n0 = indep_mad(0.0)
    mad2, n2 = indep_mad(2.0)
    s3b = {"window": "|az|<15deg, |el|<8deg, src==CAM_F0", "n_px": n0, "mad_px_independent_remap": round(mad0, 3),
           "control_shift_2deg_mad": round(mad2, 3), "n_px_control": n2, "distortion_ignored": True}
    s3b["PASS"] = bool(mad0 < 12.0 and mad2 > 2.0 * mad0)
    rep["S3b_independent_remap"] = s3b
    s3["PASS"] = bool(s3["PASS"] and s3b["PASS"])
    rep["S3_independent_geometry"] = s3
    print("[S3]", s3, "\n[S3b]", s3b, flush=True)

    # ---- S4: one navtest token from the 416 bank -------------------------------------------------------------------- #
    idx = json.load(open(NAVTEST_BANK / "index.json", encoding="utf-8"))
    assert (idx["frame"]["h"], idx["frame"]["w"], idx["frame"]["projection"]) == (416, 1024, "cylindrical")
    assert abs(idx["frame"]["f_ref"] - F_REF) < 1e-9
    tok = a.navtest_token or sorted(idx["tokens"])[0]
    r = idx["tokens"][tok]
    mm = np.load(NAVTEST_BANK / r["file"], mmap_mode="r")
    nt = np.ascontiguousarray(mm[r["rows"]])
    del mm
    s4 = {"token": tok, "log_name": r["log_name"], "rig_key": r["rig_key"], "shape": list(nt.shape),
          "sha16": sha16(nt), "index_sha16": r["sha256"], "observed_frac": r["observed_frac"]}
    s4["SHA_OK"] = s4["sha16"] == s4["index_sha16"]
    rep["S4_navtest_bank"] = s4
    print("[S4]", s4, flush=True)

    # ---- S5: one PhysicalAI eval139 frame ----------------------------------------------------------------------------- #
    import torch
    import torchvision.io as tvio
    clips = sorted(PAI_EVAL.glob("*.v2ep.pt"))
    cp = PAI_EVAL / a.pai_clip if a.pai_clip else clips[0]
    d = torch.load(cp, map_location="cpu", weights_only=False)
    fr = d.get("frame")
    lens = d["jpeg_len"].to(torch.int64)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    n = len(lens)
    j = n // 2
    codec = str(d.get("codec", "jpeg"))
    dec = tvio.decode_png if codec == "png" else tvio.decode_jpeg
    pai = dec(d["jpeg_buf"][int(offs[j]):int(offs[j + 1])], mode=tvio.ImageReadMode.RGB).permute(1, 2, 0).numpy()
    s5 = {"clip": cp.name, "frame_dict": fr, "codec": codec, "n_frames": n, "frame_index": j,
          "shape": list(pai.shape)}
    s5["SAME_CANONICAL_FRAME"] = (isinstance(fr, dict) and int(fr["height"]) == 416 and int(fr["width"]) == 1024
                                  and abs(float(fr["f_ref"]) - F_REF) < 1e-6
                                  and fr.get("projection", "pinhole") == "cylindrical")
    rep["S5_physicalai_eval139"] = s5
    del d
    print("[S5]", s5, flush=True)

    # ---- S6: the side-by-side PNG + stats ----------------------------------------------------------------------------- #
    from PIL import Image, ImageDraw
    W, H = FRAME_416x1024.width, FRAME_416x1024.height
    ext = json.load(open(EXTR_JSON, encoding="utf-8")).get(cp.name.split(".")[0])
    pai_pitch_down = float(ext["_read"]["pitch_down_deg"]) if ext else None
    rep["S5_physicalai_eval139"]["camera_extrinsics_read"] = ext["_read"] if ext else "ABSENT"
    nav_pitch_down = s3["virtual_pitch_down_deg"]

    def hz_row(pd_deg):
        return None if pd_deg is None else (H - 1) / 2 - math.tan(math.radians(pd_deg)) * F_REF
    panels = [("PhysicalAI-AV eval139 (training domain): " + cp.name[:13], pai, None, hz_row(pai_pitch_down)),
              ("NAVSIM warmup (re-stitched from raw L0/F0/R0): " + key, arr[-1], src, hz_row(nav_pitch_down)),
              ("NAVSIM navtest (banked 416 stitch, t0): " + tok, nt[-1], None, None)]
    rep["S6_panel_stats"] = {}
    tiles = []
    for title, img, s, hz in panels:
        st = panel_stats(img, s)
        st["analytic_horizon_row"] = None if hz is None else round(hz, 1)
        rep["S6_panel_stats"][title.split(":")[0]] = st
        im = Image.fromarray(img).convert("RGB")
        dr = ImageDraw.Draw(im)
        for azd in (-60, -30, 0, 30, 60):
            u = (W - 1) / 2 + math.radians(azd) * F_REF
            u = min(max(u, 0), W - 1)
            dr.line([(u, 0), (u, 14)], fill=(255, 255, 0), width=2)
            dr.line([(u, H - 15), (u, H - 1)], fill=(255, 255, 0), width=2)
            dr.text((min(u + 3, W - 30), 16), f"{azd:+d}", fill=(255, 255, 0))
        hc = (H - 1) / 2
        dr.line([(0, hc), (18, hc)], fill=(0, 255, 255), width=2)
        dr.line([(W - 19, hc), (W - 1, hc)], fill=(0, 255, 255), width=2)
        if hz is not None:
            dr.line([(W // 3, hz), (2 * W // 3, hz)], fill=(255, 0, 255), width=2)
        band = Image.new("RGB", (W, 22), (20, 20, 20))
        ImageDraw.Draw(band).text((6, 5), title + f"   mean_px {st['mean_px']}  black rows top/bot "
                                  f"{st['black_rows_top']}/{st['black_rows_bottom']}", fill=(255, 255, 255))
        tiles += [band, im]
    canvas = Image.new("RGB", (W, sum(t.height for t in tiles) + 26), (0, 0, 0))
    y = 0
    for t in tiles:
        canvas.paste(t, (0, y))
        y += t.height
    ImageDraw.Draw(canvas).text((6, y + 6), "refcv8 input frame 416x1024 CYLINDRICAL f_ref 488.924 px (hfov 120 deg). "
                                "yellow ticks = azimuth -60/-30/0/+30/+60 deg (analytic); cyan = optical-axis row; "
                                "magenta = analytic horizon row from the camera pitch (flat road)", fill=(200, 200, 200))
    png = out / "camera_smoke_side_by_side.png"
    canvas.save(png)
    rep["png"] = str(png)
    rep["png_sha256"] = hashlib.sha256(png.read_bytes()).hexdigest()
    rep["t_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rep["VERDICT"] = ("PASS" if (s2["REPRODUCED"] and s3["PASS"] and s4["SHA_OK"] and s5["SAME_CANONICAL_FRAME"])
                      else "FAIL")
    (out / "camera_smoke.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("[DONE]", rep["VERDICT"], png, flush=True)
    return 0 if rep["VERDICT"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
