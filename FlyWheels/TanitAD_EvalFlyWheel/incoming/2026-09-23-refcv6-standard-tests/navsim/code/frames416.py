#!/usr/bin/env python3
"""refcv6's camera input from NavSim: the 3-camera stitch into ``FRAME_416x1024`` (TANITAD VENV, CPU).

⭐ WHAT THE FRAME IS (stated, never re-derived from a pinhole formula):
``tanitad.models.trunk_shapes.FRAME_416x1024`` = ``frame_for_width(1024, 416)`` =
``CanonicalFrame(416, 1024, f_ref = 305.5774907364391 * 1024 / 640 = 488.92398517830253,
projection="cylindrical")``. **Cylindrical**: the column is LINEAR IN AZIMUTH,
``phi = (u - (W-1)/2) / f_ref`` (``calib.cylindrical_rays``), so the horizontal field is
``2 * (W/2) / f_ref = 120.0000 deg``; the vertical axis is pinhole per column,
``y_n = (v - (H-1)/2) / f_ref`` -> ``2*atan((H/2)/f_ref) = 46.0921 deg``. It is the frame of
the ``physicalai-b1-w120-416x1024cyl`` cache refcv6 trains on (config.json ``image_hw
[416, 1024]``; ``trunk_shapes`` docstring: *"The builder's own manifest reports the same
46.09213171161337, independently computed"*). The pinhole formula would read 92.6 deg — the
CLAUDE.md cylindrical-FOV trap — and :func:`assert_frame` refuses anything but this object.

⭐ HOW NAVSIM'S CAMERAS BECOME THAT FRAME — THE GEOMETRY IS IMPORTED, NOT RE-DERIVED.
E2's ``2026-09-19-navsim-refcv4b-bridge/code/build_frames.py`` holds a VERBATIM copy of the
DataFlyWheel's stitch (``distort``, ``rig_key``, ``build_map``, ``sample``, ``build_scene``),
proven bit-exact against the DataFlyWheel's 256x640 bank (E2 control KB, 6/6). That module
builds its ray table for ``PHYSICALAI_WIDE120_256x640`` at import; :func:`configure` re-points
its three frame globals (``RAY_CAN``, ``H``, ``W``) at another ``CanonicalFrame`` and changes
NOTHING else. The only difference between the refcv4b stitch and this one is therefore the
output frame, and two controls prove it:

* **KB-256** — configured for 256x640, this module must still reproduce the DataFlyWheel bank
  bit-exactly (so the re-pointing did not change the geometry);
* **KG-416** — configured for 416x1024, every output pixel's ray must equal
  ``cylindrical_rays(FRAME_416x1024)`` (so the frame really is the model's).

Stitch semantics (the DataFlyWheel's, restated): a single VIRTUAL cylindrical camera whose
boresight is CAM_F0's boresight in the lidar/ego frame, rolled level (``right = fwd x up``,
``up = +z``); every output ray is sent through the three pinhole+Brown-distortion cameras
CAM_L0 / CAM_F0 / CAM_R0 and takes the camera where it lands with the largest ``cos``
(bilinear ``grid_sample``); rays no camera sees stay BLACK (``src == -1``). It is a
PURE-ROTATION reprojection (camera centres differ by ~0.3 m; ignored, as in every E2 number).
nuPlan cameras are 1920x1080 at f ~ 1545 px (vfov ~ 38.5 deg < 46.1 deg), so bands above
and below the camera field stay unobserved — the same as E2's 256x640 bank (89.29 % observed).

History timing: NavSim hands 4 history frames at 2 Hz (t0-1.5, -1.0, -0.5, 0 s); refcv6 reads
10 raw frames at 10 Hz (8 window rows x a 3-frame D-015 stack, t0-0.9 ... t0). This builder
stores the LAST ``--keep`` history frames per scene (index -1 = t0); which NavSim frame fills
which 10 Hz slot is the bridge's ``slot_sources`` (ST: every slot <- t0; NT: nearest in time).

Keys: stage-2 scenes by ``scene_token`` (the pickle stem, E2's convention); stage-1 / navtest
tokens by the SCORER token. Outputs under ``--out``: ``frames/<key>.npy`` u8 ``[K,416,1024,3]``,
``frames/<key>.src.npy`` int8 ``[416,1024]`` (camera index per pixel, -1 unobserved),
``frames_provenance.parquet`` (sha256[:16] of the array, rig key, observed fraction, mean px)
and ``BUILD_REPORT.json`` (+ the rig table the lift needs: per rig, CAM_F0's
``sensor2lidar_translation`` and the virtual camera's rotation).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo() -> str:
    env = os.environ.get("TANITAD_REPO")
    if env:
        return os.path.abspath(env)
    d = HERE
    for _ in range(12):
        if os.path.isdir(os.path.join(d, "stack", "tanitad")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("⛔ cannot find the repo root (a dir holding stack/tanitad); set TANITAD_REPO")


REPO = _find_repo()
if os.path.join(REPO, "stack") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "stack"))
E2_CODE = os.path.join(REPO, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming",
                       "2026-09-19-navsim-refcv4b-bridge", "code")


def _load(name: str, path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: E2's verbatim stitch, IMPORTED by path (never copied).
E2BF = _load("e2_build_frames", os.path.join(E2_CODE, "build_frames.py"))

from tanitad.data.calib import (PHYSICALAI_WIDE120_256x640,  # noqa: E402
                                CanonicalFrame, cylindrical_rays)
from tanitad.models.trunk_shapes import FRAME_416x1024  # noqa: E402

CAMS = E2BF.CAMS                      # ("cam_l0", "cam_f0", "cam_r0")


def assert_frame(frame: CanonicalFrame) -> None:
    """The model frame, and nothing else. A frame is not its pixel count."""
    if frame != FRAME_416x1024:
        raise SystemExit(f"⛔ frame {frame} is not trunk_shapes.FRAME_416x1024 {FRAME_416x1024}")
    if abs(frame.hfov_deg - 120.0) > 1e-9:
        raise SystemExit(f"⛔ hfov {frame.hfov_deg} != 120")


def configure(frame: CanonicalFrame) -> dict:
    """Re-point E2's stitch at ``frame``. Changes ONLY the three frame globals it reads
    (``RAY_CAN`` in ``build_map``; ``H``, ``W`` in ``sample``) — the same expression E2's
    module evaluates at import, with the frame swapped."""
    xc, yc, zc = (t.double().numpy() for t in cylindrical_rays(frame))
    ray = np.stack([xc, yc, zc], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    E2BF.XC, E2BF.YC, E2BF.ZC = xc, yc, zc
    E2BF.RAY_CAN = ray
    E2BF.H, E2BF.W = int(frame.height), int(frame.width)
    E2BF.FRAME = frame
    return {"frame": frame.to_dict(), "tag": frame.tag(), "hfov_deg": frame.hfov_deg,
            "vfov_deg": frame.vfov_deg}


def virtual_rotation(cd) -> np.ndarray:
    """``M`` (3x3, columns = the virtual camera's right / down / fwd axes in the LIDAR frame),
    the SAME expression as E2's ``build_map`` (which returns it as its third output; the unit
    test asserts equality). Needed by the BEV lift (``rig6.py``)."""
    f0 = E2BF.cam_get(cd, "cam_f0")
    r_f0 = np.asarray(f0["sensor2lidar_rotation"], dtype=np.float64)
    fwd = r_f0 @ np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    down /= np.linalg.norm(down)
    return np.stack([right, down, fwd / np.linalg.norm(fwd)], axis=1)


def rig_record(cd) -> dict:
    """What the lift needs from one rig: CAM_F0's mount (the virtual camera centre) and the
    virtual camera rotation, plus the intrinsics fingerprint E2's ``rig_key`` hashes."""
    f0 = E2BF.cam_get(cd, "cam_f0")
    return {"rig_key": E2BF.rig_key(cd),
            "f0_sensor2lidar_translation": np.asarray(f0["sensor2lidar_translation"],
                                                      dtype=np.float64).tolist(),
            "f0_sensor2lidar_rotation": np.asarray(f0["sensor2lidar_rotation"],
                                                   dtype=np.float64).tolist(),
            "virtual_R_cam_to_lidar": virtual_rotation(cd).tolist(),
            "f0_intrinsic": np.asarray(f0["cam_intrinsic"], dtype=np.float64).tolist()}


def build_one(frame_cams: list, sensor_root: pathlib.Path, cache: dict, keep: int):
    """The last ``keep`` history frames' camera dicts -> ``(u8 [keep,H,W,3], src, rig_key,
    observed_frac, rig_record)``. Uses E2's ``sample`` / ``build_map`` / ``rig_key``."""
    fc = list(frame_cams)[-keep:]
    out, srcs, obs, rk, rrec = [], None, [], None, None
    for cd in fc:
        paths = [sensor_root / str(E2BF.cam_get(cd, c)["data_path"]) for c in CAMS]
        miss = [str(p) for p in paths if not p.exists()]
        if miss:
            raise FileNotFoundError(f"{len(miss)} source jpg(s) absent, e.g. {miss[0]}")
        from PIL import Image
        with Image.open(paths[1]) as im:
            iw, ih = im.size
        rk = E2BF.rig_key(cd)
        if rk not in cache:
            cache[rk] = E2BF.build_map(cd, (ih, iw))
        grids, src, _ = cache[rk]
        out.append(E2BF.sample(paths, grids, src))
        srcs = src
        obs.append(float((src >= 0).mean()))
        rrec = rig_record(cd)
    return np.stack(out), srcs, rk, min(obs), rrec


def jobs_for(split: str, stage: str, data_root: pathlib.Path, inputs: str | None,
             stage1_sensors: str | None) -> list:
    """``(key, stage, log, frame camera dicts, sensor root)`` per scene."""
    jobs = []
    if stage in ("2", "both"):
        syn = data_root / split
        for sp in sorted((syn / "synthetic_scene_pickles").glob("*.pkl")):
            sc = E2BF.load(sp)
            m = sc["scene_metadata"]
            jobs.append((m["scene_token"], 2, m["log_name"],
                         [fr["camera_dict"] for fr in sc["frames"]], syn / "sensor_blobs"))
    if stage in ("1", "both"):
        if not inputs:
            raise SystemExit("--inputs (the export JSON) is required for stage 1")
        doc = json.load(open(inputs, encoding="utf-8"))
        osens = pathlib.Path(stage1_sensors or (data_root / "sensor_blobs" / "test"))
        logs: dict = {}
        for tok, r in sorted(doc["tokens"].items()):
            if r["stage"] != 1:
                continue
            ln = r["log_name"]
            if ln not in logs:
                fr = E2BF.load(data_root / "navsim_logs" / "test" / f"{ln}.pkl")
                logs[ln] = ({f["token"]: i for i, f in enumerate(fr)}, fr)
            idx, fr = logs[ln]
            i = idx[tok]
            hist = fr[i - 3:i + 1]
            if [f["token"] for f in hist] != r["frame_tokens"]:
                raise SystemExit(f"{tok}: log window != exported frame tokens — refusing")
            jobs.append((tok, 1, ln, [f["cams"] for f in hist], osens))
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup_two_stage")
    ap.add_argument("--stage", choices=("1", "2", "both"), default="2")
    ap.add_argument("--data-root", default="C:/Users/Admin/navsim-crun/data/openscene")
    ap.add_argument("--stage1-sensors", default=None,
                    help="root of the stage-1 ORIGINAL jpgs (navhard: the extraction root)")
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, default=1, help="history frames stored per scene (1 = t0)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frame", choices=("416x1024", "256x640"), default="416x1024")
    ap.add_argument("--verify-bank", default=None, help="256x640 only: DataFlyWheel bank")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(argv)
    frame = FRAME_416x1024 if a.frame == "416x1024" else PHYSICALAI_WIDE120_256x640
    if a.frame == "416x1024":
        assert_frame(frame)
    frec = configure(frame)
    out = pathlib.Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(a.split, a.stage, pathlib.Path(a.data_root), a.inputs, a.stage1_sensors)
    if a.limit:
        jobs = jobs[:a.limit]
    cache, prov, fails, rigs = {}, [], [], {}
    t0 = time.time()
    for n, (key, stage, log, fcams, sroot) in enumerate(jobs):
        fp = out / "frames" / f"{key}.npy"
        try:
            if a.skip_existing and fp.exists():
                arr = np.load(fp)
                src = np.load(out / "frames" / f"{key}.src.npy")
                rk = None
                obs = float((src >= 0).mean())
                rrec = rig_record(fcams[-1])
                rk = rrec["rig_key"]
            else:
                arr, src, rk, obs, rrec = build_one(fcams, sroot, cache, a.keep)
                np.save(fp, arr)
                np.save(out / "frames" / f"{key}.src.npy", src)
            rigs.setdefault(rk, rrec)
            mean_px = float(arr.mean())
            if mean_px < 1.0:
                raise ValueError(f"mean px {mean_px:.3f} — an all-black stitch")
            prov.append({"scene_token": key, "stage": stage, "log_name": log, "rig_key": rk,
                         "n_frames": int(arr.shape[0]), "observed_frac": round(obs, 6),
                         "mean_px": round(mean_px, 2),
                         "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16]})
        except Exception as e:                                          # noqa: BLE001
            fails.append({"key": key, "stage": stage,
                          "reason": f"{type(e).__name__}: {str(e)[:200]}"})
        if (n + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[frames416] {n+1}/{len(jobs)} {el/(n+1):.3f} s/scene fails={len(fails)}",
                  flush=True)
    import pandas as pd
    pd.DataFrame(prov).to_parquet(out / "frames_provenance.parquet", index=False)
    rep = {"split": a.split, "stage": a.stage, "keep": a.keep, "n_jobs": len(jobs),
           "n_built": len(prov), "n_failed": len(fails), "failures": fails[:50],
           "n_rigs": len(rigs), "rigs": rigs, "seconds": round(time.time() - t0, 1),
           "frame": {"h": frame.height, "w": frame.width, "f_ref": frame.f_ref,
                     "projection": frame.projection, **frec},
           "stitch_source": {"module": os.path.join(E2_CODE, "build_frames.py"),
                             "configured_globals": ["RAY_CAN", "H", "W"]},
           "data_root": a.data_root, "stage1_sensors": a.stage1_sensors}
    if a.verify_bank:
        if a.frame != "256x640":
            raise SystemExit("--verify-bank compares against the 256x640 DataFlyWheel bank")
        bank = pd.read_parquet(os.path.join(a.verify_bank, "frames_provenance.parquet"))
        want = dict(zip(bank.scene_token, bank.sha256))
        # the DataFlyWheel bank stores ALL 4 history frames; compare like with like
        got = {}
        for p in prov:
            if p["stage"] != 2:
                continue
            got[p["scene_token"]] = p["sha256"]
        common = sorted(set(got) & set(want))
        rep["reproduction_control_KB256"] = {
            "bank": a.verify_bank, "n_compared": len(common),
            "n_bit_exact": sum(1 for k in common if got[k] == want[k]),
            "pass": bool(common) and all(got[k] == want[k] for k in common),
            "note": "requires --keep 4 (the bank stores all 4 history frames)"}
    with open(out / "BUILD_REPORT.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("failures", "rigs")}, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
