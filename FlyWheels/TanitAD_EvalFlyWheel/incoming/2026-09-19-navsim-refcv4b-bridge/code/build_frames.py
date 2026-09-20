#!/usr/bin/env python3
"""Split-parameterised NavSim frame builder for the bridge (TANITAD VENV, CPU).

WHY: the DataFlyWheel bank covers ONLY warmup's 204 STAGE-2 synthetic scenes
(``build_navsim_eval.py:155`` globs ``synthetic_scene_pickles``). navhard_two_stage
(450 stage-1 + 5,462 stage-2) needs frames for BOTH stages, and stage 1 comes from
the original LOGS (``navsim_logs/<data_split>/<log>.pkl`` + ``sensor_blobs``), not
from scene pickles.

⛔ THE GEOMETRY IS NOT RE-DERIVED. ``distort``, ``rig_key``, ``build_map`` and
``sample`` below are VERBATIM copies of the DataFlyWheel's
``TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-28-navsim-corpus-adaptation/code/build_navsim_eval.py``
(lines 82-151; that file is a SCRIPT that builds on import and carries a stale G:
sys.path line, so it cannot be imported). The copy is proven faithful by the
REPRODUCTION CONTROL: ``--verify-bank <warmup corpus>`` rebuilds warmup stage-2
scenes and requires sha256[:16] == the bank's ``frames_provenance.parquet``
(bit-exact), before this builder may be trusted on a new split.

Keys: stage-2 frames are named by ``scene_token`` (the pickle stem — the bank's
convention); stage-1 frames by the SCORER token (frame-3 token of the log window).
Only cam_l0 + cam_f0 + cam_r0 are read; output = PHYSICALAI_WIDE120_256x640.

    python code/build_frames.py --split warmup_two_stage --stage 2 --limit 6 \
        --out <scratch> --verify-bank C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, os.path.join(REPO, "stack"))
from tanitad.data.calib import (  # noqa: E402
    PHYSICALAI_WIDE120_256x640 as FRAME, cylindrical_rays)

CAMS = ("cam_l0", "cam_f0", "cam_r0")


class _Stub:
    def __init__(self, *a, **k):
        self.__dict__.update(k)


class _Tol(pickle.Unpickler):
    """nuplan.* is not installed here; stub what we do not read (DataFlyWheel's)."""

    def find_class(self, mod, name):
        if mod == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        try:
            return super().find_class(mod, name)
        except Exception:                                   # noqa: BLE001
            return type(name, (_Stub,), {"__module__": mod})


def load(p):
    with open(p, "rb") as f:
        return _Tol(f).load()


def cam_get(d, key):
    """camera_dict keys are lower in synthetic pickles, UPPER in meta ones."""
    return d.get(key, d.get(key.upper()))


# --- VERBATIM from build_navsim_eval.py:76-151 (the backward map, per RIG) -------
XC, YC, ZC = (t.double().numpy() for t in cylindrical_rays(FRAME))
RAY_CAN = np.stack([XC, YC, ZC], -1)                        # [H,W,3]
RAY_CAN /= np.linalg.norm(RAY_CAN, axis=-1, keepdims=True)
H, W = FRAME.height, FRAME.width


def distort(xn, yn, dist):
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    r2 = xn * xn + yn * yn
    rad = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return xd, yd


def rig_key(cd):
    h = hashlib.sha256()
    for c in CAMS:
        e = cam_get(cd, c)
        for f in ("cam_intrinsic", "sensor2lidar_rotation", "sensor2lidar_translation",
                  "distortion"):
            h.update(np.asarray(e[f], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def build_map(cd, img_hw):
    """-> grids [C,H,W,2] in grid_sample coords, src [H,W] int8 (-1 = unobserved)."""
    f0 = cam_get(cd, "cam_f0")
    R_f0 = np.asarray(f0["sensor2lidar_rotation"], dtype=np.float64)
    fwd = R_f0 @ np.array([0.0, 0.0, 1.0])                  # F0 boresight in lidar
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    down /= np.linalg.norm(down)
    M = np.stack([right, down, fwd / np.linalg.norm(fwd)], axis=1)   # can -> lidar

    ray_lidar = RAY_CAN @ M.T                               # [H,W,3]
    grids, coss = [], []
    for c in CAMS:
        e = cam_get(cd, c)
        R = np.asarray(e["sensor2lidar_rotation"], dtype=np.float64)
        K = np.asarray(e["cam_intrinsic"], dtype=np.float64)
        d = np.asarray(e["distortion"], dtype=np.float64).ravel()
        r = ray_lidar @ R                                   # == (R^T @ ray)  [H,W,3]
        z = r[..., 2]
        ok = z > 1e-6
        xn = np.where(ok, r[..., 0] / np.where(ok, z, 1.0), 0.0)
        yn = np.where(ok, r[..., 1] / np.where(ok, z, 1.0), 0.0)
        xd, yd = distort(xn, yn, d)
        u = K[0, 0] * xd + K[0, 2]
        v = K[1, 1] * yd + K[1, 2]
        ih, iw = img_hw
        inb = ok & (u >= 0) & (u <= iw - 1) & (v >= 0) & (v <= ih - 1)
        cos = np.where(inb, z, -1.0)                        # ray is unit -> z == cos
        grids.append(np.stack([2 * u / (iw - 1) - 1, 2 * v / (ih - 1) - 1], -1))
        coss.append(cos)
    coss = np.stack(coss)                                   # [C,H,W]
    src = np.where(coss.max(0) > 0, coss.argmax(0), -1).astype(np.int8)
    return np.stack(grids).astype(np.float32), src, M


def sample(paths, grids, src):
    out = np.zeros((H, W, 3), np.uint8)
    for i, p in enumerate(paths):
        m = src == i
        if not m.any():
            continue
        im = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(
            2, 0, 1)[None].float()
        g = torch.from_numpy(grids[i])[None]
        s = torch.nn.functional.grid_sample(
            im, g, mode="bilinear", padding_mode="zeros", align_corners=True)
        s = s[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
        out[m] = s[m]
    return out
# --- end of the verbatim block --------------------------------------------------


def build_scene(frame_cams: list, sensor_root: pathlib.Path, cache: dict):
    """``frame_cams`` = the 4 history frames' camera dicts -> (frames u8 [4,H,W,3], src, rk, obs)."""
    stack_out, srcs, obs, rk = [], None, [], None
    for cd in frame_cams:
        paths = [sensor_root / str(cam_get(cd, c)["data_path"]) for c in CAMS]
        miss = [str(p) for p in paths if not p.exists()]
        if miss:
            raise FileNotFoundError(f"{len(miss)} source jpg(s) absent, e.g. {miss[0]}")
        ih, iw = np.asarray(Image.open(paths[1]).convert("RGB")).shape[:2]
        rk = rig_key(cd)
        if rk not in cache:
            cache[rk] = build_map(cd, (ih, iw))
        grids, src, _ = cache[rk]
        stack_out.append(sample(paths, grids, src))
        srcs = src
        obs.append(float((src >= 0).mean()))
    return np.stack(stack_out), srcs, rk, min(obs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="warmup_two_stage")
    ap.add_argument("--stage", choices=("1", "2", "both"), default="both")
    ap.add_argument("--data-root", default="C:/Users/Admin/navsim/data/openscene")
    ap.add_argument("--orig-sensors", default=None, help="default <data-root>/sensor_blobs/<data_split>")
    ap.add_argument("--inputs", default=None, help="the export JSON for this split (stage-1 cams)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-bank", default=None)
    a = ap.parse_args(argv)
    root = pathlib.Path(a.data_root)
    out = pathlib.Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    jobs = []            # (key, stage, log, frame camera dicts, sensor root)
    if a.stage in ("2", "both"):
        syn = root / a.split
        for sp in sorted((syn / "synthetic_scene_pickles").glob("*.pkl")):
            sc = load(sp)
            m = sc["scene_metadata"]
            jobs.append((m["scene_token"], 2, m["log_name"],
                         [fr["camera_dict"] for fr in sc["frames"]], syn / "sensor_blobs"))
    if a.stage in ("1", "both"):
        if not a.inputs:
            sys.exit("--inputs (the export JSON) is required for stage 1: it names the tokens")
        doc = json.load(open(a.inputs, encoding="utf-8"))
        osens = pathlib.Path(a.orig_sensors or (root / "sensor_blobs" / "test"))
        logs: dict = {}
        for tok, r in sorted(doc["tokens"].items()):
            if r["stage"] != 1:
                continue
            ln = r["log_name"]
            if ln not in logs:     # calibration lives in the LOG pickle's `cams` (UPPER keys)
                fr = load(root / "navsim_logs" / "test" / f"{ln}.pkl")
                logs[ln] = ({f["token"]: i for i, f in enumerate(fr)}, fr)
            idx, fr = logs[ln]
            i = idx[tok]
            hist = fr[i - 3:i + 1]                  # the devkit window: frames i-3..i (t0 = i)
            if [f["token"] for f in hist] != r["frame_tokens"]:
                raise SystemExit(f"{tok}: log window != exported frame tokens — refusing")
            jobs.append((tok, 1, ln, [f["cams"] for f in hist], osens))
    if a.limit:
        jobs = jobs[:a.limit]
    cache, prov, fails = {}, [], []
    t0 = time.time()
    for key, stage, log, fcams, sroot in jobs:
        try:
            arr, src, rk, obs = build_scene(fcams, sroot, cache)
            np.save(out / "frames" / f"{key}.npy", arr)
            np.save(out / "frames" / f"{key}.src.npy", src)
            prov.append({"scene_token": key, "stage": stage, "log_name": log, "rig_key": rk,
                         "n_frames": int(arr.shape[0]), "observed_frac": round(obs, 6),
                         "mean_px": round(float(arr.mean()), 2),
                         "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16]})
        except Exception as e:                              # noqa: BLE001
            fails.append({"key": key, "stage": stage, "reason": f"{type(e).__name__}: {str(e)[:160]}"})
    import pandas as pd
    pd.DataFrame(prov).to_parquet(out / "frames_provenance.parquet", index=False)
    rep = {"split": a.split, "stage": a.stage, "n_jobs": len(jobs), "n_built": len(prov),
           "n_failed": len(fails), "failures": fails[:40], "n_rigs": len(cache),
           "seconds": round(time.time() - t0, 1),
           "frame": {"h": FRAME.height, "w": FRAME.width, "f_ref": FRAME.f_ref,
                     "projection": FRAME.projection}}
    if a.verify_bank:
        bank = pd.read_parquet(os.path.join(a.verify_bank, "frames_provenance.parquet"))
        want = dict(zip(bank.scene_token, bank.sha256))
        got = {p["scene_token"]: p["sha256"] for p in prov if p["stage"] == 2}
        common = sorted(set(got) & set(want))
        rep["reproduction_control"] = {
            "bank": a.verify_bank, "n_compared": len(common),
            "n_bit_exact": sum(1 for k in common if got[k] == want[k]),
            "pass": bool(common) and all(got[k] == want[k] for k in common)}
    with open(out / "BUILD_REPORT.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "failures"}, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
