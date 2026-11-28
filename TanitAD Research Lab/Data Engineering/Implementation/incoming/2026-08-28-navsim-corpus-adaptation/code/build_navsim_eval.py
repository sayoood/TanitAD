"""Build the NavSim warmup EVAL corpus in the TanitAD training frame.

Commission: "adapt the data to what we need ... generate an eval data set in
production grade" (PI 2026-08-28, via Master Mind).

Design decisions this implements (all pre-registered, all measured):
  * FRAME_DECISION.md  -> output = PHYSICALAI_WIDE120_256x640, the object from
    `tanitad.data.calib`, NEVER a re-declared literal. 120 deg cylindrical.
  * A single NavSim camera spans 63.7 deg, so the 120 deg frame is STITCHED
    from cam_l0 + cam_f0 + cam_r0, with the measured 5-param distortion applied
    (k1 = -0.356) before sampling.
  * EGO_SIDECAR.md -> ego state goes to a schema-distinct parquet; the frame
    stream carries frames + calib only (vision-only at inference).

Geometry chain, explicit so it can be audited:
    canonical pixel -> ray (calib.cylindrical_rays; +x right, +y down, +z fwd)
    -> lidar/ego frame via M = [right | down | fwd], fwd = CAM_F0 boresight,
       right = fwd x up_lidar  (roll removed, camera PITCH retained -- this is
       how the PhysicalAI training frames were built)
    -> source camera i via R_i^T           (R_i = sensor2lidar_rotation)
    -> pinhole K + radtan distortion       -> source pixel
    per output pixel the camera with the SMALLEST angle off its own boresight
    wins (least distortion, deterministic seams); the choice is recorded.
"""
import hashlib
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.data.calib import (  # noqa: E402
    PHYSICALAI_WIDE120_256x640 as FRAME, cylindrical_rays)

ROOT = pathlib.Path("C:/Users/Admin/navsim/data/openscene/warmup_two_stage")
BLOBS = ROOT / "sensor_blobs"
OUT = pathlib.Path("C:/Users/Admin/tanitad-wt/_s2build/navsim/corpus")
CAMS = ("cam_l0", "cam_f0", "cam_r0")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "frames").mkdir(exist_ok=True)

pathlib.PosixPath = pathlib.WindowsPath


class _Stub:
    def __init__(self, *a, **k):
        self.__dict__.update(k)


class _Tol(pickle.Unpickler):
    """nuplan.* is not installed here; stub what we do not read."""

    def find_class(self, mod, name):
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


# --- the backward map, computed once per distinct RIG ------------------------
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


# --- run ---------------------------------------------------------------------
scenes = sorted((ROOT / "synthetic_scene_pickles").glob("*.pkl"))
print(f"synthetic scenes: {len(scenes)} | frame {FRAME} | hfov {FRAME.hfov_deg:.2f}",
      flush=True)

cache: dict = {}
ego_rows, prov, fails = [], [], []
t0 = time.time()

for si, sp in enumerate(scenes):
    try:
        sc = load(sp)
        meta = sc["scene_metadata"]
        tok, log = meta["scene_token"], meta["log_name"]
        frames = sc["frames"]
        stack_out, srcs, obs = [], None, []
        for fi, fr in enumerate(frames):
            cd = fr["camera_dict"]
            paths = [BLOBS / cam_get(cd, c)["data_path"] for c in CAMS]
            miss = [str(p) for p in paths if not p.exists()]
            if miss:
                raise FileNotFoundError(f"{len(miss)} source jpg(s) absent")
            ih, iw = np.asarray(Image.open(paths[1]).convert("RGB")).shape[:2]
            rk = rig_key(cd)
            if rk not in cache:
                cache[rk] = build_map(cd, (ih, iw))
            grids, src, M = cache[rk]
            stack_out.append(sample(paths, grids, src))
            srcs = src
            obs.append(float((src >= 0).mean()))
            es = fr["ego_status"]
            ego_rows.append({
                "scene_token": tok, "log_name": log, "frame_idx": fi,
                "t_us": int(fr["timestamp"]),
                "ego_x": float(es["ego_pose"][0]), "ego_y": float(es["ego_pose"][1]),
                "ego_heading": float(es["ego_pose"][2]),
                "vx": float(es["ego_velocity"][0]), "vy": float(es["ego_velocity"][1]),
                "ax": float(es["ego_acceleration"][0]),
                "ay": float(es["ego_acceleration"][1]),
                "driving_command": int(np.argmax(es["driving_command"])),
                "source_pkl": sp.name})
        arr = np.stack(stack_out)                           # [T,H,W,3]
        np.save(OUT / "frames" / f"{tok}.npy", arr)
        np.save(OUT / "frames" / f"{tok}.src.npy", srcs)
        prov.append({"scene_token": tok, "log_name": log, "rig_key": rk,
                     "n_frames": len(frames), "observed_frac": round(min(obs), 6),
                     "mean_px": round(float(arr.mean()), 2),
                     "sha256": hashlib.sha256(arr.tobytes()).hexdigest()[:16]})
    except Exception as e:                                  # noqa: BLE001
        fails.append({"pkl": sp.name, "reason": f"{type(e).__name__}: {str(e)[:110]}"})
    if (si + 1) % 20 == 0:
        print(f"  [{si+1}/{len(scenes)}] rigs={len(cache)} fails={len(fails)} "
              f"{(time.time()-t0)/60:.1f} min", flush=True)

import pandas as pd                                          # noqa: E402
pd.DataFrame(ego_rows).to_parquet(OUT / "navsim_ego_sidecar.parquet", index=False)
pd.DataFrame(prov).to_parquet(OUT / "frames_provenance.parquet", index=False)
cid = hashlib.sha256("".join(sorted(p["scene_token"] for p in prov)).encode()
                     ).hexdigest()[:16]
json.dump({"built": "2026-08-29", "corpus_id": cid, "parity": "NON-PARITY (NavSim)",
           "n_scenes": len(prov), "n_failed": len(fails),
           "n_ego_rows": len(ego_rows), "n_rigs": len(cache),
           "frame": {"h": FRAME.height, "w": FRAME.width, "f_ref": FRAME.f_ref,
                     "projection": FRAME.projection,
                     "hfov_deg": round(FRAME.hfov_deg, 4),
                     "source": "tanitad.data.calib.PHYSICALAI_WIDE120_256x640"},
           "cameras": list(CAMS), "failures": fails[:40],
           "observed_frac_min": min((p["observed_frac"] for p in prov), default=0.0)},
          open(OUT / "BUILD.json", "w"), indent=1)
print(f"\nNAVSIM BUILD DONE: {len(prov)}/{len(scenes)} scenes, {len(fails)} failed, "
      f"{len(cache)} rigs, corpus {cid}, {(time.time()-t0)/60:.1f} min", flush=True)
