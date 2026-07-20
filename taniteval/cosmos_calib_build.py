"""Build EXACT per-ep camera calibration for the cosmos val cache (v2 rebuild).

Chain (all derived from the source dataset's own files, no global constants):
  native 1920x1080  --[x*2/3, y=(y-VCROP_TOP)*2/3]-->  generation 1280x704
  generation        --[PER-CLIP crop c @ (top,left), *256/c]--> cache 256x256

The generation inherits the SOURCE clip rig: pinhole_intrinsic.camera_front_wide_120fov
[fx fy cx cy w h] in native px + per-frame camera pose (camera-to-world, OpenCV
axes x-right/y-down/z-fwd). Extrinsic cam->vehicle = inv(vehicle_pose) @ cam_pose,
averaged over frames.

v2 (2026-07-19): the cache is now built by build_cosmos_v2.py with a PER-CLIP
principal-point-centered crop (c = round(fx_gen*256/266) ~606, f_eff ~266,
replicate-padded bottom overflow) and chunk-corrected pose pairing. The as-run
crop (c, top, left) per ep is READ FROM the cache's build_manifest.json rather
than re-derived here, so builder and calib cannot drift. The old global
constants (C=356, TOP=174, LEFT=462 — the ~1.70x-zoomed geometric-center crop
of cosmos-val-a7a8527ba14e) are gone.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

PAIRS = Path("/root/cosmos_data/pairs")
OUT = Path("/root/taniteval/results/cosmos_calib.json")
EPDIR = Path(sys.argv[1] if len(sys.argv) > 1 else
             "/root/valdata/cosmos-val-COSMOS_V2_HASH")

# native -> generation vertical crop before the 2/3 scale (1080 -> 1056 center)
VCROP_TOP = 12.0
SX = 2.0 / 3.0

WEATHER = ("Foggy", "Golden_hour", "Morning", "Night", "Rainy", "Snowy", "Sunny")


def clip_of(stem: str):
    for w in WEATHER:
        if stem.lower().endswith("_" + w.lower()):
            stem = stem[: -(len(w) + 1)]
            break
    base, chunk = stem.rsplit("_", 1)
    return base, int(chunk)


def cam_to_vehicle(cid: str):
    """Mean cam->vehicle 4x4 over frame-matched pose files."""
    pose_dir = next((PAIRS / "pose" / cid).rglob("*.pose.camera_front_wide_120fov.npy")).parent
    veh_dir = PAIRS / "vehicle_pose" / cid
    Ts = []
    for cf in sorted(pose_dir.glob("*.pose.camera_front_wide_120fov.npy")):
        fidx = cf.name.split(".")[1]
        vf = veh_dir / f"{cid}.{fidx}.vehicle_pose.npy"
        if not vf.exists():
            continue
        Tc = np.load(cf).reshape(4, 4)
        Tv = np.load(vf).reshape(4, 4)
        Ts.append(np.linalg.inv(Tv) @ Tc)
    Ts = np.stack(Ts)
    T = Ts.mean(0)
    # re-orthonormalize R via SVD
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    spread = float(np.abs(Ts - T).max())
    return R, T[:3, 3].copy(), len(Ts), spread


def main():
    man = json.loads((EPDIR / "build_manifest.json").read_text())
    eps = man["eps"]
    mp4s = sorted((PAIRS / "generation").glob("*.mp4"))
    # ep index = position in sorted mp4 list; 42/43 (e4ae6dee) excluded upstream
    table = {}
    clip_cache = {}
    for i, mp4 in enumerate(mp4s):
        base, chunk = clip_of(mp4.stem)
        m = eps.get(str(i))
        if m is None:
            table[i] = dict(mp4=mp4.name, clip=base, chunk=chunk, calib=None,
                            reason="no pinhole_intrinsic in source dataset "
                                   "(excluded from v2 cache)")
            continue
        assert m["mp4"] == mp4.name, (i, m["mp4"], mp4.name)
        if base not in clip_cache:
            clip_cache[base] = cam_to_vehicle(base)
        R, t, nfr, spread = clip_cache[base]
        fx, fy, cx, cy = m["fx"], m["fy"], m["cx"], m["cy"]
        C, TOP, LEFT = m["crop_c"], m["crop_top"], m["crop_left"]
        SCALE = 256.0 / C

        # derived canonical-frame scalars (for the report; projection uses chain)
        f_eff = fx * SX * SCALE
        cx256 = (cx * SX - LEFT) * SCALE
        cy256 = ((cy - VCROP_TOP) * SX - TOP) * SCALE
        # horizon: vehicle-forward horizontal ray (1,0,0) in cam frame
        d = R.T @ np.array([1.0, 0.0, 0.0])
        u_nat = fx * d[0] / d[2] + cx
        v_nat = fy * d[1] / d[2] + cy
        hrow = ((v_nat - VCROP_TOP) * SX - TOP) * SCALE
        hcol = (u_nat * SX - LEFT) * SCALE
        pitch_deg = math.degrees(math.asin(-R[2, 2]))  # cam z-axis vs horizontal

        table[i] = dict(
            mp4=mp4.name, clip=base, chunk=chunk,
            calib="exact:pinhole_intrinsic+pose",
            fx=fx, fy=fy, cx=cx, cy=cy, w=1920.0, h=1080.0,
            R=R.tolist(), t=t.tolist(),
            n_pose_frames=nfr, extrinsic_spread=round(spread, 5),
            vcrop_top=VCROP_TOP, sx=SX, crop_c=C, crop_top=TOP,
            crop_left=LEFT, pad_bottom=m.get("pad_bottom", 0),
            f_eff_256=round(f_eff, 2), cx_256=round(cx256, 2),
            cy_256=round(cy256, 2), horizon_row_256=round(hrow, 2),
            horizon_col_256=round(hcol, 2), cam_pitch_down_deg=round(pitch_deg, 3),
            cam_height_m=round(float(t[2]), 4), cam_fwd_m=round(float(t[0]), 4),
        )
    OUT.write_text(json.dumps(table, indent=1))
    print(f"wrote {OUT} ({len(table)} mp4 entries) for cache {EPDIR.name}")
    for i, e in table.items():
        if e.get("calib"):
            print(f"mp4#{i:02d} {e['clip'][:8]} ch{e['chunk']} f_eff={e['f_eff_256']} "
                  f"cx={e['cx_256']} cy={e['cy_256']} horizon_row={e['horizon_row_256']} "
                  f"pitch_dn={e['cam_pitch_down_deg']} h={e['cam_height_m']} "
                  f"spread={e['extrinsic_spread']}")
        else:
            print(f"mp4#{i:02d} {e['clip'][:8]} NO CALIB ({e['reason']})")


if __name__ == "__main__":
    main()
